"""Muon + Adam optimizer 테스트 (issue #28)."""

import importlib.util
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn

from minipfn.recipes import tabpfn_v3
from minipfn.training.config import TrainConfig
from minipfn.training.loop import check_resume_step, train
from minipfn.training.muon import MuonAdam, _newton_schulz, muon_parameters
from minipfn.utils import lr_factor


def _names(model, params):
    ids = {id(p) for p in params}
    return {n for n, p in model.named_parameters() if id(p) in ids}


def test_muon_takes_block_matrices_only():
    # v2: 블록 안 Linear weight는 모두 Muon, embedding·decoder·bias는 Adam. qkv는 셋으로 나눈다
    model = TrainConfig.smoke().build_model()
    pairs, adam = muon_parameters(model)
    muon = [p for p, _ in pairs]
    assert muon and all(p.ndim == 2 for p in muon)
    assert all(n.startswith("layers.") and n.endswith(".weight") for n in _names(model, muon))
    split = {n: s for n, p in model.named_parameters() for q, s in pairs if p is q}
    assert split["layers.0.sample_attention.qkv.weight"] == 3
    assert split["layers.0.sample_attention.out.weight"] == 1
    assert split["layers.0.mlp.0.weight"] == 1
    assert not any(
        n.startswith("layers.") and p.ndim == 2
        for n, p in model.named_parameters()
        if id(p) in {id(q) for q in adam}
    )
    assert len(muon) + len(adam) == len(list(model.parameters()))
    # v3.5: 학습하는 토큰(inducing·CLS)과 스칼라를 펴는 층(입력 1개)은 Adam
    v35 = tabpfn_v3.with_architecture(TrainConfig.smoke("multitask"), "v3.5", keep_size=True)
    model = v35.build_model()
    muon, adam = muon_parameters(model)
    muon_names, adam_names = _names(model, [p for p, _ in muon]), _names(model, adam)
    assert "column_blocks.0.summarize.qkv.weight" in muon_names
    assert {"cls_tokens", "column_blocks.0.inducing"} <= adam_names
    assert "column_blocks.0.summarize.qassmax.base_mlp.0.weight" in adam_names
    assert not any(n.startswith(("cell_encoder", "cell_target", "icl_target")) for n in muon_names)


def test_muon_adam_is_muon_plus_adam():
    # 묶은 optimizer는 같은 파라미터를 torch.optim.Muon·Adam으로 따로 학습한 것과 비트 단위로 같다.
    # qkv는 q·k·v 조각을 따로 둔 torch.optim.Muon과 같다 (scheduler가 바꾼 lr도 모든 묶음에
    # 들어간다)
    torch.manual_seed(0)
    model = TrainConfig.smoke().build_model()
    opt = MuonAdam(model, lr=1e-3, weight_decay=0.01)
    pairs, adam = muon_parameters(model)
    pieces = {  # 모델 파라미터 → 따로 학습할 조각들
        p: [nn.Parameter(w.clone()) for w in p.detach().chunk(s)] for p, s in pairs
    }
    pieces.update({p: [nn.Parameter(p.detach().clone())] for p in adam})
    ref_opts = [
        torch.optim.Muon(
            [w for p, _ in pairs for w in pieces[p]],
            lr=1e-3,
            weight_decay=0.01,
            momentum=0.95,
            adjust_lr_fn="match_rms_adamw",
        ),
        torch.optim.Adam([pieces[p][0] for p in adam], lr=1e-3),
    ]
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(s, 2, 6))
    ref_scheds = [
        torch.optim.lr_scheduler.LambdaLR(o, lambda s: lr_factor(s, 2, 6)) for o in ref_opts
    ]
    for _ in range(6):
        for p, ws in pieces.items():
            p.grad = torch.randn_like(p)
            for w, g in zip(ws, p.grad.chunk(len(ws)), strict=True):
                w.grad = g.clone()
        opt.step()
        sched.step()
        for o, s in zip(ref_opts, ref_scheds, strict=True):
            o.step()
            s.step()
        opt.zero_grad()
    for name, p in model.named_parameters():
        assert torch.equal(p, torch.cat(list(pieces[p]))), name
    assert {g["lr"] for g in opt.param_groups} == {o.param_groups[0]["lr"] for o in ref_opts}


def test_qkv_is_orthogonalized_per_block():
    # 조각마다 직교 행렬 쪽으로 간다: 묶어서 하면 각 조각의 특이값이 1보다 한참 작다
    torch.manual_seed(0)
    g = torch.randn(3 * 16, 16)
    split = torch.cat([_newton_schulz(u) for u in g.split(16)]).float()
    joint = _newton_schulz(g).float()
    for i in range(3):
        block = slice(16 * i, 16 * (i + 1))
        assert torch.linalg.svdvals(split[block]).min() > 0.5
        assert torch.linalg.svdvals(joint[block]).mean() < 0.75


def test_muon_takes_multihead_attention_projections():
    # nn.MultiheadAttention의 입력 projection(q·k·v 묶음)도 out_proj처럼 Muon이 맡는다
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Linear(1, 8)
            self.block = nn.MultiheadAttention(8, 2, batch_first=True)

        def repeated_blocks(self):
            return [self.block]

    model = Model()
    muon, adam = muon_parameters(model)
    assert [(n, s) for n, p in model.named_parameters() for q, s in muon if p is q] == [
        ("block.in_proj_weight", 3),
        ("block.out_proj.weight", 1),
    ]
    assert {"embed.weight", "block.in_proj_bias"} <= _names(model, adam)


def test_muon_adam_state_round_trip():
    torch.manual_seed(0)
    model = TrainConfig.smoke().build_model()
    opt = MuonAdam(model, lr=1e-3)
    for p in model.parameters():
        p.grad = torch.randn_like(p)
    opt.step()
    assert len(opt.state) == len(list(model.parameters()))
    state = opt.state_dict()
    # 이어 학습할 때 weight decay는 지금 설정을 쓴다 (lr은 학습 루프가 맞춘다)
    other = MuonAdam(model, lr=5e-4, weight_decay=0.05)
    other.load_state_dict(state)
    assert [g["lr"] for g in other.param_groups] == [1e-3] * len(other.param_groups)
    assert [g["weight_decay"] for g in other.muon.param_groups] == [0.05, 0.05]
    assert other.param_groups[0] is other.muon.param_groups[0]  # scheduler가 바꿀 group
    with pytest.raises(NotImplementedError):
        other.add_param_group({"params": []})
    with pytest.raises(ValueError, match="MuonAdam"):
        other.load_state_dict(torch.optim.Adam(model.parameters()).state_dict())


def _smoke(**changes):
    return replace(
        TrainConfig.smoke("classification"),
        steps=6,
        val_every=3,
        log_every=1,
        checkpoint_every=3,
        keep_every=None,
        collapse_threshold=None,
        **changes,
    )


def test_muon_trains_and_resumes_like_one_run(tmp_path):
    adam = train(_smoke(), tmp_path / "adam", mlflow_log=False)
    full = train(_smoke(optimizer="muon"), tmp_path / "full", mlflow_log=False)
    assert all(torch.isfinite(torch.tensor(full["history"]["train_loss"])))
    assert full["history"]["train_loss"] != adam["history"]["train_loss"]
    state = torch.load(tmp_path / "full" / "latest.pt", weights_only=True)
    assert state["optimizer_type"] == "muon" and set(state["optimizer"]) == {"muon", "adam"}
    # step 3에서 끊었다 잇는다. 이어 갈 때 데이터 시드가 바뀌어(Adam도 같다) 손실은 step 3까지만
    # 한 번에 학습한 것과 같다
    train(_smoke(optimizer="muon", stop_step=3), tmp_path / "a", mlflow_log=False)
    resumed = train(
        _smoke(optimizer="muon"),
        tmp_path / "b",
        resume_from=tmp_path / "a" / "latest.pt",
        mlflow_log=False,
    )
    assert resumed["history"]["step"] == list(range(1, 7))
    assert resumed["history"]["train_loss"][:3] == full["history"]["train_loss"][:3]
    assert all(torch.isfinite(torch.tensor(resumed["history"]["train_loss"])))


def test_resume_needs_the_same_optimizer(tmp_path):
    train(_smoke(optimizer="muon", stop_step=3), tmp_path / "muon", mlflow_log=False)
    muon_state = torch.load(tmp_path / "muon" / "latest.pt", weights_only=True)
    with pytest.raises(ValueError, match="optimizer=muon"):
        check_resume_step(muon_state, _smoke())
    train(_smoke(stop_step=3), tmp_path / "adam", mlflow_log=False)
    adam_state = torch.load(tmp_path / "adam" / "latest.pt", weights_only=True)
    with pytest.raises(ValueError, match="optimizer=adam"):
        check_resume_step(adam_state, _smoke(optimizer="muon"))
    # 이 키가 생기기 전의 checkpoint는 Adam이다
    del adam_state["optimizer_type"]
    check_resume_step(adam_state, _smoke())


def test_muon_trains_the_multitask_model(tmp_path):
    # task별 출력층은 두 step에 한 번 gradient를 받는다(Adam 묶음, gradient가 없으면 건너뛴다)
    config = replace(
        tabpfn_v3.with_architecture(TrainConfig.smoke("multitask"), "v3.5", keep_size=True),
        steps=4,
        val_every=2,
        log_every=1,
        optimizer="muon",
    )
    result = train(config, tmp_path, mlflow_log=False)
    assert all(torch.isfinite(torch.tensor(result["history"]["train_loss"])))


def test_muon_with_qk_clip(tmp_path):
    # QK-clip은 optimizer step 뒤에 Wq·Wk를 줄인다 (Kimi K2의 MuonClip과 같은 조합)
    result = train(
        _smoke(optimizer="muon", qk_clip=1.0, qk_clip_every=1), tmp_path, mlflow_log=False
    )
    assert all(torch.isfinite(torch.tensor(result["history"]["train_loss"])))


def test_config_checks():
    assert TrainConfig().optimizer == "adam" and TrainConfig().muon_weight_decay == 0.0
    with pytest.raises(ValueError, match="optimizer"):
        TrainConfig(optimizer="sgd")
    with pytest.raises(ValueError, match="schedule_free"):
        TrainConfig(optimizer="muon", schedule_free=True)
    with pytest.raises(ValueError, match="muon_weight_decay"):
        TrainConfig(muon_weight_decay=0.1)  # Adam이면 쓰지 않는다
    TrainConfig(optimizer="muon", muon_weight_decay=0.1, fused_adam=True)


def _script(name):
    path = Path(__file__).parents[1] / "scripts" / "tabpfn_v2" / "train.py"
    spec = importlib.util.spec_from_file_location(name, path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    return script


def test_train_script_takes_the_optimizer(monkeypatch):
    from minipfn.hub import run_architecture

    for name in ("TABPFN_V2_OPTIMIZER", "TABPFN_V2_MUON_WEIGHT_DECAY", "TABPFN_V2_CONFIG_NAME"):
        monkeypatch.delenv(name, raising=False)
    script = _script("train_script_muon")
    config, _ = script.build_config(script.parse_args(["--config-name", "smoke"]))
    assert config.optimizer == "adam"
    monkeypatch.setenv("TABPFN_V2_OPTIMIZER", "muon")
    monkeypatch.setenv("TABPFN_V2_MUON_WEIGHT_DECAY", "0.05")
    config, _ = script.build_config(script.parse_args(["--config-name", "smoke"]))
    assert config.optimizer == "muon" and config.muon_weight_decay == 0.05
    monkeypatch.delenv("TABPFN_V2_OPTIMIZER")
    monkeypatch.delenv("TABPFN_V2_MUON_WEIGHT_DECAY")
    # 이어 학습하면 원래 run의 optimizer를 따른다
    options = script.source_speed_options({"optimizer": "muon", "muon_weight_decay": "0.0"})
    assert options == {"optimizer": "muon", "muon_weight_decay": 0.0}
    source = run_architecture({k: str(v) for k, v in vars(TrainConfig.smoke()).items()})
    args = script.parse_args(["--config-name", "smoke", "--resume-run-id", "r"])
    config, _ = script.build_config(args, source, options)
    assert config.optimizer == "muon"


def test_train_script_switches_or_keeps_the_source_optimizer(monkeypatch):
    # 원래 run의 optimizer 설정은 한 묶음: 같은 종류면 이어 쓰고(값만 인자로 바꾼다), 다른 종류를
    # 주면 통째로 버린다. 설정 검사는 원래 run의 옵션과 인자를 합친 마지막 설정만 본다
    from minipfn.hub import run_architecture

    for name in (
        "TABPFN_V2_OPTIMIZER",
        "TABPFN_V2_MUON_WEIGHT_DECAY",
        "TABPFN_V2_SCHEDULE_FREE",
        "TABPFN_V2_CONFIG_NAME",
    ):
        monkeypatch.delenv(name, raising=False)
    script = _script("train_script_muon_switch")
    source = run_architecture({k: str(v) for k, v in vars(TrainConfig.smoke()).items()})
    muon = {"optimizer": "muon", "muon_weight_decay": 0.05, "schedule_free": False}

    def build(*flags, options=muon):
        args = script.parse_args(["--config-name", "smoke", "--init-run-id", "r", *flags])
        return script.build_config(args, source, options)[0]

    config = build("--muon-weight-decay", "0.1")  # --optimizer muon 없이 값만 바꾼다
    assert config.optimizer == "muon" and config.muon_weight_decay == 0.1
    config = build("--optimizer", "muon")  # 같은 종류: 원래 weight decay를 이어 쓴다
    assert config.muon_weight_decay == 0.05
    config = build("--optimizer", "adam")  # 다른 종류: 원래 weight decay를 버린다
    assert config.optimizer == "adam" and config.muon_weight_decay == 0.0
    schedule_free = {"optimizer": "adam", "schedule_free": True}
    config = build("--optimizer", "muon", options=schedule_free)
    assert config.optimizer == "muon" and not config.schedule_free
    assert build(options=schedule_free).schedule_free
    config = build("--schedule-free")  # muon run에서 schedule-free로
    assert config.schedule_free and config.optimizer == "adam"
    with pytest.raises(SystemExit, match="schedule_free"):
        build("--schedule-free", "--optimizer", "muon")
