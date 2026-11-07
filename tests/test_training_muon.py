"""Muon + Adam optimizer 테스트 (issue #28)."""

import importlib.util
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from minipfn.recipes import tabpfn_v3
from minipfn.training.config import TrainConfig
from minipfn.training.loop import check_resume_step, train
from minipfn.training.muon import MuonAdam, muon_parameters
from minipfn.utils import lr_factor


def _names(model, params):
    ids = {id(p) for p in params}
    return {n for n, p in model.named_parameters() if id(p) in ids}


def test_muon_takes_block_matrices_only():
    # v2: 블록 안 Linear weight는 모두 Muon, embedding·decoder·bias는 Adam
    model = TrainConfig.smoke().build_model()
    muon, adam = muon_parameters(model)
    assert muon and all(p.ndim == 2 for p in muon)
    assert all(n.startswith("layers.") and n.endswith(".weight") for n in _names(model, muon))
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
    muon_names, adam_names = _names(model, muon), _names(model, adam)
    assert "column_blocks.0.summarize.qkv.weight" in muon_names
    assert {"cls_tokens", "column_blocks.0.inducing"} <= adam_names
    assert "column_blocks.0.summarize.qassmax.base_mlp.0.weight" in adam_names
    assert not any(n.startswith(("cell_encoder", "cell_target", "icl_target")) for n in muon_names)


def test_muon_adam_is_muon_plus_adam():
    # 묶은 optimizer는 같은 파라미터를 torch.optim.Muon·Adam으로 따로 학습한 것과 비트 단위로 같다
    # (scheduler가 바꾼 lr도 두 묶음에 들어간다)
    torch.manual_seed(0)
    model = TrainConfig.smoke().build_model()
    ref = TrainConfig.smoke().build_model()
    ref.load_state_dict(model.state_dict())
    opt = MuonAdam(model, lr=1e-3, weight_decay=0.01)
    ref_muon, ref_adam = muon_parameters(ref)
    ref_opts = [
        torch.optim.Muon(
            ref_muon, lr=1e-3, weight_decay=0.01, momentum=0.95, adjust_lr_fn="match_rms_adamw"
        ),
        torch.optim.Adam(ref_adam, lr=1e-3),
    ]
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(s, 2, 6))
    ref_scheds = [
        torch.optim.lr_scheduler.LambdaLR(o, lambda s: lr_factor(s, 2, 6)) for o in ref_opts
    ]
    for _ in range(6):
        for p, q in zip(model.parameters(), ref.parameters(), strict=True):
            g = torch.randn_like(p)
            p.grad, q.grad = g.clone(), g.clone()
        opt.step()
        sched.step()
        for o, s in zip(ref_opts, ref_scheds, strict=True):
            o.step()
            s.step()
        opt.zero_grad()
    for (name, p), q in zip(model.named_parameters(), ref.parameters(), strict=True):
        assert torch.equal(p, q), name
    assert [g["lr"] for g in opt.param_groups] == [o.param_groups[0]["lr"] for o in ref_opts]


def test_muon_adam_state_round_trip():
    torch.manual_seed(0)
    model = TrainConfig.smoke().build_model()
    opt = MuonAdam(model, lr=1e-3)
    for p in model.parameters():
        p.grad = torch.randn_like(p)
    opt.step()
    state = opt.state_dict()
    other = MuonAdam(model, lr=5e-4)
    other.load_state_dict(state)
    assert [g["lr"] for g in other.param_groups] == [1e-3, 1e-3]
    assert other.param_groups[0] is other.muon.param_groups[0]  # scheduler가 바꿀 group
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
