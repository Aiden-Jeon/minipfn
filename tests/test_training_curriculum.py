"""TabICLv2 3단계 학습을 옮긴 설정 (`recipes.tabpfn_v3.curriculum`)과 그 부품: 모든 파라미터 Muon,
표 크기 규칙 "train_fraction", 클래스 수의 binary 비율, prior filter."""

import importlib.util
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from minipfn.priors import graph_scm as P
from minipfn.priors.generators import graph_scm as G
from minipfn.priors.hyperparameters import ClassCount
from minipfn.priors.pipeline import Rng
from minipfn.priors.table_size import long_table_feature_cap, sample_split_size
from minipfn.recipes import tabpfn_v3
from minipfn.training.config import TrainConfig
from minipfn.training.loop import train
from minipfn.training.muon import MuonAdam, muon_parameters

# ---------------------------------------------------------------------------
# 표 크기
# ---------------------------------------------------------------------------


def _split(rng, **changes):
    kw = dict(
        min_rows=1024,
        max_rows=1024,
        min_train_fraction=0.3,
        max_train_fraction=0.9,
        max_features=100,
        max_cells=1024 * 100,
    )
    return sample_split_size(rng, **{**kw, **changes})


def test_stage1_sizes():
    # 행 1,024개 고정, 학습 30~90%, feature U(1, 100)
    rng = np.random.default_rng(0)
    sizes = [_split(rng) for _ in range(2000)]
    assert all(n_train + n_eval == 1024 for n_train, n_eval, _ in sizes)
    train_frac = np.array([n_train for n_train, _, _ in sizes]) / 1024
    # int(1024 × 0.3) = 307이라 하한은 0.2998
    assert 0.299 <= train_frac.min() < 0.31 and 0.89 < train_frac.max() < 0.9
    features = np.array([f for _, _, f in sizes])
    assert features.min() == 1 and features.max() == 100
    assert abs(features.mean() - 50.5) < 2


def test_long_table_sizes():
    # stage 3: 400~60,000행 log 균등, 학습 79~81%, 행이 많으면 feature 상한을 줄인다
    rng = np.random.default_rng(1)
    rows, features = [], []
    for _ in range(3000):
        n_train, n_eval, n_features = _split(
            rng,
            min_rows=400,
            max_rows=60_000,
            min_train_fraction=0.79,
            max_train_fraction=0.81,
            log_uniform_rows=True,
            max_cells=60_000 * 100,
        )
        n_rows = n_train + n_eval
        assert 0.78 <= n_train / n_rows <= 0.81
        assert n_features <= long_table_feature_cap(n_rows, 100)
        rows.append(n_rows)
        features.append(n_features)
    rows = np.array(rows)
    assert 400 <= rows.min() < 450 and 50_000 < rows.max() <= 60_000
    # log 균등: 중앙값이 기하평균 근처
    assert abs(np.log(np.median(rows)) - np.log(np.sqrt(400 * 60_000))) < 0.15
    assert max(f for r, f in zip(rows, features, strict=True) if r > 50_000) <= 20
    assert long_table_feature_cap(10_240, 100) == 100
    assert long_table_feature_cap(10_241, 100) == 80
    assert long_table_feature_cap(60_000, 10) == 10
    assert long_table_feature_cap(100_000, 100) == 10


def test_split_size_respects_cells_and_minimum():
    rng = np.random.default_rng(2)
    for _ in range(200):
        n_train, n_eval, n_features = _split(rng, max_cells=1024 * 7, min_train_rows=400)
        assert n_features <= 7 and n_train >= 400 and n_eval >= 1


# ---------------------------------------------------------------------------
# 클래스 수
# ---------------------------------------------------------------------------


def test_class_count_binary_share():
    # binary_prob 0이면 예전과 같은 난수열, 0.5면 binary가 0.5 + 0.5/9
    old, new = Rng.from_seed(0), Rng.from_seed(0)
    a = [ClassCount(10)(old, 100, 5)["num_classes"] for _ in range(200)]
    b = [ClassCount(10, binary_prob=0.0)(new, 100, 5)["num_classes"] for _ in range(200)]
    assert a == b
    rng = Rng.from_seed(1)
    counts = np.array([ClassCount(10, 0.5)(rng, 100, 5)["num_classes"] for _ in range(20_000)])
    assert abs((counts == 2).mean() - (0.5 + 0.5 / 9)) < 0.015
    assert counts.max() == 10
    with pytest.raises(ValueError, match="binary_prob"):
        ClassCount(10, 1.5)


# ---------------------------------------------------------------------------
# prior
# ---------------------------------------------------------------------------


def _prior(task="classification", **changes):
    kw = dict(size_rule="train_fraction", min_rows=60, max_rows=60, max_features=6, max_cells=600)
    return P.GraphSCMPrior(task, seed=0, **{**kw, **changes})


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_train_fraction_prior_batches(task):
    prior = _prior(task, min_rows=40, max_rows=80, log_uniform_rows=True)
    seen = set()
    for _ in range(6):
        batch = prior.sample_batch(3)  # x (데이터셋, 행, feature)
        n_rows, n_features = batch.x.shape[1], batch.x.shape[2]
        assert 40 <= n_rows < 80 and 1 <= n_features <= 6
        assert 0 < batch.n_train < n_rows
        seen.add(n_rows - batch.n_train)
    assert len(seen) > 1  # 검증 행 수가 batch마다 다르다
    # 크기를 직접 주면 (학습 행, feature[, 검증 행])
    assert prior.sample_batch(2, size=(30, 4, 10)).x.shape == (2, 40, 4)
    assert prior.sample_batch(2, size=(30, 4)).x.shape == (2, 30 + prior.n_eval_rows, 4)
    with pytest.raises(ValueError, match="sample_shape"):
        prior.sample_size()


def test_size_rule_checks():
    with pytest.raises(ValueError, match="size_rule"):
        _prior(size_rule="rows")
    with pytest.raises(ValueError, match="학습 행 비율"):
        _prior(min_train_fraction=0.9, max_train_fraction=0.3)
    with pytest.raises(ValueError, match="max_cells"):
        _prior(max_cells=10)
    with pytest.raises(ValueError, match="size_rule"):
        TrainConfig(size_rule="rows")
    # 기본 규칙(eval_rows)에서는 새 필드를 보지 않는다
    TrainConfig(min_rows=10_000_000)


def test_prior_filter_retries_unpredictable_datasets(monkeypatch):
    # graph·dataset filter를 켜면 TabICL PriorConfig에 넘기고, dataset filter가 계속 걸러도 데이터셋
    # 하나에 FILTER_ATTEMPTS번까지만 다시 만든다 (상한을 줄여 본다)
    prior = _prior(filter_unpredictable=True)
    generator = prior.pipeline.generator
    assert generator._config.filter_unpredictable_graphs
    assert generator._config.filter_unpredictable_datasets
    calls = []

    def always(x, y, config, *, is_classif):
        calls.append(is_classif)
        return True

    monkeypatch.setattr(generator, "_should_filter", always)
    monkeypatch.setattr(G, "FILTER_ATTEMPTS", 6)
    batch = prior.sample_batch(2)
    assert batch.x.shape[0] == 2
    assert len(calls) >= 2 * 6 and all(calls)
    assert len(calls) % 6 == 0  # 데이터셋마다 정확히 상한만큼 (그 뒤에는 받는다)
    off = _prior()
    assert not off.pipeline.generator._config.filter_unpredictable_datasets


def test_prior_filter_keeps_predictable_datasets(monkeypatch):
    prior = _prior("regression", filter_unpredictable=True)
    generator = prior.pipeline.generator
    calls = []
    monkeypatch.setattr(
        generator, "_should_filter", lambda *a, **k: calls.append(1) or False
    )  # 받기만 한다
    prior.sample_batch(3)
    assert len(calls) == 3


def test_real_filter_runs():
    # 실제 ExtraTrees filter로도 데이터셋이 나온다 (회귀 구간 경계도 같은 filter를 쓴다)
    prior = _prior("classification", filter_unpredictable=True)
    batch = prior.sample_batch(2)
    assert torch.isfinite(batch.x).all() or torch.isnan(batch.x).any()
    borders = P.regression_borders(
        20,
        8,
        filter_unpredictable=True,
        size_rule="train_fraction",
        min_rows=60,
        max_rows=60,
        max_features=6,
        max_cells=600,
    )
    assert borders.shape == (21,) and (borders.diff() > 0).all()


# ---------------------------------------------------------------------------
# 모든 파라미터 Muon
# ---------------------------------------------------------------------------


class _Toy(nn.Module):
    # 2차원 행렬, 1차원 bias·gain, 3차원 토큰 (TabICL Muon과 견줄 수 있게 0차원은 없다)
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(6, 5)
        self.gain = nn.Parameter(torch.ones(5))
        self.tokens = nn.Parameter(torch.randn(3, 2, 4))


def test_all_scope_takes_every_parameter():
    config = tabpfn_v3.with_architecture(TrainConfig.smoke("multitask"), "v3.5", keep_size=True)
    model = config.build_model()
    muon, adam = muon_parameters(model, "all")
    assert adam == [] and len(muon) == len(list(model.parameters()))
    blocks, _ = muon_parameters(model)
    splits = dict((id(p), s) for p, s in muon)
    assert all(splits[id(p)] == s for p, s in blocks)  # q·k·v는 지금처럼 셋으로 나눈다
    assert any(p.ndim == 1 for p, _ in muon)
    opt = MuonAdam(model, lr=1e-2, weight_decay=0.01, scope="all", momentum=0.9)
    assert opt.adam is None and all(g["momentum"] == 0.9 for g in opt.param_groups)
    with pytest.raises(ValueError, match="Muon 범위"):
        muon_parameters(model, "some")


def test_all_scope_matches_tabicl_muon():
    # TabICLv2 학습 코드의 Muon(모든 파라미터, momentum 0.9 = --beta1, Nesterov, match_rms 0.2,
    # weight decay 0.01)과 같은 갱신이다. momentum을 합으로 두느냐 평균으로 두느냐는 직교화 전에
    # 크기로 나누므로 결과가 같다(bf16 반올림 차이만)
    from tabicl.train._muon import Muon as TabICLMuon

    torch.manual_seed(0)
    ours, theirs = _Toy(), _Toy()
    theirs.load_state_dict(ours.state_dict())
    start = {n: p.detach().clone() for n, p in ours.named_parameters()}
    opt = MuonAdam(ours, lr=1e-2, weight_decay=0.01, scope="all", momentum=0.9)
    ref = TabICLMuon(
        param_groups=[dict(params=list(theirs.parameters()), use_muon=True)],
        lr=1e-2,
        weight_decay=0.01,
        matched_adamw_rms=0.2,
        momentum=0.9,
        nesterov=True,
        ns_steps=5,
    )
    for step in range(4):
        gen = torch.Generator().manual_seed(step)
        for a, b in zip(ours.parameters(), theirs.parameters(), strict=True):
            a.grad = torch.randn(a.shape, generator=gen)
            b.grad = a.grad.clone()
        opt.step()
        ref.step()
    for (name, a), b in zip(ours.named_parameters(), theirs.parameters(), strict=True):
        # 4 step 동안 움직인 양에 견줘 bf16 직교화 차이만큼(몇 %)만 다르다
        moved = (a - start[name]).abs().max()
        assert (a - b).abs().max() < 0.05 * moved, name


def test_scalar_parameter_and_state_round_trip():
    class Scalar(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(3, 3)
            self.scale = nn.Parameter(torch.tensor(1.0))

    model = Scalar()
    opt = MuonAdam(model, lr=0.1, scope="all")
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    before = model.scale.item()
    opt.step()
    # (1, 1)을 직교화하면 부호만 남고, lr 보정 0.2·sqrt(1)
    assert model.scale.item() == pytest.approx(before - 0.1 * 0.2, rel=1e-2)
    state = opt.state_dict()
    assert state["adam"] is None
    again = MuonAdam(model, lr=0.1, scope="all")
    again.load_state_dict(state)
    # 블록 범위 optimizer에 모든 파라미터 Muon 상태를 넣으면 멈춘다
    config = tabpfn_v3.with_architecture(TrainConfig.smoke("multitask"), "v3.5", keep_size=True)
    with pytest.raises(ValueError, match="muon_scope"):
        MuonAdam(config.build_model(), lr=0.1).load_state_dict(state)


def test_config_checks():
    assert TrainConfig().muon_scope == "blocks" and TrainConfig().muon_momentum == 0.95
    with pytest.raises(ValueError, match="muon_scope"):
        TrainConfig(muon_scope="all")  # Adam이면 쓰지 않는다
    with pytest.raises(ValueError, match="muon_scope"):
        TrainConfig(optimizer="muon", muon_scope="layers")
    with pytest.raises(ValueError, match="muon_momentum"):
        TrainConfig(optimizer="muon", muon_momentum=1.0)
    with pytest.raises(ValueError, match="binary_prob"):
        TrainConfig(binary_prob=-0.1)


# ---------------------------------------------------------------------------
# recipe와 학습
# ---------------------------------------------------------------------------


def test_curriculum_recipe():
    s1, s2, s3 = (tabpfn_v3.curriculum(n) for n in (1, 2, 3))
    for c in (s1, s2, s3):
        assert c.task == "multitask" and c.optimizer == "muon" and c.muon_scope == "all"
        assert c.muon_momentum == 0.9 and c.muon_weight_decay == 0.01 and c.batch_size == 256
        assert c.size_rule == "train_fraction" and c.max_features == 100
        assert c.binary_prob == 0.5 and c.filter_unpredictable and c.phases == ()
        assert c.warmup_steps == c.steps // 100
        assert c.max_cells == c.max_rows * 100
        assert tabpfn_v3.architecture_name(c) == "v3.5"
    assert (s1.steps, s1.lr, s1.grad_clip, s1.micro_batch) == (250_000, 4e-4, 10.0, 8)
    assert (s1.min_rows, s1.max_rows, s1.min_train_fraction, s1.max_train_fraction) == (
        1024,
        1024,
        0.3,
        0.9,
    )
    assert (s2.steps, s2.lr, s2.grad_clip, s2.max_rows, s2.log_uniform_rows) == (
        20_000,
        5e-5,
        10.0,
        10_240,
        True,
    )
    assert (s3.steps, s3.lr, s3.grad_clip, s3.max_rows) == (5_000, 1e-5, 1.0, 60_000)
    assert all(c.logit_softcap == 50.0 for c in (s1, s2, s3))
    # task당 데이터셋 수가 TabICLv2(500K step × batch 64)와 같다
    assert s1.steps * s1.batch_size // 2 == 500_000 * 64
    assert s2.steps * s2.batch_size // 2 == 40_000 * 64
    assert s3.steps * s3.batch_size // 2 == 10_000 * 64
    with pytest.raises(ValueError, match="v3.5 multitask"):
        tabpfn_v3.curriculum(1, "classification")
    with pytest.raises(ValueError, match="stage"):
        tabpfn_v3.curriculum(4)


def _smoke_curriculum(**changes):
    # 작은 크기로 stage 1의 코드 경로 전부: 모든 파라미터 Muon, train_fraction, binary, filter
    config = tabpfn_v3.with_architecture(TrainConfig.smoke("multitask"), "v3.5", keep_size=True)
    stage = {k: v for k, v in tabpfn_v3.CURRICULUM_COMMON.items() if k != "batch_size"}
    small = dict(
        max_features=8,
        min_rows=48,
        max_rows=80,
        log_uniform_rows=True,
        max_cells=80 * 8,
        steps=4,
        warmup_steps=1,
        val_every=2,
        log_every=1,
        checkpoint_every=2,
        keep_every=None,
        collapse_threshold=None,
        border_datasets=20,
    )
    return replace(config, **{**stage, **small, **changes})  # 인자로 준 값이 recipe 값을 덮는다


def test_curriculum_trains_and_resumes(tmp_path, monkeypatch):
    # 회귀 구간 경계를 process pool로 만들어서, 그 process들의 문자열 hash 시드를 고정해야 run끼리
    # 경계가 같다 (TabICL prior, `generators.graph_scm` 주석)
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    full = train(_smoke_curriculum(), tmp_path / "full", mlflow_log=False)
    assert all(np.isfinite(full["history"]["train_loss"]))
    state = torch.load(tmp_path / "full" / "latest.pt", weights_only=True)
    assert state["optimizer"]["adam"] is None  # 모든 파라미터가 Muon
    # 이어 갈 때 데이터 시드가 바뀌므로 손실은 끊은 step까지만 같다 (test_training_muon과 같다)
    train(_smoke_curriculum(stop_step=2), tmp_path / "a", mlflow_log=False)
    resumed = train(
        _smoke_curriculum(),
        tmp_path / "b",
        resume_from=tmp_path / "a" / "latest.pt",
        mlflow_log=False,
    )
    assert resumed["history"]["step"] == list(range(1, 5))
    assert resumed["history"]["train_loss"][:2] == full["history"]["train_loss"][:2]
    assert all(np.isfinite(resumed["history"]["train_loss"]))


def _script(name):
    path = Path(__file__).parents[1] / "scripts" / "tabpfn_v2" / "train.py"
    spec = importlib.util.spec_from_file_location(name, path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    return script


def test_train_script_stages(monkeypatch):
    for name in ("TABPFN_V2_CONFIG_NAME", "TABPFN_V2_OPTIMIZER", "TABPFN_V2_INIT_RUN_ID"):
        monkeypatch.delenv(name, raising=False)
    script = _script("train_script_curriculum")
    args = ["--config-name", "stage1", "--arch", "v3.5", "--task", "multitask"]
    config, arch = script.build_config(script.parse_args(args))
    assert arch == "v3.5" and config == tabpfn_v3.curriculum(1)
    with pytest.raises(SystemExit):  # 앞 단계 가중치 없이 stage 2는 시작하지 않는다
        script.parse_args(["--config-name", "stage2", "--arch", "v3.5", "--task", "multitask"])
    with pytest.raises(SystemExit, match="v3.5 multitask"):
        script.build_config(script.parse_args(["--config-name", "stage1", "--arch", "v3.5"]))
    # 이어 학습하면 Muon 범위·momentum을 원래 run에서 잇는다
    options = script.source_speed_options({"muon_scope": "all", "muon_momentum": "0.9"})
    assert options == {"muon_scope": "all", "muon_momentum": 0.9}
    # soft-cap은 모델 구조 param이 아니라 여기서 잇는다. 끈 run("None")은 preset 값을 두고 간다
    assert script.source_speed_options({"logit_softcap": "50.0"}) == {"logit_softcap": 50.0}
    assert script.source_speed_options({"logit_softcap": "None"}) == {}


def test_logit_softcap():
    # 끄면 SDPA와 같고, 아주 큰 c면 거의 같으며, 작은 c면 attention이 덜 날카롭다
    from minipfn.models import config_dict
    from minipfn.models.column_row_transformer import Attention

    torch.manual_seed(0)
    x = torch.randn(2, 7, 16) * 30
    base = Attention(16, 2, qk_norm=True, zero_out=False)
    loose = Attention(16, 2, qk_norm=True, zero_out=False, logit_softcap=1e6)
    tight = Attention(16, 2, qk_norm=True, zero_out=False, logit_softcap=0.01)
    for m in (loose, tight):
        m.load_state_dict(base.state_dict())
    with torch.no_grad():
        ref, near, flat = base(x, n_context=5), loose(x, n_context=5), tight(x, n_context=5)
    assert torch.allclose(ref, near, atol=1e-4)
    # c가 아주 작으면 logit이 거의 0이 되어 attention이 평균에 가깝다
    assert (flat - ref).abs().max() > 1e-3
    # 모델 설정: 기본값(None)이면 저장하지 않고, 켠 값은 저장한다. 가중치 모양과 상관없어
    # 다른 값의 checkpoint에서도 가중치를 가져온다(INFO_FIELDS)
    config = tabpfn_v3.with_architecture(TrainConfig.smoke("multitask"), "v3.5", keep_size=True)
    assert "logit_softcap" not in config_dict(config.model_config())
    capped = replace(config, logit_softcap=50.0)
    assert config_dict(capped.model_config())["logit_softcap"] == 50.0
    model = capped.build_model()
    model.load_state_dict(config.build_model().state_dict())
    assert all(m.logit_softcap == 50.0 for m in model.modules() if isinstance(m, Attention))
    # 0은 NaN을 내고, soft-cap이 없는 모델에 주면 조용히 빠지므로 막는다
    with pytest.raises(ValueError, match="logit_softcap"):
        replace(config, logit_softcap=0.0)
    with pytest.raises(ValueError, match="logit_softcap"):  # 끄려면 inf가 아니라 None
        replace(config, logit_softcap=float("inf"))
    # 생성 뒤에 cap을 정해도 flex attention의 score_mod를 compile 밖에서 미리 만든다
    from minipfn.models import column_row_transformer as M

    attn = Attention(16, 2)
    attn.logit_softcap = 37.5
    assert 37.5 in M._SCORE_MODS and attn.logit_softcap == 37.5
    with pytest.raises(ValueError, match="logit_softcap"):
        replace(TrainConfig.smoke("classification"), logit_softcap=50.0)


def test_logit_softcap_inference_chunks(monkeypatch):
    # 추론(gradient 없음)에서 logit 행렬이 크면 query를 나눠 계산해도 결과가 같다
    from minipfn.models import column_row_transformer as M

    torch.manual_seed(0)
    attn = M.Attention(16, 2, qk_norm=True, zero_out=False, logit_softcap=5.0)
    x = torch.randn(2, 37, 16) * 10
    with torch.no_grad():
        whole = attn(x, n_context=29)
        monkeypatch.setattr(M, "SOFTCAP_CHUNK_LOGITS", 50)
        chunked = attn(x, n_context=29)
    assert torch.allclose(whole, chunked, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="flex attention backward는 GPU만")
def test_logit_softcap_gpu_kernels_match_manual(monkeypatch):
    # GPU의 flex attention kernel이 logit을 다 만드는 구현과 출력·gradient가 같다 (issue #77).
    # multi-query 테스트 행 경로, cap이 걸리는 큰 logit, batch를 나누는 경로, head 폭 16(작은
    # block)과 64(기본 block)
    from minipfn.models import column_row_transformer as M

    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    torch.manual_seed(0)
    for dim, heads in ((64, 4), (256, 4)):
        attn = M.Attention(
            dim, heads, qk_norm=True, qassmax=True, zero_out=False, logit_softcap=5.0
        )
        attn = attn.cuda()
        x = torch.randn(5, 150, dim, device="cuda") * 10
        runs = []
        for kernel, max_batch in (("manual", 65_535), ("flex", 65_535), ("flex", 2)):
            monkeypatch.setattr(M, "SOFTCAP_KERNEL", kernel)
            monkeypatch.setattr(M, "FLEX_MAX_BATCH", max_batch)
            x_ = x.clone().requires_grad_()
            attn.zero_grad(set_to_none=True)
            out = attn(x_, n_context=120)
            out.square().mean().backward()
            grads = [p.grad for p in attn.parameters() if p.grad is not None]
            runs.append([out, x_.grad, *grads])
        for other in runs[1:]:
            for a, b in zip(runs[0], other, strict=True):
                assert torch.allclose(a, b, rtol=1e-4, atol=1e-5 * a.abs().max().item())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="flex attention backward는 GPU만")
def test_logit_softcap_flex_trains_compiled(tmp_path, monkeypatch):
    # 가속 옵션 compile(블록마다 dynamic shape)로 soft-cap 학습이 돈다: 블록 compile 안의 flex
    # attention과 모델의 inductor 설정(`compile_options`). 표 크기가 step마다 다르다. 폭은
    # TabPFN-3.5와 같게 해서 열·행 단계(head 폭 16, 작은 block)와 ICL(64)이 모두 flex를 쓴다.
    # GPU 노드의 pytest에서는 OMP_NUM_THREADS=1로 돌린다(회귀 구간 경계의 process pool이 멈춘다)
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    config = _smoke_curriculum(logit_softcap=50.0, compile=True, stop_step=3, emsize=128, nhead=16)
    model_config = config.model_config()
    assert model_config.emsize // model_config.column_heads == 16
    assert model_config.icl_dim // model_config.nhead == 64
    out = train(config, tmp_path, mlflow_log=False)
    assert all(np.isfinite(out["history"]["train_loss"]))


def test_logit_softcap_trains_from_uncapped_weights(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    # soft-cap 없이 학습한 가중치를 soft-cap을 켜고 가져온다 (모델 설정의 INFO_FIELDS)
    train(_smoke_curriculum(stop_step=2, logit_softcap=None), tmp_path / "a", mlflow_log=False)
    out = train(
        _smoke_curriculum(logit_softcap=50.0, stop_step=2),
        tmp_path / "b",
        init_from=tmp_path / "a" / "latest.pt",
        mlflow_log=False,
    )
    assert all(np.isfinite(out["history"]["train_loss"]))
