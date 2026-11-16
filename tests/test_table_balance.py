"""GPU끼리 표 크기 나눠 받기 (`TrainConfig.balance_tables`, `training.table_balance`, issue #91)와
batch가 큰 flex attention을 compile 밖에서 나눠 부르기."""

import importlib.util
import itertools
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from minipfn.recipes import tabpfn_v3
from minipfn.training.config import TrainConfig
from minipfn.training.data import PriorBatches, derive_seed
from minipfn.training.loop import train
from minipfn.training.table_balance import TableBalance, assign, table_cost

STAGE4 = tabpfn_v3.curriculum(4)


def _stage4_prior(task: str):
    # stage 4의 크기 규칙 그대로인 prior (표는 만들지 않고 크기만 뽑는다)
    return STAGE4.task_configs()[task].make_prior(0)


def _small(task: str = "classification", **changes) -> TrainConfig:
    # 작은 크기로 stage 4의 크기 규칙(train_fraction, log 균등, 셀 상한 feature)
    config = tabpfn_v3.with_architecture(TrainConfig.smoke("multitask"), "v3.5", keep_size=True)
    small = dict(
        size_rule="train_fraction",
        min_rows=24,
        max_rows=120,
        min_train_fraction=0.79,
        max_train_fraction=0.81,
        log_uniform_rows=True,
        max_features=8,
        max_cells=400,
        feature_cap_rule="cells",
        border_datasets=20,
    )
    return replace(config, **{**small, **changes}).task_configs()[task]


def _loader_shapes(config, balance, rank, num_workers, n):
    loader = DataLoader(
        PriorBatches(config, seed=(0, 0, rank), balance=balance),
        batch_size=None,
        num_workers=num_workers,
    )
    return [(x.shape, n_train) for x, _, n_train in itertools.islice(iter(loader), n)]


# ---------------------------------------------------------------------------
# 나누기 (LPT)
# ---------------------------------------------------------------------------


def test_assign_partitions_and_balances():
    rng = np.random.default_rng(0)
    world, per = 32, 8
    prior = _stage4_prior("regression")
    worst_off, worst_on = [], []
    for _ in range(30):
        shapes = [prior.sample_shape(rng) for _ in range(world * per)]
        costs = [table_cost(*s) for s in shapes]
        members = assign(costs, world, per)
        # 모든 표를 한 번씩, GPU마다 같은 수
        assert sorted(i for m in members for i in m) == list(range(world * per))
        assert all(len(m) == per for m in members)
        loads = [sum(costs[i] for i in m) for m in members]
        # 가장 느린 GPU는 평균과 가장 비싼 표 가운데 큰 것 근처다 (그보다 빠를 수는 없다)
        bound = max(sum(costs) / world, max(costs))
        assert bound <= max(loads) <= 1.1 * bound
        worst_on.append(max(loads))
        # 끈 것: GPU마다 연속한 표 per개 (따로 뽑은 것과 같은 분포)
        worst_off.append(max(sum(costs[r * per : (r + 1) * per]) for r in range(world)))
    assert np.mean(worst_on) < 0.7 * np.mean(worst_off)
    # 같은 입력이면 같은 결과 (모든 GPU가 따로 계산한다)
    assert assign(costs, world, per) == members
    with pytest.raises(ValueError, match="표 수"):
        assign(costs[:-1], world, per)


def test_cost_model_matches_measurement():
    # 1×H100 측정 (activation checkpointing 켬). 모델에 넣지 않은 표마다 같은 시간(맞춘 값 0.74초)을
    # 더하면 측정과 5% 안이다
    def cost(rows, features):
        n_train = int(rows * 0.8)
        return table_cost(n_train, features, rows - n_train) + 0.74

    for rows, features, seconds in (
        (40_000, 100, 6.0),
        (60_000, 20, 11.7),
        (100_000, 40, 31.4),
        (200_000, 10, 121.6),
        (200_000, 20, 121.8),
    ):
        assert cost(rows, features) == pytest.approx(seconds, rel=0.05)


# ---------------------------------------------------------------------------
# data worker
# ---------------------------------------------------------------------------


def test_step_shapes_follow_the_prior_size_rule():
    # 한 step의 표 크기 전부가 prior의 크기 규칙(`sample_shape`)에서 독립으로 뽑은 것이고,
    # GPU끼리 겹치지 않게 나눈 것이다
    prior = _stage4_prior("classification")
    world, per = 4, 3
    balances = [TableBalance(seed=(7, 0), rank=r, world_size=world, per_rank=per) for r in range(4)]
    for group in range(5):
        rng = np.random.default_rng(derive_seed(7, 0, group))
        expected = [prior.sample_shape(rng) for _ in range(world * per)]
        got = [s for b in balances for s in b.rank_shapes(prior, group)]
        assert sorted(got) == sorted(expected)
        assert all(len(b.rank_shapes(prior, group)) == per for b in balances)


def test_size_distribution_is_unchanged():
    # 켠 것과 끈 것의 표 크기 분포가 같다 (같은 규칙, 다른 난수). 행 수는 log 균등
    prior = _stage4_prior("regression")
    world, per, steps = 8, 8, 40
    on = [
        s
        for g in range(steps)
        for r in range(world)
        for s in TableBalance((1,), r, world, per).rank_shapes(prior, g)
    ]
    other = STAGE4.task_configs()["regression"].make_prior(1)
    off = [other.sample_shape() for _ in range(len(on))]
    for shapes in (on, off):
        rows = np.log([n_train + n_eval for n_train, _, n_eval in shapes])
        assert abs(rows.mean() - np.log(np.sqrt(400 * 200_000))) < 0.12
        frac = np.array([n_train / (n_train + n_eval) for n_train, _, n_eval in shapes])
        assert 0.78 <= frac.min() and frac.max() <= 0.81
        assert all(f * (t + e) <= 4_000_000 for t, f, e in shapes)

    def stats(shapes):
        rows = np.log([t + e for t, _, e in shapes])
        feats = np.array([f for _, f, _ in shapes])
        return np.quantile(rows, [0.25, 0.5, 0.75]), feats.mean()

    (q_on, f_on), (q_off, f_off) = stats(on), stats(off)
    assert np.allclose(q_on, q_off, atol=0.3) and abs(f_on - f_off) < 0.1 * f_off


@pytest.mark.parametrize("num_workers", [0, 2])
def test_ranks_split_one_step_of_sizes(num_workers):
    # GPU마다 받은 크기를 모으면 그 step의 크기 전부이고, 표 내용은 GPU마다 다르다.
    # worker 수와 상관없이 같은 순서다
    config = _small("classification")
    world, per, steps = 3, 2, 3
    balances = [TableBalance((5,), r, world, per) for r in range(world)]
    got = [_loader_shapes(config, b, r, num_workers, per * steps) for r, b in enumerate(balances)]
    prior = config.make_prior(0)
    for group in range(steps):
        expected = sorted(balances[0].step_shapes(prior, group))
        step = [shape for rank in got for shape in rank[group * per : (group + 1) * per]]
        # (x 모양, n_train) → (n_train, feature, 검증 행)
        sizes = [(n, s[2], s[1] - n) for s, n in step]
        assert sorted(sizes) == expected
    if num_workers:
        assert got == [_loader_shapes(config, b, r, 0, per * steps) for r, b in enumerate(balances)]


def test_without_balance_sizes_come_from_the_prior():
    # 끄면 예전 코드 경로 그대로: 크기는 GPU마다 자기 prior의 난수로 뽑는다 (rank마다 다르다)
    config = _small("regression")

    def sizes(rank):
        loader = DataLoader(PriorBatches(config, seed=(0, 0, rank)), batch_size=None)
        return [(x.shape, n) for x, _, n in itertools.islice(iter(loader), 6)]

    first = sizes(0)
    # 표 내용을 만드는 데 쓴 난수와 같은 prior 난수에서 크기를 뽑는다
    assert first == sizes(0) and first != sizes(1)


# ---------------------------------------------------------------------------
# 설정·학습·스크립트
# ---------------------------------------------------------------------------


def test_stage4_recipe_balances_and_takes_steps():
    assert STAGE4.balance_tables and STAGE4.steps == tabpfn_v3.STAGE4_DEFAULT_STEPS
    assert not any(tabpfn_v3.curriculum(n).balance_tables for n in (1, 2, 3))
    assert not TrainConfig().balance_tables
    low, high = tabpfn_v3.STAGE4_STEPS
    assert low <= STAGE4.steps <= high
    for steps in (500, 1_000):
        c = tabpfn_v3.curriculum(4, steps=steps)
        assert c.steps == steps and c.warmup_steps == steps // 100
        assert steps % c.checkpoint_every == 0 and steps % c.keep_every == 0
        assert c.log_every <= 5
    for steps in (499, 1_001):
        with pytest.raises(ValueError, match="범위"):
            tabpfn_v3.curriculum(4, steps=steps)
    # 다른 stage는 범위 없이 step 수만 바꾼다
    assert tabpfn_v3.curriculum(3, steps=100).warmup_steps == 1
    with pytest.raises(ValueError, match="graph_scm"):
        TrainConfig.smoke("classification").__class__(prior_type="prior_bag", balance_tables=True)


def test_multitask_training_with_balanced_sizes(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    config = tabpfn_v3.with_architecture(TrainConfig.smoke("multitask"), "v3.5", keep_size=True)
    config = replace(
        config,
        steps=4,
        warmup_steps=1,
        log_every=2,
        val_every=4,
        checkpoint_every=4,
        keep_every=None,
        collapse_threshold=None,
        border_datasets=20,
        balance_tables=True,
        rank_timing=True,
    )
    out = train(config, tmp_path, mlflow_log=False)
    assert len(out["history"]["train_loss"]) == 2
    assert all(np.isfinite(out["history"]["train_loss"]))


def _script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "tabpfn_v2" / "train.py"
    spec = importlib.util.spec_from_file_location("train_script_balance", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_script_steps_flags_and_resume(monkeypatch):
    for name in ("TABPFN_V2_CONFIG_NAME", "TABPFN_V2_STEPS", "TABPFN_V2_BALANCE_TABLES"):
        monkeypatch.delenv(name, raising=False)
    script = _script()
    base = ["--config-name", "stage4", "--arch", "v3.5", "--task", "multitask"]
    base += ["--init-run-id", "stage3"]
    config, _ = script.build_config(script.parse_args([*base, "--steps", "500"]))
    # warmup·cosine이 준 step 수에 맞는다
    assert config == tabpfn_v3.curriculum(4, steps=500) and config.warmup_steps == 5
    config, _ = script.build_config(
        script.parse_args([*base, "--balance-tables", "off", "--log-every", "2"])
    )
    assert not config.balance_tables and config.log_every == 2
    with pytest.raises(SystemExit, match="범위"):
        script.build_config(script.parse_args([*base, "--steps", "2000"]))
    # 이어 학습하면 원래 run의 값을 따른다
    assert script.source_speed_options({"balance_tables": "True"}) == {"balance_tables": True}


# ---------------------------------------------------------------------------
# batch가 큰 flex attention (compile 밖에서 나눠 부른다)
# ---------------------------------------------------------------------------


def test_large_flex_batch_does_not_recompile_per_size(monkeypatch):
    # 블록 compile 안에서 batch를 나누면 dynamo가 batch 크기를 graph에 고정해 크기마다 다시
    # compile했다. 나누는 부분은 compile 밖이라, 상한보다 큰 batch는 크기가 달라도 graph가 같다.
    # flex attention kernel은 GPU만이라 kernel 자리에 같은 일을 하는 함수를 넣는다
    from torch._dynamo.testing import CompileCounter

    from minipfn.models import column_row_transformer as M

    calls = []

    def fake_attend(q, k, v, score_mod):
        calls.append(q.shape[0])
        logits = score_mod(q @ k.transpose(-1, -2), None, None, None, None)
        return logits.softmax(-1) @ v

    monkeypatch.setattr(M, "FLEX_MAX_BATCH", 8)
    monkeypatch.setattr(M, "_eager_flex_attend", fake_attend)

    def block(x):
        h = x * 2
        return M._flex_softcap(h, h, h, 5.0) + 1

    M._softcap_score_mod(5.0)  # 모델처럼 compile 밖에서 미리 만든다 (`Attention.logit_softcap`)
    torch._dynamo.reset()
    counter = CompileCounter()
    compiled = torch.compile(block, backend=counter, dynamic=True)
    for n in (9, 11, 13, 17, 20, 23, 29, 31):
        x = torch.randn(n, 2, 5, 4)
        calls.clear()
        out = compiled(x)
        assert calls == [min(8, n - i) for i in range(0, n, 8)]
        assert torch.allclose(out, block(x))
    # 나누기 전과 뒤의 graph 두 개 (크기마다 다시 compile하면 크기 수만큼 는다)
    assert counter.frame_count <= 3
    torch._dynamo.reset()
