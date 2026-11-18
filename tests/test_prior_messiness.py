"""graph_scm prior의 현실의 지저분함·클래스 불균형 옵션 (issue #103): 결측 패턴, 범주 값이 많은
열, 격자로 반올림한 feature, 꼬리가 긴 회귀 target, 이진 소수 클래스. 기본값이면 예전 부품·같은
난수다."""

import importlib.util
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy import stats

from minipfn.priors import graph_scm as P
from minipfn.priors.generators.imbalance import ImbalancedBinary
from minipfn.priors.pipeline import Group, Rng
from minipfn.priors.transforms import (
    CoarseFeatures,
    HighCardinality,
    LongTailTarget,
    Missing,
    PatternMissing,
)
from minipfn.priors.transforms import features as F
from minipfn.priors.transforms import missing as M
from minipfn.training.config import TrainConfig

SMALL = dict(size_rule="train_fraction", min_rows=80, max_rows=80, max_features=8, max_cells=640)
OPTIONS = dict(
    missing_patterns=True,
    coarse_feature_prob=0.5,
    high_cardinality_prob=0.5,
    long_tail_target_prob=0.5,
    imbalance_prob=0.5,
)


def _prior(task="classification", seed=0, **options):
    return P.GraphSCMPrior(task, seed=seed, **SMALL, **options)


def _same(a, b):
    return torch.equal(torch.nan_to_num(a, nan=-7.0), torch.nan_to_num(b, nan=-7.0))


def _group(x, y=None):
    """데이터셋 하나의 그룹 (행, 1, feature)."""
    x = torch.as_tensor(x, dtype=torch.float32)
    y = torch.zeros(len(x)) if y is None else torch.as_tensor(y, dtype=torch.float32)
    return Group(x[:, None], y[:, None], len(x) // 2, {})


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_defaults_build_the_same_parts_and_data_as_before(task):
    prior = _prior(task)
    assert type(prior._missing) is Missing
    kinds = [type(t) for t in prior.pipeline.transforms]
    assert not {HighCardinality, CoarseFeatures, LongTailTarget, PatternMissing} & set(kinds)
    assert not isinstance(prior.pipeline.generator, ImbalancedBinary)
    a = [prior.sample_batch(2) for _ in range(2)]
    off = dict.fromkeys(OPTIONS, 0.0) | {"missing_patterns": False}
    explicit = _prior(task, **off)
    b = [explicit.sample_batch(2) for _ in range(2)]
    for p, q in zip(a, b, strict=True):
        assert _same(p.x, q.x) and torch.equal(p.y, q.y)


def test_options_go_only_to_their_task():
    cls = _prior("classification", **OPTIONS)
    reg = _prior("regression", **OPTIONS)
    assert isinstance(cls.pipeline.generator, ImbalancedBinary)
    assert not isinstance(reg.pipeline.generator, ImbalancedBinary)
    assert LongTailTarget in [type(t) for t in reg._accept]
    assert LongTailTarget not in [type(t) for t in cls.pipeline.transforms]
    for prior in (cls, reg):
        kinds = [type(t) for t in prior.pipeline.transforms]
        # 범주 열 → 반올림 → 결측 순서, 결측은 맨 끝
        assert kinds[-3:] == [HighCardinality, CoarseFeatures, PatternMissing]
        assert prior.max_missing_rate == 0.5
        batch = prior.sample_batch(2)
        assert batch.x.shape[:2] == (2, 80) and torch.isfinite(batch.y).all()


def _mask_of(pattern, monkeypatch, x, y, classification=False, seed=0):
    monkeypatch.setattr(M, "MISSING_PATTERNS", (pattern,))
    t = PatternMissing(1.0, classification=classification, min_strength=4.0, max_strength=4.0)
    return t.mask(x, y, np.random.default_rng(seed))


def test_pattern_missing_rates_and_columns(monkeypatch):
    r = np.random.default_rng(0)
    x = r.normal(size=(400, 10))
    for seed in range(20):
        mask = _mask_of("mcar", monkeypatch, x, np.zeros(400), seed=seed)
        cols = mask.any(0)
        assert 1 <= cols.sum() and (mask.mean(0) <= 0.5 + 1e-9).all()


@pytest.mark.parametrize("pattern", ["mar", "mnar", "mnar_y"])
def test_pattern_missing_follows_its_driver(pattern, monkeypatch):
    r = np.random.default_rng(1)
    x = r.normal(size=(2000, 2))
    y = x[:, 0] + 0.1 * r.normal(size=2000)
    # 결측이 driver 값에 따라 생기면 결측 행과 아닌 행의 driver 평균이 크게 다르다 (mcar는 같다)
    gaps = {"self": [], "other": [], "y": []}
    for seed in range(40):
        mask = _mask_of(pattern, monkeypatch, x, y, seed=seed)
        for c in range(2):
            m = mask[:, c]
            if 0.05 < m.mean() < 0.45:
                gaps["self"].append(abs(x[m, c].mean() - x[~m, c].mean()))
                gaps["other"].append(abs(x[m, 1 - c].mean() - x[~m, 1 - c].mean()))
                gaps["y"].append(abs(y[m].mean() - y[~m].mean()))
    key = {"mar": "other", "mnar": "self", "mnar_y": "y"}[pattern]
    assert np.median(gaps[key]) > 1.0
    mcar = _mask_of("mcar", monkeypatch, x, y)
    for c in range(2):
        m = mcar[:, c]
        if m.any():
            assert abs(x[m, c].mean() - x[~m, c].mean()) < 0.3


def test_pattern_missing_by_class(monkeypatch):
    r = np.random.default_rng(2)
    y = r.integers(0, 3, 3000).astype(float)
    x = r.normal(size=(3000, 1))
    spread = []
    for seed in range(20):
        m = _mask_of("mnar_y", monkeypatch, x, y, classification=True, seed=seed)[:, 0]
        if 0.05 < m.mean() < 0.45:
            rates = [m[y == k].mean() for k in range(3)]
            spread.append(max(rates) - min(rates))
    assert np.median(spread) > 0.3


def test_pattern_missing_puts_nan_in_the_group(monkeypatch):
    r = np.random.default_rng(3)
    group = Group(torch.randn(50, 3, 4), torch.randn(50, 3), 25, {})
    out = PatternMissing(1.0, classification=False)(group, Rng.from_seed(0))
    assert out.x.shape == group.x.shape and torch.isnan(out.x).any()
    assert torch.equal(out.y, group.y)
    keep = PatternMissing(0.0, classification=False)(group, Rng.from_seed(0))
    assert torch.equal(keep.x, group.x)
    del r


def test_high_cardinality_makes_unordered_category_columns():
    r = np.random.default_rng(4)
    x = r.normal(size=(1024, 6))
    x[:, 5] = r.integers(0, 3, 1024)  # 값이 셋뿐인 열은 고르지 않는다
    t = HighCardinality(1.0)
    rhos = []
    for seed in range(30):
        v = x.copy()
        cols = t.convert(v, np.random.default_rng(seed))
        assert 1 <= len(cols) <= 3 and 5 not in cols
        untouched = [c for c in range(6) if c not in cols]
        assert np.array_equal(v[:, untouched], x[:, untouched])
        for c in cols:
            n = len(np.unique(v[:, c]))
            assert 20 <= n <= 256  # K ≤ 1,024 // 4, Zipf로 작은 구간은 빌 수 있다
            assert abs(v[:, c].mean()) < 1e-4 and abs(v[:, c].std() - 1) < 1e-3
            rhos.append(abs(stats.spearmanr(v[:, c], x[:, c]).statistic))
    # 같은 범주의 행은 원래 값이 이어진 구간이지만, 번호에는 순서가 없다 (큰 구간 하나가 행 대부분을
    # 차지하면 우연히 상관이 커질 수 있어 중앙값으로 본다)
    assert np.median(rhos) < 0.15


def test_high_cardinality_skips_small_tables_and_records_columns():
    t = HighCardinality(1.0)
    small = np.random.default_rng(0).normal(size=(150, 3))  # 150 // 4 < 50
    assert t.convert(small.copy(), np.random.default_rng(0)) == []
    group = Group(torch.randn(400, 2, 5), torch.randn(400, 2), 200, {})
    out = t(group, Rng.from_seed(0))
    changed = out.hp[F.HIGH_CARDINALITY_KEY]
    assert set(changed) == {0, 1} and all(changed.values())
    assert F.HIGH_CARDINALITY_KEY not in group.hp  # 원래 hp는 그대로


def test_coarse_features_round_to_a_grid_and_skip_category_columns():
    r = np.random.default_rng(5)
    x = r.normal(size=(1024, 8))
    t = CoarseFeatures(1.0)
    for seed in range(10):
        v = x.copy()
        cols = t.coarsen(v, np.random.default_rng(seed), skip=[0])
        assert cols and 0 not in cols
        for c in cols:
            n = len(np.unique(v[:, c]))
            assert 2 <= n <= 120
            assert np.abs(v[:, c] - x[:, c]).max() <= x[:, c].std() * 2.0 / 2 + 1e-5
            gaps = np.diff(np.unique(v[:, c]))
            assert np.allclose(gaps / gaps.min(), np.round(gaps / gaps.min()), atol=1e-3)
        untouched = [c for c in range(8) if c not in cols]
        assert np.array_equal(v[:, untouched], x[:, untouched])


def test_coarse_features_leave_high_cardinality_columns_alone():
    group = Group(torch.randn(1024, 1, 6), torch.randn(1024, 1), 512, {})
    rng = Rng.from_seed(0)
    group = HighCardinality(1.0)(group, rng)
    cols = group.hp[F.HIGH_CARDINALITY_KEY][0]
    before = group.x[:, 0, cols].clone()
    out = CoarseFeatures(1.0, min_fraction=1.0, max_fraction=1.0)(group, rng)
    assert torch.equal(out.x[:, 0, cols], before)


def test_long_tail_target_is_monotone_skewed_and_standardized():
    y = torch.randn(2000, 6).clamp(-4, 4)
    out = LongTailTarget(1.0)(Group(torch.zeros(2000, 6, 1), y, 1000, {}), Rng.from_seed(0)).y
    for d in range(6):
        assert stats.spearmanr(out[:, d], y[:, d]).statistic > 0.999  # 관계 방향 그대로
        assert abs(out[:, d].mean()) < 1e-4 and abs(out[:, d].std(unbiased=False) - 1) < 1e-3
        assert abs(stats.skew(out[:, d].numpy())) > abs(stats.skew(y[:, d].numpy())) + 0.5
    skews = [stats.skew(out[:, d].numpy()) for d in range(6)]
    assert min(skews) < 0 < max(skews)  # 오른쪽·왼쪽 꼬리 둘 다


class _Toy:
    """minority 비율 0.3의 이진 데이터셋을 내는 생성기 (만든 행 수를 기록한다)."""

    def __init__(self):
        self.rows = []

    def __call__(self, n_rows, n_features, n_datasets, hp, rng):
        self.rows.append(n_rows)
        x = torch.arange(n_rows, dtype=torch.float32)[:, None, None].expand(-1, n_datasets, 2)
        y = (torch.rand(n_rows, n_datasets, generator=rng.torch) < 0.3).float()
        return x.clone(), y


def test_imbalanced_binary_lowers_the_minority_ratio():
    toy = _Toy()
    gen = ImbalancedBinary(toy, 1.0)
    rng = Rng.from_seed(0)
    ratios = []
    for _ in range(200):
        x, y = gen(1000, 2, 1, {"num_classes": 2}, rng)
        assert x.shape == (1000, 1, 2) and y.shape == (1000, 1)
        assert len(torch.unique(x[:, 0, 0])) == 1000  # 같은 행을 두 번 쓰지 않는다
        ratios.append(min(y.mean().item(), 1 - y.mean().item()))
    assert set(toy.rows) == {2000}
    ratios = np.array(ratios)
    assert ratios.min() >= 0.01 - 1e-9 and ratios.max() <= 0.15 + 1e-9
    assert 0.03 < np.median(ratios) < 0.06  # logU(0.01, 0.15)의 중앙값 ≈ 0.039


def test_imbalanced_binary_passes_other_groups_through():
    toy = _Toy()
    gen = ImbalancedBinary(toy, 0.0)
    a, b = Rng.from_seed(1), Rng.from_seed(1)
    x, y = gen(100, 2, 1, {"num_classes": 2}, a)
    x2, y2 = _Toy()(100, 2, 1, {"num_classes": 2}, b)
    assert torch.equal(y, y2)
    # 다중 분류는 난수를 더 쓰지 않는다 (감싸지 않은 생성기와 같은 다음 난수)
    gen = ImbalancedBinary(_Toy(), 1.0)
    a, b = Rng.from_seed(2), Rng.from_seed(2)
    gen(100, 2, 1, {"num_classes": 3}, a)
    _Toy()(100, 2, 1, {"num_classes": 3}, b)
    assert a.numpy.random() == b.numpy.random()


def test_imbalanced_binary_keeps_at_least_two_minority_rows():
    gen = ImbalancedBinary(_Toy(), 1.0, min_ratio=0.001, max_ratio=0.001)
    rng = Rng.from_seed(3)
    for _ in range(20):
        _, y = gen(80, 2, 1, {"num_classes": 2}, rng)
        assert min(y.sum().item(), (1 - y).sum().item()) == 2


def test_graph_scm_prior_with_imbalance_makes_imbalanced_binary_tables():
    prior = P.GraphSCMPrior(
        "classification",
        seed=0,
        size_rule="train_fraction",
        min_rows=400,
        max_rows=400,
        max_features=6,
        max_cells=2400,
        binary_prob=1.0,
        imbalance_prob=1.0,
    )
    for _ in range(4):
        batch = prior.sample_batch(1)
        y = batch.y[0]
        assert set(torch.unique(y).tolist()) == {0.0, 1.0}
        assert min(y.mean().item(), 1 - y.mean().item()) <= 0.16
        # 검증 행의 클래스는 모두 학습 행에 있다 (CoverClasses)
        train = set(y[: batch.n_train].tolist())
        assert set(y[batch.n_train :].tolist()) <= train


def test_options_are_checked():
    for name in ("coarse_feature_prob", "high_cardinality_prob", "long_tail_target_prob"):
        with pytest.raises(ValueError, match=name):
            _prior(**{name: 1.5})
        with pytest.raises(ValueError, match=name):
            TrainConfig(**{name: -0.1})
    with pytest.raises(ValueError, match="imbalance_prob"):
        TrainConfig(imbalance_prob=2.0)


def test_training_prior_and_borders_get_the_options(monkeypatch):
    config = TrainConfig(task="regression", **SMALL, **OPTIONS)
    prior = P.make_prior(config, seed=0)
    assert isinstance(prior._missing, PatternMissing)
    assert LongTailTarget in [type(t) for t in prior._accept]
    seen = {}

    def capture(n_buckets, n_datasets, seed, **kwargs):
        seen.update(kwargs)
        return torch.zeros(n_buckets + 1)

    monkeypatch.setattr(P, "regression_borders", capture)
    P.make_borders(config, 10, seed=0)
    # 회귀 target을 바꾸는 옵션만 구간 경계로 간다
    assert seen["long_tail_target_prob"] == 0.5
    for name in ("missing_patterns", "coarse_feature_prob", "high_cardinality_prob"):
        assert name not in seen
    assert "imbalance_prob" not in seen


def test_long_tail_borders_spread_further():
    plain = P.regression_borders(50, 16, seed=1, **SMALL)
    tail = P.regression_borders(50, 16, seed=1, long_tail_target_prob=1.0, **SMALL)
    assert tail.shape == (51,) and (tail.diff() > 0).all()
    assert (tail[-1] - tail[0]) > (plain[-1] - plain[0])


def _train_script():
    path = Path(__file__).parents[1] / "scripts" / "tabpfn_v2" / "train.py"
    spec = importlib.util.spec_from_file_location("train_tabpfn_v2_script_messiness", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_train_script_sets_the_options_from_env(monkeypatch):
    for name in ("TABPFN_V2_ARCH", "TABPFN_V2_CONFIG_NAME", "TABPFN_V2_TASK"):
        monkeypatch.delenv(name, raising=False)
    args = ["--config-name", "stage1", "--arch", "v3.5", "--task", "multitask"]
    script = _train_script()
    base, _ = script.build_config(script.parse_args(args))
    env = {
        "TABPFN_V2_MISSING_PATTERNS": "1",
        "TABPFN_V2_COARSE_FEATURE_PROB": "0.3",
        "TABPFN_V2_HIGH_CARDINALITY_PROB": "0.3",
        "TABPFN_V2_LONG_TAIL_TARGET_PROB": "0.3",
        "TABPFN_V2_IMBALANCE_PROB": "0.3",
    }
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    script = _train_script()
    config, _ = script.build_config(script.parse_args(args))
    assert config.missing_patterns
    probs = ("coarse_feature_prob", "high_cardinality_prob", "long_tail_target_prob")
    assert all(getattr(config, name) == 0.3 for name in (*probs, "imbalance_prob"))
    off = dict.fromkeys((*probs, "imbalance_prob"), 0.0)
    assert replace(config, **off, missing_patterns=False) == base
