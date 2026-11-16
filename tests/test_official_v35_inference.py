"""공식 TabPFN-3.5 추론 설정에 맞추는 옵션 (inference/ensemble.py, inference/preprocessing.py,
eval.py의 --official-v35 등) 테스트. 기본값은 예전과 같아야 한다."""

import importlib.util
import json
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from minipfn.inference.ensemble import (
    EnsembleClassifier,
    EnsembleRegressor,
    SafePowerTarget,
    _official_border_map,
    _probs_on_borders,
    _probs_on_full_support_borders,
)
from minipfn.inference.preprocessing import (
    Member,
    MemberPreprocessor,
    PreprocessOptions,
    balanced_feature_subsets,
    constant_columns,
    encode_low_cardinality,
    low_cardinality_values,
    row_fingerprint,
)
from minipfn.models.bar_distribution import FullSupportBarDistribution
from minipfn.models.cell_transformer import CellTransformer, CellTransformerConfig

OFFICIAL = dict(
    softmax_temperature=1.0,
    shuffle_categories=True,
    fingerprint="hash",
    outlier_clip="soft",
    infer_categorical=True,
    drop_constant=True,
    feature_subsampling="balanced",
)


def _tiny_model(n_out: int = 10) -> CellTransformer:
    torch.manual_seed(0)
    model = CellTransformer(CellTransformerConfig(emsize=16, nlayers=2, nhead=2, n_out=n_out))
    for p in model.parameters():
        torch.nn.init.normal_(p, std=0.3)
    return model.eval()


def _frame(n: int = 260, seed: int = 0) -> pd.DataFrame:
    g = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "a": g.normal(size=n),
            "b": g.lognormal(size=n),
            "flag": g.choice([0.0, 1.0], n),
            "d": np.where(g.random(n) < 0.2, np.nan, g.normal(size=n)),
            "const": np.full(n, 3.0),
            "color": g.choice(["red", "blue", "green", None], n),
        }
    )
    df.loc[5, "a"] = 80.0
    return df


def _member(kind: str = "standard", seed: int = 0) -> Member:
    return Member(kind=kind, class_permutation=None, target_power=False, seed=seed)


def test_default_options_are_the_old_behavior():
    df, y = _frame(), np.random.default_rng(1).choice(["x", "y", "z"], 260)
    old = EnsembleClassifier(_tiny_model(), n_estimators=3, member_kinds="standard")
    new = EnsembleClassifier(
        _tiny_model(),
        n_estimators=3,
        member_kinds="standard",
        shuffle_categories=False,
        fingerprint="random",
        outlier_clip="hard",
        infer_categorical=False,
        drop_constant=False,
        feature_subsampling="random",
    )
    a = old.fit(df[:200], y[:200]).predict_proba(df[200:])
    b = new.fit(df[:200], y[:200]).predict_proba(df[200:])
    assert np.array_equal(a, b)
    with pytest.raises(ValueError, match="fingerprint"):
        PreprocessOptions(fingerprint="sha")
    with pytest.raises(ValueError, match="target_transform"):
        EnsembleRegressor(
            _tiny_model(32), torch.linspace(-4, 4, 33), n_estimators=2, target_transform="log"
        )


def test_shuffle_categories_permutes_codes_in_standard_members():
    x = np.zeros((200, 2))
    x[:, 0] = np.arange(200) % 5  # 범주 5개
    x[:, 1] = np.random.default_rng(0).normal(size=200)
    plain = MemberPreprocessor(_member(), cat_ix=[0], max_one_hot_width=100).fit(x)
    assert plain.cat_maps == {}  # 기본: standard 멤버는 번호 그대로
    options = PreprocessOptions(shuffle_categories=True)
    perms = []
    for seed in range(4):
        prep = MemberPreprocessor(_member(seed=seed), [0], 100, options).fit(x)
        assert sorted(prep.cat_maps[0]) == list(range(5))
        codes = prep._base(prep._clip(x))[:, 0]
        assert np.array_equal(codes, prep.cat_maps[0][x[:, 0].astype(int)])
        perms.append(tuple(prep.cat_maps[0]))
    assert len(set(perms)) > 1  # 멤버마다 다르다
    # one-hot으로 펼치는 열은 섞지 않는다
    member = Member("standard", None, False, 0, one_hot=True)
    assert MemberPreprocessor(member, [0], 100, options).fit(x).cat_maps == {}


def test_hash_fingerprint_matches_duplicates_of_training_rows():
    rows = np.array([[1.0, 2.0], [1.0, 2.0], [3.0, np.nan], [1.0, 2.0]])
    train = row_fingerprint(rows, salt=8, unique=True)
    assert len(set(train)) == 4 and ((train >= 0) & (train <= 1)).all()
    test = row_fingerprint(np.array([[1.0, 2.0], [3.0, np.nan], [5.0, 5.0]]), salt=8, unique=False)
    assert test[0] == train[0] and test[1] == train[2]  # 같은 행은 처음 학습 행과 같은 값
    assert test[2] not in train
    assert not np.array_equal(row_fingerprint(rows, 9, True), train)  # salt가 다르면 다르다
    # 값이 모두 결측인 행도 번호를 더해 다른 값이 된다 (충돌 다시 풀기만 하지 않는다)
    missing = row_fingerprint(np.full((2, 2), np.nan), salt=4, unique=True)
    assert np.isfinite(missing).all() and missing[0] != missing[1]
    # 멤버 전처리: 식별자 열이 난수가 아니라 hash (학습 행 변환은 rng와 상관없다)
    x = np.random.default_rng(0).normal(size=(50, 3))
    prep = MemberPreprocessor(_member(), [], 100, PreprocessOptions(fingerprint="hash")).fit(x)
    a = prep.transform(x, np.random.default_rng(1))
    b = prep.transform(x, np.random.default_rng(2))
    assert np.array_equal(a, b) and a.shape == (50, 4)
    test_rows = prep.transform(x[:5], np.random.default_rng(3), is_test=True)
    assert np.array_equal(test_rows, a[:5])  # 학습 행과 같은 테스트 행은 같은 식별자


def test_soft_outlier_clip_compresses_values_beyond_two_pass_bounds():
    rng = np.random.default_rng(0)
    x = np.column_stack([rng.normal(size=1000), np.arange(1000) % 3])
    x[0, 0] = 1e6
    hard = MemberPreprocessor(_member(), [1], 100).fit(x)
    soft = MemberPreprocessor(_member(), [1], 100, PreprocessOptions(outlier_clip="soft")).fit(x)
    # 두 번째 경계는 이상치를 뺀 표준편차로 구해 훨씬 좁다
    assert soft.upper[0] < hard.upper[0] / 10 and np.isinf(soft.upper[1])
    clipped = soft._clip(x)
    inside = np.abs(x[:, 0]) < soft.upper[0]
    assert np.array_equal(clipped[inside], x[inside])  # 경계 안과 범주형 열은 그대로
    assert np.isclose(clipped[0, 0], soft.upper[0] + np.log1p(1e6))  # 경계에서 log(1 + |x|)만큼
    grid = np.linspace(-1e4, 1e4, 2001)[:, None].repeat(2, 1)
    out = soft._clip(grid)[:, 0]
    assert (np.diff(out) >= 0).all()  # 순서는 그대로
    with_nan = soft._clip(np.array([[np.nan, 1.0]]))
    assert np.isnan(with_nan[0, 0]) and with_nan[0, 1] == 1.0


def test_low_cardinality_numeric_columns_become_categorical():
    x = np.column_stack(
        [np.arange(150) % 2 * 5.0, np.arange(150) % 4, np.full(150, 1.0), np.arange(150.0)]
    )
    x[3, 1] = np.nan
    values = low_cardinality_values(x, cat_ix=[])
    # 값 2개 → 범주형, 값 4개는 수치(4개보다 적어야 한다), 상수 열은 상수 열 단계가 맡는다
    assert list(values) == [0] and np.array_equal(values[0], [0.0, 5.0])
    assert low_cardinality_values(x[:100], cat_ix=[]) == {}  # 학습 행 100개 이하는 추론하지 않음
    # 결측도 값 하나로 센다: 값 2개 + 결측 = 3 → 범주형
    y = x.copy()
    y[:, 1] = np.where(np.arange(150) % 3 == 0, np.nan, np.arange(150) % 2)
    assert 1 in low_cardinality_values(y, cat_ix=[])
    encoded = encode_low_cardinality(np.array([[5.0], [0.0], [7.0], [np.nan]]), {0: values[0]})
    assert np.array_equal(encoded[:2, 0], [1.0, 0.0]) and np.isnan(encoded[2:, 0]).all()


def test_constant_columns_keep_declared_categoricals():
    x = np.column_stack([np.ones(10), np.full(10, np.nan), np.arange(10.0), np.zeros(10)])
    x[0, 3] = np.nan  # 값 하나 + 결측 = 서로 다른 값 2개: 상수가 아니다
    assert constant_columns(x) == [0, 1]
    assert constant_columns(x, keep=[0]) == [1]


def test_ensemble_column_steps_apply_to_test_rows():
    df = _frame()
    y = np.random.default_rng(1).choice(["x", "y", "z"], 260)
    clf = EnsembleClassifier(
        _tiny_model(), n_estimators=4, member_kinds="standard", one_hot="off", **OFFICIAL
    )
    proba = clf.fit(df[:200], y[:200]).predict_proba(df[200:])
    # const(4번 열)는 빠지고, flag(2번)는 범주형, color(5번 → 4번)는 원래 범주형
    assert list(clf.kept_columns_) == [0, 1, 2, 3, 5]
    assert clf.cat_ix_ == [2, 4] and clf.encoded_ix_ == [5]
    assert proba.shape == (60, 3) and np.allclose(proba.sum(1), 1, atol=1e-5)
    # 같은 설정은 다시 해도 같다 (식별자 열이 hash)
    again = EnsembleClassifier(
        _tiny_model(), n_estimators=4, member_kinds="standard", one_hot="off", **OFFICIAL
    )
    assert np.array_equal(again.fit(df[:200], y[:200]).predict_proba(df[200:]), proba)


def test_balanced_feature_subsets_use_every_column_evenly():
    rng = np.random.default_rng(0)
    subsets = balanced_feature_subsets(20, 6, 8, rng)
    counts = np.bincount(np.concatenate(subsets), minlength=20)
    assert all(len(set(s)) == 6 and (np.diff(s) > 0).all() for s in subsets)
    assert counts.max() - counts.min() <= 1 and counts.sum() == 48
    assert balanced_feature_subsets(5, 6, 3, rng) == [None, None, None]
    x = rng.normal(size=(120, 20))
    y = (x[:, 0] > 0).astype(int)
    clf = EnsembleClassifier(
        _tiny_model(),
        n_estimators=4,
        member_kinds="standard",
        max_features_per_member=6,
        feature_subsampling="balanced",
    ).fit(x[:90], y[:90])
    assert all(len(s) == 6 for s in clf.feature_subsets_)
    assert clf.predict_proba(x[90:]).shape == (30, 2)


def test_safepower_fits_the_raw_target_and_inverts_back():
    rng = np.random.default_rng(0)
    y = rng.lognormal(5, 1.0, 400)  # 원래 단위로는 치우친 양수 target
    mean, std = y.mean(), y.std()
    z = (y - mean) / std
    target = SafePowerTarget(mean, std)
    u = target.fit_transform(z.reshape(-1, 1)).ravel()
    assert abs(u.mean()) < 1e-9 and abs(u.std() - 1) < 1e-9
    assert target.lam < 0.5  # 원래 단위 y에 맞춘 λ (z에 맞추면 z의 음수 쪽 때문에 다르다)
    assert np.allclose(target.inverse_transform(u.reshape(-1, 1)).ravel(), z, atol=1e-8)
    # 거의 상수인 target은 λ = 1
    const = SafePowerTarget(2.0, 1e-12)
    const.fit_transform(np.zeros((10, 1)))
    assert const.lam == 1.0


def test_official_border_map_repairs_borders_outside_the_domain():
    borders = np.linspace(-6, 6, 13)
    identity = types.SimpleNamespace(inverse_transform=lambda b: b)
    z, cancel, descending = _official_border_map(identity, borders)
    assert np.allclose(z.numpy(), borders) and cancel is None and not descending

    def broken_ends(b):  # 오른쪽 두 경계는 정의역 밖, 왼쪽 하나는 너무 멀다
        out = b.copy().ravel()
        out[-2:] = np.nan
        out[0] = -5e3
        return out

    z, cancel, _ = _official_border_map(
        types.SimpleNamespace(inverse_transform=broken_ends), borders
    )
    z = z.numpy()
    assert np.isfinite(z).all() and (np.diff(z) >= 0).all()
    assert z[0] == z[1] - 1.0 and z[-1] == z[-2] + 1.0
    assert cancel[0] and cancel[-1] and cancel[-2] and not cancel[1:-2].any()
    reverse = types.SimpleNamespace(inverse_transform=lambda b: -b.ravel())
    z, _, descending = _official_border_map(reverse, borders)
    assert descending and np.allclose(z.numpy(), borders)


def test_full_support_remap_spreads_tails_and_keeps_mass():
    borders = torch.tensor([-20.0, -2.0, -1.0, 0.0, 1.0, 2.0, 20.0])
    probs = torch.softmax(torch.randn(4, 6), dim=-1)
    same = _probs_on_full_support_borders(probs, borders, borders)
    assert torch.allclose(same, probs, atol=1e-6)
    # 더 넓은 경계로: 꼬리 확률이 half-normal로 안쪽 구간까지 퍼진다 (균일로 보면 다르다)
    wide = torch.tensor([-40.0, -10.0, -1.0, 0.0, 1.0, 10.0, 40.0])
    full = _probs_on_full_support_borders(probs, borders, wide)
    uniform = _probs_on_borders(probs, borders, wide)
    assert torch.allclose(full.sum(-1), torch.ones(4), atol=1e-6)
    assert not torch.allclose(full, uniform, atol=1e-3)
    # half-normal 꼬리의 중앙값은 맨 끝 구간 폭: 첫 구간 확률의 절반이 -2 - 18 = -20보다 바깥
    only_low = torch.zeros(1, 6)
    only_low[0, 0] = 1.0
    edges = torch.tensor([-60.0, -20.0, -2.0, 0.0, 1.0, 2.0, 20.0])
    moved = _probs_on_full_support_borders(only_low, borders, edges)
    assert torch.isclose(moved[0, 0], torch.tensor(0.5), atol=1e-5)


@pytest.mark.parametrize("target_transform", ["power", "safepower", "none"])
def test_mixture_mean_is_the_mean_of_the_mixture_distribution(target_transform):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 3))
    y = rng.lognormal(2, 1.0, 200)
    borders = torch.linspace(-4, 4, 33)
    reg = EnsembleRegressor(
        _tiny_model(32),
        borders,
        n_estimators=4,
        member_kinds="standard",
        target_transform=target_transform,
        regression_mean="mixture",
    ).fit(x[:150], y[:150])
    assert sum(m is not None for m in reg.official_maps_) == (
        0 if target_transform == "none" else 2
    )
    mean, log_probs, b = reg.predict_full(x[150:])
    assert np.allclose(mean, reg.predict(x[150:])) and np.isfinite(mean).all()
    assert torch.allclose(log_probs.exp().sum(-1), torch.ones(50), atol=1e-4)
    centers = FullSupportBarDistribution(borders).bucket_centers.double()
    expected = (log_probs.exp().double() @ centers).numpy() * reg.y_std_ + reg.y_mean_
    assert np.allclose(mean, expected, rtol=1e-4)
    assert torch.allclose(b, borders.double() * reg.y_std_ + reg.y_mean_)
    # target 변환이 없으면 멤버 경계가 공통 경계라, 섞는 분포는 기본(member)의 것과 같다
    if target_transform == "none":
        member = EnsembleRegressor(
            _tiny_model(32),
            borders,
            n_estimators=4,
            member_kinds="standard",
            target_transform="none",
        ).fit(x[:150], y[:150])
        assert torch.allclose(member.predict_distribution(x[150:])[0], log_probs, atol=1e-5)


def _eval_script():
    path = Path(__file__).parents[1] / "scripts" / "tabpfn_v2" / "eval.py"
    spec = importlib.util.spec_from_file_location("eval_tabpfn_v2_official", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_eval_official_v35_flags(monkeypatch, tmp_path):
    script = _eval_script()
    default = script.parse_args([])
    assert default.one_hot == ["all"] and default.max_features_per_member is None
    assert default.softmax_temperature == 0.9 and script.inference_option_params(default) == {}
    official = script.parse_args(["--official-v35"])
    assert official.one_hot == ["off"] and official.max_features_per_member == 768
    assert script.inference_option_params(official) == script.OFFICIAL_V35
    # 따로 준 옵션이 먼저
    mixed = script.parse_args(["--official-v35", "--no-shuffle-categories", "--one-hot", "all"])
    assert mixed.shuffle_categories is False and mixed.one_hot == ["all"]
    single = script.parse_args(["--softmax-temperature", "1.0"])
    assert script.inference_option_params(single) == {"softmax_temperature": 1.0}
    with pytest.raises(SystemExit):
        script.parse_args(["--target-transform", "safepower", "--test-time", "greedy"])
    # 회귀 전용 옵션은 분류 estimator에 넘기지 않는다
    assert "target_transform" not in script.estimator_options(official, "classification")
    reg_kw = script.estimator_options(official, "regression")
    assert reg_kw["regression_mean"] == "mixture" and reg_kw["fingerprint"] == "hash"
    # estimator와 arena fingerprint
    model = _tiny_model()
    monkeypatch.setattr(script, "load_model", lambda *a, **k: (model, None))
    official.classifier_path, official.out_dir = "unused", str(tmp_path)
    make = script.make_estimator_factory("classification", official)
    estimator = make(types.SimpleNamespace(categorical_indices=[]), "off")
    assert estimator.softmax_temperature == 1.0 and estimator.drop_constant
    assert estimator.preprocess_options == PreprocessOptions(True, "hash", "soft")
    paths, kinds = {"classification": "unused"}, {"classification": ("standard",)}
    factories = {"classification": types.SimpleNamespace(model=model)}
    printed = json.loads(script.arena_fingerprint(official, paths, factories, kinds, "off"))
    assert printed["target_transform"] == "safepower" and printed["max_features_per_member"] == 768
    plain = json.loads(script.arena_fingerprint(default, paths, factories, kinds, "all"))
    assert set(plain) == {"paths", "member_kinds", "one_hot"}  # 기본은 예전 fingerprint 그대로
