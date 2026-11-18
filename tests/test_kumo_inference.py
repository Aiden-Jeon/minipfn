"""Kumo-Tabular 추론에서 가져온 옵션 (issue #100: 범주 빈도 열·드문 범주 결측, logit 평균, 회귀
trimmed 합치기·target 부호 뒤집기) 테스트. 기본값은 예전과 같아야 한다."""

import importlib.util
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from minipfn.inference.ensemble import EnsembleClassifier, EnsembleRegressor
from minipfn.inference.preprocessing import (
    apply_category_steps,
    category_count_tables,
    rare_category_maps,
)
from minipfn.models.cell_transformer import CellTransformer, CellTransformerConfig


def _tiny_model(n_out: int = 10) -> CellTransformer:
    torch.manual_seed(0)
    model = CellTransformer(CellTransformerConfig(emsize=16, nlayers=2, nhead=2, n_out=n_out))
    for p in model.parameters():
        torch.nn.init.normal_(p, std=0.3)
    return model.eval()


def _frame(n: int = 260, seed: int = 0) -> pd.DataFrame:
    g = np.random.default_rng(seed)
    many = np.array([f"v{i}" for i in range(80)])
    return pd.DataFrame(
        {
            "a": g.normal(size=n),
            "b": g.lognormal(size=n),
            "city": many[np.minimum(g.geometric(0.04, n), 79)],  # 값 많고 드문 값도 있다
            "color": g.choice(["red", "blue", "green", None], n),
        }
    )


def _eval_script():
    path = Path(__file__).parents[1] / "scripts" / "tabpfn_v2" / "eval.py"
    spec = importlib.util.spec_from_file_location("eval_tabpfn_v2_kumo", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_options_are_the_old_behavior():
    df = _frame()
    y_cls = np.random.default_rng(1).choice(["x", "y", "z"], len(df))
    old = EnsembleClassifier(_tiny_model(), n_estimators=3, member_kinds="standard")
    new = EnsembleClassifier(
        _tiny_model(),
        n_estimators=3,
        member_kinds="standard",
        category_counts=None,
        rare_category_min_count=None,
        class_average="proba",
    )
    a = old.fit(df[:200], y_cls[:200]).predict_proba(df[200:])
    b = new.fit(df[:200], y_cls[:200]).predict_proba(df[200:])
    assert np.array_equal(a, b)
    y = np.random.default_rng(2).normal(size=len(df))
    borders = torch.linspace(-4, 4, 33)
    kw = dict(n_estimators=4, member_kinds="standard")
    old_reg = EnsembleRegressor(_tiny_model(32), borders, **kw).fit(df[:200], y[:200])
    new_reg = EnsembleRegressor(
        _tiny_model(32), borders, regression_trim=0.0, target_flip=False, **kw
    ).fit(df[:200], y[:200])
    assert np.array_equal(old_reg.predict(df[200:]), new_reg.predict(df[200:]))
    assert torch.equal(
        old_reg.predict_distribution(df[200:])[0], new_reg.predict_distribution(df[200:])[0]
    )
    with pytest.raises(ValueError, match="class_average"):
        EnsembleClassifier(_tiny_model(), n_estimators=2, class_average="median")
    with pytest.raises(ValueError, match="regression_trim"):
        EnsembleRegressor(_tiny_model(32), borders, n_estimators=2, regression_trim=0.5)
    with pytest.raises(ValueError, match="target_flip"):
        EnsembleRegressor(
            _tiny_model(32), borders, n_estimators=2, target_flip=True, regression_mean="mixture"
        )


def test_category_counts_are_log_train_frequencies():
    x = np.array([[0.0, 1.0], [0.0, 2.0], [1.0, 3.0], [np.nan, 4.0], [2.0, 5.0], [0.0, 6.0]])
    tables = category_count_tables(x, [0], min_cardinality=2)
    assert list(tables) == [0]  # 값 3개 > 2
    assert category_count_tables(x, [0], min_cardinality=3) == {}
    out = apply_category_steps(x, tables, {})
    assert out.shape == (6, 3) and np.array_equal(out[:, :2], x, equal_nan=True)
    expected = np.log1p([3, 3, 1, 1, 1, 3])  # 결측은 학습 행 결측 수(1)
    assert np.allclose(out[:, 2], expected)
    # 다른 행: 결측과 학습에 없던 값(-1)은 결측 칸을 같이 쓴다
    test = np.array([[1.0, 0.0], [np.nan, 0.0], [-1.0, 0.0]])
    assert np.allclose(apply_category_steps(test, tables, {})[:, 2], np.log1p([1, 1, 1]))


def test_rare_categories_become_missing_and_codes_stay_contiguous():
    x = np.array([[0.0], [0.0], [1.0], [2.0], [2.0], [3.0], [np.nan]])
    maps = rare_category_maps(x, [0], min_count=2)
    out = apply_category_steps(x, {}, maps)[:, 0]
    assert np.array_equal(out, [0, 0, np.nan, 1, 1, np.nan, np.nan], equal_nan=True)
    assert rare_category_maps(x, [0], min_count=1) == {}  # 드문 값이 없으면 표도 없다
    # 학습에 없던 값(-1)은 그대로, 빈도는 원래 번호로 센다
    test = apply_category_steps(np.array([[-1.0], [3.0]]), {}, maps)
    assert np.array_equal(test[:, 0], [-1.0, np.nan], equal_nan=True)
    both = apply_category_steps(x, category_count_tables(x, [0], 1), maps)
    assert np.allclose(both[:, 1], np.log1p([2, 2, 1, 2, 2, 1, 1]))


def test_category_steps_in_the_ensemble():
    df = _frame()
    y = np.random.default_rng(3).choice([0, 1], len(df))
    clf = EnsembleClassifier(
        _tiny_model(),
        n_estimators=2,
        member_kinds="standard",
        category_counts=50,
        rare_category_min_count=2,
    ).fit(df[:200], y[:200])
    # city(값이 50개 넘음)에 빈도 열 하나가 붙는다. color는 값이 3개라 빈도 열이 없다
    assert list(clf.count_tables_) == [2] and 2 in clf.rare_maps_
    proba = clf.predict_proba(df[200:])
    assert proba.shape == (60, 2) and np.allclose(proba.sum(1), 1)
    x_train = clf._fit_columns(df[:200])
    assert x_train.shape[1] == df.shape[1] + 1
    rare = clf.rare_maps_[2]
    assert np.isnan(x_train[:, 2]).sum() == int(np.isnan(rare[1]).sum())  # 한 번 나온 값마다 한 행
    # 같은 열 단계를 테스트 행에도 (학습 행을 다시 넣으면 학습 때와 같다)
    from minipfn.inference.preprocessing import apply_encoder

    again = clf._columns(apply_encoder(df[:200], clf.encoded_ix_, clf.encoder_))
    assert np.array_equal(again, x_train, equal_nan=True)


def test_logit_average_is_softmax_of_mean_logits():
    df = _frame()
    y = np.random.default_rng(4).choice(["x", "y", "z"], len(df))
    clf = EnsembleClassifier(
        _tiny_model(), n_estimators=3, member_kinds="standard", class_average="logit"
    ).fit(df[:200], y[:200])
    logits = torch.stack([lg / clf.softmax_temperature for _, lg in clf.member_logits(df[200:])])
    expected = torch.softmax(logits.mean(0), -1).numpy()
    assert np.allclose(clf.predict_proba(df[200:]), expected, atol=1e-6)
    plain = EnsembleClassifier(_tiny_model(), n_estimators=3, member_kinds="standard")
    probs = plain.fit(df[:200], y[:200]).predict_proba(df[200:])
    assert not np.allclose(probs, expected)  # 확률 평균과는 다르다


def test_regression_trim_drops_extreme_member_means():
    df = _frame()
    y = np.random.default_rng(5).normal(size=len(df))
    reg = EnsembleRegressor(
        _tiny_model(32),
        torch.linspace(-4, 4, 33),
        n_estimators=8,
        member_kinds="standard",
        regression_trim=0.2,
    ).fit(df[:200], y[:200])
    means = torch.stack([reg._member_mean(i, p) for i, p in reg._member_probs(df[200:])])
    kept = means.sort(0).values[1:7]  # int(0.2 × 8) = 1개씩
    expected = kept.mean(0).numpy() * reg.y_std_ + reg.y_mean_
    assert np.allclose(reg.predict(df[200:]), expected, atol=1e-6)
    full = reg.predict_full(df[200:])[0]
    assert np.allclose(full, expected, atol=1e-6)
    reg.regression_trim = 0.0
    assert not np.allclose(reg.predict(df[200:]), expected)


class _PeakAtFirstTarget(nn.Module):
    """학습 행 첫 target(모델이 받은 값)이 든 구간에 확률을 모으는 가짜 모델."""

    def __init__(self, borders: torch.Tensor) -> None:
        super().__init__()
        self.borders = borders
        self.config = types.SimpleNamespace(n_out=len(borders) - 1, max_features=100)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        n_test = x.shape[1] - y.shape[1]
        k = int(torch.searchsorted(self.borders, y[0, 0].float()).clamp(1, len(self.borders) - 1))
        logits = torch.full((1, n_test, len(self.borders) - 1), -30.0)
        logits[..., k - 1] = 30.0
        return logits


@pytest.mark.parametrize("target_transform", ["none", "power"])
def test_target_flip_members_predict_in_the_original_direction(target_transform):
    # 경계가 0에 대해 대칭이 아니어야 뒤집기의 경계 처리를 확인할 수 있다
    borders = torch.cat([torch.linspace(-4, 0, 41)[:-1], torch.linspace(0, 6, 121)])
    g = np.random.default_rng(6)
    X = g.normal(size=(120, 3))
    y = g.normal(size=120) * 2.0 + 5.0
    y[0] = 7.3
    kw = dict(n_estimators=4, member_kinds="standard", target_transform=target_transform)
    plain = EnsembleRegressor(_PeakAtFirstTarget(borders), borders, **kw).fit(X[:100], y[:100])
    flip = EnsembleRegressor(_PeakAtFirstTarget(borders), borders, target_flip=True, **kw)
    flip.fit(X[:100], y[:100])
    assert [p.member.target_flip for p in flip.preprocessors_] == [False, True, False, True]
    # 뒤집은 멤버는 -u를 받는다
    assert np.allclose(flip.y_trains_[1], -plain.y_trains_[1])
    width = float((borders[1:] - borders[:-1]).max()) * plain.y_std_
    a, b = plain.predict(X[100:]), flip.predict(X[100:])
    assert np.allclose(a, 7.3, atol=width) and np.allclose(b, 7.3, atol=width)
    # 분포도 같은 곳에 모인다 (공통 경계 위)
    log_p, edges = flip.predict_distribution(X[100:])
    center = (edges[1:] + edges[:-1]) / 2
    mode = center[log_p[0].argmax()].item()
    assert abs(mode - 7.3) < 2 * width


def test_eval_kumo_flags(monkeypatch):
    script = _eval_script()
    default = script.parse_args([])
    assert script.inference_option_params(default) == {}
    args = script.parse_args(
        [
            "--category-counts",
            "50",
            "--rare-category-min-count",
            "2",
            "--class-average",
            "logit",
            "--regression-trim",
            "0.2",
            "--target-flip",
        ]
    )
    params = script.inference_option_params(args)
    assert params == {
        "category_counts": 50,
        "rare_category_min_count": 2,
        "class_average": "logit",
        "regression_trim": 0.2,
        "target_flip": True,
    }
    cls_kw = script.estimator_options(args, "classification")
    reg_kw = script.estimator_options(args, "regression")
    assert "regression_trim" not in cls_kw and "target_flip" not in cls_kw
    assert cls_kw["class_average"] == "logit" and "class_average" not in reg_kw
    assert reg_kw["regression_trim"] == 0.2 and reg_kw["category_counts"] == 50
    with pytest.raises(SystemExit):
        script.parse_args(["--target-flip", "--regression-mean", "mixture"])
    with pytest.raises(SystemExit):
        script.parse_args(["--class-average", "logit", "--test-time", "greedy"])
