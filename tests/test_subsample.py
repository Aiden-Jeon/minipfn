"""추론 때 학습 행을 줄이는 subsample (`minipfn.inference.subsample`, `--subsample-train-rows`)."""

from __future__ import annotations

import importlib.util
import json
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from minipfn.inference.ensemble import EnsembleClassifier, EnsembleRegressor
from minipfn.inference.subsample import SubsampledEstimator, subsample_indices
from minipfn.models.cell_transformer import CellTransformer, CellTransformerConfig


def _eval_script():
    path = Path(__file__).parents[1] / "scripts" / "tabpfn_v2" / "eval.py"
    spec = importlib.util.spec_from_file_location("eval_tabpfn_v2_subsample", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tiny_model(n_out: int = 10) -> CellTransformer:
    torch.manual_seed(0)
    model = CellTransformer(CellTransformerConfig(emsize=16, nlayers=2, nhead=2, n_out=n_out))
    for p in model.parameters():
        torch.nn.init.normal_(p, std=0.3)
    return model.eval()


class _Recorder:
    """fit에 들어온 인자를 남기는 estimator."""

    classes_ = np.array([0, 1])

    def fit(self, X, y, **kwargs):
        self.X, self.y, self.kwargs = X, y, kwargs
        return self

    def predict(self, X):
        return np.zeros(len(X))


def test_regression_subsample_size_and_order():
    ix = subsample_indices(np.arange(1000.0), 100, stratify=False, seed=0)
    assert len(ix) == 100 and len(np.unique(ix)) == 100
    assert np.all(np.diff(ix) > 0) and ix.min() >= 0 and ix.max() < 1000
    # 행이 n개 이하면 모두, 순서 그대로
    small = np.arange(50.0)
    assert np.array_equal(subsample_indices(small, 50, stratify=False, seed=0), np.arange(50))
    assert np.array_equal(subsample_indices(small, 80, stratify=True, seed=0), np.arange(50))
    with pytest.raises(ValueError):
        subsample_indices(np.arange(5.0), 0, stratify=False, seed=0)


def test_stratified_subsample_keeps_class_ratios_and_rare_classes():
    # 클래스 a 900, b 99, c 1행 → 100행: a 90·b 9 근처, c는 최소 1행
    y = np.array(["a"] * 900 + ["b"] * 99 + ["c"])
    ix = subsample_indices(y, 100, stratify=True, seed=0)
    assert len(ix) == 100 and len(np.unique(ix)) == 100
    picked = pd.Series(y[ix]).value_counts()
    assert picked["c"] == 1 and abs(picked["a"] - 90) <= 1 and abs(picked["b"] - 9.9) <= 1
    # 몫의 합이 n보다 작을 때(내림)와 클 때(최소 1행) 모두 정확히 n
    y = np.repeat(np.arange(7), [400, 300, 200, 50, 30, 15, 5])
    for n in (7, 10, 13, 333):
        ix = subsample_indices(y, n, stratify=True, seed=1)
        assert len(ix) == n and set(y[ix]) == set(range(7))
    # 클래스 수가 n보다 많으면 클래스마다 1행
    ix = subsample_indices(np.repeat(np.arange(5), 3), 3, stratify=True, seed=0)
    assert sorted(np.repeat(np.arange(5), 3)[ix]) == [0, 1, 2, 3, 4]


def test_subsample_is_reproducible_by_seed():
    y = np.random.default_rng(0).choice(3, 5000)
    for stratify in (True, False):
        a = subsample_indices(y, 500, stratify=stratify, seed=3)
        assert np.array_equal(a, subsample_indices(y, 500, stratify=stratify, seed=3))
        assert not np.array_equal(a, subsample_indices(y, 500, stratify=stratify, seed=4))


def test_wrapper_subsamples_rows_and_row_aligned_fit_args():
    X = pd.DataFrame({"a": np.arange(300.0), "b": np.arange(300.0) * 2})
    y = np.repeat([0, 1], 150)
    groups = np.arange(300) // 10
    inner = _Recorder()
    est = SubsampledEstimator(inner, 40, stratify=True, seed=0).fit(X, y, groups=groups, other=3)
    assert est.train_rows_ == (300, 40)
    rows = inner.X["a"].to_numpy().astype(int)
    assert len(rows) == 40 and np.array_equal(inner.y, y[rows])
    assert np.array_equal(inner.kwargs["groups"], groups[rows]) and inner.kwargs["other"] == 3
    # 이름이 행 인자가 아니면 길이가 학습 행 수와 같아도 자르지 않는다
    weights = np.arange(300.0)
    inner = _Recorder()
    SubsampledEstimator(inner, 40, stratify=True, seed=0).fit(X, y, per_feature=weights)
    assert inner.kwargs["per_feature"] is weights
    assert (inner.y == 0).sum() == 20  # 층화
    # 나머지 속성은 감싼 estimator의 것
    assert np.array_equal(est.classes_, [0, 1]) and len(est.predict(X)) == 300
    # 행이 max_rows개 이하면 그대로 넘긴다 (같은 객체)
    inner = _Recorder()
    SubsampledEstimator(inner, 300, stratify=True).fit(X, y, groups=groups)
    assert inner.X is X and inner.y is y and inner.kwargs["groups"] is groups


def test_ensemble_on_subsample_equals_fit_on_the_chosen_rows():
    g = np.random.default_rng(0)
    X = g.normal(size=(200, 4))
    y = (X[:, 0] + 0.3 * g.normal(size=200) > 0).astype(int)
    model = _tiny_model()
    kw = dict(n_estimators=2, device="cpu", random_state=0)
    wrapped = SubsampledEstimator(EnsembleClassifier(model, **kw), 60, stratify=True, seed=5)
    wrapped.fit(X, y)
    ix = subsample_indices(y, 60, stratify=True, seed=5)
    direct = EnsembleClassifier(model, **kw).fit(X[ix], y[ix])
    assert np.array_equal(wrapped.predict_proba(X[:50]), direct.predict_proba(X[:50]))


def test_eval_flag_off_is_the_old_estimator(monkeypatch, tmp_path):
    script = _eval_script()
    model = _tiny_model()
    monkeypatch.setattr(script, "load_model", lambda *a, **k: (model, None))
    paths, kinds = {"classification": "unused"}, {"classification": ("standard",)}
    factories = {"classification": types.SimpleNamespace(model=model)}
    task = types.SimpleNamespace(categorical_indices=[])
    g = np.random.default_rng(1)
    X, y = g.normal(size=(120, 3)), g.choice(3, 120)

    off = script.parse_args(["--classifier-path", "unused", "--out-dir", str(tmp_path)])
    assert off.subsample_train_rows is None
    plain = script.make_estimator_factory("classification", off)(task, "all")
    assert type(plain) is EnsembleClassifier
    printed = json.loads(script.arena_fingerprint(off, paths, factories, kinds, "all"))
    assert "subsample_train_rows" not in printed  # 기본은 예전 fingerprint 그대로

    on = script.parse_args(
        ["--classifier-path", "unused", "--out-dir", str(tmp_path), "--subsample-train-rows", "50"]
    )
    wrapped = script.make_estimator_factory("classification", on)(task, "all")
    assert isinstance(wrapped, SubsampledEstimator) and wrapped.stratify
    printed = json.loads(script.arena_fingerprint(on, paths, factories, kinds, "all"))
    assert printed["subsample_train_rows"] == 50
    # 학습 행이 N개 이하면 줄이지 않은 것과 비트 동일
    big = script.parse_args(
        ["--classifier-path", "unused", "--out-dir", str(tmp_path), "--subsample-train-rows", "500"]
    )
    same = script.make_estimator_factory("classification", big)(task, "all").fit(X, y)
    plain.fit(X, y)
    assert np.array_equal(same.predict_proba(X[:30]), plain.predict_proba(X[:30]))
    with pytest.raises(SystemExit):
        script.parse_args(["--subsample-train-rows", "0"])


def test_eval_regression_subsample_is_not_stratified(monkeypatch, tmp_path):
    script = _eval_script()
    model = _tiny_model(n_out=16)
    borders = torch.linspace(-3, 3, 17)
    monkeypatch.setattr(script, "load_model", lambda *a, **k: (model, borders))
    args = script.parse_args(
        ["--regressor-path", "unused", "--out-dir", str(tmp_path), "--subsample-train-rows", "40"]
    )
    est = script.make_estimator_factory("regression", args)(
        types.SimpleNamespace(categorical_indices=[]), "all"
    )
    assert isinstance(est, SubsampledEstimator) and not est.stratify
    assert isinstance(est.estimator, EnsembleRegressor)
    X = np.random.default_rng(2).normal(size=(90, 3))
    est.fit(X, X[:, 0])
    assert est.train_rows_ == (90, 40) and est.predict(X[:5]).shape == (5,)


def test_expands_one_hot_looks_at_the_subsampled_rows():
    # 학습 행 전체로는 범주 100개(>20)라 펼치지 않지만, 10행으로 줄이면 범주가 10개 이하라 펼친다.
    # 줄인 행으로 보지 않으면 one-hot 모드를 한 묶음으로 보고 결과를 잘못 다시 쓴다
    script = _eval_script()
    X = pd.DataFrame({"a": [f"c{i}" for i in range(100)]})
    task = types.SimpleNamespace(
        X=X,
        y=pd.Series(np.zeros(100)),
        categorical_indices=[0],
        task_type="regression",
        folds=lambda n_folds=10: [(0, np.arange(100), np.arange(0))],
    )
    assert not script.expands_one_hot(task)
    assert script.expands_one_hot(task, subsample_rows=10)
    assert not script.expands_one_hot(task, subsample_rows=100)  # 줄이지 않으면 그대로
