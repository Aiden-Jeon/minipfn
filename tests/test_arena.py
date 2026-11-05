"""TabArena·BeyondArena 하네스(`minipfn.benchmarks.arena`)의 규칙 테스트. 데이터와 baseline 결과를
받지 않는다. estimator를 감싸는 system 테스트는 extra `arena`(tabarena)가 있을 때만 돈다."""

import argparse
import importlib.util
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from minipfn.benchmarks import arena, beyondarena, tabarena


def _row(**kw):
    base = {"problem_type": "binary", "max_train_rows": 1000, "n_features": 10, "n_classes": 2}
    return pd.Series({**base, **kw})


def test_limits_mark_splits_too_big_for_the_model():
    lim = arena.ModelLimits(max_train_rows=10_000, max_features=500, max_classes=10)
    assert lim.skip_reason(_row()) is None
    assert lim.skip_reason(_row(max_train_rows=10_001)) == "train rows 10001 > 10000"
    assert lim.skip_reason(_row(n_features=501)) == "features 501 > 500"
    assert lim.skip_reason(_row(problem_type="multiclass", n_classes=11)) == "classes 11 > 10"
    # 회귀의 n_classes는 -1이라 클래스 상한에 걸리지 않는다
    assert lim.skip_reason(_row(problem_type="regression", n_classes=-1)) is None
    assert arena.ModelLimits().skip_reason(_row(max_train_rows=10**7)) is None


def test_limits_per_problem_type_skip_types_without_a_model():
    limits = {"regression": arena.ModelLimits(max_train_rows=1000)}
    assert arena.skip_reason(limits, _row()) == "problem_type binary"
    assert arena.skip_reason(limits, _row(problem_type="regression", n_classes=-1)) is None
    reason = arena.skip_reason(limits, _row(problem_type="regression", max_train_rows=1001))
    assert reason == "train rows 1001 > 1000"


def test_display_name_strips_link_and_imputed_tag():
    assert arena.display_name("[TabPFNv2 (default)](https://x) [35.29% IMPUTED]") == (
        "TabPFNv2 (default)"
    )
    assert arena.display_name("TabFM+") == "TabFM+"


def _per_split():
    rows = []
    for d in ("a", "b", "c"):
        rows.append({"dataset": d, "fold": 0, "method": "RF (default)", "metric_error": 0.5})
        rows.append({"dataset": d, "fold": 0, "method": "XGB (default)", "metric_error": 0.4})
    rows.append({"dataset": "a", "fold": 0, "method": "new", "metric_error": 0.3})
    rows.append({"dataset": "b", "fold": 0, "method": "new", "metric_error": 0.4})
    return pd.DataFrame(rows)


def test_missing_splits_are_filled_with_the_fillna_method():
    ours = arena.with_imputed(_per_split(), "new", "RF (default)")
    assert list(ours["dataset"]) == ["a", "b", "c"]
    assert list(ours["imputed"]) == [False, False, True]
    assert ours.loc[2, "metric_error"] == 0.5  # c는 RF의 결과
    assert (ours["method"] == "new").all()


def test_winrate_counts_ties_as_half_and_reports_run_only_splits():
    per_split = _per_split()
    ours = arena.with_imputed(per_split, "new", "RF (default)")
    w = arena.winrates(per_split, ours)["XGBoost (default)"]
    # a 승, b 무, c(imputed RF 0.5 vs 0.4) 패
    assert w["all"] == pytest.approx((1 + 0.5 + 0) / 3)
    assert w["run"] == pytest.approx(0.75)
    assert (w["splits"], w["run_splits"]) == (3, 2)
    assert "TabPFNv2 (default)" not in arena.winrates(per_split, ours)  # 결과가 없는 방법
    # BeyondArena 결과처럼 앞에 "TA-"가 붙고 대소문자가 달라도 같은 방법으로 본다
    renamed = per_split.replace({"method": {"XGB (default)": "TA-xgb (default)"}})
    assert arena.winrates(renamed, ours)["XGBoost (default)"] == w


def test_skip_reasons_are_counted_by_kind():
    reasons = pd.Series([None, "train rows 11640 > 10000", "train rows 20000 > 10000", "failed"])
    assert arena._reason_counts(reasons) == {"train rows > 10000": 2, "run": 1, "failed": 1}


def test_comparison_puts_public_and_recomputed_scores_side_by_side():
    website = pd.DataFrame(
        {
            "Model": ["[XGBoost (default)](u)", "new"],
            "Elo [⬆️]": [1180.0, 1300.0],
            "Score [⬆️]": [0.05, 0.2],
        }
    )
    public = pd.DataFrame({"Model": ["[XGBoost (default)](u)", "[Other (default)](u)"]})
    public["Elo [⬆️]"] = [1181.0, 1500.0]
    public["Score [⬆️]"] = [0.05, 0.6]
    public["name"] = public["Model"].map(arena.display_name)
    out = arena.compare_with_public(website, public, "new")
    assert list(out["name"]) == ["new", "XGBoost (default)", "Other (default)"]
    assert out.loc[1, "elo_public"] == 1181.0
    assert bool(out.loc[0, "is_new"]) and np.isnan(out.loc[0, "elo_public"])
    assert np.isnan(out.loc[2, "elo"])  # 이 실행의 pool에 없는 방법


def test_suites_point_at_the_official_splits_and_public_tables():
    assert tabarena.SUITE.subset == ("lite",)
    assert "splits_lite" in tabarena.SUITE.public_csv
    assert "imputation_yes" in tabarena.SUITE.public_csv
    assert tabarena.SUITE.breakdown["regression"][0] == ("lite", "regression")
    core = beyondarena.suite()
    assert core.subset == ("core2k",) and core.name == "beyondarena"
    assert core.breakdown["split_temporal"][0] == ("core2k", "temporal")
    assert "split_temporal/size_all" in core.breakdown["split_temporal"][1]
    lite = beyondarena.suite("lite")
    assert lite.subset == ("core2k", "lite")
    assert lite.breakdown["size_tiny"][0] == ("core2k", "lite", "tiny")


def _system():
    pytest.importorskip("tabarena")
    from minipfn.benchmarks import _arena_system

    return _arena_system


class _Constant:
    """학습 행의 클래스 비율을 그대로 예측하는 분류기 (bag마다 본 클래스가 다르게)."""

    def fit(self, X, y):
        self.classes_, counts = np.unique(y, return_counts=True)
        self.p_ = counts / counts.sum()
        self.n_fit_ = len(X)
        return self

    def predict_proba(self, X):
        return np.tile(self.p_, (len(X), 1))


def test_prepare_frame_turns_dates_into_numbers_and_strings_into_categoricals():
    system = _system()
    X = pd.DataFrame(
        {
            "num": [1.0, 2.0],
            "when": pd.to_datetime(["2020-01-01", None]),
            "text": ["a b", "c"],
            "flag": [True, False],
        }
    )
    out, categorical = system.prepare_frame(X)
    assert categorical == [2]
    assert out["when"].iloc[0] == pytest.approx(1577836800.0) and np.isnan(out["when"].iloc[1])
    assert out["flag"].tolist() == [1.0, 0.0]
    assert X["when"].dtype.kind == "M"  # 원래 frame은 그대로


def test_bagging_fits_k_estimators_and_aligns_their_classes():
    system = _system()
    key = system.register_factory(lambda problem_type, categorical: _Constant())
    model = object.__new__(system.EstimatorSystem)
    model.factory, model.bagging = key, 4
    X = pd.DataFrame({"x": np.arange(20.0)}, index=np.arange(100, 120))
    y = pd.Series(["a"] * 17 + ["b"] * 3, index=X.index)  # b가 4개보다 적어 층화하지 않는다
    model._fit_system(X, y, problem_type="binary", random_state=0)
    assert [e.n_fit_ for e in model.estimators_] == [15, 15, 15, 15]
    proba = model._predict_proba(X.iloc[:5])
    assert list(proba.columns) == ["a", "b"]
    assert list(proba.index) == list(X.index[:5])
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    np.testing.assert_allclose(proba["b"], 3 / 20, atol=0.06)

    model.bagging = 0
    model._fit_system(X, y, problem_type="binary", random_state=None)
    assert [e.n_fit_ for e in model.estimators_] == [20]


def _eval_script():
    path = Path(__file__).parents[1] / "scripts" / "tabpfn_v2" / "eval.py"
    spec = importlib.util.spec_from_file_location("eval_tabpfn_v2_arena", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_eval_script_limits_follow_the_model_architecture(monkeypatch):
    script = _eval_script()
    archs = {"classification": "v2.5", "regression": None}
    monkeypatch.setattr(script, "model_architecture", lambda model, task: archs[task])
    clf = types.SimpleNamespace(config=types.SimpleNamespace(n_out=10))
    args = argparse.Namespace(max_train_rows=None, max_features=None)
    limits = script.arena_limits({"classification": (clf, None), "regression": (clf, "b")}, args)
    assert limits["binary"] == arena.ModelLimits(50_000, 2_000, 10)
    assert limits["multiclass"] is limits["binary"]
    assert limits["regression"] == arena.ModelLimits(10_000, 500)  # 모르는 구조는 v2
    only = script.arena_limits(
        {"regression": (clf, "b")}, argparse.Namespace(max_train_rows=2048, max_features=None)
    )
    assert only == {"regression": arena.ModelLimits(2048, 500)}  # 분류 체크포인트가 없으면 빠진다


def test_eval_script_accepts_arena_suites():
    script = _eval_script()
    args = script.parse_args(["--suite", "beyondarena-lite", "--bagging", "8"])
    assert (args.suite, args.bagging) == ("beyondarena-lite", 8)
    assert script.arena_suite("beyondarena").subset == ("core2k",)
    assert script.arena_suite("tabarena-lite").name == "tabarena-lite"
