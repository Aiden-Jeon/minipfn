"""예측 분포 baseline (`minipfn.benchmarks.baselines`)과 `scripts/baselines/` 테스트.

GBDT 테스트는 extra baselines가 있을 때만 돈다 (macOS에서는 libomp도 필요하다). 공개 TFM(TabPFN,
TabICL)은 checkpoint를 내려받아야 해서 여기서는 돌리지 않는다.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from minipfn.benchmarks import baselines as bl
from minipfn.benchmarks import uncertainty as unc

SCRIPTS = Path(__file__).parents[1] / "scripts" / "baselines"


def _script(name: str):
    spec = importlib.util.spec_from_file_location(f"baselines_{name}", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _importable(name: str) -> bool:
    try:
        importlib.import_module(name)
    except Exception:  # 설치가 안 됐거나 OpenMP가 없다
        return False
    return True


def _toy(n=160, seed=0):
    g = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "a[0]": g.normal(size=n),
            "c": pd.Categorical(g.choice(["u", "v", "w"], n)),
            "o": g.choice(["p", "q", None], n).astype(object),
        }
    )
    X.loc[::13, "a[0]"] = np.nan
    y = 2 * np.nan_to_num(X["a[0]"].to_numpy()) + 3 * (X["c"] == "u") + g.normal(size=n)
    return X, np.asarray(y, dtype=float)


def test_frame_encoder_fixes_training_categories_and_renames_columns():
    X = pd.DataFrame({"x<1>": [1.0, 2.0, np.nan], "c": ["a", "b", None]})
    enc = bl.FrameEncoder().fit(X)
    assert enc.categorical_columns == ["f1"]
    test = pd.DataFrame({"x<1>": [3.0], "c": ["z"]})  # 처음 보는 범주
    out = enc.transform(test, "lightgbm")
    assert list(out.columns) == ["f0", "f1"]
    assert list(out["f1"].cat.categories) == ["a", "b"] and out["f1"].isna().all()
    cat = enc.transform(X, "catboost")
    assert cat["f1"].tolist() == ["a", "b", "nan"]
    with pytest.raises(ValueError, match="열"):
        enc.transform(X.rename(columns={"c": "d"}), "lightgbm")


def test_make_baseline_checks_the_method_for_the_task():
    assert "catboost_gaussian" in bl.methods("regression")
    assert "catboost_gaussian" not in bl.methods("classification")
    with pytest.raises(ValueError, match="baseline"):
        bl.make_baseline("catboost_gaussian", "classification")
    assert isinstance(bl.make_baseline("tabicl", "regression"), bl.TabICLBaseline)
    assert isinstance(bl.make_baseline("tabpfn_v2", "classification"), bl.OfficialTabPFNv2)


def test_gbdt_quantile_levels_hold_every_central_interval():
    levels = set(bl.GBDT_QUANTILE_LEVELS)
    for level in unc.COVERAGE_LEVELS:
        assert round((1 - level) / 2, 3) in levels and round((1 + level) / 2, 3) in levels
    assert list(bl.GBDT_QUANTILE_LEVELS) == sorted(levels)


@pytest.mark.parametrize("backend", bl.GBDTS)
def test_gbdt_baselines_fit_and_give_distributions(backend):
    if not _importable(backend):
        pytest.skip(f"{backend}를 불러올 수 없다")
    X, y = _toy()
    reg = bl.make_baseline(backend, "regression").fit(X[:120], y[:120])
    mean, log_probs, borders = reg.predict_full(X[120:])
    k = len(bl.GBDT_QUANTILE_LEVELS)
    assert mean.shape == (40,) and log_probs.shape == (40, k + 1) and borders.shape == (40, k + 2)
    assert np.allclose(mean, reg.predict(X[120:]))
    out = unc.regression_metrics(np.exp(log_probs), borders, y[120:])
    assert np.isfinite(out["crps"]) and 0 <= out["coverage_90"] <= 1

    labels = np.where(y > 1, "hi", np.where(y > -1, "mid", "lo"))
    clf = bl.make_baseline(backend, "classification").fit(X[:120], labels[:120])
    proba = clf.predict_proba(X[120:])
    assert list(clf.classes_) == ["hi", "lo", "mid"]
    assert proba.shape == (40, 3) and np.allclose(proba.sum(axis=1), 1.0)


def test_catboost_gaussian_quantiles_are_the_normal_quantiles():
    if not _importable("catboost"):
        pytest.skip("catboost를 불러올 수 없다")
    X, y = _toy()
    reg = bl.make_baseline("catboost_gaussian", "regression").fit(X[:120], y[:120])
    mean, var = reg.predict_mean_var(X[120:])
    _, log_probs, borders = reg.predict_full(X[120:])
    h = unc.bar_histogram(np.exp(log_probs), borders)
    q = unc.quantile(h, [0.05, 0.5, 0.95])
    assert np.allclose(q[:, 1], mean, atol=1e-6)
    assert np.allclose(q[:, 2] - q[:, 1], 1.6448536 * np.sqrt(var), rtol=1e-5)


def test_eval_script_skips_suites_without_the_method():
    script = _script("eval")
    args = script.parse_args(["--method", "catboost_gaussian"])
    tasks = script.selected_tasks(args)
    assert tasks and {suite for suite, _ in tasks} == {"regression"}
    with pytest.raises(SystemExit, match="없다"):
        script.selected_tasks(
            script.parse_args(["--method", "catboost_gaussian", "--suite", "classification"])
        )
    with pytest.raises(SystemExit, match="task id"):
        script.selected_tasks(script.parse_args(["--method", "lightgbm", "--tasks", "1"]))


def _write_run(path: Path, rows: list[dict], with_score: bool) -> None:
    path.mkdir(parents=True)
    table = pd.DataFrame(rows)
    if with_score:
        table.to_csv(path / "eval_uncertainty_per_dataset.csv", index=False)
    else:
        table.drop(columns=["score"]).to_csv(path / "eval_uncertainty_per_dataset.csv", index=False)
        table[["suite", "task_id", "dataset", "score"]].to_csv(
            path / "eval_per_dataset.csv", index=False
        )


def test_compare_uses_common_datasets_and_ranks_methods(tmp_path, capsys):
    def row(task_id, rmse, crps, coverage):
        return {
            "suite": "regression",
            "task_id": task_id,
            "dataset": f"d{task_id}",
            "score": -rmse,
            "rmse_scaled": rmse,
            "crps_scaled": crps,
            "coverage_90": coverage,
            "pit_ks": 0.1,
        }

    # a는 점 예측이 좋고 b는 분포가 좋다. 데이터셋 3은 a에만 있어 빠진다
    _write_run(
        tmp_path / "a",
        [row(1, 0.5, 0.4, 0.80), row(2, 0.5, 0.4, 0.85), row(3, 1, 1, 1)],
        with_score=False,
    )
    _write_run(tmp_path / "b", [row(1, 0.6, 0.3, 0.92), row(2, 0.6, 0.3, 0.97)], with_score=True)
    out = tmp_path / "out"
    _script("compare").main([f"a={tmp_path / 'a'}", f"b={tmp_path / 'b'}", "--out-dir", str(out)])
    ranks = pd.read_csv(out / "compare_ranks.csv", index_col=0)
    assert ranks.loc["a", "rmse_scaled"] == 1 and ranks.loc["a", "crps_scaled"] == 2
    # coverage는 목표 0.9와의 거리: b(0.02, 0.07)가 a(0.10, 0.05)보다 데이터셋 1에서만 낫다
    assert ranks.loc["a", "coverage_90"] == 1.5 and ranks.loc["b", "coverage_90"] == 1.5
    assert ranks["datasets"].eq(2).all()
    assert ranks["rank_spearman"].eq(-1.0).all()  # 두 순위가 늘 반대
    assert "Spearman 평균: -1.000" in capsys.readouterr().out
    # 출처#suite는 그 suite만 쓴다: 회귀만 있는 결과에서 분류를 고르면 남는 것이 없다
    assert _script("compare").load(f"{tmp_path / 'a'}#classification").empty
    with pytest.raises(SystemExit, match="이름=출처"):
        _script("compare").main([str(tmp_path / "a")])


def _regression_row(task_id, **metrics):
    return {
        "suite": "regression",
        "task_id": task_id,
        "dataset": f"d{task_id}",
        "score": -1.0,
        "rmse_scaled": 1.0,
        "crps_scaled": 1.0,
        **metrics,
    }


def test_compare_scores_worst_group_coverage_by_distance_to_the_target(tmp_path):
    # 너무 넓은 구간(묶음별 최솟값 0.99)이 목표에 가까운 0.88을 이기지 않는다
    _write_run(tmp_path / "wide", [_regression_row(1, coverage_min_by_width_90=0.99)], True)
    _write_run(tmp_path / "good", [_regression_row(1, coverage_min_by_width_90=0.88)], True)
    out = tmp_path / "out"
    args = [f"wide={tmp_path / 'wide'}", f"good={tmp_path / 'good'}", "--out-dir", str(out)]
    _script("compare").main(args)
    ranks = pd.read_csv(out / "compare_ranks.csv", index_col=0)
    assert ranks.loc["good", "coverage_min_by_width_90"] == 1
    assert ranks.loc["wide", "coverage_min_by_width_90"] == 2


def test_compare_ranks_only_datasets_where_every_method_has_the_metric(tmp_path, capsys):
    # 데이터셋 2에서 b의 pit_ks가 NaN이면 그 데이터셋은 pit_ks 순위에서 모두 빠진다
    _write_run(
        tmp_path / "a", [_regression_row(1, pit_ks=0.1), _regression_row(2, pit_ks=0.5)], True
    )
    _write_run(
        tmp_path / "b", [_regression_row(1, pit_ks=0.2), _regression_row(2, pit_ks=None)], True
    )
    out = tmp_path / "out"
    _script("compare").main([f"a={tmp_path / 'a'}", f"b={tmp_path / 'b'}", "--out-dir", str(out)])
    ranks = pd.read_csv(out / "compare_ranks.csv", index_col=0)
    assert ranks.loc["a", "pit_ks"] == 1 and ranks.loc["b", "pit_ks"] == 2
    assert "{'pit_ks': 1}" in capsys.readouterr().out


def test_compare_suite_suffix_and_empty_inputs(tmp_path):
    compare = _script("compare")
    run = tmp_path / "run#3"  # 경로 안의 #는 suite 필터가 아니다
    _write_run(run, [_regression_row(1, pit_ks=0.1)], True)
    assert len(compare.load(str(run))) == 1
    assert len(compare.load(f"{run}#regression")) == 1
    with pytest.raises(SystemExit, match="결과가 없다"):
        compare.collect([f"a={run}#classification"])
    other = tmp_path / "other"
    _write_run(other, [_regression_row(2, pit_ks=0.1)], True)
    with pytest.raises(SystemExit, match="데이터셋이 없다"):
        compare.common_datasets(compare.collect([f"a={run}", f"b={other}"]))


def test_dataset_means_average_folds_and_skip_columns():
    rows = [
        {"suite": "regression", "task_id": 1, "dataset": "d", "fold": f, "crps": c, "method": "m"}
        for f, c in ((0, 1.0), (1, 3.0))
    ]
    table = unc.dataset_means(rows, exclude=("method",))
    assert list(table.columns) == ["suite", "task_id", "dataset", "crps"]
    assert table["crps"].tolist() == [2.0]
