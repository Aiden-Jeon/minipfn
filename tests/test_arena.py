"""TabArena·BeyondArena 하네스(`minipfn.benchmarks.arena`)의 규칙 테스트. 데이터와 baseline 결과를
받지 않는다. estimator를 감싸는 system 테스트는 extra `arena`(tabarena)가 있을 때만 돈다."""

import argparse
import importlib.util
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


def test_comparison_without_public_table_or_new_method():
    website = pd.DataFrame({"Model": ["[XGBoost (default)](u)"], "Elo [⬆️]": [1180.0]})
    out = arena.compare_with_public(website, None, None)
    assert list(out["name"]) == ["XGBoost (default)"]
    assert np.isnan(out.loc[0, "elo_public"]) and not out["is_new"].any()


def test_public_table_failure_does_not_raise(monkeypatch):
    def fail(path):
        raise OSError("HTTP 503")

    monkeypatch.setattr(arena, "load_public_leaderboard", fail)
    assert arena.try_public_leaderboard("data/x.csv") is None


def test_run_id_changes_with_the_estimator_fingerprint():
    a = arena._run_id("clf=step_100k", "minipfn", 0)
    assert a == arena._run_id("clf=step_100k", "minipfn", 0)
    assert a != arena._run_id("clf=step_200k", "minipfn", 0)
    assert a != arena._run_id("clf=step_100k", "minipfn", 8)


def test_method_without_results_has_no_name():
    board = pd.DataFrame({"method": ["RF (default)", "minipfn_c1"]})
    assert arena._new_method_name(board, "minipfn") == "minipfn_c1"
    assert arena._new_method_name(board.iloc[:1], "minipfn") is None


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
    # parquet에서 읽은 datetime64[us]·[s]도 같은 초 단위
    for unit in ("us", "s"):
        coarse, _ = system.prepare_frame(X.assign(when=X["when"].astype(f"datetime64[{unit}]")))
        assert coarse["when"].iloc[0] == pytest.approx(1577836800.0)
    assert out["flag"].tolist() == [1.0, 0.0]
    assert X["when"].dtype.kind == "M"  # 원래 frame은 그대로


def test_probability_columns_follow_the_natural_label_order():
    # 정수 라벨 2와 10: 문자열 순서("10" < "2")로 두면 TabArena가 둘째 열을 클래스 10으로 읽는다
    system = _system()
    key = system.register_factory(lambda problem_type, categorical: _Constant())
    model = object.__new__(system.EstimatorSystem)
    model.factory, model.bagging = key, 0
    X = pd.DataFrame({"x": np.arange(10.0)})
    y = pd.Series([2] * 8 + [10] * 2)
    model._fit_system(X, y, problem_type="binary", random_state=0)
    proba = model._predict_proba(X)
    assert list(proba.columns) == [2, 10]
    np.testing.assert_allclose(proba.iloc[:, 1], 0.2)
    labels = pd.Series(list(range(12)) * 2)  # 다중 분류 0..11
    model._fit_system(
        pd.DataFrame({"x": np.arange(24.0)}), labels, problem_type="multiclass", random_state=0
    )
    assert list(model._predict_proba(X).columns) == list(range(12))


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


def test_eval_script_limits_follow_the_model_architecture():
    from dataclasses import replace

    from minipfn.recipes import tabpfn_v2
    from minipfn.training.config import TrainConfig

    script = _eval_script()
    # 분류는 2.5 구조(층 수만 줄인 것), 회귀는 어느 구조와도 맞지 않는 모델
    clf = tabpfn_v2.with_architecture(TrainConfig.smoke(), "v2.5", keep_size=True).build_model()
    reg = replace(TrainConfig.smoke("regression"), features_per_group=4).build_model()
    args = argparse.Namespace(max_train_rows=None, max_features=None)
    limits = script.arena_limits({"classification": (clf, None), "regression": (reg, "b")}, args)
    assert limits["binary"] == arena.ModelLimits(50_000, 2_000, 10)
    assert limits["multiclass"] is limits["binary"]
    assert limits["regression"] == arena.ModelLimits(10_000, 500)  # 모르는 구조는 v2
    only = script.arena_limits(
        {"regression": (reg, "b")}, argparse.Namespace(max_train_rows=2048, max_features=None)
    )
    assert only == {"regression": arena.ModelLimits(2048, 500)}  # 분류 체크포인트가 없으면 빠진다


def test_eval_script_accepts_arena_suites():
    script = _eval_script()
    args = script.parse_args(["--suite", "beyondarena-lite", "--bagging", "8"])
    assert (args.suite, args.bagging) == ("beyondarena-lite", 8)
    assert script.arena_suite("beyondarena").subset == ("core2k",)
    assert script.arena_suite("tabarena-lite").name == "tabarena-lite"


def test_post_predict_calls_after_predict_with_one_test_row():
    system = _system()
    seen = []

    class _Hooked(_Constant):
        def after_predict(self, row):
            seen.append(row)

    key = system.register_factory(lambda problem_type, categorical: _Hooked())
    model = object.__new__(system.EstimatorSystem)
    model.factory, model.bagging = key, 0
    X = pd.DataFrame({"x": np.arange(10.0), "when": pd.date_range("2020-01-01", periods=10)})
    y = pd.Series([0, 1] * 5)
    model._fit_system(X, y, problem_type="binary", random_state=0)
    model.post_fit(X, y, X.iloc[3:])
    model._predict_proba(X)
    assert seen == []  # 예측(시간을 재는 곳)에서는 부르지 않는다
    model.post_predict()
    assert len(seen) == 1 and len(seen[0]) == 1
    assert seen[0]["x"].iloc[0] == 3.0 and seen[0]["when"].dtype.kind == "f"  # 같은 전처리


def test_dataset_queue_puts_expensive_datasets_first_and_skips_unrun_splits():
    # 데이터셋 비용 = 돌릴 split의 셀 수 합. 비싼 것부터 (같은 grid면 같은 답)
    rows = []
    for name, n_rows, n_features, splits in [
        ("big", 100_000, 50, 1),
        ("mid_a", 10_000, 100, 3),
        ("mid_b", 20_000, 50, 3),
        ("small", 1_000, 10, 30),
        ("tiny", 100, 5, 10),
    ]:
        rows += [
            {
                "dataset": name,
                "max_train_rows": n_rows,
                "n_features": n_features,
                "skip_reason": None,
            }
        ] * splits
    rows.append(
        {"dataset": "skipped", "max_train_rows": 10**7, "n_features": 10, "skip_reason": "x"}
    )
    grid = pd.DataFrame(rows)
    queue = arena.dataset_queue(grid)
    assert queue == arena.dataset_queue(grid.sample(frac=1, random_state=0))
    # mid_a·mid_b는 셀 3백만으로 같다: 이름 순
    assert queue == [("big", 1), ("mid_a", 3), ("mid_b", 3), ("small", 30), ("tiny", 10)]


def test_shard_worker_command_replaces_dataset_and_output_flags(tmp_path):
    script = _eval_script()
    argv = [
        "--suite", "beyondarena", "--classifier-path", "c.pt", "--shards", "4",
        "--datasets=a,b", "--out-dir", "out", "--mlflow", "--method-name", "m",
        "--heartbeat-minutes", "1",
    ]  # fmt: skip
    worker_dir = tmp_path / "shards" / "shard1"
    cmd = script.shard_worker_command(argv, worker_dir)
    assert cmd == [
        "--suite", "beyondarena", "--classifier-path", "c.pt", "--method-name", "m",
        "--out-dir", str(worker_dir), "--shard-worker",
    ]  # fmt: skip
    args = script.parse_args(cmd)
    assert args.shard_worker and args.shards == 1 and args.datasets is None
    jobs = script.shard_jobs([("big/one", 2), ("b", 1)], tmp_path, "beyondarena")
    # out-dir은 worker가 아니라 데이터셋으로 정한다 (다시 돌리면 어느 worker든 캐시를 쓴다)
    dataset_dir = tmp_path / "shards" / "datasets" / "big_one" / "beyondarena"
    assert jobs[0] == {
        "dataset": "big/one",
        "splits": 2,
        "out_dir": str(dataset_dir),
        "results": str(dataset_dir / "results.pkl"),
    }
    with pytest.raises(SystemExit):
        script.main(["--suite", "classification", "--shards", "2"])


# 가짜 worker: stdin으로 일을 받아 `sleep`초 쉰 뒤 결과 파일을 남긴다. "bad"를 받으면 죽는다
FAKE_WORKER = """
import json, pickle, sys, time
from pathlib import Path
for line in iter(sys.stdin.readline, ""):
    job = json.loads(line)
    if job["dataset"] == "bad":
        sys.exit(3)
    time.sleep(job["sleep"])
    Path(job["out_dir"]).mkdir(parents=True, exist_ok=True)
    with open(job["results"], "wb") as f:
        pickle.dump([{"dataset": job["dataset"], "worker": sys.argv[1]}], f)
"""


def _fake_jobs(tmp_path, spec):
    jobs = []
    for name, splits, sleep in spec:
        out = tmp_path / name
        jobs.append(
            {
                "dataset": name,
                "splits": splits,
                "out_dir": str(out),
                "results": str(out / "results.pkl"),
                "sleep": sleep,
            }
        )
    return jobs


def _fake_spawn(i):
    import subprocess
    import sys

    return subprocess.Popen(
        [sys.executable, "-c", FAKE_WORKER, str(i)], stdin=subprocess.PIPE, text=True
    )


def test_run_queue_gives_the_next_dataset_to_the_worker_that_is_free(tmp_path, capsys):
    import pickle

    script = _eval_script()
    # worker 0이 비싼 것(1초)을 잡는 동안 worker 1이 나머지를 모두 돈다 (고정 분배라면 반씩)
    spec = [("big", 4, 1.0)] + [(f"small{k}", 1, 0.01) for k in range(5)]
    jobs = _fake_jobs(tmp_path, spec)
    (tmp_path / "small0").mkdir()
    (tmp_path / "small0" / "results.pkl").write_bytes(b"stale")  # 지난 실행의 결과는 지운다
    missing, failed = script.run_queue(jobs, _fake_spawn, 2, heartbeat_s=0.2, poll_s=0.01)
    assert missing == [] and failed == {}
    workers = {}
    for job in jobs:
        with open(job["results"], "rb") as f:
            (row,) = pickle.load(f)
        workers[row["dataset"]] = row["worker"]
    assert workers["big"] == "0"
    assert all(workers[f"small{k}"] == "1" for k in range(5))
    out = capsys.readouterr().out
    assert "[heartbeat" in out and "splits 9/9 (datasets 6/6, queue 0)" in out
    assert "worker 0: big splits 0/4" in out  # 일하는 동안 worker별 데이터셋·경과 시간


def test_run_queue_reports_a_dead_worker_and_its_dataset(tmp_path):
    script = _eval_script()
    jobs = _fake_jobs(tmp_path, [("bad", 2, 0.0), ("a", 1, 0.2), ("b", 1, 0.01), ("c", 1, 0.01)])
    missing, failed = script.run_queue(jobs, _fake_spawn, 2, heartbeat_s=60, poll_s=0.01)
    # 죽은 worker의 데이터셋은 결과가 없다고 낸다 (조용히 imputed가 되지 않도록). 나머지는 끝난다
    assert missing == ["bad"]
    assert failed == {0: "exit 3 while running bad"}
    assert all(Path(j["results"]).exists() for j in jobs[1:])


def test_register_factory_gives_the_same_key_to_the_same_function():
    pytest.importorskip("tabarena")
    from minipfn.benchmarks import _arena_system as system

    def make(problem_type, categorical):
        return None

    key = system.register_factory(make)
    assert system.register_factory(make) == key
    assert system.register_factory(lambda p, c: None) != key
