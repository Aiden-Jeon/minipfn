"""예측 분포 baseline(GBDT·공개 TFM)을 AMLB에서 v2 `--uncertainty`와 같은 fold·지표로 평가한다.

issue #48. `scripts/tabpfn_v2/eval.py --uncertainty`와 같은
`amlb.evaluate_task(..., uncertainty=True)`로 같은 OpenML 10-fold에서 점 예측 점수(분류 ROC AUC,
회귀 negative RMSE)와 예측 분포·불확실성 지표(`minipfn.benchmarks.uncertainty`)를 낸다. 방법은
`minipfn.benchmarks.baselines`에 있다: lightgbm, xgboost, catboost(분류·회귀),
catboost_gaussian(회귀만), tabpfn_v2(공식 패키지), tabicl. 모두 기본 설정이다. 실행 하나가 방법
하나를 평가한다. 그 방법이 없는 suite는 건너뛴다.

예) 저장소 루트에서 (extra baselines가 필요하다)
  uv run --extra baselines python scripts/baselines/eval.py --method lightgbm
  uv run --extra baselines python scripts/baselines/eval.py --method tabicl --suite regression \
      --tasks 359944 --folds 2

결과는 v2 평가와 같은 이름이다: `eval_rows.csv`(fold마다), `eval_per_dataset.csv`(점 예측 점수와
논문 TabPFN 점수), `eval_uncertainty_per_dataset.csv`(지표의 데이터셋별 fold 평균, `score` 포함).
MLflow 지표도 `<suite>_mean_roc_auc`·`<suite>_mean_neg_rmse`, `<suite>_mean_<지표>`로 같다(회귀의
CRPS·폭·interval score는 `_scaled`만 평균한다). 방법끼리, 그리고 v2와의 비교는 `compare.py`가
한다.

task 하나가 실패하면 나머지는 계속 돌리고, 끝에 실패한 task를 모아 0이 아닌 코드로 끝낸다(그때까지의
결과는 저장·기록한다).
"""

from __future__ import annotations

import argparse
import contextlib
import os
import time
import traceback
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch

from minipfn.benchmarks import amlb as bm
from minipfn.benchmarks import baselines as bl
from minipfn.benchmarks import uncertainty as unc
from minipfn.utils import log_run_source

METHODS = tuple(dict.fromkeys(bl.CLASSIFIERS + bl.REGRESSORS))
SUITES = ("classification", "regression")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--suite", default="all", choices=["all", *SUITES])
    parser.add_argument("--tasks", help="쉼표로 구분한 OpenML task id (suite 안에서 고른다)")
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", default="baselines_eval")
    parser.add_argument("--mlflow", action="store_true", help="로컬에서도 MLflow에 기록한다")
    return parser.parse_args(argv)


def selected_tasks(args: argparse.Namespace) -> list[tuple[str, int]]:
    """(suite, task id). 방법이 없는 suite(분류의 catboost_gaussian)는 뺀다."""
    suites = list(SUITES) if args.suite == "all" else [args.suite]
    suites = [s for s in suites if args.method in bl.methods(s)]
    if not suites:
        raise SystemExit(f"{args.method}는 --suite {args.suite}에 없다.")
    chosen = None if args.tasks is None else {int(t) for t in args.tasks.split(",")}
    out = [
        (suite, task_id)
        for suite in suites
        for task_id in bm.task_names(suite)
        if chosen is None or task_id in chosen
    ]
    if chosen is not None:
        unknown = sorted(chosen - {task_id for _, task_id in out})
        if unknown:
            raise SystemExit(f"{args.method}·--suite {args.suite}에 없는 task id: {unknown}")
    if not out:
        raise SystemExit("평가할 task가 없다.")
    return out


def per_dataset_tables(rows: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(점 예측 점수와 논문 TabPFN 점수, 불확실성 지표의 데이터셋별 fold 평균 + score)."""
    df = pd.DataFrame(rows)
    keys = unc.DATASET_KEYS
    metrics = [c for c in df.columns if c not in {*keys, "fold", "method"}]
    per_dataset = df.groupby(keys)[metrics].mean().reset_index()
    paper = {**bm.paper_scores("classification"), **bm.paper_scores("regression")}
    scores = per_dataset[[*keys, "score"]].copy()
    scores["paper"] = scores["task_id"].map(paper)
    scores["diff"] = scores["score"] - scores["paper"]
    return scores, per_dataset


@contextlib.contextmanager
def mlflow_run(use_mlflow: bool):
    """AI Runtime에서는 플랫폼이 만든 run(`MLFLOW_RUN_ID`), 로컬에서는 `BASELINES_EXPERIMENT`에 새
    run."""
    if not use_mlflow:
        yield
        return
    if "MLFLOW_RUN_ID" not in os.environ:
        mlflow.set_experiment(os.environ.get("BASELINES_EXPERIMENT", "tabpfn-v2-reproduction"))
    with mlflow.start_run(run_id=os.environ.get("MLFLOW_RUN_ID")):
        log_run_source(__file__, env_prefix="BASELINES_")
        yield


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    tasks = selected_tasks(args)  # 고른 task가 틀렸으면 run을 만들기 전에 끝낸다
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    use_mlflow = args.mlflow or "MLFLOW_RUN_ID" in os.environ
    with mlflow_run(use_mlflow):
        failures = run(args, tasks, out_dir, use_mlflow)
    if failures:
        raise SystemExit(f"실패한 task {len(failures)}개: {failures}")


def run(args: argparse.Namespace, tasks: list, out_dir: Path, use_mlflow: bool) -> list:
    print(
        f"method: {args.method}, tasks: {len(tasks)}, folds: {args.folds}, device: {args.device}",
        flush=True,
    )
    if use_mlflow:
        mlflow.log_params(
            {"method": args.method, "suite": args.suite, "folds": args.folds, "seed": args.seed}
        )
        mlflow.set_tag("method", args.method)
    rows, failures = [], []
    for i, (suite, task_id) in enumerate(tasks, start=1):
        start = time.time()
        try:
            task = bm.load_openml_task(task_id)

            def make(t: bm.OpenMLTask):
                return bl.make_baseline(args.method, t.task_type, args.device, args.seed)

            task_rows = bm.evaluate_task(task, make, n_folds=args.folds, uncertainty=True)
        except Exception:  # task 하나의 실패로 나머지를 버리지 않는다
            traceback.print_exc()
            failures.append(task_id)
            print(f"[{i}/{len(tasks)}] {suite:14s} task {task_id} 실패", flush=True)
            continue
        seconds = time.time() - start
        rows.extend(
            {**r, "suite": suite, "method": args.method, "seconds": seconds / len(task_rows)}
            for r in task_rows
        )
        score = float(np.nanmean([r["score"] for r in task_rows]))
        print(
            f"[{i}/{len(tasks)}] {suite:14s} {task.name:40s} {task.X.shape} "
            f"score {score:.4f} ({seconds:.0f}s)",
            flush=True,
        )
        pd.DataFrame(rows).to_csv(out_dir / "eval_rows.csv", index=False)
    if not rows:
        return failures

    scores, per_dataset = per_dataset_tables(rows)
    scores.to_csv(out_dir / "eval_per_dataset.csv", index=False)
    per_dataset.to_csv(out_dir / "eval_uncertainty_per_dataset.csv", index=False)
    print(scores.round(4).to_string(index=False))
    for suite, group in per_dataset.groupby("suite"):
        metric = "roc_auc" if suite == "classification" else "neg_rmse"
        means = unc.suite_means(group.drop(columns=["score"]))
        means["seconds_per_fold"] = means.pop("seconds")
        print(
            f"{suite} ({args.method}, datasets {len(group)}): mean score {group.score.mean():.4f}"
        )
        print(pd.Series(means).round(4).to_string())
        if use_mlflow:
            mlflow.log_metric(f"{suite}_mean_{metric}", float(group["score"].mean()))
            mlflow.log_metrics({f"{suite}_mean_{k}": v for k, v in means.items()})
    if use_mlflow:
        for name in ("eval_rows.csv", "eval_per_dataset.csv", "eval_uncertainty_per_dataset.csv"):
            mlflow.log_artifact(str(out_dir / name), "eval")
        if failures:
            mlflow.set_tag("failed_tasks", ",".join(map(str, failures)))
    return failures


if __name__ == "__main__":
    main()
