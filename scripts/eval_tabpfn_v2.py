"""TabPFN v2를 논문 benchmark에서 평가하고 논문 수치와 비교한다.

benchmark는 AMLB 분류 29개와 AMLB·CTR23 회귀 28개다. 데이터셋마다 OpenML task의 10-fold 교차검증
평균을 낸다(분류 ROC AUC, 회귀 negative RMSE). 모델은 `model.save_checkpoint`로 저장한 체크포인트다.

예) 저장소 루트에서 `EVAL=scripts/eval_tabpfn_v2.py`라고 두면
  uv run python $EVAL --classifier-path clf.pt --regressor-path reg.pt            # 57개 × 10 fold
  uv run python $EVAL --suite classification --classifier-path clf.pt --tasks 359955 --folds 2

AI Runtime(`runs/eval_tabpfn_v2.yaml`)에서는 플랫폼이 만든 MLflow run(`MLFLOW_RUN_ID`)에 기록한다.
로컬에서 `--mlflow`를 주면 `TABPFN_V2_EXPERIMENT`(기본 tabpfn-v2-reproduction) experiment에 새
run을 만든다.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import mlflow
import pandas as pd
import torch

from minipfn.tabpfn_v2 import benchmark as bm
from minipfn.tabpfn_v2.inference import TabPFNv2Classifier, TabPFNv2Regressor
from minipfn.tabpfn_v2.model import load_checkpoint
from minipfn.utils import log_run_source


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--suite", choices=["classification", "regression", "all"], default="all")
    parser.add_argument("--tasks", help="쉼표로 구분한 OpenML task id (suite 안에서 고른다)")
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--classifier-path", help="분류 체크포인트 (suite에 분류가 있으면 필요)")
    parser.add_argument("--regressor-path", help="회귀 체크포인트 (suite에 회귀가 있으면 필요)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", default="tabpfn_v2_eval")
    parser.add_argument("--mlflow", action="store_true", help="로컬에서도 MLflow에 기록한다")
    return parser.parse_args(argv)


def selected_tasks(args: argparse.Namespace) -> list[tuple[str, int]]:
    suites = ["classification", "regression"] if args.suite == "all" else [args.suite]
    chosen = None if args.tasks is None else {int(t) for t in args.tasks.split(",")}
    out = []
    for suite in suites:
        for task_id in bm.task_names(suite):
            if chosen is None or task_id in chosen:
                out.append((suite, task_id))
    return out


def make_estimator_factory(suite: str, args: argparse.Namespace):
    path = args.classifier_path if suite == "classification" else args.regressor_path
    if path is None:
        flag = "--classifier-path" if suite == "classification" else "--regressor-path"
        raise SystemExit(f"{suite} 평가에는 {flag}가 필요하다.")
    model, borders = load_checkpoint(path, map_location=args.device)

    def make(task: bm.OpenMLTask):
        kw = {"categorical_features_indices": task.categorical_indices, "device": args.device}
        if suite == "classification":
            return TabPFNv2Classifier(model, **kw)
        return TabPFNv2Regressor(model, borders, **kw)

    return make


def summarize(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    per_dataset = df.groupby(["suite", "task_id", "dataset"])["score"].mean().reset_index()
    paper = {**bm.paper_scores("classification"), **bm.paper_scores("regression")}
    per_dataset["paper"] = per_dataset["task_id"].map(paper)
    per_dataset["diff"] = per_dataset["score"] - per_dataset["paper"]
    return per_dataset


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = selected_tasks(args)
    print(f"tasks: {len(tasks)}, folds: {args.folds}, device: {args.device}", flush=True)

    use_mlflow = args.mlflow or "MLFLOW_RUN_ID" in os.environ
    if use_mlflow:
        if "MLFLOW_RUN_ID" not in os.environ:
            mlflow.set_experiment(os.environ.get("TABPFN_V2_EXPERIMENT", "tabpfn-v2-reproduction"))
        mlflow.start_run(run_id=os.environ.get("MLFLOW_RUN_ID"))
        log_run_source(__file__, env_prefix="TABPFN_V2_")
        mlflow.log_params({"suite": args.suite, "folds": args.folds})

    factories = {suite: make_estimator_factory(suite, args) for suite in {s for s, _ in tasks}}
    rows = []
    for i, (suite, task_id) in enumerate(tasks, start=1):
        start = time.time()
        task = bm.load_openml_task(task_id)
        task_rows = bm.evaluate_task(task, factories[suite], n_folds=args.folds)
        for row in task_rows:
            row["suite"] = suite
        rows.extend(task_rows)
        mean = sum(r["score"] for r in task_rows) / len(task_rows)
        paper = bm.paper_scores(suite)[task_id]
        print(
            f"[{i}/{len(tasks)}] {suite:14s} {task.name:40s} {task.X.shape} "
            f"score {mean:.4f} paper {paper:.4f} ({time.time() - start:.0f}s)",
            flush=True,
        )
        pd.DataFrame(rows).to_csv(out_dir / "eval_rows.csv", index=False)

    per_dataset = summarize(rows)
    per_dataset.to_csv(out_dir / "eval_per_dataset.csv", index=False)
    print(per_dataset.round(4).to_string(index=False))
    for suite, group in per_dataset.groupby("suite"):
        metric = "roc_auc" if suite == "classification" else "neg_rmse"
        print(
            f"{suite}: datasets {len(group)}, mean score {group['score'].mean():.4f}, "
            f"paper {group['paper'].mean():.4f}, mean |diff| {group['diff'].abs().mean():.4f}"
        )
        if use_mlflow:
            mlflow.log_metric(f"{suite}_mean_{metric}", group["score"].mean())
            mlflow.log_metric(f"{suite}_paper_mean_{metric}", group["paper"].mean())
            mlflow.log_metric(f"{suite}_mean_abs_diff", group["diff"].abs().mean())
    if use_mlflow:
        for name in ("eval_rows.csv", "eval_per_dataset.csv"):
            mlflow.log_artifact(str(out_dir / name), "eval")
        mlflow.end_run()


if __name__ == "__main__":
    main()
