"""TabPFN v2를 논문 benchmark에서 평가하고 논문 수치와 비교한다.

benchmark는 AMLB 분류 29개와 AMLB·CTR23 회귀 28개다. 데이터셋마다 OpenML task의 10-fold 교차검증
평균을 낸다(분류 ROC AUC, 회귀 negative RMSE). 모델은 `model.save_checkpoint`로 저장한 체크포인트,
학습 run이 MLflow에 남긴 checkpoint(`runs:/<run_id>/checkpoints/step_XXXXXXX.pt`), Hub repo id
(`hf://owner/name`, `hub.save_pretrained`로 올린 것) 가운데 하나다.

예) 저장소 루트에서 `EVAL=scripts/tabpfn_v2/eval.py`라고 두면
  uv run python $EVAL --classifier-path clf.pt --regressor-path reg.pt            # 57개 × 10 fold
  uv run python $EVAL --suite classification --classifier-path clf.pt --tasks 359955 --folds 2
  uv run python $EVAL --suite classification --classifier-path runs:/<run_id>/checkpoints/latest.pt

범주형 열의 one-hot(`--one-hot`)은 기본으로 추론의 기본값(`DEFAULT_ONE_HOT`) 하나만 평가한다.
쉼표로 여러 모드를 주면(`--one-hot off,power,all`) 모드마다 평가해
`eval_per_dataset_by_one_hot.csv`와 `<suite>_mean_<metric>_one_hot_<mode>` 지표에 남긴다. 이때도
`eval_per_dataset.csv`와 MLflow의 평균 지표는 기본값 모드(목록에 없으면 첫 모드)의 결과다.
test fold 점수로 모드를 고르면 결과가 좋게 나오고, checkpoint마다 모드가 달라지면 step별 곡선을
견줄 수 없어서다. one-hot으로 펼칠 열(범주가 `MAX_ONE_HOT_CATEGORIES`개 이하인 범주형 열)이 없는
데이터셋은 모드와 상관없이 결과가 같아 한 번만 평가하고, 그 결과를 다른 모드에도 `reused=True`로
쓴다.

AI Runtime(`scripts/tabpfn_v2/eval.yaml`)에서는 플랫폼이 만든 MLflow run(`MLFLOW_RUN_ID`)에
기록한다. 로컬에서 `--mlflow`를 주면 `TABPFN_V2_EXPERIMENT`(기본 tabpfn-v2-reproduction)
experiment에 새 run을 만든다.
"""

from __future__ import annotations

import argparse
import functools
import os
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch

from minipfn.tabpfn_v2 import benchmark as bm
from minipfn.tabpfn_v2.hub import load_model
from minipfn.tabpfn_v2.inference import (
    DEFAULT_ONE_HOT,
    ONE_HOT_MODES,
    TabPFNv2Classifier,
    TabPFNv2Regressor,
    _one_hot_sizes,
)
from minipfn.utils import log_run_source


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--suite", choices=["classification", "regression", "all"], default="all")
    parser.add_argument("--tasks", help="쉼표로 구분한 OpenML task id (suite 안에서 고른다)")
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument(
        "--classifier-path", help="분류 체크포인트 (파일, 폴더, runs:/ URI, hf://owner/name)"
    )
    parser.add_argument(
        "--regressor-path", help="회귀 체크포인트 (파일, 폴더, runs:/ URI, hf://owner/name)"
    )
    parser.add_argument(
        "--one-hot",
        type=_one_hot_modes,
        default=[DEFAULT_ONE_HOT],
        help=f"평가할 one-hot 모드, 쉼표로 구분 (기본: {DEFAULT_ONE_HOT}, 비교: off,power,all)",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", default="tabpfn_v2_eval")
    parser.add_argument("--mlflow", action="store_true", help="로컬에서도 MLflow에 기록한다")
    return parser.parse_args(argv)


def _one_hot_modes(value: str) -> list[str]:
    modes = [m.strip() for m in value.split(",") if m.strip()]
    unknown = [m for m in modes if m not in ONE_HOT_MODES]
    if unknown or not modes:
        raise argparse.ArgumentTypeError(f"one-hot 모드는 {ONE_HOT_MODES} 가운데서: {value}")
    return list(dict.fromkeys(modes))


def headline_mode(modes: list[str]) -> str:
    """`eval_per_dataset.csv`와 평균 지표에 쓸 모드: 추론의 기본값, 목록에 없으면 첫 모드."""
    return DEFAULT_ONE_HOT if DEFAULT_ONE_HOT in modes else modes[0]


def expands_one_hot(task: bm.OpenMLTask) -> bool:
    """one-hot 모드에 따라 결과가 달라질 수 있는지.

    추론이 범주형으로 다루는 열(`categorical_indices`이거나 숫자가 아닌 열) 가운데 추론이 펼칠
    열(범주가 `MAX_ONE_HOT_CATEGORIES`개 이하이고, 펼친 폭이 상한 안)이 있는지 본다 (전체 행 기준).
    """
    X = task.X
    cat = set(task.categorical_indices)
    cat |= {i for i, dtype in enumerate(X.dtypes) if not pd.api.types.is_numeric_dtype(dtype)}
    return bool(_one_hot_sizes({i: X.iloc[:, i].nunique() for i in cat}, X.shape[1]))


def selected_tasks(args: argparse.Namespace) -> list[tuple[str, int]]:
    suites = ["classification", "regression"] if args.suite == "all" else [args.suite]
    chosen = None if args.tasks is None else {int(t) for t in args.tasks.split(",")}
    out = []
    for suite in suites:
        for task_id in bm.task_names(suite):
            if chosen is None or task_id in chosen:
                out.append((suite, task_id))
    if chosen is not None:
        unknown = sorted(chosen - {task_id for _, task_id in out})
        if unknown:
            raise SystemExit(f"--suite {args.suite}에 없는 task id: {unknown}")
    if not out:
        raise SystemExit("평가할 task가 없다.")
    return out


def make_estimator_factory(suite: str, args: argparse.Namespace):
    path = args.classifier_path if suite == "classification" else args.regressor_path
    if path is None:
        flag = "--classifier-path" if suite == "classification" else "--regressor-path"
        raise SystemExit(f"{suite} 평가에는 {flag}가 필요하다.")
    # CPU로 읽고, 추론할 때 모델만 장치로 옮긴다
    model, borders = load_model(path, suite, Path(args.out_dir) / "checkpoints" / suite)

    def make(task: bm.OpenMLTask, one_hot: str):
        kw = {
            "categorical_features_indices": task.categorical_indices,
            "one_hot": one_hot,
            "device": args.device,
        }
        if suite == "classification":
            return TabPFNv2Classifier(model, **kw)
        return TabPFNv2Regressor(model, borders, **kw)

    return make


def summarize(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    keys = ["one_hot", "suite", "task_id", "dataset"]
    per_dataset = df.groupby(keys)["score"].mean().reset_index()
    paper = {**bm.paper_scores("classification"), **bm.paper_scores("regression")}
    per_dataset["paper"] = per_dataset["task_id"].map(paper)
    per_dataset["diff"] = per_dataset["score"] - per_dataset["paper"]
    return per_dataset


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = selected_tasks(args)
    headline = headline_mode(args.one_hot)
    print(
        f"tasks: {len(tasks)}, folds: {args.folds}, device: {args.device}, "
        f"one-hot: {','.join(args.one_hot)} (결과: {headline})",
        flush=True,
    )

    use_mlflow = args.mlflow or "MLFLOW_RUN_ID" in os.environ
    if use_mlflow:
        if "MLFLOW_RUN_ID" not in os.environ:
            mlflow.set_experiment(os.environ.get("TABPFN_V2_EXPERIMENT", "tabpfn-v2-reproduction"))
        mlflow.start_run(run_id=os.environ.get("MLFLOW_RUN_ID"))
        log_run_source(__file__, env_prefix="TABPFN_V2_")
        mlflow.log_params(
            {
                "suite": args.suite,
                "folds": args.folds,
                "one_hot": headline,
                "one_hot_modes": ",".join(args.one_hot),
            }
        )
        paths = {"classifier_path": args.classifier_path, "regressor_path": args.regressor_path}
        mlflow.log_params({k: v for k, v in paths.items() if v is not None})

    factories = {suite: make_estimator_factory(suite, args) for suite in {s for s, _ in tasks}}
    rows = []
    for i, (suite, task_id) in enumerate(tasks, start=1):
        start = time.time()
        task = bm.load_openml_task(task_id)
        expands = expands_one_hot(task)
        scores = {}
        for mode in args.one_hot if expands else [headline]:
            make = functools.partial(factories[suite], one_hot=mode)
            task_rows = bm.evaluate_task(task, make, n_folds=args.folds)
            scores[mode] = float(np.nanmean([r["score"] for r in task_rows]))
            rows.extend({**r, "suite": suite, "one_hot": mode, "reused": False} for r in task_rows)
        if not expands:
            # 펼칠 열이 없으면 모든 모드가 같은 결과라 한 번 낸 것을 다른 모드에도 쓴다
            for mode in args.one_hot:
                if mode != headline:
                    rows.extend(
                        {**r, "suite": suite, "one_hot": mode, "reused": True} for r in task_rows
                    )
        paper = bm.paper_scores(suite)[task_id]
        shown = " ".join(f"{m} {v:.4f}" for m, v in scores.items())
        print(
            f"[{i}/{len(tasks)}] {suite:14s} {task.name:40s} {task.X.shape} "
            f"score {shown} paper {paper:.4f} ({time.time() - start:.0f}s)",
            flush=True,
        )
        pd.DataFrame(rows).to_csv(out_dir / "eval_rows.csv", index=False)

    by_mode = summarize(rows)
    by_mode.to_csv(out_dir / "eval_per_dataset_by_one_hot.csv", index=False)
    for (suite, mode), group in by_mode.groupby(["suite", "one_hot"]):
        metric = "roc_auc" if suite == "classification" else "neg_rmse"
        print(
            f"{suite} one-hot {mode}: mean score {group['score'].mean():.4f}, "
            f"mean |diff| {group['diff'].abs().mean():.4f}"
        )
        if use_mlflow and len(args.one_hot) > 1:
            mlflow.log_metric(f"{suite}_mean_{metric}_one_hot_{mode}", group["score"].mean())
            mlflow.log_metric(f"{suite}_mean_abs_diff_one_hot_{mode}", group["diff"].abs().mean())

    per_dataset = by_mode[by_mode.one_hot == headline]
    per_dataset.to_csv(out_dir / "eval_per_dataset.csv", index=False)
    print(per_dataset.round(4).to_string(index=False))
    for suite, group in per_dataset.groupby("suite"):
        metric = "roc_auc" if suite == "classification" else "neg_rmse"
        print(
            f"{suite}: one-hot {headline}, datasets {len(group)}, "
            f"mean score {group['score'].mean():.4f}, paper {group['paper'].mean():.4f}, "
            f"mean |diff| {group['diff'].abs().mean():.4f}"
        )
        if use_mlflow:
            mlflow.log_metric(f"{suite}_mean_{metric}", group["score"].mean())
            mlflow.log_metric(f"{suite}_paper_mean_{metric}", group["paper"].mean())
            mlflow.log_metric(f"{suite}_mean_abs_diff", group["diff"].abs().mean())
    if use_mlflow:
        for name in ("eval_rows.csv", "eval_per_dataset.csv", "eval_per_dataset_by_one_hot.csv"):
            mlflow.log_artifact(str(out_dir / name), "eval")
        mlflow.end_run()


if __name__ == "__main__":
    main()
