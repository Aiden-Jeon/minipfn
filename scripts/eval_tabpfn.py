"""학습한 TabPFN을 OpenML-CC18(논문 Table 1)에서 평가하고 논문 수치와 비교한다.

예) 저장소 루트에서 `EVAL=scripts/eval_tabpfn.py`라고 두면
  uv run python $EVAL --run-id <학습 run_id>                # MLflow run의 모델
  uv run python $EVAL --model-path tabpfn_run/tabpfn_final.pt --datasets 11,37
  uv run python $EVAL --hub-repo aiden-jeon/minipfn-tabpfn-v1   # Hugging Face Hub의 모델
  uv run python $EVAL --hub-repo aiden-jeon/minipfn-tabpfn-v1 --hub-revision <commit>
  uv run python $EVAL --baselines-only                      # sklearn baseline만

`--run-id`를 주면 결과(평균 ROC AUC와 데이터셋별 표)와 평가에 쓴 코드를 그 run의
`eval_<datasets>/`에 함께 기록한다.
"""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

import mlflow
import pandas as pd
import torch

from minipfn.tabpfn import spec
from minipfn.tabpfn.evaluate import (
    evaluate_methods,
    load_openml_dataset,
    sklearn_baselines,
    tabpfn_predict_proba,
)
from minipfn.tabpfn.hub import from_pretrained
from minipfn.tabpfn.model import configure_attention_backends
from minipfn.tabpfn.train import load_tabpfn
from minipfn.utils import log_run_source


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--run-id", help="학습 run (artifact model/tabpfn_final.pt)")
    src.add_argument("--model-path", help="로컬 tabpfn_final.pt")
    src.add_argument("--hub-repo", help="Hugging Face Hub repo id")
    src.add_argument("--baselines-only", action="store_true")
    parser.add_argument(
        "--datasets", default="table1", help="table1 | all | 쉼표로 구분한 OpenML did"
    )
    parser.add_argument("--hub-revision", help="--hub-repo의 branch, tag 또는 commit")
    parser.add_argument("--n-ensemble", type=int, default=spec.N_ENSEMBLE)
    parser.add_argument("--splits", type=int, default=spec.N_SPLITS)
    parser.add_argument("--no-baselines", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", default="tabpfn_eval")
    return parser.parse_args(argv)


def select_datasets(arg: str):
    dids = (
        spec.OPENML_CC18_TEST_DIDS if arg in ("table1", "all") else [int(d) for d in arg.split(",")]
    )
    datasets = [load_openml_dataset(d) for d in dids]
    if arg == "table1":
        datasets = [d for d in datasets if d.is_table1]
        if len(datasets) != spec.TABLE1_N_DATASETS:
            print(f"주의: Table 1 조건을 만족하는 데이터셋이 {len(datasets)}개 (논문 18개)")
    return datasets


def summarize(rows: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.DataFrame(rows)
    per_dataset = df.groupby(["method", "dataset"])[["roc_auc", "accuracy"]].mean().reset_index()
    summary = per_dataset.groupby("method")[["roc_auc", "accuracy"]].mean()
    summary["paper_roc_auc"] = [spec.PAPER_TABLE1_ROC_AUC.get(m) for m in summary.index]
    return per_dataset, summary.sort_values("roc_auc", ascending=False)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    datasets = select_datasets(args.datasets)
    print("datasets:", ", ".join(f"{d.name}({d.did})" for d in datasets))

    methods = {}
    if not args.baselines_only:
        configure_attention_backends()
        if args.hub_repo:
            model, _ = from_pretrained(
                args.hub_repo, revision=args.hub_revision, map_location=args.device
            )
        else:
            path = args.model_path or mlflow.artifacts.download_artifacts(
                run_id=args.run_id, artifact_path="model/tabpfn_final.pt", dst_path=str(out_dir)
            )
            model, _ = load_tabpfn(path, map_location=args.device)
        predict = partial(tabpfn_predict_proba, model, device=args.device)
        methods["TabPFN"] = partial(predict, n_ensemble=args.n_ensemble)
        methods["TabPFN (no ensemble)"] = partial(predict, n_ensemble=1)
    if not args.no_baselines:
        for name, make in sklearn_baselines().items():
            methods[f"{name} (default)"] = lambda xtr, ytr, xte, make=make: (
                make().fit(xtr, ytr).predict_proba(xte)
            )

    rows = evaluate_methods(datasets, methods, n_splits=args.splits)
    per_dataset, summary = summarize(rows)
    pd.DataFrame(rows).to_csv(out_dir / "eval_rows.csv", index=False)
    per_dataset.to_csv(out_dir / "eval_per_dataset.csv", index=False)
    summary.to_csv(out_dir / "eval_summary.csv")
    print(
        per_dataset.pivot(index="dataset", columns="method", values="roc_auc").round(3).to_string()
    )
    print()
    print(summary.round(4).to_string())

    if args.run_id:
        # MLflow metric 이름에는 쉼표를 쓸 수 없다 (예: --datasets 11,37 → 11-37)
        tag = args.datasets.replace(",", "-")
        with mlflow.start_run(run_id=args.run_id):
            log_run_source(__file__, env_prefix="TABPFN_", artifact_path=f"eval_{tag}/code")
            for method, row in summary.iterrows():
                key = method.replace(" ", "_").replace("(", "").replace(")", "")
                mlflow.log_metric(f"{tag}_roc_auc_{key}", row["roc_auc"])
                mlflow.log_metric(f"{tag}_acc_{key}", row["accuracy"])
            for name in ("eval_rows.csv", "eval_per_dataset.csv", "eval_summary.csv"):
                mlflow.log_artifact(str(out_dir / name), f"eval_{tag}")


if __name__ == "__main__":
    main()
