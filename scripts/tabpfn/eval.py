"""학습한 TabPFN을 OpenML-CC18(논문 Table 1)에서 평가하고 논문 수치와 비교한다.

모델은 `save_tabpfn`으로 저장한 `tabpfn_final.pt`, 학습 run이 MLflow에 남긴 모델·checkpoint
(`runs:/<run_id>/model/tabpfn_final.pt`, `runs:/<run_id>/checkpoints/step_XXXXXXX.pt`) 또는
Hugging Face Hub의 모델이다.

예) 저장소 루트에서 `EVAL=scripts/tabpfn/eval.py`라고 두면
  uv run python $EVAL --model-path runs:/<run_id>/model/tabpfn_final.pt   # 학습 run의 최종 모델
  uv run python $EVAL --model-path runs:/<run_id>/checkpoints/step_0003000.pt
  uv run python $EVAL --model-path tabpfn_run/tabpfn_final.pt --datasets 11,37
  uv run python $EVAL --hub-repo aiden-jeon/minipfn-tabpfn-v1   # Hugging Face Hub의 모델
  uv run python $EVAL --hub-repo aiden-jeon/minipfn-tabpfn-v1 --hub-revision <commit>
  uv run python $EVAL --baselines-only                      # sklearn baseline만

AI Runtime(`scripts/tabpfn/eval.yaml`)에서는 플랫폼이 만든 MLflow run(`MLFLOW_RUN_ID`)에 기록한다.
로컬에서 `--mlflow`를 주면 `TABPFN_EXPERIMENT`(기본 tabpfn-reproduction) experiment에 새 run을
만든다. 결과(평균 ROC AUC와 데이터셋별 표)는 그 run의 `eval/`에 남는다.
"""

from __future__ import annotations

import argparse
import os
from functools import partial
from pathlib import Path

import mlflow
import pandas as pd
import torch

from minipfn.benchmarks.cc18 import (
    evaluate_methods,
    load_openml_dataset,
    sklearn_baselines,
    tabpfn_predict_proba,
)
from minipfn.hub.tabpfn_v1 import from_pretrained
from minipfn.models.tabpfn_v1 import TabPFNTransformer, configure_attention_backends
from minipfn.recipes import tabpfn_v1 as spec
from minipfn.training.tabpfn_v1 import TabPFNConfig, build_tabpfn_model, tabpfn_from_state
from minipfn.utils import log_run_source

# 학습 run의 param 중 모델 구조를 정하는 것 (학습 중 checkpoint에는 config가 없다)
MODEL_PARAMS = {
    "emsize": int,
    "nhead": int,
    "nhid": int,
    "nlayers": int,
    "dropout": float,
    "num_features": int,
    "n_out": int,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--model-path", help="tabpfn_final.pt·checkpoint 파일 또는 runs:/ URI")
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
    parser.add_argument("--mlflow", action="store_true", help="로컬에서도 MLflow에 기록한다")
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


def load_model(
    path_or_uri: str, out_dir: Path, map_location: str | torch.device = "cpu"
) -> TabPFNTransformer:
    """모델을 읽는다. `runs:/<run_id>/<경로>`면 MLflow에서 받는다.

    학습 중에 남긴 checkpoint(`checkpoints/step_XXXXXXX.pt`, `latest.pt`)에는 모델 설정이 없어서
    그 run에 기록된 param(emsize·nlayers 등)으로 모델을 만든다.
    """
    run_id, path = None, path_or_uri
    if path_or_uri.startswith("runs:/"):
        run_id = path_or_uri.removeprefix("runs:/").split("/", 1)[0]
        path = mlflow.artifacts.download_artifacts(
            artifact_uri=path_or_uri, dst_path=str(out_dir / "model")
        )
    state = torch.load(path, map_location=map_location, weights_only=True)
    if "config" in state:
        return tabpfn_from_state(state["config"], state["state_dict"], map_location)[0]
    if run_id is None:
        raise SystemExit(
            f"{path_or_uri}는 학습 중 checkpoint라 모델 설정이 없다. runs:/<run_id>/...로 준다."
        )
    params = mlflow.MlflowClient().get_run(run_id).data.params
    missing = [k for k in MODEL_PARAMS if k not in params]
    if missing:
        # 같은 run에서 이어서 학습한 run은 config가 학습을 처음 시작한 run에만 기록돼 있다
        raise SystemExit(
            f"run {run_id}에 모델 설정 param({', '.join(missing)})이 없다. 설정이 함께 저장된 "
            f"runs:/{run_id}/model/tabpfn_final.pt를 주거나, "
            "학습을 처음 시작한 run의 param을 확인한다."
        )
    config = TabPFNConfig(**{k: cast(params[k]) for k, cast in MODEL_PARAMS.items()})
    model = build_tabpfn_model(config)
    model.load_state_dict(state["model"])
    return model.to(map_location).eval()


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

    use_mlflow = args.mlflow or "MLFLOW_RUN_ID" in os.environ
    if use_mlflow:
        if "MLFLOW_RUN_ID" not in os.environ:
            mlflow.set_experiment(os.environ.get("TABPFN_EXPERIMENT", "tabpfn-reproduction"))
        mlflow.start_run(run_id=os.environ.get("MLFLOW_RUN_ID"))
        log_run_source(__file__, env_prefix="TABPFN_")
        mlflow.log_params(
            {"datasets": args.datasets, "splits": args.splits, "n_ensemble": args.n_ensemble}
        )
        sources = {
            "model_path": args.model_path,
            "hub_repo": args.hub_repo,
            "hub_revision": args.hub_revision,
        }
        mlflow.log_params({k: v for k, v in sources.items() if v is not None})

    methods = {}
    if not args.baselines_only:
        configure_attention_backends()
        if args.hub_repo:
            model, _ = from_pretrained(
                args.hub_repo, revision=args.hub_revision, map_location=args.device
            )
        else:
            model = load_model(args.model_path, out_dir, map_location=args.device)
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

    if use_mlflow:
        # MLflow metric 이름에는 쉼표를 쓸 수 없다 (예: --datasets 11,37 → 11-37)
        tag = args.datasets.replace(",", "-")
        for method, row in summary.iterrows():
            key = method.replace(" ", "_").replace("(", "").replace(")", "")
            mlflow.log_metric(f"{tag}_roc_auc_{key}", row["roc_auc"])
            mlflow.log_metric(f"{tag}_acc_{key}", row["accuracy"])
        for name in ("eval_rows.csv", "eval_per_dataset.csv", "eval_summary.csv"):
            mlflow.log_artifact(str(out_dir / name), "eval")
        mlflow.end_run()


if __name__ == "__main__":
    main()
