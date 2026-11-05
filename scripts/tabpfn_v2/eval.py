"""TabPFN v2를 논문 benchmark에서 평가하고 논문 수치와 비교한다.

benchmark는 AMLB 분류 29개와 AMLB·CTR23 회귀 28개다. 데이터셋마다 OpenML task의 10-fold 교차검증
평균을 낸다(분류 ROC AUC, 회귀 negative RMSE). 모델은 `minipfn.models.save_checkpoint`로 저장한
체크포인트, 학습 run이 MLflow에 남긴 checkpoint(`runs:/<run_id>/checkpoints/step_XXXXXXX.pt`),
Hub repo id(`hf://owner/name`, `hub.save_pretrained`로 올린 것) 가운데 하나다.

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
쓴다. 펼칠 열이 있는지는 fold마다 그 fold의 학습 행으로 본다
(추론이 멤버 전처리를 학습 행으로 맞추므로).

앙상블 멤버 수와 전처리는 기본으로 모델이 정한다(`minipfn.recipes.default_inference`: 모델 종류의
recipe가 정한다. Hub config.json의 `inference`가 먼저이고, 없으면 v2·2.5 모델은 구조가 2.5면 v2.5,
아니면 v2. TabPFN-3·3.5 모델은 3이면 quantile,robust, 3.5면 standard 멤버 8개).
`--member-kinds`로 주면 멤버 종류만 그 값을 쓴다(v2의 quantile,power, `v2.5`는 TabPFN-2.5의 robust
scaling + soft clipping을 쓰는 quantile,robust). 멤버 수는 `--n-estimators`로 따로 준다.
TabPFN-2.5 checkpoint를 로컬에서 `--mlflow`로 평가하면
`TABPFN_V2_EXPERIMENT=tabpfn-v2-5-reproduction`을 준다.

`--suite tabarena-lite`, `beyondarena`, `beyondarena-lite`는 TabArena-Lite·BeyondArena다
(`minipfn.benchmarks.arena`, extra `arena`가 필요하다. pandas 2.3이 필요해 dev group과 따로
설치하고 `uv run`에도 같은 옵션을 준다: `uv sync --no-dev --extra arena --group arena-dev`).
공식 split을 돌리고 공개 leaderboard와 같은 방식으로 Elo·정규화 점수·
improvability·승률을 낸다. 분류·회귀 체크포인트
가운데 주지 않은 쪽과 논문 권장 크기(recipe의 `SUPPORTED_SIZE`; 클래스 수는 모델 출력 수)를
넘는 split은 돌리지 않고 imputed로 둔다(`--max-train-rows`·`--max-features`로 바꾼다). 기본은 학습
split 전체에 한 번 fit하고, `--bagging 8`은 TabArena처럼 8-fold로 나눠 8개의 예측을 평균한다.
`--datasets`로 일부만 돌릴 수 있다(나머지는 imputed). 결과는 `<out-dir>/<suite>/`
(`minipfn.benchmarks.arena.run`).

  uv run --no-dev --extra arena --group arena-dev python $EVAL --suite tabarena-lite \
    --classifier-path hf://aiden-jeon/minipfn-tabpfn-v2-classifier \
    --regressor-path hf://aiden-jeon/minipfn-tabpfn-v2-regressor

AI Runtime(`scripts/tabpfn_v2/eval.yaml`)에서는 플랫폼이 만든 MLflow run(`MLFLOW_RUN_ID`)에
기록한다. 로컬에서 `--mlflow`를 주면 `TABPFN_V2_EXPERIMENT`(기본 tabpfn-v2-reproduction)
experiment에 새 run을 만든다.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import os
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch

from minipfn.benchmarks import amlb as bm
from minipfn.hub import load_model
from minipfn.inference.ensemble import (
    DEFAULT_ONE_HOT,
    EnsembleClassifier,
    EnsembleRegressor,
    max_features_of,
)
from minipfn.inference.preprocessing import (
    ONE_HOT_MODES,
    one_hot_kinds,
    one_hot_sizes,
    to_numeric,
)
from minipfn.priors.table_size import MAX_FEATURES
from minipfn.recipes import default_inference, resolve_member_kinds, supported_size
from minipfn.utils import log_run_source

ARENA_SUITES = ("tabarena-lite", "beyondarena", "beyondarena-lite")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--suite", choices=["classification", "regression", "all", *ARENA_SUITES], default="all"
    )
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
    parser.add_argument(
        "--member-kinds",
        type=_member_kinds,
        default=None,
        help="앙상블 멤버 전처리 종류, 쉼표로 구분해 차례로 쓴다 (기본: 모델이 정한다."
        " v2 = quantile,power, v2.5·v3 = quantile,robust, v3.5 = standard)."
        " 멤버 수는 바꾸지 않는다",
    )
    parser.add_argument(
        "--n-estimators", type=int, default=None, help="앙상블 멤버 수 (기본: 모델이 정한다)"
    )
    parser.add_argument(
        "--bagging", type=int, default=0, help="TabArena·BeyondArena: k-fold bagging (0: 안 함)"
    )
    parser.add_argument(
        "--datasets", help="TabArena·BeyondArena: 쉼표로 구분한 데이터셋 이름 (나머지는 imputed)"
    )
    parser.add_argument(
        "--max-train-rows", type=int, help="TabArena·BeyondArena: 학습 행 상한 (기본: 모델 구조)"
    )
    parser.add_argument(
        "--max-features", type=int, help="TabArena·BeyondArena: feature 수 상한 (기본: 모델 구조)"
    )
    parser.add_argument(
        "--method-name", default="minipfn-tabpfn-v2", help="TabArena·BeyondArena 결과의 방법 이름"
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


def _member_kinds(value: str) -> tuple[str, ...]:
    try:
        return resolve_member_kinds(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from None


def headline_mode(modes: list[str]) -> str:
    """`eval_per_dataset.csv`와 평균 지표에 쓸 모드: 추론의 기본값, 목록에 없으면 첫 모드."""
    return DEFAULT_ONE_HOT if DEFAULT_ONE_HOT in modes else modes[0]


def fold_expands_one_hot(X: pd.DataFrame, categorical_indices: list[int], max_width: int) -> bool:
    """학습 행 하나에서 추론이 범주형 열을 펼칠 열이 있는지.
    `MemberPreprocessor.fit`과 같은 규칙이다: `to_numeric`으로 번호를 매긴 뒤
    (범주 수는 학습 행의 범주만 센다) `one_hot_sizes`가 고른 열이 하나라도 있으면 펼친다."""
    X_num, cat_ix, _ = to_numeric(X, categorical_indices)
    n_categories = {
        i: int(np.nanmax(X_num[:, i])) + 1 if np.isfinite(X_num[:, i]).any() else 1 for i in cat_ix
    }
    return bool(one_hot_sizes(n_categories, X_num.shape[1], max_width))


def expands_one_hot(task: bm.OpenMLTask, max_width: int = MAX_FEATURES, n_folds: int = 10) -> bool:
    """one-hot 모드에 따라 결과가 달라질 수 있는지: 어느 fold의 학습 행에서든 펼칠 열이 있으면 True.

    전체 행으로 보면 안 된다. 범주가 전체로는 `MAX_ONE_HOT_CATEGORIES`개를 넘어도
    fold의 학습 행에서는 그 이하일 수 있고, 멤버 전처리는 fold마다 학습 행으로 맞춘다.
    `n_folds`는 평가에 쓰는 fold 수와 같아야 한다. 상한 `max_width`는 모델이 학습 때 본
    feature 수다 (`max_features_of`).
    """
    return any(
        fold_expands_one_hot(task.X.iloc[train_ix], task.categorical_indices, max_width)
        for _, train_ix, _ in task.folds(n_folds)
    )


def same_result_modes(
    modes: list[str], member_kinds: tuple[str, ...], expands: bool
) -> list[list[str]]:
    """결과가 같은 one-hot 모드끼리 묶는다. 묶음마다 headline 모드(`headline_mode`)가 있으면 그것이
    맨 앞이다. 펼칠 열이 없으면(`expands_one_hot`) 모두 한 묶음이고, 있으면 범주형 열을 펼치는 멤버
    종류(`one_hot_kinds`)가 같은 모드끼리 묶인다(예: 분위수 멤버만 쓰면 off와 power가 같다)."""
    headline = headline_mode(modes)
    groups: dict[frozenset, list[str]] = {}
    for mode in sorted(modes, key=lambda m: m != headline):  # headline을 먼저
        key = one_hot_kinds(mode, member_kinds) if expands else frozenset()
        groups.setdefault(key, []).append(mode)
    return list(groups.values())


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
    # 주지 않으면 모델의 추론 설정 (Hub config.json의 inference, 없으면 모델 구조로)
    inference = default_inference(model, suite)
    member_kinds = args.member_kinds or inference["member_kinds"]
    n_estimators = args.n_estimators or inference["n_estimators"]

    def make(task: bm.OpenMLTask, one_hot: str):
        kw = {
            "n_estimators": n_estimators,
            "categorical_features_indices": task.categorical_indices,
            "one_hot": one_hot,
            "member_kinds": member_kinds,
            "device": args.device,
        }
        if suite == "classification":
            return EnsembleClassifier(model, **kw)
        return EnsembleRegressor(model, borders, **kw)

    make.max_one_hot_width = max_features_of(model)  # expands_one_hot이 같은 상한을 쓰도록
    make.member_kinds = member_kinds
    make.n_estimators = n_estimators
    make.model, make.borders = model, borders  # TabArena·BeyondArena의 크기 상한(arena_limits)
    return make


def arena_suite(name: str):
    from minipfn.benchmarks import beyondarena, tabarena

    if name == "tabarena-lite":
        return tabarena.SUITE
    return beyondarena.suite("lite" if name == "beyondarena-lite" else "core2k")


def arena_limits(models: dict, args: argparse.Namespace) -> dict:
    """문제 종류마다 돌릴 split의 상한. 체크포인트가 없는 쪽(분류·회귀)은 빠진다(imputed).
    크기는 모델 구조의 논문 권장값(`minipfn.recipes.supported_size`: recipe의 `SUPPORTED_SIZE`,
    모르는 구조는 그 recipe의 첫 구조), 클래스 수는 모델 출력 수."""
    from minipfn.benchmarks.arena import ModelLimits

    limits = {}
    for task, (model, _) in models.items():
        size = supported_size(model, task)
        if args.max_train_rows is not None:
            size["max_train_rows"] = args.max_train_rows
        if args.max_features is not None:
            size["max_features"] = args.max_features
        if task == "classification":
            lim = ModelLimits(**size, max_classes=int(model.config.n_out))
            limits.update(binary=lim, multiclass=lim)
        else:
            limits["regression"] = ModelLimits(**size)
    return limits


def run_arena(args: argparse.Namespace, out_dir: Path, use_mlflow: bool) -> dict:
    """TabArena-Lite·BeyondArena를 돌리고 점수(`summary.json`의 내용)를 돌려준다. estimator는
    AMLB 평가와 같은 `make_estimator_factory`로 만든다(문제 종류와 범주형 열만 바꿔 넘긴다)."""
    import json
    import types

    from minipfn.benchmarks import arena

    suite = arena_suite(args.suite)
    paths = {"classification": args.classifier_path, "regression": args.regressor_path}
    factories = {
        task: make_estimator_factory(task, args) for task, path in paths.items() if path is not None
    }
    if not factories:
        raise SystemExit(f"{args.suite}에는 --classifier-path나 --regressor-path가 필요하다.")
    one_hot = headline_mode(args.one_hot)
    kinds = {task: make.member_kinds for task, make in factories.items()}
    print(f"member kinds: { {t: ','.join(k) for t, k in kinds.items()} }, one-hot: {one_hot}")

    def make(problem_type: str, categorical: list[int]):
        task = "regression" if problem_type == "regression" else "classification"
        return factories[task](types.SimpleNamespace(categorical_indices=categorical), one_hot)

    # 끝난 split을 다시 쓸지 정하는 값: 체크포인트와 추론 설정이 같을 때만 다시 쓴다
    fingerprint = json.dumps(
        {
            "paths": {t: paths[t] for t in factories},
            "member_kinds": {t: list(k) for t, k in kinds.items()},
            "one_hot": one_hot,
        },
        sort_keys=True,
    )
    models = {task: (f.model, f.borders) for task, f in factories.items()}
    datasets = None if args.datasets is None else args.datasets.split(",")
    summary = arena.run(
        suite,
        make,
        limits=arena_limits(models, args),
        out_dir=out_dir / args.suite,
        method=args.method_name,
        bagging=args.bagging,
        datasets=datasets,
        fingerprint=fingerprint,
    )
    print_arena_summary(summary)
    if use_mlflow:
        log_arena_mlflow(args, summary, out_dir / args.suite, kinds, one_hot)
    return summary


def print_arena_summary(summary: dict) -> None:
    def line(label: str, s: dict) -> str:
        if s.get("elo") is None:
            return f"{label:24s} 결과가 있는 split이 없다 (imputed {s['imputed_pct']:.1f}%)"
        xgb = s.get("winrate", {}).get("XGBoost (default)", {})
        return (
            f"{label:24s} elo {s['elo']:.0f} (+{s['elo_ci_plus']:.0f}/-{s['elo_ci_minus']:.0f}) "
            f"score {s['score']:.3f} improvability {s['improvability']:.2f}% "
            f"imputed {s['imputed_pct_leaderboard']:.1f}% "
            f"win vs XGB(default) {xgb.get('all', float('nan')):.3f} "
            f"position {s['position']}/{s['n_methods']}"
        )

    print(line(summary["suite"], summary))
    for label, s in summary["breakdown"].items():
        print(line(f"  {label}", s))


def log_arena_mlflow(args, summary: dict, out_dir: Path, kinds: dict, one_hot: str) -> None:
    prefix = summary["suite"].replace("-", "_")
    mlflow.log_params(
        {
            "suite": args.suite,
            "bagging": args.bagging,
            "method_name": args.method_name,
            "run_id_cache": summary["run_id"],
            "one_hot": one_hot,
            **{f"member_kinds_{t}": ",".join(k) for t, k in kinds.items()},
        }
    )
    paths = {"classifier_path": args.classifier_path, "regressor_path": args.regressor_path}
    mlflow.log_params({k: v for k, v in paths.items() if v is not None})

    def metrics(p: str, s: dict) -> dict:
        keys = ("elo", "score", "improvability", "rank", "imputed_pct_leaderboard")
        out = {f"{p}_{k}": s[k] for k in keys if s.get(k) is not None}
        out[f"{p}_imputed_pct_splits"] = s["imputed_pct"]
        for baseline, w in s.get("winrate", {}).items():
            key = baseline.split(" (")[0].lower().replace("-", "").replace(".", "")
            sub = baseline.split("(")[-1].rstrip(")").replace(" + ", "_").replace(" ", "_")
            if w["all"] == w["all"]:  # NaN은 남기지 않는다
                out[f"{p}_winrate_{key}_{sub}"] = w["all"]
        return out

    mlflow.log_metrics(metrics(prefix, summary))
    for label, s in summary["breakdown"].items():
        mlflow.log_metrics(metrics(f"{prefix}_{label.replace('-', '_')}", s))
    for path in sorted(out_dir.glob("*.csv")) + [out_dir / "summary.json"]:
        mlflow.log_artifact(str(path), f"eval/{summary['suite']}")


def summarize(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    keys = ["one_hot", "suite", "task_id", "dataset"]
    per_dataset = df.groupby(keys)["score"].mean().reset_index()
    paper = {**bm.paper_scores("classification"), **bm.paper_scores("regression")}
    per_dataset["paper"] = per_dataset["task_id"].map(paper)
    per_dataset["diff"] = per_dataset["score"] - per_dataset["paper"]
    return per_dataset


@contextlib.contextmanager
def mlflow_run(use_mlflow: bool):
    """평가 run 하나. AI Runtime에서는 플랫폼이 만든 run(`MLFLOW_RUN_ID`), 로컬에서는
    `TABPFN_V2_EXPERIMENT`에 새 run. 도중에 예외가 나면 run을 FAILED로 끝낸다."""
    if not use_mlflow:
        yield
        return
    if "MLFLOW_RUN_ID" not in os.environ:
        mlflow.set_experiment(os.environ.get("TABPFN_V2_EXPERIMENT", "tabpfn-v2-reproduction"))
    with mlflow.start_run(run_id=os.environ.get("MLFLOW_RUN_ID")):
        log_run_source(__file__, env_prefix="TABPFN_V2_")
        yield


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    use_mlflow = args.mlflow or "MLFLOW_RUN_ID" in os.environ
    if args.suite in ARENA_SUITES:
        with mlflow_run(use_mlflow):
            run_arena(args, out_dir, use_mlflow)
        return
    tasks = selected_tasks(args)  # 고른 task가 틀렸으면 run을 만들기 전에 끝낸다
    with mlflow_run(use_mlflow):
        run_amlb(args, out_dir, tasks, use_mlflow)


def run_amlb(args: argparse.Namespace, out_dir: Path, tasks: list, use_mlflow: bool) -> None:
    """논문 benchmark(AMLB 분류 29개, AMLB·CTR23 회귀 28개)를 평가하고 논문 수치와 비교한다."""
    headline = headline_mode(args.one_hot)
    print(
        f"tasks: {len(tasks)}, folds: {args.folds}, device: {args.device}, "
        f"one-hot: {','.join(args.one_hot)} (결과: {headline})",
        flush=True,
    )

    if use_mlflow:
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
    kinds = {suite: ",".join(make.member_kinds) for suite, make in sorted(factories.items())}
    print("member kinds:", kinds, flush=True)
    if use_mlflow:
        # suite마다 모델이 정하므로 다를 수 있다. 같으면 예전처럼 member_kinds 하나로 남긴다
        if len(set(kinds.values())) == 1:
            mlflow.log_param("member_kinds", next(iter(kinds.values())))
        else:
            mlflow.log_params({f"member_kinds_{suite}": v for suite, v in kinds.items()})
        mlflow.log_params(
            {f"n_estimators_{suite}": make.n_estimators for suite, make in factories.items()}
        )
    rows = []
    for i, (suite, task_id) in enumerate(tasks, start=1):
        start = time.time()
        task = bm.load_openml_task(task_id)
        scores = {}
        expands = expands_one_hot(task, factories[suite].max_one_hot_width, args.folds)
        for group in same_result_modes(args.one_hot, factories[suite].member_kinds, expands):
            # 한 묶음의 모드는 결과가 같으므로 첫 모드만 평가하고 나머지에는 그 결과를 쓴다
            make = functools.partial(factories[suite], one_hot=group[0])
            task_rows = bm.evaluate_task(task, make, n_folds=args.folds)
            scores[group[0]] = float(np.nanmean([r["score"] for r in task_rows]))
            for mode in group:
                rows.extend(
                    {**r, "suite": suite, "one_hot": mode, "reused": mode != group[0]}
                    for r in task_rows
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


if __name__ == "__main__":
    main()
