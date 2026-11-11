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

`--uncertainty`를 주면 점 예측 점수와 함께 예측 분포·불확실성 지표
(`minipfn.benchmarks.uncertainty`)를 fold마다 낸다. 회귀는 CRPS, 가운데 구간(50·80·90·95%)의
coverage·폭·interval score, 폭 5분위별 coverage의 최솟값, PIT의 KS 통계량이고, 분류는 ECE,
log loss, Brier, split conformal(LAC·APS, α = 0.1)의 coverage·집합 크기·SSCS·클래스 조건부
coverage gap이다. 데이터셋별 표는 `eval_uncertainty_per_dataset.csv`, suite 평균은
`<suite>_mean_<지표>`로 남긴다(회귀의 CRPS·폭·interval score는 학습 target 표준편차로 나눈
`_scaled`만 평균한다).

  uv run python $EVAL --suite regression --regressor-path reg.pt --uncertainty

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
가운데 주지 않은 쪽과 논문 권장 크기(recipe의 `SUPPORTED_SIZE`; 클래스 수는 모델이 받는 수
`max_classes_of`)를
넘는 split은 돌리지 않고 imputed로 둔다(`--max-train-rows`·`--max-features`로 바꾼다). 기본은 학습
split 전체에 한 번 fit하고, `--bagging 8`은 TabArena처럼 8-fold로 나눠 8개의 예측을 평균한다.
`--datasets`로 일부만 돌릴 수 있다(나머지는 imputed). 결과는 `<out-dir>/<suite>/`
(`minipfn.benchmarks.arena.run`).

  uv run --no-dev --extra arena --group arena-dev python $EVAL --suite tabarena-lite \
    --classifier-path hf://aiden-jeon/minipfn-tabpfn-v2-classifier \
    --regressor-path hf://aiden-jeon/minipfn-tabpfn-v2-regressor

`--seed`(기본 0)는 앙상블 멤버의 난수(feature·클래스 순서, 식별자 열 등)를 바꾼다. 같은 모델을
seed만 바꿔 다시 평가하면 benchmark 점수의 노이즈를 잴 수 있다(issue #67). arena 결과 cache는
seed마다 따로다.

`--max-features-per-member 6000`은 feature가 6,000개보다 많은 표에서 앙상블 멤버마다 6,000개를
무작위로 골라 넣는다(공식 TabPFN의 `max_features_per_estimator`). 주면 TabArena·BeyondArena의
feature 수 상한을 없애서, 권장 크기보다 넓은 split(BeyondArena의 6,772~22,215개)도
돌린다(issue #66).

`--test-time`은 추론에 연산을 더 쓰는 방법이다(issue #43, arXiv 2610.12005). `greedy`는 멤버 pool을
학습 행 안의 out-of-fold 손실로 greedy 선택하고(`minipfn.inference.ensemble_selection`,
`--selection`·`--pool-size`), `diagscale`은 데이터셋마다 attention의 대각 scale만 학습한다
(`minipfn.inference.diag_scale`). `diagscale+greedy`는 둘 다. fit마다의 선택·학습 기록은
`<out-dir>/test_time.jsonl`에 남긴다.

  uv run --no-dev --extra arena --group arena-dev python $EVAL --suite tabarena-lite \
    --classifier-path hf://aiden-jeon/minipfn-tabpfn-v2-classifier --test-time greedy \
    --method-name minipfn-tabpfn-v2-greedy

greedy의 out-of-fold 과적합을 막는 설정(issue #52): `--min-improvement`(기본 0.01)는 out-of-fold
손실이 기본보다 그 비율 이상 낮지 않으면 기본 앙상블을 쓰고, `--oof-split task`(기본)는
TabArena·BeyondArena split의 group·시간 열로 out-of-fold fold를 나누고, `--min-temperature`(기본
없음)는 그보다 낮은 온도를 pool에서 뺀다. PR #49(논문 방식)와 같게 하려면
`--min-improvement 0 --oof-split random`.

`--distill`은 teacher(위의 추론 estimator)를 데이터셋마다 GBDT 학생으로 증류해 학생으로만 예측한다
(issue #44, `minipfn.inference.distillation`, extra `distill`). `oof`는 out-of-fold teacher 라벨
(arXiv 2605.18654), `synthetic`은 학습 행 전체를 context로 학습 행 + CutMix synthetic 행을 라벨링
(arXiv 2610.01435), `labels`는 teacher 없이 정답 라벨만 배운 같은 학생(기본 GBDT 비교용)이다.
학생은 `--student`(xgboost, lightgbm, hist)이고 CPU에서 예측한다. fit마다 teacher·학생 학습 시간과
나무 수를 `<out-dir>/distill_<suite>.jsonl`에 남긴다(이 실행의 fit만: 실행마다 새로 쓴다).
학생은 평균만 내므로 회귀에서는 `--uncertainty`와 같이 쓸 수 없다.
`--latency`(TabArena·BeyondArena)는 fit마다 테스트 행 하나의 예측 시간(`LATENCY_REPEATS`번의
중앙값)도 남긴다(teacher만 쓸 때도). 예측이 끝난 뒤 시간을 재지 않는 hook에서 재므로 TabArena의
학습·추론 시간에는 들지 않는다.

  uv run --no-dev --extra arena --extra distill --group arena-dev python $EVAL \
    --suite tabarena-lite --classifier-path hf://aiden-jeon/minipfn-tabpfn-v2-classifier \
    --regressor-path hf://aiden-jeon/minipfn-tabpfn-v2-regressor \
    --distill synthetic --student xgboost --method-name minipfn-tabpfn-v2-kd-xgb

AI Runtime(`scripts/tabpfn_v2/eval.yaml`)에서는 플랫폼이 만든 MLflow run(`MLFLOW_RUN_ID`)에
기록한다. 로컬에서 `--mlflow`를 주면 `TABPFN_V2_EXPERIMENT`(기본 tabpfn-v2-reproduction)
experiment에 새 run을 만든다.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import sys
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch

from minipfn.benchmarks import amlb as bm
from minipfn.benchmarks import uncertainty as unc
from minipfn.hub import load_model
from minipfn.inference.diag_scale import DiagScale
from minipfn.inference.distillation import (
    METHODS,
    STUDENTS,
    DistilledClassifier,
    DistilledRegressor,
)
from minipfn.inference.ensemble import (
    DEFAULT_ONE_HOT,
    EnsembleClassifier,
    EnsembleRegressor,
    max_classes_of,
    max_features_of,
)
from minipfn.inference.ensemble_selection import (
    MIN_IMPROVEMENT,
    POOL_SIZE,
    SELECTIONS,
    GreedyEnsembleClassifier,
    GreedyEnsembleRegressor,
    check_selection,
)
from minipfn.inference.preprocessing import (
    ONE_HOT_MODES,
    one_hot_kinds,
    one_hot_sizes,
    to_numeric,
)
from minipfn.priors.table_size import MAX_FEATURES
from minipfn.recipes import default_inference, resolve_member_kinds, supported_size
from minipfn.utils import mlflow_eval_run

ARENA_SUITES = ("tabarena-lite", "beyondarena", "beyondarena-lite")
TEST_TIME = ("none", "greedy", "diagscale", "diagscale+greedy")
OOF_SPLITS = ("random", "task")
LATENCY_REPEATS = 5


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
        "--seed",
        type=int,
        default=0,
        help="앙상블 멤버의 난수 seed (feature·클래스 순서, 식별자 열 등. 같은 모델의 평가 노이즈를"
        " 잴 때 바꾼다)",
    )
    parser.add_argument(
        "--uncertainty",
        action="store_true",
        help="AMLB: 예측 분포·불확실성 지표도 낸다 (회귀 CRPS·coverage·PIT, 분류 ECE·conformal)",
    )
    parser.add_argument(
        "--test-time",
        choices=TEST_TIME,
        default="none",
        help="추론에 연산을 더 쓰는 방법: 멤버 pool greedy 선택, DiagScale, 또는 둘 다",
    )
    parser.add_argument(
        "--selection",
        choices=SELECTIONS,
        default="greedy",
        help="--test-time greedy의 선택 방식 (best: 하나, uniform: 균일 평균)",
    )
    parser.add_argument(
        "--pool-size", type=int, default=POOL_SIZE, help="--test-time greedy의 pool 멤버 수"
    )
    parser.add_argument(
        "--min-improvement",
        type=float,
        default=None,
        help="--test-time greedy: 고른 조합의 out-of-fold 손실이 기본보다 이 비율 이상 낮지 않으면"
        f" 기본 앙상블을 쓴다 (기본 {MIN_IMPROVEMENT}, 0: 조건 없음, 논문 방식)",
    )
    parser.add_argument(
        "--min-temperature",
        type=float,
        default=None,
        help="--test-time greedy: 이보다 낮은 softmax 온도를 pool에서 뺀다 (기본: 빼지 않는다)",
    )
    parser.add_argument(
        "--oof-split",
        choices=OOF_SPLITS,
        default=None,
        help="--test-time greedy의 out-of-fold fold (기본 task: TabArena·BeyondArena split의"
        " group·시간 열로 나눈다. 무작위 split과 AMLB는 그대로 무작위. random: 논문 방식."
        " diagscale+greedy에서도 DiagScale의 검증 split은 무작위다)",
    )
    parser.add_argument(
        "--distill",
        choices=["none", *METHODS],
        default="none",
        help="teacher를 GBDT 학생으로 증류해 학생으로 예측한다 (oof, synthetic, labels: 정답만)",
    )
    parser.add_argument(
        "--student", choices=STUDENTS, default="xgboost", help="--distill의 학생 (CPU)"
    )
    parser.add_argument(
        "--synthetic-ratio",
        type=float,
        default=None,
        help="--distill: synthetic 행 수 / 학습 행 수 (기본: 방법이 정한다. oof 0, synthetic 2)",
    )
    parser.add_argument(
        "--distill-alpha",
        type=float,
        default=None,
        help="--distill: teacher target의 비중, 나머지는 정답 라벨 (기본: oof 0.7, synthetic 1)",
    )
    parser.add_argument(
        "--latency",
        action="store_true",
        help="TabArena·BeyondArena: fit마다 한 행 예측 시간을 distill_<suite>.jsonl에 남긴다",
    )
    parser.add_argument(
        "--bagging", type=int, default=0, help="TabArena·BeyondArena: k-fold bagging (0: 안 함)"
    )
    parser.add_argument(
        "--datasets", help="TabArena·BeyondArena: 쉼표로 구분한 데이터셋 이름 (나머지는 imputed)"
    )
    parser.add_argument(
        "--shards",
        type=int,
        default=1,
        help="TabArena·BeyondArena: 데이터셋을 이만큼 나눠 GPU마다 worker 하나로 동시에 돌린다"
        " (GPU 수 이하, 1이면 지금처럼 한 프로세스)",
    )
    # 내부용: worker가 split 결과를 남길 파일 (`--shards`가 넘긴다)
    parser.add_argument("--shard-results", help=argparse.SUPPRESS)
    parser.add_argument(
        "--max-train-rows", type=int, help="TabArena·BeyondArena: 학습 행 상한 (기본: 모델 구조)"
    )
    parser.add_argument(
        "--max-features", type=int, help="TabArena·BeyondArena: feature 수 상한 (기본: 모델 구조)"
    )
    parser.add_argument(
        "--max-features-per-member",
        type=int,
        default=None,
        help="feature가 이보다 많으면 앙상블 멤버마다 이만큼만 무작위로 고른다(공식 TabPFN의"
        " max_features_per_estimator). 주면 TabArena·BeyondArena의 feature 수 상한은 없앤다"
        "(--max-features로 다시 둘 수 있다)",
    )
    parser.add_argument(
        "--method-name", default="minipfn-tabpfn-v2", help="TabArena·BeyondArena 결과의 방법 이름"
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", default="tabpfn_v2_eval")
    parser.add_argument("--mlflow", action="store_true", help="로컬에서도 MLflow에 기록한다")
    args = parser.parse_args(argv)
    # greedy 설정은 greedy에서만 준다 (주지 않으면 greedy의 기본값)
    given = [
        n
        for n in ("min_improvement", "min_temperature", "oof_split")
        if getattr(args, n) is not None
    ]
    if given and "greedy" not in args.test_time:
        parser.error("--min-improvement·--min-temperature·--oof-split은 greedy에서만 쓴다.")
    if args.min_improvement is None:
        args.min_improvement = MIN_IMPROVEMENT
    if args.oof_split is None:
        args.oof_split = "task"
    return args


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
    if "greedy" in args.test_time:
        # 여기서 멈춘다: arena에서는 estimator를 만들 때 실패하면 모든 split이 조용히 imputed가 된다
        try:
            check_selection(
                n_estimators=n_estimators,
                pool_size=args.pool_size,
                selection=args.selection,
                min_improvement=args.min_improvement,
                min_temperature=args.min_temperature,
            )
        except ValueError as e:
            raise SystemExit(f"{suite}: {e}") from e
    log_path = Path(args.out_dir) / "test_time.jsonl"
    # 같은 out-dir에 다른 방법의 기록이 쌓여도 구분되도록 줄마다 남긴다
    log_fields = {
        "suite": suite,
        "method": args.method_name,
        "seed": args.seed,
        **test_time_params(args),
    }
    fits_log = fits_path(args)

    def make(task: bm.OpenMLTask, one_hot: str):
        kw = {
            "n_estimators": n_estimators,
            "categorical_features_indices": task.categorical_indices,
            "one_hot": one_hot,
            "member_kinds": member_kinds,
            "device": args.device,
            "random_state": args.seed,
            "max_features_per_member": args.max_features_per_member,
        }
        if "greedy" in args.test_time:
            # 기본 앙상블(후보 0번, pool 앞 멤버)의 멤버 수는 n_estimators
            kw.update(
                selection=args.selection,
                pool_size=args.pool_size,
                min_improvement=args.min_improvement,
                min_temperature=args.min_temperature,
            )
            if suite == "classification":
                estimator = GreedyEnsembleClassifier(model, **kw)
            else:
                estimator = GreedyEnsembleRegressor(model, borders, **kw)
        elif suite == "classification":
            estimator = EnsembleClassifier(model, **kw)
        else:
            estimator = EnsembleRegressor(model, borders, **kw)
        if args.test_time.startswith("diagscale"):
            estimator = DiagScale(estimator)
        if args.test_time != "none":
            fields = dict(log_fields)
            if getattr(task, "name", None):  # AMLB task만 이름이 있다
                fields["dataset"] = task.name
            record_fits(estimator, log_path, **fields)
        if args.distill != "none":
            distilled = DistilledClassifier if suite == "classification" else DistilledRegressor
            estimator = distilled(
                estimator,
                method=args.distill,
                student=args.student,
                synthetic_ratio=args.synthetic_ratio,
                alpha=args.distill_alpha,
                categorical_features_indices=task.categorical_indices,
                random_state=args.seed,
            )
        if args.distill != "none" or args.latency:
            record = {"suite": args.suite, "task": suite, "dataset": getattr(task, "name", None)}
            estimator = RecordedEstimator(estimator, record, fits_log, latency=args.latency)
        return estimator

    # arena system이 fit에 split의 group 번호(groups)를 넘긴다
    make.fit_groups = "greedy" in args.test_time and args.oof_split == "task"
    make.max_one_hot_width = max_features_of(model)  # expands_one_hot이 같은 상한을 쓰도록
    make.member_kinds = member_kinds
    make.n_estimators = n_estimators
    make.model, make.borders = model, borders  # TabArena·BeyondArena의 크기 상한(arena_limits)
    return make


def record_fits(estimator, path: Path, **fields) -> None:
    """fit할 때마다 `fields`(suite, 방법, test-time 설정 등)와 test-time 방법의 기록(greedy 선택
    `selection_`, DiagScale `adaptation_`), 표 크기, fit 시간을 `path`(JSONL)에 한 줄씩 덧붙인다."""
    fit = estimator.fit

    def fit_and_record(X, y, **fit_kwargs):
        start = time.perf_counter()
        fitted = fit(X, y, **fit_kwargs)
        record = {**fields, "rows": len(X), "features": int(np.shape(X)[1])}
        for key in ("selection_", "adaptation_"):
            value = getattr(fitted, key, None)
            if value is not None:
                record[key.rstrip("_")] = value
        record["fit_seconds"] = time.perf_counter() - start
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(record, default=float) + "\n")
        return fitted

    estimator.fit = fit_and_record


def test_time_params(args: argparse.Namespace) -> dict:
    """test-time 방법을 정하는 인자 (MLflow param, arena 결과를 다시 쓸지 정하는 값)."""
    params = {"test_time": args.test_time}
    if "greedy" in args.test_time:
        params.update(selection=args.selection, pool_size=args.pool_size)
        # issue #52의 설정은 논문 방식과 다를 때만 (논문 방식의 arena 결과 cache는 그대로 다시 쓴다)
        if args.min_improvement:
            params["min_improvement"] = args.min_improvement
        if args.min_temperature is not None:
            params["min_temperature"] = args.min_temperature
        if args.oof_split != "random":
            params["oof_split"] = args.oof_split
    return params


def fits_path(args: argparse.Namespace) -> Path:
    """fit 기록 파일 (suite마다 하나, `main`이 실행마다 비운다)."""
    return Path(args.out_dir) / f"distill_{args.suite}.jsonl"


class RecordedEstimator:
    """estimator를 감싸 fit마다 기록을 `path`(JSONL)에 한 줄씩 남긴다: `record`(suite, task,
    데이터셋), 표 크기, fit 시간, 증류 기록(`fit_record_`: teacher·학생 학습 시간, 나무 수).

    `latency`면 한 행 예측 시간(ms, `LATENCY_REPEATS`번의 중앙값)을 `after_predict`에서 재고 그때
    쓴다. TabArena system이 예측이 끝난 뒤 시간을 재지 않는 hook(`post_predict`)에서 테스트 행
    하나로 부른다. 나머지 속성(`predict`, `predict_proba`, `classes_` 등)은 감싼 estimator의 것이다.
    """

    def __init__(self, estimator, record: dict, path: Path, latency: bool) -> None:
        self.estimator = estimator
        self.record = record
        self.path = path
        self.latency = latency

    def __getattr__(self, name: str):
        # 복사·pickle 도중(estimator가 아직 없을 때)에는 감싼 estimator를 찾지 않는다
        if name.startswith("__") or name == "estimator":
            raise AttributeError(name)
        return getattr(self.estimator, name)

    def fit(self, X, y, **fit_kwargs) -> RecordedEstimator:
        start = time.perf_counter()
        self.estimator.fit(X, y, **fit_kwargs)
        self.record_ = {
            **self.record,
            "rows": len(X),
            "features": int(np.shape(X)[1]),
            "fit_seconds": time.perf_counter() - start,
            **getattr(self.estimator, "fit_record_", {}),
        }
        if not self.latency:
            self._write()
        return self

    def after_predict(self, row) -> None:
        if not self.latency:
            return
        predict = (
            self.estimator.predict
            if self.record["task"] == "regression"
            else self.estimator.predict_proba
        )
        times = []
        for _ in range(LATENCY_REPEATS):
            t = time.perf_counter()
            predict(row)
            times.append(time.perf_counter() - t)
        self.record_["latency_ms"] = 1000 * float(np.median(times))
        self._write()

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as f:
            f.write(json.dumps(self.record_, default=float) + "\n")


def distill_params(args: argparse.Namespace) -> dict:
    """증류를 정하는 인자 (MLflow param, arena 결과를 다시 쓸지 정하는 값)."""
    params = {"distill": args.distill}
    if args.distill != "none":
        params["student"] = args.student
        for key in ("synthetic_ratio", "distill_alpha"):
            if getattr(args, key) is not None:
                params[key] = getattr(args, key)
    return params


def arena_suite(name: str):
    from minipfn.benchmarks import beyondarena, tabarena

    if name == "tabarena-lite":
        return tabarena.SUITE
    return beyondarena.suite("lite" if name == "beyondarena-lite" else "core2k")


def arena_limits(models: dict, args: argparse.Namespace) -> dict:
    """문제 종류마다 돌릴 split의 상한. 체크포인트가 없는 쪽(분류·회귀)은 빠진다(imputed).
    크기는 모델 구조의 논문 권장값(`minipfn.recipes.supported_size`: recipe의 `SUPPORTED_SIZE`,
    모르는 구조는 그 recipe의 첫 구조), 클래스 수는 모델이 받는 수(`max_classes_of`: 출력 수,
    column_row_transformer는 정답 embedding을 묶어 `MAX_CLASSES`개까지)."""
    from minipfn.benchmarks.arena import ModelLimits

    limits = {}
    for task, (model, _) in models.items():
        size = supported_size(model, task)
        if args.max_train_rows is not None:
            size["max_train_rows"] = args.max_train_rows
        if args.max_features is not None:
            size["max_features"] = args.max_features
        elif getattr(args, "max_features_per_member", None) is not None:
            size["max_features"] = None  # 멤버마다 feature를 골라 넣으므로 상한이 없다
        if task == "classification":
            lim = ModelLimits(**size, max_classes=max_classes_of(model))
            limits.update(binary=lim, multiclass=lim)
        else:
            limits["regression"] = ModelLimits(**size)
    return limits


def arena_fingerprint(
    args: argparse.Namespace, paths: dict, factories: dict, kinds: dict, one_hot: str
) -> str:
    """끝난 split을 다시 쓸지 정하는 값: 체크포인트와 추론 설정이 같을 때만 다시 쓴다. 기본값인
    설정은 넣지 않아서 예전 결과를 그대로 다시 쓴다. 다만 추론 계산이 바뀐 모델
    (`supports_inference_options`, column_row_transformer: 큰 표를 나눠 계산하고 분류 decoder를
    float32 attention으로 바꿨다, issue #66)은 표시를 더해 예전 계산의 split과 섞이지 않게 한다."""
    inference = any(
        getattr(f.model, "supports_inference_options", False) for f in factories.values()
    )
    return json.dumps(
        {
            "paths": {t: paths[t] for t in factories},
            "member_kinds": {t: list(k) for t, k in kinds.items()},
            "one_hot": one_hot,
            # 기본 추론(none)은 예전 fingerprint 그대로 둔다 (끝난 split을 다시 쓴다)
            **({} if args.test_time == "none" else test_time_params(args)),
            # 증류하지 않으면 예전 fingerprint 그대로 둔다 (끝난 split을 다시 쓴다)
            **({} if args.distill == "none" else distill_params(args)),
            # seed 0(기본)도 예전 fingerprint 그대로
            **({"seed": args.seed} if args.seed else {}),
            **({"inference": "chunked"} if inference else {}),
            # 주지 않으면(기본) 예전 fingerprint 그대로
            **(
                {"max_features_per_member": args.max_features_per_member}
                if args.max_features_per_member is not None
                else {}
            ),
        },
        sort_keys=True,
    )


def run_arena(args: argparse.Namespace, out_dir: Path, use_mlflow: bool) -> dict:
    """TabArena-Lite·BeyondArena를 돌리고 점수(`summary.json`의 내용)를 돌려준다. estimator는
    AMLB 평가와 같은 `make_estimator_factory`로 만든다(문제 종류와 범주형 열만 바꿔 넘긴다)."""
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

    # arena system은 이 함수를 받으므로 split의 group 번호를 넘길지도 여기에 단다
    make.fit_groups = any(getattr(f, "fit_groups", False) for f in factories.values())

    fingerprint = arena_fingerprint(args, paths, factories, kinds, one_hot)
    models = {task: (f.model, f.borders) for task, f in factories.items()}
    datasets = None if args.datasets is None else args.datasets.split(",")
    limits = arena_limits(models, args)
    common = dict(
        limits=limits, method=args.method_name, bagging=args.bagging, fingerprint=fingerprint
    )
    if args.shard_results:  # worker: 맡은 데이터셋만 돌리고 결과를 남긴다
        arena.run(
            suite,
            make,
            out_dir=out_dir / args.suite,
            datasets=datasets,
            results_path=args.shard_results,
            **common,
        )
        return {}
    results = None
    if args.shards > 1:
        results = run_shards(args, suite, limits, datasets, out_dir)
    summary = arena.run(
        suite, make, out_dir=out_dir / args.suite, datasets=datasets, results=results, **common
    )
    print_arena_summary(summary)
    if use_mlflow:
        log_arena_mlflow(args, summary, out_dir / args.suite, kinds, one_hot)
    return summary


def _strip_flags(argv: list[str], flags: dict[str, int]) -> list[str]:
    """`argv`에서 `flags`(이름: 값 개수)를 뺀다. `--name=value` 꼴도 뺀다."""
    out, i = [], 0
    while i < len(argv):
        name = argv[i].split("=", 1)[0]
        if name in flags:
            i += 1 if "=" in argv[i] else 1 + flags[name]
            continue
        out.append(argv[i])
        i += 1
    return out


def shard_commands(
    argv: list[str], shards: list[list[str]], out_dir: Path
) -> list[tuple[list[str], Path, Path]]:
    """worker마다 (명령 인자, out-dir, 결과 파일). 원래 인자에서 나누는 데 쓰는 인자와 MLflow 기록을
    빼고 맡은 데이터셋·자기 out-dir·결과 파일을 더한다."""
    base = _strip_flags(
        argv,
        {"--shards": 1, "--datasets": 1, "--out-dir": 1, "--mlflow": 0, "--shard-results": 1},
    )
    out = []
    for i, names in enumerate(shards):
        worker_dir = out_dir / "shards" / f"shard{i}"
        results = worker_dir / "results.pkl"
        cmd = [
            *base,
            "--datasets",
            ",".join(names),
            "--out-dir",
            str(worker_dir),
            "--shard-results",
            str(results),
        ]
        out.append((cmd, worker_dir, results))
    return out


def run_shards(args, suite, limits, datasets, out_dir: Path) -> list:
    """데이터셋을 `--shards`개로 나눠(`arena.shard_datasets`) GPU마다 worker 프로세스 하나로 동시에
    돌리고, worker들의 split 결과를 모아 돌려준다. worker의 출력은 `<out-dir>/shards/shard<i>.log`,
    test-time·증류 기록(jsonl)은 끝나면 이 실행의 파일 뒤에 붙인다. worker가 하나라도 실패하면
    멈춘다(그 데이터셋이 조용히 imputed가 되지 않도록)."""
    import pickle
    import subprocess

    from minipfn.benchmarks import arena

    n_gpus = torch.cuda.device_count()
    if n_gpus and args.shards > n_gpus:
        raise SystemExit(f"--shards {args.shards}가 GPU 수 {n_gpus}보다 많다.")
    # 부모가 이미 GPU를 골라 두었으면(CUDA_VISIBLE_DEVICES) worker i는 그 목록의 i번째를 쓴다
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    devices = visible.split(",") if visible else [str(i) for i in range(max(n_gpus, args.shards))]
    grid = arena.task_grid(arena.make_context(suite), suite, limits)
    if datasets is not None:
        # worker를 띄우기 전에 멈춘다 (점수를 낼 때의 `arena.run`도 같은 이유로 멈춘다)
        unknown = sorted(set(datasets) - set(grid["dataset"]))
        if unknown:
            raise SystemExit(f"{suite.name}에 없는 데이터셋: {unknown}")
        grid = grid[grid["dataset"].isin(datasets)]
    shards = arena.shard_datasets(grid, args.shards)
    if not shards:  # 돌릴 split이 없다: 점수만 낸다 (모두 imputed)
        return []
    commands = shard_commands(args.argv, shards, out_dir)
    threads = max(1, (os.cpu_count() or 1) // len(commands))
    # 부모가 받아 둔 checkpoint(runs:/)를 worker가 다시 받지 않도록 같은 폴더를 가리킨다
    checkpoints = (out_dir / "checkpoints").resolve()
    procs = []
    try:
        for i, (cmd, worker_dir, results_file) in enumerate(commands):
            worker_dir.mkdir(parents=True, exist_ok=True)
            # 지난 실행의 기록·결과가 이 실행의 파일에 섞이지 않도록
            for stale in (worker_dir / "test_time.jsonl", results_file):
                stale.unlink(missing_ok=True)
            link = worker_dir / "checkpoints"
            if checkpoints.is_dir() and not link.exists():
                link.symlink_to(checkpoints, target_is_directory=True)
            env = {k: v for k, v in os.environ.items() if k != "MLFLOW_RUN_ID"}
            env.update(CUDA_VISIBLE_DEVICES=devices[i], OMP_NUM_THREADS=str(threads))
            log = open(out_dir / "shards" / f"shard{i}.log", "w")  # noqa: SIM115 (끝날 때까지 연다)
            print(f"shard {i}: {len(shards[i])} datasets → GPU {devices[i]}", flush=True)
            proc = subprocess.Popen(
                [sys.executable, __file__, *cmd], env=env, stdout=log, stderr=subprocess.STDOUT
            )
            procs.append((proc, log))
        failed = []
        for i, (proc, log) in enumerate(procs):
            code = proc.wait()
            log.close()
            if code:
                failed.append(i)
    finally:
        # 부모가 도중에 멈추면(예외, Ctrl-C) worker가 GPU를 잡고 남지 않도록
        for proc, log in procs:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            log.close()
    if failed:
        tails = {
            i: (out_dir / "shards" / f"shard{i}.log").read_text().splitlines()[-20:] for i in failed
        }
        raise SystemExit(f"shard {failed}가 실패했다: {tails}")
    results = []
    for _, worker_dir, path in commands:
        with open(path, "rb") as f:
            results.extend(pickle.load(f))
        for name in ("test_time.jsonl", f"distill_{args.suite}.jsonl"):
            part = worker_dir / name
            if part.exists():
                with open(out_dir / name, "a") as f:
                    f.write(part.read_text())
    print(f"shards: {len(results)} split results from {len(commands)} workers", flush=True)
    return results


def print_arena_summary(summary: dict) -> None:
    def line(label: str, s: dict) -> str:
        if s.get("elo") is None:
            return f"{label:24s} 결과가 있는 split이 없다 (imputed {s['imputed_pct']:.1f}%)"
        xgb = s.get("winrate", {}).get("XGBoost (default)", {})
        t = s.get("time", {})  # split 평균
        return (
            f"{label:24s} elo {s['elo']:.0f} (+{s['elo_ci_plus']:.0f}/-{s['elo_ci_minus']:.0f}) "
            f"score {s['score']:.3f} improvability {s['improvability']:.2f}% "
            f"imputed {s['imputed_pct_leaderboard']:.1f}% "
            f"win vs XGB(default) {xgb.get('all', float('nan')):.3f} "
            f"position {s['position']}/{s['n_methods']} "
            f"time train {t.get('time_train_s_mean', float('nan')):.1f}s "
            f"infer {t.get('time_infer_s_mean', float('nan')):.1f}s"
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
            "seed": args.seed,
            "max_features_per_member": args.max_features_per_member,
            **test_time_params(args),
            **distill_params(args),
            **{f"member_kinds_{t}": ",".join(k) for t, k in kinds.items()},
        }
    )
    paths = {"classifier_path": args.classifier_path, "regressor_path": args.regressor_path}
    mlflow.log_params({k: v for k, v in paths.items() if v is not None})

    def metrics(p: str, s: dict) -> dict:
        keys = ("elo", "score", "improvability", "rank", "imputed_pct_leaderboard")
        out = {f"{p}_{k}": s[k] for k in keys if s.get(k) is not None}
        out[f"{p}_imputed_pct_splits"] = s["imputed_pct"]
        out.update({f"{p}_{k}": v for k, v in s.get("time", {}).items()})
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
    log_test_time_artifact(args)
    log_distill_artifact(args)


def log_test_time_artifact(args: argparse.Namespace) -> None:
    path = Path(args.out_dir) / "test_time.jsonl"
    if path.exists():
        mlflow.log_artifact(str(path), "eval")


def log_distill_artifact(args: argparse.Namespace) -> None:
    path = fits_path(args)
    if path.exists():
        mlflow.log_artifact(str(path), "eval")


def summarize(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    keys = ["one_hot", "suite", "task_id", "dataset"]
    per_dataset = df.groupby(keys)["score"].mean().reset_index()
    paper = {**bm.paper_scores("classification"), **bm.paper_scores("regression")}
    per_dataset["paper"] = per_dataset["task_id"].map(paper)
    per_dataset["diff"] = per_dataset["score"] - per_dataset["paper"]
    return per_dataset


def mlflow_run(use_mlflow: bool):
    """평가 run 하나 (`minipfn.utils.mlflow_eval_run`, 로컬 experiment는 `TABPFN_V2_EXPERIMENT`)."""
    return mlflow_eval_run(use_mlflow, __file__, "TABPFN_V2_", "tabpfn-v2-reproduction")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.argv = list(sys.argv[1:] if argv is None else argv)  # `--shards`의 worker 명령에 쓴다
    if args.shards < 1:
        raise SystemExit("--shards는 1 이상이어야 한다.")
    if args.shards > 1 and args.suite not in ARENA_SUITES:
        raise SystemExit("--shards는 TabArena·BeyondArena suite에서만 쓴다.")
    if args.uncertainty and args.suite in ARENA_SUITES:
        # arena는 공식 패키지의 지표만 낸다. 조용히 무시하지 않고 멈춘다
        raise SystemExit("--uncertainty는 AMLB suite(classification, regression, all)에서만 쓴다.")
    if args.uncertainty and args.distill != "none" and args.suite in ("regression", "all"):
        # 회귀 학생은 평균만 내고 예측 분포가 없다
        raise SystemExit("--distill은 회귀 --uncertainty와 같이 쓸 수 없다 (학생은 평균만 낸다).")
    if args.test_time != "none" and args.distill != "none":
        # test-time 기록(record_fits)이 fit을 덮어써서, 증류가 teacher를 복사하면 원래 객체를 쓴다
        raise SystemExit("--distill은 --test-time none에서만 쓴다.")
    if args.latency and args.suite not in ARENA_SUITES:
        raise SystemExit("--latency는 TabArena·BeyondArena suite에서만 쓴다.")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fits_path(args).unlink(missing_ok=True)  # 이 실행의 fit만 남긴다
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
                "seed": args.seed,
                "max_features_per_member": args.max_features_per_member,
                **test_time_params(args),
                **distill_params(args),
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
            task_rows = bm.evaluate_task(
                task, make, n_folds=args.folds, uncertainty=args.uncertainty
            )
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
    artifacts = ["eval_rows.csv", "eval_per_dataset.csv", "eval_per_dataset_by_one_hot.csv"]
    if args.uncertainty:
        report_uncertainty(rows, headline, out_dir, use_mlflow)
        artifacts.append("eval_uncertainty_per_dataset.csv")
    if use_mlflow:
        for name in artifacts:
            mlflow.log_artifact(str(out_dir / name), "eval")
        log_test_time_artifact(args)
        log_distill_artifact(args)


def summarize_uncertainty(rows: list[dict], headline: str) -> pd.DataFrame:
    """headline 모드의 불확실성 지표(`minipfn.benchmarks.uncertainty`)의 데이터셋별 fold 평균."""
    rows = [r for r in rows if r["one_hot"] == headline]
    return unc.dataset_means(rows, exclude=("score", "one_hot", "reused"))


def report_uncertainty(rows: list[dict], headline: str, out_dir: Path, use_mlflow: bool) -> None:
    """데이터셋별 표를 저장하고 suite마다 데이터셋 평균을 출력·기록한다 (`<suite>_mean_<지표>`).
    단위가 있는 회귀 지표(CRPS, 폭, interval score)는 학습 target 표준편차로 나눈 `_scaled`만
    평균한다."""
    per_dataset = summarize_uncertainty(rows, headline)
    per_dataset.to_csv(out_dir / "eval_uncertainty_per_dataset.csv", index=False)
    for suite, group in per_dataset.groupby("suite"):
        means = unc.suite_means(group)
        print(f"{suite} uncertainty (one-hot {headline}, datasets {len(group)}):")
        print(pd.Series(means).round(4).to_string())
        if use_mlflow:
            mlflow.log_metrics({f"{suite}_mean_{k}": v for k, v in means.items()})


if __name__ == "__main__":
    main()
