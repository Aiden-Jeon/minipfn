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
아니면 v2. TabPFN-3·3.5 모델은 3이면 quantile,robust, 3.5면 standard 멤버 8개와 멤버당 768열
balanced).
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

TabPFN-3.5 모델은 `--max-features-per-member`를 주지 않으면 모델의 추론 설정(recipe
`default_inference`)대로 멤버마다 768열을 balanced로 고른다(issue #99, `model_feature_cap`. 분류·
회귀 모델마다 따로 정한다. test-time 방법에서는 예전처럼 모든 열). `--max-features-per-member 0`은
상한을 끈다(모든 열, 예전 기본값. arena의 feature 수 상한도 모델 구조의 값 그대로).
`--max-features-per-member 6000`은 feature가 6,000개보다 많은 표에서 앙상블 멤버마다 6,000개를
무작위로 골라 넣는다(공식 TabPFN의 `max_features_per_estimator`). 주면 TabArena·BeyondArena의
feature 수 상한을 없애서, 권장 크기보다 넓은 split(BeyondArena의 6,772~22,215개)도
돌린다(issue #66).

`--subsample-train-rows N`은 fit마다 학습 행이 N개보다 많으면 N개만 무작위로 골라 문맥으로 쓴다
(`minipfn.inference.subsample`, issue #91: 큰 표에서 긴 문맥을 쓰는지 보는 실험). 분류는 클래스
층화로 고르고 클래스마다 최소 1행을 남긴다. 고르는 난수는 `--seed`로 정해지고, 앙상블 멤버는 모두
같은 행을 쓴다. `--max-train-rows`는 이와 따로, 원래 학습 행 수로 split을 거른다(넘으면 돌리지
않고 imputed). 그래서 둘을 함께 주면 원래 행이 `--max-train-rows` 이하인 split만 돌리고, 그 가운데
N개보다 많은 split만 줄인다. `--bagging`이면 bag마다 따로 줄인다.

  uv run --no-dev --extra arena --group arena-dev python $EVAL --suite beyondarena \
    --classifier-path hf://aiden-jeon/minipfn-tabpfn-v3-5 \
    --regressor-path hf://aiden-jeon/minipfn-tabpfn-v3-5 \
    --subsample-train-rows 30000 --method-name minipfn-tabpfn-v3-5-ctx30k

공식 TabPFN-3.5(PriorLabs/TabPFN v9 이후) checkpoint에 든 추론 설정에 맞추는 옵션은 항목마다 따로
켠다(기본은 모두 main과 같다): `--softmax-temperature 1.0`, `--one-hot off --shuffle-categories`
(범주를 번호로 두고 멤버마다 섞기), `--target-transform safepower`·`--regression-mean mixture`(회귀
target 변환, 공통 경계로 옮긴 분포의 평균), `--max-features-per-member 768 --feature-subsampling
balanced`, `--fingerprint-feature hash`·`--outlier-clip soft`·`--infer-categorical`·
`--drop-constant`(`minipfn.inference.preprocessing`). `--official-v35`는 이것을 모두 켠다(따로 준
옵션이 먼저). `--test-time none`에서만 쓴다. 진단용으로 `--unseen-category minus1`(학습에 없던
범주를 결측 대신 번호 -1로, TabICLv2 방식. 공식 3.5는 NaN)과 `--drop-group-columns`
(TabArena·BeyondArena grouped split의 group 열을 fit·예측에서 뺀다)가 있다.

  uv run --no-dev --extra arena --group arena-dev python $EVAL --suite tabarena-lite \
    --classifier-path hf://aiden-jeon/minipfn-tabpfn-v3-5 \
    --regressor-path hf://aiden-jeon/minipfn-tabpfn-v3-5 \
    --official-v35 --method-name minipfn-tabpfn-v3-5-official-inference

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
import re
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
    REGRESSION_MEANS,
    SOFTMAX_TEMPERATURE,
    TARGET_TRANSFORMS,
    EnsembleClassifier,
    EnsembleRegressor,
    max_classes_of,
    max_features_of,
    quantile_regressor,
    uses_categorical,
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
    FEATURE_SUBSAMPLINGS,
    FINGERPRINTS,
    ONE_HOT_MODES,
    OUTLIER_CLIPS,
    UNSEEN_CATEGORIES,
    one_hot_kinds,
    one_hot_sizes,
    to_numeric,
)
from minipfn.inference.subsample import SubsampledEstimator, subsample_indices
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
        default=None,
        help=f"평가할 one-hot 모드, 쉼표로 구분 (기본: {DEFAULT_ONE_HOT}, --official-v35면 off,"
        " 비교: off,power,all)",
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
        " 잴 때 바꾼다). --subsample-train-rows면 고르는 학습 행도 이 seed로 정한다",
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
        help="TabArena·BeyondArena: GPU마다 worker 하나를 띄워 데이터셋을 큐에서 하나씩 나눠 준다"
        " (비싼 것부터. GPU 수 이하, 1이면 한 프로세스)",
    )
    parser.add_argument(
        "--bag-gpus",
        type=int,
        default=0,
        help="TabArena·BeyondArena --bagging: GPU마다 worker 하나를 띄워 split 하나의 bag들을"
        " 동시에 맞추고 예측한다 (데이터셋은 차례로. 0이면 이 프로세스에서 bag을 차례로,"
        " --shards와 따로)",
    )
    parser.add_argument(
        "--heartbeat-minutes",
        type=float,
        default=5.0,
        help="--shards: 이 간격(분)마다 진행률(끝난 split, worker별 데이터셋·경과 시간)을 찍는다",
    )
    parser.add_argument(
        "--host-memory-budget-gb",
        type=float,
        default=None,
        help="--shards: CPU 셀 상태를 쓰는 큰 표(stage_layout=interleaved 모델)는 동시에 도는 합이"
        " 이 GB 안일 때만 준다 (기본: 컨테이너 메모리 한도(cgroup)의 50%%, 한도를 모르면 끔."
        " 0: 끔). 큐 순서만 바뀐다",
    )
    # 내부용: stdin으로 받은 데이터셋을 하나씩 돌리는 worker (`--shards`가 띄운다)
    parser.add_argument("--shard-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--max-train-rows",
        type=int,
        help="TabArena·BeyondArena: 원래 학습 행이 이보다 많은 split은 돌리지 않는다(imputed)"
        " (기본: 모델 구조. --subsample-train-rows로 줄이기 전의 행 수로 본다)",
    )
    parser.add_argument(
        "--subsample-train-rows",
        type=int,
        default=None,
        help="fit마다 학습 행이 이보다 많으면 이만큼만 무작위로 골라 문맥으로 쓴다(분류는 클래스"
        " 층화, 클래스마다 최소 1행, 난수는 --seed, 멤버는 모두 같은 행, --bagging이면 bag마다)."
        " --max-train-rows로 거르고 남은 split에 쓴다 (기본: 줄이지 않는다)",
    )
    parser.add_argument(
        "--max-features", type=int, help="TabArena·BeyondArena: feature 수 상한 (기본: 모델 구조)"
    )
    parser.add_argument(
        "--max-features-per-member",
        type=int,
        default=None,
        help="feature가 이보다 많으면 앙상블 멤버마다 이만큼만 고른다(공식 TabPFN의"
        " max_features_per_estimator). 기본: 모델이 정한다(TabPFN-3.5는 768, 그 밖은 끔)."
        " 0이면 끈다(모든 열). 상한이 있으면 TabArena·BeyondArena의 feature 수 상한은 없앤다"
        "(--max-features로 다시 둘 수 있다)",
    )
    # 공식 TabPFN-3.5 추론 설정에 맞추는 옵션 (docstring). 주지 않으면 None → 기본값
    # (`--official-v35`면 `OFFICIAL_V35`)
    parser.add_argument(
        "--official-v35",
        action="store_true",
        help="공식 TabPFN-3.5 checkpoint의 추론 설정을 모두 켠다"
        " (아래 옵션을 따로 주면 그것이 먼저)",
    )
    parser.add_argument(
        "--softmax-temperature",
        type=float,
        default=None,
        help=f"logit을 나누는 온도 (기본 {SOFTMAX_TEMPERATURE}, 공식 3.5는 1.0)",
    )
    parser.add_argument(
        "--shuffle-categories",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="분위수가 아닌 멤버도 범주 번호를 멤버마다 섞는다 (공식 3.5의 ordinal_shuffled)",
    )
    parser.add_argument(
        "--fingerprint-feature",
        choices=FINGERPRINTS,
        default=None,
        help="샘플 식별자 열: random(기본, 표준정규 난수) 또는 hash(공식, 행 값의 hash)",
    )
    parser.add_argument(
        "--outlier-clip",
        choices=OUTLIER_CLIPS,
        default=None,
        help="이상치: hard(기본, 평균 ± 12σ로 자름) 또는 soft(공식, 두 번 구한 경계 밖을"
        " log로 누름)",
    )
    parser.add_argument(
        "--infer-categorical",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="학습 행이 100개보다 많으면 값이 4개보다 적은 수치 열을 범주형으로 본다"
        " (공식 type 추론)",
    )
    parser.add_argument(
        "--drop-constant",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="학습 행에서 값이 하나뿐인 열을 뺀다 (공식 RemoveConstantFeaturesStep)",
    )
    parser.add_argument(
        "--feature-subsampling",
        choices=FEATURE_SUBSAMPLINGS,
        default=None,
        help="--max-features-per-member로 열을 고르는 방법: random(기본) 또는 balanced(공식 3.5)",
    )
    parser.add_argument(
        "--target-transform",
        choices=TARGET_TRANSFORMS,
        default=None,
        help="회귀 멤버 절반의 target 변환: power(기본, z에 Yeo-Johnson), safepower(공식), none",
    )
    parser.add_argument(
        "--unseen-category",
        choices=UNSEEN_CATEGORIES,
        default=None,
        help="학습 행에 없던 범주: nan(기본, 결측과 같게. 공식 3.5도 NaN) 또는 minus1(번호 -1,"
        " TabICLv2 방식. one-hot 블록은 모두 0)",
    )
    parser.add_argument(
        "--drop-group-columns",
        action="store_true",
        help="TabArena·BeyondArena 진단: grouped split의 group 열(validation_metadata.group_on)을"
        " fit·예측 양쪽에서 뺀다",
    )
    parser.add_argument(
        "--regression-mean",
        choices=REGRESSION_MEANS,
        default=None,
        help="회귀 평균: member(기본, 멤버마다 자기 구간 값) 또는 mixture(공식, 공통 경계로 옮긴"
        " 분포의 평균, half-normal 꼬리)",
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
    # 모델의 기본 멤버당 열 상한(`model_feature_cap`)이 따로 준 나누기 방법을 덮지 않도록
    args.feature_subsampling_given = args.feature_subsampling is not None or args.official_v35
    if args.max_features_per_member is not None and args.max_features_per_member < 0:
        parser.error("--max-features-per-member는 0(끔) 이상이어야 한다.")
    resolve_inference_options(args)
    if inference_option_params(args) and args.test_time != "none":
        # greedy pool·DiagScale의 view는 이 옵션을 넘기지 않는다
        parser.error("공식 3.5 추론 옵션은 --test-time none에서만 쓴다.")
    if args.subsample_train_rows is not None and args.subsample_train_rows < 1:
        parser.error("--subsample-train-rows는 1 이상이어야 한다.")
    if args.drop_group_columns and args.suite not in ARENA_SUITES:
        # AMLB split에는 group 열이 없어 아무 일도 하지 않는다
        parser.error("--drop-group-columns는 TabArena·BeyondArena suite에서만 쓴다.")
    return args


# 공식 TabPFN-3.5 checkpoint(PriorLabs/TabPFN v9 이후)에 든 inference config에 맞춘 값
# (`--official-v35`). 범주형 열은 one-hot 대신 번호로 두고(`--one-hot off`) 멤버마다 섞는다
OFFICIAL_V35 = {
    "softmax_temperature": 1.0,
    "shuffle_categories": True,
    "fingerprint_feature": "hash",
    "outlier_clip": "soft",
    "infer_categorical": True,
    "drop_constant": True,
    "feature_subsampling": "balanced",
    "target_transform": "safepower",
    "regression_mean": "mixture",
}
OFFICIAL_V35_MAX_FEATURES_PER_MEMBER = 768
# 위 옵션의 기본값 (main의 추론 그대로)
INFERENCE_DEFAULTS = {
    "softmax_temperature": SOFTMAX_TEMPERATURE,
    "shuffle_categories": False,
    "fingerprint_feature": "random",
    "outlier_clip": "hard",
    "infer_categorical": False,
    "drop_constant": False,
    "feature_subsampling": "random",
    "target_transform": "power",
    "regression_mean": "member",
    "unseen_category": "nan",
}
REGRESSION_ONLY = ("target_transform", "regression_mean")


def resolve_inference_options(args: argparse.Namespace) -> None:
    """주지 않은(None) 추론 옵션을 기본값으로 채운다. `--official-v35`면 공식 3.5의 값과
    `--one-hot off`, `--max-features-per-member 768`로 (따로 준 옵션은 그대로)."""
    official = getattr(args, "official_v35", False)
    for name, default in INFERENCE_DEFAULTS.items():
        if getattr(args, name, None) is None:
            # 공식 3.5는 학습에 없던 범주를 NaN으로 둔다 (unseen_category 기본값과 같다)
            setattr(args, name, OFFICIAL_V35.get(name, default) if official else default)
    if getattr(args, "one_hot", None) is None:
        args.one_hot = ["off"] if official else [DEFAULT_ONE_HOT]
    if official and args.max_features_per_member is None:
        args.max_features_per_member = OFFICIAL_V35_MAX_FEATURES_PER_MEMBER


def inference_option_params(args: argparse.Namespace) -> dict:
    """기본값과 다른 추론 옵션 (arena fingerprint·MLflow param에 남긴다. 기본이면 빈 dict라 예전
    fingerprint 그대로)."""
    return {
        name: getattr(args, name)
        for name, default in INFERENCE_DEFAULTS.items()
        if getattr(args, name, default) != default
    }


def estimator_options(args: argparse.Namespace, suite: str) -> dict:
    """estimator에 줄 추론 옵션 (회귀 전용 옵션은 회귀에만). 기본값인 것은 넘기지 않는다."""
    out = {}
    for name, value in inference_option_params(args).items():
        if name in REGRESSION_ONLY and suite != "regression":
            continue
        out["fingerprint" if name == "fingerprint_feature" else name] = value
    return out


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


def expands_one_hot(
    task: bm.OpenMLTask,
    max_width: int = MAX_FEATURES,
    n_folds: int = 10,
    subsample_rows: int | None = None,
    seed: int = 0,
) -> bool:
    """one-hot 모드에 따라 결과가 달라질 수 있는지: 어느 fold의 학습 행에서든 펼칠 열이 있으면 True.

    전체 행으로 보면 안 된다. 범주가 전체로는 `MAX_ONE_HOT_CATEGORIES`개를 넘어도
    fold의 학습 행에서는 그 이하일 수 있고, 멤버 전처리는 fold마다 학습 행으로 맞춘다.
    `n_folds`는 평가에 쓰는 fold 수와 같아야 한다. 상한 `max_width`는 모델이 학습 때 본
    feature 수다 (`max_features_of`). `subsample_rows`(`--subsample-train-rows`)면 fit이 보는
    것처럼 fold의 학습 행을 `SubsampledEstimator`와 같은 규칙(`subsample_indices`)으로 줄여 본다.
    """
    for _, train_ix, _ in task.folds(n_folds):
        if subsample_rows is not None:
            y = task.y.iloc[train_ix].to_numpy()
            stratify = task.task_type == "classification"
            train_ix = train_ix[subsample_indices(y, subsample_rows, stratify=stratify, seed=seed)]
        if fold_expands_one_hot(task.X.iloc[train_ix], task.categorical_indices, max_width):
            return True
    return False


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
    quantile = suite == "regression" and quantile_regressor(model)
    if (uses_categorical(model) or quantile) and args.test_time != "none":
        # 범주 열 표시·회귀 분위수 출력(issue #110)은 기본 앙상블만 맞췄다
        # (`check_basic_ensemble_only`). 여기서 멈춘다(arena에서는 estimator를 만들 때 실패하면
        # 모든 split이 조용히 imputed가 된다). 증류는 teacher의 예측만 쓰므로 막지 않는다
        raise SystemExit("범주 표시·회귀 분위수 출력 모델은 --test-time none에서만 쓴다.")
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
            "max_features_per_member": make.max_features_per_member,
            **estimator_options(args, suite),
            "feature_subsampling": make.feature_subsampling,
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
        if args.subsample_train_rows is not None:
            # 맨 바깥: test-time 방법·증류·기록 모두 고른 행만 본다
            estimator = SubsampledEstimator(
                estimator,
                args.subsample_train_rows,
                stratify=suite == "classification",
                seed=args.seed,
            )
        return estimator

    # arena system이 fit에 split의 group 번호(groups)를 넘긴다
    make.fit_groups = "greedy" in args.test_time and args.oof_split == "task"
    make.max_one_hot_width = max_features_of(model)  # expands_one_hot이 같은 상한을 쓰도록
    make.member_kinds = member_kinds
    make.n_estimators = n_estimators
    # 이 task의 멤버당 열 상한과 나누기 방법 (`model_feature_cap`. TabPFN-3.5는 768·balanced)
    make.max_features_per_member, make.feature_subsampling = model_feature_cap(args, inference)
    make.model, make.borders = model, borders  # TabArena·BeyondArena의 크기 상한(arena_limits)
    return make


def model_feature_cap(args: argparse.Namespace, inference: dict) -> tuple[int | None, str]:
    """한 task의 (멤버당 열 상한, 나누기 방법). `--max-features-per-member`를 주면 그것(0은 끔:
    모든 열)이고, 주지 않으면 모델의 추론 설정(recipe의 `default_inference`, TabPFN-3.5는
    768·balanced)이다.
    나누기 방법은 `--feature-subsampling`을 따로 줬으면 그것이다. 분류·회귀 모델마다 따로 정하므로
    v3.5와 다른 모델을 같이 평가해도 각자의 값을 쓴다. test-time 방법(greedy·DiagScale)은 멤버
    view가 나누기 방법을 넘기지 않으므로 모델 값을 쓰지 않는다(예전처럼 모든 열)."""
    cap, subsampling = args.max_features_per_member, args.feature_subsampling
    if cap == 0:
        return None, subsampling
    if cap is not None or args.test_time != "none":
        return cap, subsampling
    model_cap = inference.get("max_features_per_member")
    if model_cap is None:
        return None, subsampling
    if not getattr(args, "feature_subsampling_given", False):
        subsampling = inference.get("feature_subsampling", "random")
    return model_cap, subsampling


def feature_caps(args: argparse.Namespace, factories: dict) -> dict:
    """task마다 쓰는 멤버당 열 상한 (`model_feature_cap`. factory가 아닌 것은 `args`의 값)."""
    return {
        task: getattr(f, "max_features_per_member", args.max_features_per_member or None)
        for task, f in factories.items()
    }


def _one_or_by_task(values: dict):
    # task마다 같으면 값 하나, 다르면 task별 dict (fingerprint·MLflow param)
    distinct = set(values.values())
    return distinct.pop() if len(distinct) == 1 else dict(sorted(values.items()))


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


def arena_limits(models: dict, args: argparse.Namespace, caps: dict | None = None) -> dict:
    """문제 종류마다 돌릴 split의 상한. 체크포인트가 없는 쪽(분류·회귀)은 빠진다(imputed).
    크기는 모델 구조의 논문 권장값(`minipfn.recipes.supported_size`: recipe의 `SUPPORTED_SIZE`,
    모르는 구조는 그 recipe의 첫 구조), 클래스 수는 모델이 받는 수(`max_classes_of`: 출력 수,
    column_row_transformer는 정답 embedding을 묶어 `MAX_CLASSES`개까지). 멤버당 열 상한(`caps`:
    task마다, 주지 않으면 `--max-features-per-member`)이 있는 task는 feature 수 상한이 없다."""
    from minipfn.benchmarks.arena import ModelLimits

    limits = {}
    for task, (model, _) in models.items():
        size = supported_size(model, task)
        if args.max_train_rows is not None:
            size["max_train_rows"] = args.max_train_rows
        if args.max_features is not None:
            size["max_features"] = args.max_features
        elif (caps or {}).get(task, getattr(args, "max_features_per_member", None) or None):
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
    caps = feature_caps(args, factories)
    subsampling = {
        t: getattr(f, "feature_subsampling", args.feature_subsampling) for t, f in factories.items()
    }
    subsampling_changed = any(v != args.feature_subsampling for v in subsampling.values())
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
            # 상한이 없으면(모든 열) 예전 fingerprint 그대로. task마다 다르면 task별로
            **(
                {"max_features_per_member": _one_or_by_task(caps)}
                if any(v is not None for v in caps.values())
                else {}
            ),
            # 공식 3.5 추론 옵션: 기본값과 다른 것만 (모두 기본이면 예전 fingerprint 그대로)
            **inference_option_params(args),
            # 모델이 정한 나누기 방법 (`model_feature_cap`): 예전과 다를 때만
            **(
                {"feature_subsampling": _one_or_by_task(subsampling)} if subsampling_changed else {}
            ),
            **({"drop_group_columns": True} if getattr(args, "drop_group_columns", False) else {}),
            # 주지 않으면(기본) 예전 fingerprint 그대로
            **(
                {"subsample_train_rows": args.subsample_train_rows}
                if getattr(args, "subsample_train_rows", None) is not None
                else {}
            ),
        },
        sort_keys=True,
    )


def arena_estimator_factory(args: argparse.Namespace, paths: dict):
    """TabArena·BeyondArena system이 받는 `make(problem_type, categorical)`와 task별 factory.
    estimator는 AMLB 평가와 같은 `make_estimator_factory`로 만든다(문제 종류와 범주형 열만 바꿔
    넘긴다)."""
    import types

    factories = {
        task: make_estimator_factory(task, args) for task, path in paths.items() if path is not None
    }
    if not factories:
        raise SystemExit(f"{args.suite}에는 --classifier-path나 --regressor-path가 필요하다.")
    one_hot = headline_mode(args.one_hot)

    def make(problem_type: str, categorical: list[int]):
        task = "regression" if problem_type == "regression" else "classification"
        return factories[task](types.SimpleNamespace(categorical_indices=categorical), one_hot)

    # arena system은 이 함수를 받으므로 split의 group 번호를 넘길지도 여기에 단다
    make.fit_groups = any(getattr(f, "fit_groups", False) for f in factories.values())
    # 진단: grouped split의 group 열을 빼고 fit·예측 (`--drop-group-columns`)
    make.drop_group_columns = getattr(args, "drop_group_columns", False)
    make.one_hot = one_hot
    make.kinds = {task: f.member_kinds for task, f in factories.items()}
    return make, factories


def bag_worker_make(argv: list[str]):
    """`--bag-gpus`의 worker가 부른다(`minipfn.benchmarks.bag_pool`): 부모와 같은 인자로 만든
    `make(problem_type, categorical)`. worker는 GPU 하나만 본다(`CUDA_VISIBLE_DEVICES`)."""
    args = parse_args(argv)
    paths = {"classification": args.classifier_path, "regression": args.regressor_path}
    return arena_estimator_factory(args, paths)[0]


def bag_worker_command(argv: list[str], worker_dir: Path) -> list[str]:
    """bag worker의 인자: 원래 인자에서 GPU를 나누는 인자와 MLflow 기록을 빼고 자기 out-dir을
    더한다."""
    base = _strip_flags(
        argv, {"--bag-gpus": 1, "--shards": 1, "--out-dir": 1, "--mlflow": 0, "--device": 1}
    )
    return [*base, "--out-dir", str(worker_dir), "--device", "cuda"]


def start_bag_pool(args: argparse.Namespace, out_dir: Path):
    """`--bag-gpus`: GPU마다 bag worker 하나 (`minipfn.benchmarks.bag_pool.BagPool`). worker의
    test-time·증류 기록은 `<out-dir>/bag_gpus/worker<i>/`에 쓰고 끝나면 이 실행의 파일 뒤에
    붙인다."""
    from minipfn.benchmarks.bag_pool import BagPool

    n_gpus = torch.cuda.device_count()
    if args.bag_gpus > n_gpus:
        raise SystemExit(f"--bag-gpus {args.bag_gpus}가 GPU 수 {n_gpus}보다 많다.")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    devices = (visible.split(",") if visible else [str(i) for i in range(n_gpus)])[: args.bag_gpus]
    checkpoints = (out_dir / "checkpoints").resolve()
    commands = []
    for i in range(len(devices)):
        worker_dir = out_dir / "bag_gpus" / f"worker{i}"
        worker_dir.mkdir(parents=True, exist_ok=True)
        for name in ("test_time.jsonl", f"distill_{args.suite}.jsonl"):
            (worker_dir / name).unlink(missing_ok=True)  # 지난 실행의 기록이 섞이지 않도록
        link = worker_dir / "checkpoints"  # 부모가 받아 둔 checkpoint(runs:/)를 다시 받지 않는다
        if checkpoints.is_dir() and not link.exists():
            link.symlink_to(checkpoints, target_is_directory=True)
        commands.append(bag_worker_command(args.argv, worker_dir))
    threads = max(1, (os.cpu_count() or 1) // len(devices))
    print(f"bag gpus: {len(devices)} workers (GPU {','.join(devices)})", flush=True)
    return BagPool(
        str(Path(__file__).resolve()), "bag_worker_make", commands, devices, threads=threads
    )


def merge_worker_logs(args: argparse.Namespace, out_dir: Path, worker_dirs: list[Path]) -> None:
    """이 실행의 worker들(`worker_dirs`)의 test-time·증류 기록(jsonl)을 이 실행의 파일 뒤에
    붙인다. 폴더를 glob하지 않는다: 지난 실행에서 worker가 더 많았으면 남은 폴더의 기록이 섞인다."""
    for worker_dir in worker_dirs:
        for name in ("test_time.jsonl", f"distill_{args.suite}.jsonl"):
            part = worker_dir / name
            if part.exists():
                with open(out_dir / name, "a") as f:
                    f.write(part.read_text())


def run_arena(args: argparse.Namespace, out_dir: Path, use_mlflow: bool) -> dict:
    """TabArena-Lite·BeyondArena를 돌리고 점수(`summary.json`의 내용)를 돌려준다. estimator는
    AMLB 평가와 같은 `make_estimator_factory`로 만든다(문제 종류와 범주형 열만 바꿔 넘긴다)."""
    from minipfn.benchmarks import arena

    suite = arena_suite(args.suite)
    paths = {"classification": args.classifier_path, "regression": args.regressor_path}
    make, factories = arena_estimator_factory(args, paths)
    one_hot, kinds = make.one_hot, make.kinds
    print(f"member kinds: { {t: ','.join(k) for t, k in kinds.items()} }, one-hot: {one_hot}")

    fingerprint = arena_fingerprint(args, paths, factories, kinds, one_hot)
    models = {task: (f.model, f.borders) for task, f in factories.items()}
    datasets = None if args.datasets is None else args.datasets.split(",")
    limits = arena_limits(models, args, feature_caps(args, factories))
    args.feature_caps_logged = _one_or_by_task(feature_caps(args, factories))
    common = dict(
        limits=limits, method=args.method_name, bagging=args.bagging, fingerprint=fingerprint
    )
    if args.shard_worker:  # worker: stdin으로 받은 데이터셋을 하나씩 돌리고 결과를 남긴다
        for line in iter(sys.stdin.readline, ""):
            job = json.loads(line)
            print(f"shard worker: {job['dataset']} ({job['splits']} splits)", flush=True)
            # 받았다는 표시: 이것 없이 죽으면 부모가 이 일을 다른 worker에게 다시 준다
            Path(job["out_dir"]).mkdir(parents=True, exist_ok=True)
            Path(job["started"]).touch()
            arena.run(
                suite,
                make,
                out_dir=job["out_dir"],
                datasets=[job["dataset"]],
                results_path=job["results"],
                **common,
            )
        return {}
    results = None
    if args.shards > 1:
        cache_id = arena._run_id(fingerprint, args.method_name, args.bagging)  # 진행률 표시용
        estimate = host_state_estimator(factories, feature_caps(args, factories), one_hot)
        results = run_shards(args, suite, limits, datasets, out_dir, cache_id, estimate)
    pool = start_bag_pool(args, out_dir) if args.bag_gpus else None
    make.bag_pool = pool
    try:
        summary = arena.run(
            suite, make, out_dir=out_dir / args.suite, datasets=datasets, results=results, **common
        )
    finally:
        if pool is not None:
            worker_dirs = [out_dir / "bag_gpus" / f"worker{i}" for i in range(len(pool))]
            pool.close()
            make.bag_pool = None
            merge_worker_logs(args, out_dir, worker_dirs)
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


def shard_worker_command(argv: list[str], worker_dir: Path) -> list[str]:
    """worker의 명령 인자. 원래 인자에서 나누는 데 쓰는 인자와 MLflow 기록을 빼고 자기 out-dir과
    `--shard-worker`를 더한다. 데이터셋은 stdin으로 하나씩 받는다."""
    base = _strip_flags(
        argv,
        {
            "--shards": 1,
            "--datasets": 1,
            "--out-dir": 1,
            "--mlflow": 0,
            "--heartbeat-minutes": 1,
            "--shard-worker": 0,
        },
    )
    return [*base, "--out-dir", str(worker_dir), "--shard-worker"]


def shard_jobs(
    queue: list[tuple[str, int]],
    out_dir: Path,
    suite: str,
    cache_id: str = "",
    host_bytes: dict[str, int] | None = None,
) -> list[dict]:
    """큐의 데이터셋마다 worker에 넘길 일: 데이터셋, split 수, `arena.run`의 out-dir, 결과 파일,
    받았다는 표시 파일, 이 실행의 split 캐시 폴더(`runs/<cache_id>`, 진행률 표시용). `host_bytes`에
    0보다 큰 값이 있는 데이터셋은 그 값(CPU 셀 상태 어림, `host_state_bytes`)을 `host_bytes`로
    단다(`run_queue`의 예산).

    out-dir은 worker가 아니라 데이터셋으로 정한다. 그래서 다시 돌리면 어느 worker가 받아도 끝난
    split(캐시)을 다시 쓴다. 폴더 이름에는 이름의 hash를 붙인다(`a/b`와 `a_b`, 대소문자만 다른
    이름이 한 폴더가 되지 않도록)."""
    import hashlib

    jobs = []
    for dataset, splits in queue:
        digest = hashlib.sha1(dataset.encode()).hexdigest()[:8]
        folder = f"{re.sub(r'[^\w.-]', '_', dataset)}-{digest}"
        dataset_dir = out_dir / "shards" / "datasets" / folder / suite
        jobs.append(
            {
                "dataset": dataset,
                "splits": splits,
                "out_dir": str(dataset_dir),
                "results": str(dataset_dir / "results.pkl"),
                "started": str(dataset_dir / "started"),
                "cache": str(dataset_dir / "runs" / cache_id),
            }
        )
        if (host_bytes or {}).get(dataset):
            jobs[-1]["host_bytes"] = int(host_bytes[dataset])
    return jobs


def _clock(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 3600}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"


def _splits_cached(job: dict) -> int:
    # 진행률 표시용: tabarena가 이 실행의 끝난 split마다 남기는 캐시
    # (`runs/<id>/data/<method>/<task>/<repeat>_<fold>/results.pkl`)
    found = sum(1 for _ in Path(job["cache"]).glob("data/*/*/*/results.pkl*"))
    return min(found, job["splits"])


def run_queue(
    jobs: list[dict],
    spawn,
    n_workers: int,
    *,
    heartbeat_s: float = 300.0,
    poll_s: float = 2.0,
    labels: list[str] | None = None,
    host_budget: int | None = None,
) -> tuple[list[str], dict[int, str]]:
    """worker `n_workers`개(`spawn(i)`가 stdin이 PIPE인 Popen을 돌려준다)에 `jobs`를 앞에서부터
    하나씩 나눠 준다. 일을 받은 worker가 결과 파일(`job["results"]`)을 남기면 다음 일을 준다.
    모든 일이 끝나면 stdin을 닫아 worker를 끝낸다. `heartbeat_s`마다 끝난 split 수와 worker별
    데이터셋·경과 시간을 찍는다(플랫폼이 로그 없는 job을 죽이지 않도록, 사람이 진행률을 보도록).

    worker가 죽으면 남은 일은 나머지 worker가 받는다. 죽은 worker가 받았다는 표시
    (`job["started"]`)를 남기기 전이면 그 일도 다시 준다. 표시를 남긴 뒤면 다시 주지 않는다(같은
    데이터셋이 worker를 하나씩 다 죽이지 않도록). 돌려주는 것: 결과가 없는 데이터셋과 실패한 worker
    {번호: 이유}. 둘 다 비어야 성공이다.

    `host_budget`(바이트)을 주면 CPU 셀 상태를 쓰는 큰 표(`job["host_bytes"]`, `host_state_bytes`)는
    지금 도는 큰 표의 합과 더해 예산 안일 때만 준다. 그 사이 쉬는 worker는 큐에서 다음 일(작은 표,
    예산에 드는 큰 표)을 받는다. 큰 표가 하나도 돌지 않으면 예산보다 커도 준다(혼자 돈다). 큐 순서
    말고는 바뀌지 않는다 (결과는 데이터셋마다 같다)."""
    labels = labels or [f"worker {i}" for i in range(n_workers)]
    pending = list(jobs)
    total = sum(j["splits"] for j in jobs)
    current: dict[int, tuple[dict, float]] = {}  # worker → (일, 받은 시각)
    done: set[str] = set()
    failed: dict[int, str] = {}
    waiting: set[str] = set()  # 예산 때문에 미룬 큰 표 (한 번만 알린다)
    start = last_beat = time.monotonic()

    def host_in_use() -> int:
        return sum(job.get("host_bytes", 0) for job, _ in current.values())

    def next_job() -> int | None:
        # 줄 일의 큐 번호: 앞에서부터, 큰 표는 예산 안일 때만 (docstring)
        if host_budget is None:
            return 0
        in_use = host_in_use()
        for k, job in enumerate(pending):
            need = job.get("host_bytes", 0)
            if not need or not in_use or in_use + need <= host_budget:
                return k
            if job["dataset"] not in waiting:
                waiting.add(job["dataset"])
                print(
                    f"shards: {job['dataset']} waits for host memory (cell state "
                    f"{need / 1e9:.0f} GB, in use {in_use / 1e9:.0f} / budget "
                    f"{host_budget / 1e9:.0f} GB)",
                    flush=True,
                )
        return None

    def assign(i: int, proc, k: int = 0) -> None:
        job = pending.pop(k)
        # 지난 실행의 것을 이 실행의 것으로 보지 않도록
        for stale in (job["results"], job["started"]):
            Path(stale).unlink(missing_ok=True)
        try:
            proc.stdin.write(json.dumps(job) + "\n")
            proc.stdin.flush()
        except OSError:  # 이미 죽었다 (BrokenPipeError 등): 일은 다른 worker에게 돌려준다
            pending.insert(k, job)
            failed.setdefault(i, "stdin closed")
            return
        current[i] = (job, time.monotonic())
        host = f", cell state {job['host_bytes'] / 1e9:.0f} GB" if job.get("host_bytes") else ""
        print(f"shards: {labels[i]} ← {job['dataset']} ({job['splits']} splits{host})", flush=True)

    def close(proc) -> None:
        if not proc.stdin.closed:
            try:
                proc.stdin.close()
            except OSError:
                pass

    def heartbeat(now: float) -> None:
        cached = {i: _splits_cached(job) for i, (job, _) in current.items()}
        finished = sum(j["splits"] for j in jobs if j["dataset"] in done)
        print(
            f"[heartbeat {_clock(now - start)}] splits {finished + sum(cached.values())}/{total} "
            f"(datasets {len(done)}/{len(jobs)}, queue {len(pending)})",
            flush=True,
        )
        if host_budget is not None and any(j.get("host_bytes") for j in jobs):
            print(
                f"  host cell state {host_in_use() / 1e9:.0f} / budget {host_budget / 1e9:.0f} GB",
                flush=True,
            )
        for i in range(n_workers):
            if i in current:
                job, since = current[i]
                elapsed = _clock(now - since)
                state = f"{job['dataset']} splits {cached[i]}/{job['splits']}, {elapsed}"
            else:
                state = "failed" if i in failed else ("idle" if pending else "done")
            print(f"  {labels[i]}: {state}", flush=True)

    procs = [spawn(i) for i in range(n_workers)]
    try:
        while True:
            for i, proc in enumerate(procs):
                if i in current and Path(current[i][0]["results"]).exists():
                    job, since = current.pop(i)
                    done.add(job["dataset"])
                    print(
                        f"shards: {labels[i]} done {job['dataset']} "
                        f"in {_clock(time.monotonic() - since)} "
                        f"({len(done)}/{len(jobs)} datasets)",
                        flush=True,
                    )
                code = proc.poll()
                if code is not None and (code or i in current):
                    failed.setdefault(i, f"exit {code}")
                    if i in current:
                        job, _ = current.pop(i)
                        if Path(job["started"]).exists():
                            failed[i] += f" while running {job['dataset']}"
                        else:  # 받기 전에 죽었다: 다른 worker에게
                            pending.insert(0, job)
                    close(proc)
            # 쉬는 worker(살아 있고 일이 없다)에게 남은 일을 준다
            for i, proc in enumerate(procs):
                if pending and i not in current and i not in failed and proc.poll() is None:
                    k = next_job()
                    if k is None:  # 남은 일이 모두 예산을 기다린다
                        break
                    assign(i, proc, k)
            if not current and (not pending or all(i in failed for i in range(n_workers))):
                for proc in procs:
                    close(proc)
                if all(p.poll() is not None for p in procs):
                    break
            now = time.monotonic()
            if now - last_beat >= heartbeat_s:
                heartbeat(now)
                last_beat = now
            time.sleep(poll_s)
    finally:
        # 부모가 도중에 멈추면(예외, Ctrl-C) worker가 GPU를 잡고 남지 않도록
        for proc in procs:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
    heartbeat(time.monotonic())
    missing = [j["dataset"] for j in jobs if j["dataset"] not in done]
    return missing, failed


# `--host-memory-budget-gb`를 주지 않으면 컨테이너 메모리 한도(cgroup)의 이 비율을 CPU 셀 상태에
# 쓴다. 나머지는 worker들의 다른 메모리(표, 멤버마다 전처리한 학습 행 등)에 남긴다
HOST_MEMORY_FRACTION = 0.5
CGROUP_MEMORY_FILES = (
    "/sys/fs/cgroup/memory.max",  # cgroup v2
    "/sys/fs/cgroup/memory/memory.limit_in_bytes",  # cgroup v1
)


def cgroup_memory_limit(files: tuple[str, ...] = CGROUP_MEMORY_FILES) -> int | None:
    """이 컨테이너의 메모리 한도 (바이트). 읽을 수 없거나 한도가 없으면("max", v1의 아주 큰 값)
    None."""
    for name in files:
        try:
            text = Path(name).read_text().strip()
        except OSError:
            continue
        if text.isdigit() and int(text) < 1 << 60:
            return int(text)
        return None  # "max" 또는 v1의 "한도 없음"
    return None


def host_memory_budget(args: argparse.Namespace, files: tuple[str, ...] = CGROUP_MEMORY_FILES):
    """`run_queue`의 CPU 셀 상태 예산 (바이트, None이면 끔). `--host-memory-budget-gb`(0 이하면 끔),
    주지 않으면 cgroup 한도의 `HOST_MEMORY_FRACTION`(한도를 모르면 끔)."""
    given = getattr(args, "host_memory_budget_gb", None)
    if given is not None:
        return None if given <= 0 else int(given * 1e9)
    limit = cgroup_memory_limit(files)
    return None if limit is None else int(limit * HOST_MEMORY_FRACTION)


def host_state_estimator(factories: dict, caps: dict, one_hot: str = "off"):
    """`(problem_type, 행, feature, 범주 열이 있는지) → 멤버 forward 하나가 CPU에 두는 셀 상태
    바이트` 어림 (위쪽으로). 모델의 `inference_host_state_bytes`(column_row_transformer
    `stage_layout="interleaved"`만 0이 아니다)에 멤버 입력 열 수를 준다: 원래 열(멤버당 상한
    `caps`), 범주 열이 있고 `one_hot` 모드가 펼치면 one-hot 상한(`max_one_hot_width`)까지, quantile
    멤버는 분위수 열과 SVD 성분(최대 열/2)까지, 그리고 샘플 식별자 1. interleaved 모델이 없으면 None
    (큐는 예전 그대로)."""
    configs = [getattr(getattr(f, "model", None), "config", None) for f in factories.values()]
    if not any(getattr(c, "stage_layout", None) == "interleaved" for c in configs):
        return None

    def task_of(problem_type: str) -> str:
        return "regression" if problem_type == "regression" else "classification"

    def estimate(problem_type: str, rows: int, n_features: int, categorical: bool = True) -> int:
        factory = factories.get(task_of(problem_type))
        fn = getattr(getattr(factory, "model", None), "inference_host_state_bytes", None)
        if fn is None:
            return 0
        cap = caps.get(task_of(problem_type))
        columns = n_features if cap is None else min(n_features, cap)
        kinds = tuple(getattr(factory, "member_kinds", ()))
        width = getattr(factory, "max_one_hot_width", None)
        if categorical and width and one_hot_kinds(one_hot, kinds):
            # 펼친 열 수는 원래 열 수와 상한 가운데 큰 것을 넘지 않는다 (`one_hot_sizes`)
            columns = max(columns, width)
        if "quantile" in kinds:
            columns = 2 * columns + columns // 2
        return int(fn(rows, columns + 1))

    return estimate


def host_state_bytes(grid, rows: dict[str, int], estimate) -> dict[str, int]:
    """돌릴 split이 있는 데이터셋마다 CPU 셀 상태 어림 (`estimate`, 0인 것은 뺀다). 행은 split의
    학습 + 테스트 행 가운데 가장 큰 것(`arena.split_rows`), 모르면 학습 행 × 1.25."""
    todo = grid[grid["skip_reason"].isna()]
    out = {}
    for dataset, part in todo.groupby("dataset"):
        n_rows = rows.get(dataset) or int(part["max_train_rows"].max() * 1.25)
        problem_type = part["problem_type"].iloc[0]
        # 범주 열이 있는지 모르면(None, TabArena) 있다고 본다
        categorical = part["has_categorical"].iloc[0] if "has_categorical" in part else None
        categorical = True if categorical is None or pd.isna(categorical) else bool(categorical)
        need = estimate(problem_type, int(n_rows), int(part["n_features"].max()), categorical)
        if need:
            out[dataset] = need
    return out


def run_shards(
    args, suite, limits, datasets, out_dir: Path, cache_id: str = "", host_estimate=None
) -> list:
    """GPU마다 worker 프로세스를 하나씩 띄우고, 데이터셋을 비싼 것부터(`arena.dataset_queue`) 큐에서
    하나씩 나눠 준다(`run_queue`). 하나 끝낸 worker가 바로 다음 것을 받으므로 먼저 끝난 GPU가 놀지
    않는다. worker들의 split 결과를 모아 돌려준다. worker의 출력은 `<out-dir>/shards/shard<i>.log`,
    test-time·증류 기록(jsonl)은 끝나면 이 실행의 파일 뒤에 붙인다. worker가 하나라도 실패하거나
    결과가 없는 데이터셋이 있으면 멈춘다(그 데이터셋이 조용히 imputed가 되지 않도록).

    `host_estimate(problem_type, 행, feature)`(`host_state_estimator`)가 CPU 셀 상태를 쓴다고 하는
    큰 표는 동시에 도는 합이 `--host-memory-budget-gb`(`host_memory_budget`) 안일 때만 준다."""
    import pickle
    import subprocess

    from minipfn.benchmarks import arena

    n_gpus = torch.cuda.device_count()
    if n_gpus and args.shards > n_gpus:
        raise SystemExit(f"--shards {args.shards}가 GPU 수 {n_gpus}보다 많다.")
    # 부모가 이미 GPU를 골라 두었으면(CUDA_VISIBLE_DEVICES) worker i는 그 목록의 i번째를 쓴다
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    devices = visible.split(",") if visible else [str(i) for i in range(max(n_gpus, args.shards))]
    context = arena.make_context(suite)
    grid = arena.task_grid(context, suite, limits)
    if datasets is not None:
        # worker를 띄우기 전에 멈춘다 (점수를 낼 때의 `arena.run`도 같은 이유로 멈춘다)
        unknown = sorted(set(datasets) - set(grid["dataset"]))
        if unknown:
            raise SystemExit(f"{suite.name}에 없는 데이터셋: {unknown}")
        grid = grid[grid["dataset"].isin(datasets)]
    subsample = getattr(args, "subsample_train_rows", None)
    queue = arena.dataset_queue(grid, subsample)
    host_bytes = {}
    if host_estimate is not None and queue:
        host_bytes = host_state_bytes(grid, arena.split_rows(context, subsample), host_estimate)
    jobs = shard_jobs(queue, out_dir, args.suite, cache_id, host_bytes)
    if not jobs:  # 돌릴 split이 없다: 점수만 낸다 (모두 imputed)
        return []
    budget = host_memory_budget(args) if host_bytes else None
    if host_bytes:
        big = sorted(host_bytes.items(), key=lambda t: (-t[1], t[0]))
        limit = "off" if budget is None else f"{budget / 1e9:.0f} GB"
        print(
            f"shards: host cell state budget {limit}; "
            + ", ".join(f"{d} {b / 1e9:.0f} GB" for d, b in big),
            flush=True,
        )
    n_workers = min(args.shards, len(jobs))
    threads = max(1, (os.cpu_count() or 1) // n_workers)
    # 부모가 받아 둔 checkpoint(runs:/)를 worker가 다시 받지 않도록 같은 폴더를 가리킨다
    checkpoints = (out_dir / "checkpoints").resolve()
    worker_dirs = [out_dir / "shards" / f"shard{i}" for i in range(n_workers)]
    logs = []

    def spawn(i: int):
        worker_dir = worker_dirs[i]
        worker_dir.mkdir(parents=True, exist_ok=True)
        # 지난 실행의 기록이 이 실행의 파일에 섞이지 않도록 (증류 기록은 worker의 main이 비운다)
        (worker_dir / "test_time.jsonl").unlink(missing_ok=True)
        link = worker_dir / "checkpoints"
        if checkpoints.is_dir() and not link.exists():
            link.symlink_to(checkpoints, target_is_directory=True)
        env = {k: v for k, v in os.environ.items() if k != "MLFLOW_RUN_ID"}
        env.update(CUDA_VISIBLE_DEVICES=devices[i], OMP_NUM_THREADS=str(threads))
        log = open(out_dir / "shards" / f"shard{i}.log", "w")  # noqa: SIM115 (끝날 때까지 연다)
        logs.append(log)
        cmd = shard_worker_command(args.argv, worker_dir)
        return subprocess.Popen(
            [sys.executable, __file__, *cmd],
            env=env,
            stdin=subprocess.PIPE,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )

    total = sum(j["splits"] for j in jobs)
    print(f"shards: {len(jobs)} datasets ({total} splits) → {n_workers} workers", flush=True)
    labels = [f"shard {i} (GPU {devices[i]})" for i in range(n_workers)]
    try:
        missing, failed = run_queue(
            jobs,
            spawn,
            n_workers,
            heartbeat_s=args.heartbeat_minutes * 60,
            labels=labels,
            host_budget=budget,
        )
    finally:
        for log in logs:
            log.close()
    if failed or missing:
        tails = {
            i: (out_dir / "shards" / f"shard{i}.log").read_text().splitlines()[-20:] for i in failed
        }
        raise SystemExit(
            f"shard {sorted(failed)}가 실패했다 ({failed}), 결과가 없는 데이터셋 {missing}: {tails}"
        )
    results = []
    # 결과는 데이터셋 이름 순으로 모은다 (어느 worker가 받았는지와 상관없이 같은 순서)
    for job in sorted(jobs, key=lambda j: j["dataset"]):
        with open(job["results"], "rb") as f:
            results.extend(pickle.load(f))
    merge_worker_logs(args, out_dir, worker_dirs)
    print(f"shards: {len(results)} split results from {n_workers} workers", flush=True)
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
            "max_features_per_member": getattr(
                args, "feature_caps_logged", args.max_features_per_member
            ),
            **test_time_params(args),
            **distill_params(args),
            **{f"member_kinds_{t}": ",".join(k) for t, k in kinds.items()},
            **inference_option_params(args),
            "drop_group_columns": getattr(args, "drop_group_columns", False),
            "subsample_train_rows": args.subsample_train_rows,
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
    if args.bag_gpus < 0:
        raise SystemExit("--bag-gpus는 0 이상이어야 한다.")
    if args.bag_gpus:
        if args.suite not in ARENA_SUITES or args.bagging < 2:
            raise SystemExit("--bag-gpus는 TabArena·BeyondArena의 --bagging 2 이상에서만 쓴다.")
        if args.shards > 1:
            raise SystemExit("--bag-gpus는 --shards와 같이 쓰지 않는다 (둘 다 GPU마다 worker).")
        if args.latency:
            # 한 행 예측 시간은 이 프로세스의 estimator에서 잰다 (bag은 worker에 있다)
            raise SystemExit("--latency는 --bag-gpus와 같이 쓸 수 없다.")
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
    # 모델을 먼저 읽는다: 멤버당 열 상한의 기본값을 모델이 정한다 (`model_feature_cap`)
    factories = {suite: make_estimator_factory(suite, args) for suite in {s for s, _ in tasks}}

    if use_mlflow:
        mlflow.log_params(
            {
                "suite": args.suite,
                "folds": args.folds,
                "one_hot": headline,
                "one_hot_modes": ",".join(args.one_hot),
                "seed": args.seed,
                "max_features_per_member": _one_or_by_task(feature_caps(args, factories)),
                "subsample_train_rows": args.subsample_train_rows,
                **test_time_params(args),
                **distill_params(args),
                **inference_option_params(args),
            }
        )
        paths = {"classifier_path": args.classifier_path, "regressor_path": args.regressor_path}
        mlflow.log_params({k: v for k, v in paths.items() if v is not None})

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
        # 열 단계 옵션(--infer-categorical·--drop-constant)은 범주형 열과 열 수를 바꾸므로
        # expands_one_hot으로 알 수 없다: 범주형 열을 펼치는 멤버 종류로만 묶는다
        expands = (
            args.infer_categorical
            or args.drop_constant
            or expands_one_hot(
                task,
                factories[suite].max_one_hot_width,
                args.folds,
                subsample_rows=args.subsample_train_rows,
                seed=args.seed,
            )
        )
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
