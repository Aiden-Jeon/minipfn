"""TabPFN v2 학습 데이터: TabICL prior + 논문의 데이터셋 크기 규칙과 결측값.

v2 논문의 prior 코드는 공개되지 않았으므로, 같은 계열의 공개 prior인 TabICLv2 graph prior
(soda-inria/tabicl, BSD-3-Clause)로 데이터셋 하나하나를 만든다. TabICL prior는 무작위 DAG 위의
계산 그래프로 feature와 target을 만들고, 범주형 feature, 이산화, Kumaraswamy 휘기, 회귀 target을
이미 포함한다. 여기에 논문 Methods의 두 가지를 더한다.

1. 데이터셋 크기 ("Training details"): 학습 행은 2,048개까지 균등, 검증 행(정답을 가리는 행)은
   128개, feature 수는 Beta(0.95, 8.0)을 1~160으로 늘린 값, 표 하나의 셀은 75,000개 미만.
   GPU에서 묶어 계산하려고 batch 안의 데이터셋은 같은 크기를 쓴다(크기는 batch마다 새로 뽑는다).
2. 결측값 ("Post-processing"): 일부 데이터셋에서 셀마다 같은 확률로 값을 지운다(MCAR).
   비율은 논문에 없어서 여기서 정했다(`missing_dataset_prob`, `max_missing_rate`).

    prior = TabPFNv2Prior(task="classification", seed=0)
    batch = prior.sample_batch(8)  # x (8, 행, feature), y (8, 행), n_train

TabICL은 numpy·torch의 전역 난수를 쓰고, 같은 시드여도 파이썬 프로세스의 문자열 hash 시드
(`PYTHONHASHSEED`)가 다르면 다른 데이터셋을 만든다. 완전히 같은 데이터가 필요하면 둘 다 고정한다.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

# 논문 Methods "Training details"의 수치
MAX_TRAIN_ROWS = 2048
N_EVAL_ROWS = 128
MAX_FEATURES = 160
FEATURE_BETA = (0.95, 8.0)
MAX_CELLS = 75_000
MAX_CLASSES = 10
MAX_ATTEMPTS = 100  # 데이터셋 하나를 만들 때 다시 뽑는 최대 횟수 (멈추지 않고 오류를 내도록)


def _graph_scm_classes():
    """TabICL의 GraphSCM과 PriorConfig를 불러온다.

    `tabicl.prior` 패키지는 v1 prior(tree SCM)용으로 xgboost를 모듈 맨 위에서 import한다.
    graph prior는 xgboost를 쓰지 않으므로, xgboost를 불러올 수 없으면 쓰는 순간 오류를 내는
    대체 모듈을 넣는다 (macOS에서는 xgboost에 libomp가 따로 필요하다).
    """
    try:
        import xgboost  # noqa: F401
    except Exception:
        stub = types.ModuleType("xgboost")

        class XGBRegressor:
            def __init__(self, *args, **kwargs):
                raise ImportError("tree SCM prior에는 xgboost가 필요하다.")

        stub.XGBRegressor = XGBRegressor
        sys.modules["xgboost"] = stub
    from tabicl.prior._graph_scm import GraphSCM
    from tabicl.prior.graph_lib._config import PriorConfig

    return GraphSCM, PriorConfig


def sample_table_size(rng: np.random.Generator) -> tuple[int, int]:
    """(학습 행 수, feature 수). 셀 수 제한은 검증 행까지 포함해 적용한다."""
    n_train = int(rng.integers(1, MAX_TRAIN_ROWS + 1))
    n_features = 1 + int(round(rng.beta(*FEATURE_BETA) * (MAX_FEATURES - 1)))
    max_train = (MAX_CELLS - 1) // n_features - N_EVAL_ROWS
    return min(n_train, max_train), n_features


@dataclass
class PriorBatch:
    x: torch.Tensor  # (B, R, F), 결측은 NaN
    y: torch.Tensor  # (B, R). 분류: 클래스 번호, 회귀: z-정규화한 값
    n_train: int  # 앞의 n_train행이 학습 행, 나머지가 정답을 가린 행

    @property
    def y_train(self) -> torch.Tensor:
        return self.y[:, : self.n_train]

    @property
    def y_eval(self) -> torch.Tensor:
        return self.y[:, self.n_train :]


class TabPFNv2Prior:
    def __init__(
        self,
        task: Literal["classification", "regression"],
        seed: int = 0,
        *,
        missing_dataset_prob: float = 0.3,
        max_missing_rate: float = 0.3,
        min_train_rows: int = 1,
        max_train_rows: int = MAX_TRAIN_ROWS,
        n_eval_rows: int = N_EVAL_ROWS,
        max_features: int = MAX_FEATURES,
    ) -> None:
        self.task = task
        if task == "classification":
            min_train_rows = max(min_train_rows, 2)  # 클래스가 2개 이상 학습 행에 나오려면
        self.rng = np.random.default_rng(seed)
        self.missing_dataset_prob = missing_dataset_prob
        self.max_missing_rate = max_missing_rate
        self.min_train_rows = min_train_rows
        self.max_train_rows = max_train_rows
        self.n_eval_rows = n_eval_rows
        self.max_features = max_features
        # TabICL은 numpy·torch의 전역 난수를 쓴다
        np.random.seed(seed)
        torch.manual_seed(seed)
        self._graph_scm, prior_config = _graph_scm_classes()
        self._config = prior_config()

    def sample_size(self) -> tuple[int, int]:
        n_train, n_features = sample_table_size(self.rng)
        n_train = int(np.clip(n_train, self.min_train_rows, self.max_train_rows))
        return n_train, min(n_features, self.max_features)

    def _one_dataset(self, n_train: int, n_features: int) -> tuple[torch.Tensor, torch.Tensor]:
        n_rows = n_train + self.n_eval_rows
        regression = self.task == "regression"
        for _ in range(MAX_ATTEMPTS):
            # 분류: 모든 클래스가 학습 행에 나와야 하므로 클래스 수는 학습 행 수 이하
            max_classes = min(MAX_CLASSES, n_train)
            n_classes = None if regression else int(self.rng.integers(2, max_classes + 1))
            x, y = self._graph_scm(
                regression=regression,
                seq_len=n_rows,
                num_features=n_features,
                max_features=n_features,
                num_classes=n_classes,
                config=self._config,
            )()
            if (y == -100).all():
                continue  # TabICL이 NaN이 생긴 데이터셋을 버린 표시
            if regression:
                return x, y
            order = _train_rows_cover_classes(y, n_train, self.rng)
            if order is not None:
                return x[order], y[order]
        raise RuntimeError(
            f"{MAX_ATTEMPTS}번 시도해도 데이터셋을 만들지 못했다 ({n_train}, {n_features})"
        )

    def _add_missing(self, x: torch.Tensor) -> torch.Tensor:
        if self.rng.random() >= self.missing_dataset_prob:
            return x
        rate = self.rng.uniform(0.0, self.max_missing_rate)
        mask = torch.from_numpy(self.rng.random(tuple(x.shape)) < rate)
        return x.masked_fill(mask, float("nan"))

    def sample_batch(self, batch_size: int, size: tuple[int, int] | None = None) -> PriorBatch:
        n_train, n_features = size or self.sample_size()
        n_train = max(n_train, self.min_train_rows)
        xs, ys = [], []
        for _ in range(batch_size):
            x, y = self._one_dataset(n_train, n_features)
            xs.append(self._add_missing(x.float()))
            ys.append(y.float())
        return PriorBatch(torch.stack(xs), torch.stack(ys), n_train)


def _train_rows_cover_classes(
    y: torch.Tensor, n_train: int, rng: np.random.Generator
) -> torch.Tensor | None:
    """정답을 가린 행의 클래스가 모두 학습 행에도 있도록 행 순서를 정한다.

    클래스마다 한 행씩 먼저 학습 행에 넣고, 나머지 학습 행은 무작위로 채운다. 클래스가 2개보다
    적거나 학습 행 수보다 많으면 None(그 데이터셋은 버린다).
    """
    labels = y.long().numpy()
    classes = np.unique(labels)
    if len(classes) < 2 or len(classes) > n_train:
        return None
    first = np.array([rng.choice(np.flatnonzero(labels == c)) for c in classes])
    rest = rng.permutation(np.setdiff1d(np.arange(len(labels)), first))
    n_more = n_train - len(first)
    train = rng.permutation(np.concatenate([first, rest[:n_more]]))
    return torch.from_numpy(np.concatenate([train, rest[n_more:]]))


def _border_targets(args: tuple[int, int, int, int, int]) -> torch.Tensor:
    # 회귀 데이터셋 n개의 고유한 target 값 (결측값은 넣지 않는다)
    n_datasets, seed, max_train_rows, n_eval_rows, max_features = args
    prior = TabPFNv2Prior(
        "regression",
        seed=seed,
        missing_dataset_prob=0.0,
        max_train_rows=max_train_rows,
        n_eval_rows=n_eval_rows,
        max_features=max_features,
    )
    values = []
    for _ in range(n_datasets):
        n_train, n_features = prior.sample_size()
        _, y = prior._one_dataset(n_train, n_features)
        values.append(torch.unique(y.float()))
    return torch.cat(values)


def regression_borders(
    n_buckets: int = 5000,
    n_datasets: int = 2000,
    seed: int = 0,
    *,
    n_jobs: int = 1,
    max_train_rows: int = MAX_TRAIN_ROWS,
    n_eval_rows: int = N_EVAL_ROWS,
    max_features: int = MAX_FEATURES,
) -> torch.Tensor:
    """회귀 출력 구간의 경계 (Methods "Regression details").

    prior에서 데이터셋을 많이 뽑아 z-정규화한 target을 모으고, 그 분포의 1/n_buckets 분위수들을
    경계로 쓴다. 양 끝은 half-normal 꼬리가 맡는다(`FullSupportBarDistribution`).
    같은 값이 많은 데이터셋(이산화한 target, 4σ에서 잘린 값)이 분위수를 끌어당겨 폭이 0에 가까운
    구간이 생기지 않도록, 데이터셋마다 고유한 값만 모은다(여기서 정함).
    """
    # n_jobs개 프로세스로 나눠 만든다 (프로세스마다 시드가 다르다)
    sizes = [n_datasets // n_jobs + (i < n_datasets % n_jobs) for i in range(n_jobs)]
    jobs = [(n, seed + i, max_train_rows, n_eval_rows, max_features) for i, n in enumerate(sizes)]
    if n_jobs == 1:
        targets = _border_targets(jobs[0])
    else:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(n_jobs) as pool:
            targets = torch.cat(list(pool.map(_border_targets, jobs)))
    targets = targets[torch.randperm(len(targets))[: 2**24]]  # torch.quantile의 입력 크기 한도
    borders = torch.quantile(targets, torch.linspace(0, 1, n_buckets + 1))
    # 같은 값이 겹치면 구간 폭이 0이 되므로 아주 조금씩 벌린다 (경계 수는 그대로)
    for i in range(1, len(borders)):
        borders[i] = torch.maximum(borders[i], borders[i - 1] + 1e-6)
    return borders
