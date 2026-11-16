"""Graph SCM 생성기: TabICLv2의 graph SCM(soda-inria/tabicl, BSD-3-Clause)으로 표를 만든다.

무작위 DAG 위의 계산 그래프로 feature와 target을 만들고, 범주형 feature, 이산화, Kumaraswamy 휘기,
회귀 target까지 이미 포함한다. 그래서 y는 분류면 클래스 번호, 회귀면 z-정규화한 값이다(클래스로
바꾸는 후처리가 필요 없다). 표 크기 규칙과 결측값을 붙인 prior는 `minipfn.priors.graph_scm`이다.

TabICL은 numpy·torch의 전역 난수를 쓴다(`Rng`를 쓰지 않는다). 그래서 이 생성기를 만들 때 전역
시드를 고정한다. 같은 시드여도 파이썬 프로세스의 문자열 hash 시드(`PYTHONHASHSEED`)가 다르면 다른
데이터셋을 만든다.

`filter_unpredictable`(TabICLv2 학습 스크립트의 `--filter_unpredictable_graphs True
--filter_unpredictable_datasets True`): x 노드와 y 노드에 공통 조상이 없는 그래프를 다시 뽑고
(graph filter, 생성기 안), 만든 데이터셋을 ExtraTrees OOB 예측이 평균보다 낫지 않으면(bootstrap
p ≥ 0.05) 버린다(dataset filter, `tabicl.prior._dataset.should_filter`). TabICLv2의 graph prior
(`tabicl.prior._dataset.GraphPrior`)처럼 예측할 수 있는 데이터셋이 나올 때까지 다시 뽑는다(그쪽은
`while True`. 여기서는 멈추지 않는 것을 막으려 `FILTER_ATTEMPTS`번 뒤에는 받는다. mlp·tree prior인
`SCMPrior`는 처음 6번만 거른다). 논문: stage 1에서 분류 약 35%, 회귀 약 25%가
걸러지고 학습 수렴이 좋아진다. 기본은 끔(예전과 같은 데이터셋).

prior 변형 옵션 (issue #92, 기본은 모두 꺼져 예전과 같은 데이터셋):
- `corrected`: TabICL `PriorConfig`의 보정 플래그 셋(`CORRECTED_FLAGS`)을 켠다. TabICLv2 실험은
  이 플래그를 끈 채로 돌았다(TabICL 주석: "wrong version, but it was used in the experiments").
  켜면 숫자 열 warping(Kumaraswamy)이 그래프로 전파되는 값이 아니라 열 출력에 걸리고(치우친
  분포의 열), local 범주의 크기 상한이 9에서 200으로 올라가며, 범주 meta sampling이 데이터셋
  안에서 같은 가중치를 쓴다
- `mixture_root_prob`: 이 확률로 고른 데이터셋은 부모 없는 root 노드의 점을 Gaussian
  mixture(군집 `MIXTURE_MIN_CENTERS`~`MIXTURE_MAX_CENTERS`개)에서 뽑는다(`mixture_roots`). 행이
  군집·그룹을 이루는 표다. TabICL의 `RandomGaussianMixturePoints`는 정의만 있고 root 분포
  후보에는 없다
"""

from __future__ import annotations

import importlib
import sys
import types
from contextlib import contextmanager
from functools import cache
from typing import Literal

import numpy as np
import torch

from minipfn.priors.pipeline import Rng

# 데이터셋 하나에서 dataset filter를 적용하는 시도 수의 상한 (TabICL `GraphPrior`는 상한 없음,
# 멈추지 않는 것만 막는다). PR #74의 첫 두 stage 1 run은 6(`SCMPrior`의 `idx <= 5`)으로 돌았다
FILTER_ATTEMPTS = 1000
# `corrected`가 켜는 TabICL PriorConfig 플래그
CORRECTED_FLAGS = (
    "use_corrected_num_converters",
    "use_corrected_cat_sizes",
    "use_corrected_cat_meta_sampling",
)
# `mixture_root_prob`로 고른 데이터셋의 군집 수 범위 (log 균등, 여기서 정함)
MIXTURE_MIN_CENTERS = 2
MIXTURE_MAX_CENTERS = 32


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
    from tabicl.prior._dataset import should_filter
    from tabicl.prior._graph_scm import GraphSCM as TabICLGraphSCM
    from tabicl.prior.graph_lib._config import PriorConfig

    return TabICLGraphSCM, PriorConfig, should_filter


class GraphSCM:
    """Graph SCM 생성기 (`Generator`). 분류면 하이퍼파라미터의 `num_classes`를 쓴다.

    데이터셋을 하나씩 만든다. TabICL이 NaN이 생긴 데이터셋을 버리면 y가 모두 -100이다
    (`minipfn.priors.transforms.filters.DropInvalid`가 그 그룹을 다시 뽑게 한다).
    """

    def __init__(
        self,
        task: Literal["classification", "regression"],
        seed: int = 0,
        *,
        filter_unpredictable: bool = False,
        corrected: bool = False,
        mixture_root_prob: float = 0.0,
    ) -> None:
        check_probability("mixture_root_prob", mixture_root_prob)
        self.regression = task == "regression"
        self.filter_unpredictable = filter_unpredictable
        self.mixture_root_prob = mixture_root_prob
        np.random.seed(seed)
        torch.manual_seed(seed)
        self._graph_scm, prior_config, self._should_filter = _graph_scm_classes()
        self._config = prior_config(
            filter_unpredictable_graphs=filter_unpredictable,
            filter_unpredictable_datasets=filter_unpredictable,
            **(dict.fromkeys(CORRECTED_FLAGS, True) if corrected else {}),
        )

    def __call__(
        self, n_rows: int, n_features: int, n_datasets: int, hp: dict, rng: Rng
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return stack_datasets(self._dataset, n_rows, n_features, n_datasets, hp)

    def _dataset(self, n_rows: int, n_features: int, hp: dict) -> tuple[torch.Tensor, torch.Tensor]:
        make = self._graph_scm(
            regression=self.regression,
            seq_len=n_rows,
            num_features=n_features,
            max_features=n_features,
            num_classes=None if self.regression else hp["num_classes"],
            config=self._config,
        )
        # 확률을 켰을 때만 난수를 쓴다(끄면 예전과 같은 데이터셋). filter로 다시 만들어도 같은 선택
        if self.mixture_root_prob and np.random.random() < self.mixture_root_prob:
            plain = make

            def make():
                with mixture_roots():
                    return plain()

        return filtered(make, self, regression=self.regression)


def stack_datasets(
    dataset, n_rows: int, n_features: int, n_datasets: int, hp: dict
) -> tuple[torch.Tensor, torch.Tensor]:
    """`dataset(n_rows, n_features, hp)`로 하나씩 만든 데이터셋 n_datasets개를 그룹으로 쌓는다:
    x (행, 데이터셋, feature), y (행, 데이터셋). graph·tree SCM 생성기가 같이 쓴다."""
    xs, ys = [], []
    for _ in range(n_datasets):
        x, y = dataset(n_rows, n_features, hp)
        xs.append(x)
        ys.append(y)
    return torch.stack(xs, 1), torch.stack(ys, 1)


def check_probability(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name}는 0 이상 1 이하여야 한다: {value}")


def filtered(
    make, generator, *, regression: bool, attempts: int | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """`make()`로 데이터셋 하나를 만든다. `generator.filter_unpredictable`이면 예측할 수 있을
    때까지 다시 만든다(`attempts`개, 기본 `FILTER_ATTEMPTS`개를 다 거르면 그 다음 것을 받는다).
    NaN으로 버린 데이터셋(y가 모두 -100)은 거르지 않고 내보내 DropInvalid가 다시 뽑게 한다.
    `generator`는 `filter_unpredictable`·`_should_filter`·`_config`가 있는 생성기다(graph·tree
    SCM)."""
    x, y = make()
    if not generator.filter_unpredictable:
        return x, y
    for _ in range(FILTER_ATTEMPTS if attempts is None else attempts):
        if (y == -100).all() or not generator._should_filter(
            x, y, generator._config, is_classif=not regression
        ):
            break
        x, y = make()
    return x, y


@contextmanager
def mixture_roots():
    """이 안에서 만드는 graph SCM은 root 노드의 점을 Gaussian mixture에서 뽑는다
    (`mixture_root_points_class`). TabICL `RandomNodeFunction`이 쓰는 `RandomPoints` 이름만 잠시
    바꾼다(범주 embedding 등 다른 자리의 `RandomPoints`는 그대로다)."""
    module = importlib.import_module("tabicl.prior.graph_lib._node_function")
    original = module.RandomPoints
    module.RandomPoints = mixture_root_points_class()
    try:
        yield
    finally:
        module.RandomPoints = original


@cache
def mixture_root_points_class():
    """TabICL `RandomPoints`처럼 기본 점 위에 무작위 함수를 씌우되, 기본 점을 Gaussian mixture에서
    뽑는 클래스. 분포는 TabICL `RandomGaussianMixturePoints`와 같고(군집마다 중심 N(0, I), 무작위
    공분산 행렬과 배율, 군집 가중치는 `RandomWeights`), 군집 수만
    [MIXTURE_MIN_CENTERS, MIXTURE_MAX_CENTERS]에서 log 균등으로 뽑는다. TabICL 쪽은 행마다 (n, n)
    행렬을 꺼내는데, 여기서는 군집마다 곱해서 (군집, n, n)만 쓴다. tabicl을 쓸 때만 import하려고
    클래스를 함수 안에서 만든다."""
    from tabicl.prior.graph_lib._function import RandomFunction
    from tabicl.prior.graph_lib._points import RandomPoints
    from tabicl.prior.graph_lib._weights import RandomWeights

    class MixtureRootPoints(RandomPoints):
        def sample(self, n_batch: int, n: int) -> torch.Tensor:
            k = self.sampler.randint(
                "mixture_root_n_centers",
                MIXTURE_MIN_CENTERS,
                MIXTURE_MAX_CENTERS + 1,
                use_log=True,
            )
            weights = RandomWeights(self.context).sample(1, k).squeeze(0)
            member = torch.multinomial(weights, num_samples=n_batch, replacement=True)
            centers = torch.randn(k, n)
            matrices = torch.randn(k, n, n) * RandomWeights(self.context).sample(k, n)[:, None, :]
            matrices = matrices * torch.exp(torch.randn(1) + torch.randn(1) * torch.randn(k, 1, 1))
            noise = torch.randn(n_batch, n)
            points = centers[member]
            for c in range(k):
                rows = member == c
                points[rows] += noise[rows] @ matrices[c].T
            return RandomFunction(self.context, n, n)(points)

    return MixtureRootPoints
