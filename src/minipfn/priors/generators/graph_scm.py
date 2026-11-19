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

`categorical_mask`(issue #110, 범주를 아는 셀 encoder): 데이터셋마다 범주 열 표시 (feature,) bool을
더 낸다(생성기 반환이 (x, y, 표시)). TabICL `GraphSCM.__call__`은 범주 열(x_cat)을 앞에, 수치
열(x_num)을 뒤에 이어 붙인 뒤 열을 섞는데 그 표시를 돌려주지 않는다. 그래서 같은 계산을 같은 난수
순서로 다시 짜서(`_call_with_categorical`) 표시만 더 꺼낸다. 표 x·y는 끈 것과 같다(테스트로
확인). 기본은 끔(예전과 같은 경로)
"""

from __future__ import annotations

import importlib
import sys
import types
from contextlib import contextmanager
from functools import cache, partial
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
# `_call_with_categorical`이 그대로 옮긴 TabICL `GraphSCM.__call__`의 버전 (다른 버전이면 난수
# 순서가 어긋날 수 있어 멈춘다)
CATEGORICAL_TABICL_VERSION = "2.2."
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
        categorical_mask: bool = False,
    ) -> None:
        check_probability("mixture_root_prob", mixture_root_prob)
        self.regression = task == "regression"
        self.categorical_mask = categorical_mask
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

    def _dataset(self, n_rows: int, n_features: int, hp: dict) -> tuple[torch.Tensor, ...]:
        make = self._graph_scm(
            regression=self.regression,
            seq_len=n_rows,
            num_features=n_features,
            max_features=n_features,
            num_classes=None if self.regression else hp["num_classes"],
            config=self._config,
        )
        if self.categorical_mask:  # (x, y, 범주 열 표시) (모듈 docstring)
            make = partial(_call_with_categorical, make)
        # 확률을 켰을 때만 난수를 쓴다(끄면 예전과 같은 데이터셋). filter로 다시 만들어도 같은 선택
        if self.mixture_root_prob and np.random.random() < self.mixture_root_prob:
            plain = make

            def make():
                with mixture_roots():
                    return plain()

        return filtered(make, self, regression=self.regression)


def stack_datasets(
    dataset, n_rows: int, n_features: int, n_datasets: int, hp: dict
) -> tuple[torch.Tensor, ...]:
    """`dataset(n_rows, n_features, hp)`로 하나씩 만든 데이터셋 n_datasets개를 그룹으로 쌓는다:
    x (행, 데이터셋, feature), y (행, 데이터셋). graph·tree SCM 생성기가 같이 쓴다. 데이터셋이 범주
    열 표시 (feature,)도 내면(`categorical_mask`) 셋째로 (데이터셋, feature)를 더 낸다."""
    outs = [dataset(n_rows, n_features, hp) for _ in range(n_datasets)]
    x, y = torch.stack([o[0] for o in outs], 1), torch.stack([o[1] for o in outs], 1)
    if len(outs[0]) > 2:
        return x, y, torch.stack([o[2] for o in outs])
    return x, y


def check_probability(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name}는 0 이상 1 이하여야 한다: {value}")


def filtered(
    make, generator, *, regression: bool, attempts: int | None = None
) -> tuple[torch.Tensor, ...]:
    """`make()`로 데이터셋 하나를 만든다. `generator.filter_unpredictable`이면 예측할 수 있을
    때까지 다시 만든다(`attempts`개, 기본 `FILTER_ATTEMPTS`개를 다 거르면 그 다음 것을 받는다).
    NaN으로 버린 데이터셋(y가 모두 -100)은 거르지 않고 내보내 DropInvalid가 다시 뽑게 한다.
    `generator`는 `filter_unpredictable`·`_should_filter`·`_config`가 있는 생성기다(graph·tree
    SCM). `make()`가 (x, y, 범주 열 표시)를 내면 그대로 넘긴다."""
    out = make()
    if not generator.filter_unpredictable:
        return out
    for _ in range(FILTER_ATTEMPTS if attempts is None else attempts):
        x, y = out[0], out[1]
        if (y == -100).all() or not generator._should_filter(
            x, y, generator._config, is_classif=not regression
        ):
            break
        out = make()
    return out


def _call_with_categorical(scm) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """TabICL `GraphSCM.__call__`(tabicl 2.2, BSD-3-Clause)과 같은 계산을 같은 순서로 하고 범주 열
    표시 (feature,) bool을 더 낸다 (모듈 docstring). 그 함수가 쓰는 TabICL 부품을 그대로 부른다.
    범주 열은 x_cat 쪽 열이고(TabICL이 범주 크기가 2 이상인 열을 x_cat으로 모은다), 열 섞기·0
    패딩을 표시에도 똑같이 한다."""
    import importlib.metadata

    import torch.nn.functional as F
    from tabicl.prior._reg2cls import outlier_removing, standard_scaling

    version = importlib.metadata.version("tabicl")
    if not version.startswith(CATEGORICAL_TABICL_VERSION):
        raise RuntimeError(
            f"categorical_mask는 tabicl {CATEGORICAL_TABICL_VERSION}x의 GraphSCM을 옮긴 것이다"
            f" (설치된 것 {version}): `_call_with_categorical`을 그 버전과 맞춘다"
        )
    from tabicl.prior.graph_lib._base import Context, DatasetProperties
    from tabicl.prior.graph_lib._dataset import RandomDataset
    from tabicl.prior.graph_lib._properties import sample_categorical_sizes

    context = Context(config=scm.config, device=scm.device)
    properties = DatasetProperties(
        n_train=scm.seq_len,
        n_test=0,
        cat_sizes={
            "x": sample_categorical_sizes(scm.num_features, context, max_cat_size=200),
            "y": [0 if scm.regression else scm.num_classes],
        },
    )
    data = RandomDataset(context).sample(properties).get_concat_tensors()
    x_cat, x_num = data.get("x_cat", None), data.get("x_num", None)
    parts = [t for t in (x_cat, x_num) if t is not None]
    if not parts:
        raise ValueError("No features found in dataset")
    X = torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]
    n_cat = 0 if x_cat is None else x_cat.shape[-1]
    is_cat = torch.arange(X.shape[-1], device=X.device) < n_cat

    X = outlier_removing(X.float(), threshold=4)
    X = standard_scaling(X)
    if scm.permute_features:
        feat_perm = torch.randperm(scm.num_features, device=scm.device)
        X = X[..., feat_perm]
        is_cat = is_cat[feat_perm]
    if scm.num_features < scm.max_features:
        pad = scm.max_features - scm.num_features
        X = F.pad(X, (0, pad), mode="constant", value=0.0)
        is_cat = F.pad(is_cat, (0, pad), value=False)
    if scm.regression:
        y = data["y_num"]
        y = outlier_removing(y.float(), threshold=4)
        y = standard_scaling(y)
        y = y.view(-1)
    else:
        y = data["y_cat"].view(-1)
        if scm.permute_labels:
            class_perm = torch.randperm(scm.num_classes, device=scm.device)
            y = class_perm[y.long()]
    if torch.any(torch.isnan(X)) or torch.any(torch.isnan(y)):
        X = torch.zeros_like(X)
        y = torch.full_like(y, -100.0)
    return X, y, is_cat


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
