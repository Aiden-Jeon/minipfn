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
p ≥ 0.05) 버린다(dataset filter, `tabicl.prior._dataset.should_filter`). TabICLv2처럼 데이터셋마다
처음 `FILTER_ATTEMPTS`번만 걸러 그 뒤에는 받는다. 논문: stage 1에서 분류 약 35%, 회귀 약 25%가
걸러지고 학습 수렴이 좋아진다. 기본은 끔(예전과 같은 데이터셋).
"""

from __future__ import annotations

import sys
import types
from typing import Literal

import numpy as np
import torch

from minipfn.priors.pipeline import Rng

# 데이터셋 하나에서 dataset filter를 적용하는 시도 수 (TabICLv2 `PriorDataset`의 `idx <= 5`)
FILTER_ATTEMPTS = 6


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
    ) -> None:
        self.regression = task == "regression"
        self.filter_unpredictable = filter_unpredictable
        np.random.seed(seed)
        torch.manual_seed(seed)
        self._graph_scm, prior_config, self._should_filter = _graph_scm_classes()
        self._config = prior_config(
            filter_unpredictable_graphs=filter_unpredictable,
            filter_unpredictable_datasets=filter_unpredictable,
        )

    def __call__(
        self, n_rows: int, n_features: int, n_datasets: int, hp: dict, rng: Rng
    ) -> tuple[torch.Tensor, torch.Tensor]:
        xs, ys = [], []
        for _ in range(n_datasets):
            x, y = self._dataset(n_rows, n_features, hp)
            xs.append(x)
            ys.append(y)
        return torch.stack(xs, 1), torch.stack(ys, 1)

    def _dataset(self, n_rows: int, n_features: int, hp: dict) -> tuple[torch.Tensor, torch.Tensor]:
        make = self._graph_scm(
            regression=self.regression,
            seq_len=n_rows,
            num_features=n_features,
            max_features=n_features,
            num_classes=None if self.regression else hp["num_classes"],
            config=self._config,
        )
        x, y = make()
        if not self.filter_unpredictable:
            return x, y
        for _ in range(FILTER_ATTEMPTS - 1):
            # NaN으로 버린 데이터셋(y가 모두 -100)은 DropInvalid가 다시 뽑는다
            if (y == -100).all() or not self._should_filter(
                x, y, self._config, is_classif=not self.regression
            ):
                break
            x, y = make()
        return x, y
