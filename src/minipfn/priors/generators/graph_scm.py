"""Graph SCM 생성기: TabICLv2의 graph SCM(soda-inria/tabicl, BSD-3-Clause)으로 표를 만든다.

무작위 DAG 위의 계산 그래프로 feature와 target을 만들고, 범주형 feature, 이산화, Kumaraswamy 휘기,
회귀 target까지 이미 포함한다. 그래서 y는 분류면 클래스 번호, 회귀면 z-정규화한 값이다(클래스로
바꾸는 후처리가 필요 없다). 표 크기 규칙과 결측값을 붙인 prior는 `minipfn.priors.graph_scm`이다.

TabICL은 numpy·torch의 전역 난수를 쓴다(`Rng`를 쓰지 않는다). 그래서 이 생성기를 만들 때 전역
시드를 고정한다. 같은 시드여도 파이썬 프로세스의 문자열 hash 시드(`PYTHONHASHSEED`)가 다르면 다른
데이터셋을 만든다.
"""

from __future__ import annotations

import sys
import types
from typing import Literal

import numpy as np
import torch

from minipfn.priors.pipeline import Rng


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
    from tabicl.prior._graph_scm import GraphSCM as TabICLGraphSCM
    from tabicl.prior.graph_lib._config import PriorConfig

    return TabICLGraphSCM, PriorConfig


class GraphSCM:
    """Graph SCM 생성기 (`Generator`). 분류면 하이퍼파라미터의 `num_classes`를 쓴다.

    데이터셋을 하나씩 만든다. TabICL이 NaN이 생긴 데이터셋을 버리면 y가 모두 -100이다
    (`minipfn.priors.transforms.filters.DropInvalid`가 그 그룹을 다시 뽑게 한다).
    """

    def __init__(self, task: Literal["classification", "regression"], seed: int = 0) -> None:
        self.regression = task == "regression"
        np.random.seed(seed)
        torch.manual_seed(seed)
        self._graph_scm, prior_config = _graph_scm_classes()
        self._config = prior_config()

    def __call__(
        self, n_rows: int, n_features: int, n_datasets: int, hp: dict, rng: Rng
    ) -> tuple[torch.Tensor, torch.Tensor]:
        xs, ys = [], []
        for _ in range(n_datasets):
            x, y = self._graph_scm(
                regression=self.regression,
                seq_len=n_rows,
                num_features=n_features,
                max_features=n_features,
                num_classes=None if self.regression else hp["num_classes"],
                config=self._config,
            )()
            xs.append(x)
            ys.append(y)
        return torch.stack(xs, 1), torch.stack(ys, 1)
