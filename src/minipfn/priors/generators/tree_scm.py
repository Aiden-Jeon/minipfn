"""Tree SCM 생성기: TabICL v1 prior의 tree SCM(soda-inria/tabicl `tabicl.prior._tree_scm`,
BSD-3-Clause)으로 표를 만든다 (issue #92).

feature는 무작위 원인 변수(정규·균등·혼합 분포)이고, target은 그 위에 XGBoost 층 1~2개(무작위
target에 맞춘 트리 앙상블, 깊이·트리 수 4 이하)와 noise를 쌓은 값이다. TabICL v1 `SCMPrior`의
"tree_scm"과 같고, 하이퍼파라미터는 TabICL v1 기본값(`DEFAULT_FIXED_HP`·`DEFAULT_SAMPLED_HP`)을
데이터셋마다 뽑는다. 분류 target은 TabICL `Reg2Cls`가 클래스로 바꾼다(일부 feature는 범주로도
바꾼다). graph SCM의 tree 함수(무작위 leaf 값의 oblivious tree)와 달리, 실제 GBDT처럼 축에
나란한 분할을 데이터에 맞춰 만든다. Mitra(arXiv 2510.21204)는 SCM prior에 트리 계열 prior를 섞는
것이 실데이터 성능에 좋다고 보고했다.

graph SCM과 같은 규칙을 따른다: TabICL은 numpy·torch·`random`의 전역 난수를 쓰고(`Rng`를 쓰지
않는다), y는 분류면 클래스 번호, 회귀면 z-정규화한 값이다. NaN이 생긴 데이터셋은 y를 모두 -100으로
내보낸다(`DropInvalid`가 다시 뽑는다). `filter_unpredictable`이면 graph SCM과 같은 dataset
filter를 쓰되, TabICL `SCMPrior`처럼 처음 `FILTER_ATTEMPTS`번만 거른다(tree SCM 데이터셋은
filter에 자주 걸려 상한이 없으면 데이터셋 하나에 평균 6~11번 다시 만든다. 1,024행 CPU 실측).
xgboost가 필요하다(extra `distill`·`baselines`, 학습 환경의 dependencies).
"""

from __future__ import annotations

from typing import Literal

import torch

from minipfn.priors.generators.graph_scm import _graph_scm_classes, filtered
from minipfn.priors.pipeline import Rng

# dataset filter를 적용하는 시도 수 (TabICL `SCMPrior.generate_dataset`의 `idx <= 5`)
FILTER_ATTEMPTS = 6


class TreeSCM:
    """Tree SCM 생성기 (`Generator`). 분류면 하이퍼파라미터의 `num_classes`를 쓴다.

    전역 시드는 graph SCM 생성기가 고정한다(둘을 섞어 쓸 때 `GraphSCMPrior`가 같이 만든다).
    """

    def __init__(
        self,
        task: Literal["classification", "regression"],
        *,
        filter_unpredictable: bool = False,
    ) -> None:
        try:
            import xgboost
        except Exception as e:  # macOS에서는 libomp가 따로 필요하다
            raise ImportError("tree SCM prior에는 xgboost가 필요하다") from e
        if not hasattr(xgboost, "XGBModel"):  # graph SCM이 넣은 대체 모듈 (`_graph_scm_classes`)
            raise ImportError("tree SCM prior에는 xgboost가 필요하다 (대체 모듈만 있다)")
        _, prior_config, self._should_filter = _graph_scm_classes()
        from tabicl.prior._hp_sampling import HpSamplerList
        from tabicl.prior._prior_config import DEFAULT_FIXED_HP, DEFAULT_SAMPLED_HP

        self.regression = task == "regression"
        self.filter_unpredictable = filter_unpredictable
        self._config = prior_config(
            filter_unpredictable_graphs=filter_unpredictable,
            filter_unpredictable_datasets=filter_unpredictable,
        )
        self._fixed_hp = dict(DEFAULT_FIXED_HP)
        self._sampled_hp = HpSamplerList(DEFAULT_SAMPLED_HP, device="cpu")

    def __call__(
        self, n_rows: int, n_features: int, n_datasets: int, hp: dict, rng: Rng
    ) -> tuple[torch.Tensor, torch.Tensor]:
        xs, ys = [], []
        for _ in range(n_datasets):
            x, y = self._dataset(n_rows, n_features, hp)
            xs.append(x)
            ys.append(y)
        return torch.stack(xs, 1), torch.stack(ys, 1)

    def _params(self, n_rows: int, n_features: int, hp: dict) -> dict:
        # TabICL `SCMPrior.get_batch`가 데이터셋 하나에 넘기는 인자 (그룹 하나 = 데이터셋 하나)
        sampled = {k: v() if callable(v) else v for k, v in self._sampled_hp.sample().items()}
        return {
            "regression": self.regression,
            "device": "cpu",
            **self._fixed_hp,
            **sampled,
            "seq_len": n_rows,
            "max_features": n_features,
            "num_features": n_features,
            "num_classes": None if self.regression else hp["num_classes"],
            "prior_type": "tree_scm",
        }

    def _dataset(self, n_rows: int, n_features: int, hp: dict) -> tuple[torch.Tensor, torch.Tensor]:
        from tabicl.prior._reg2cls import Reg2Cls
        from tabicl.prior._tree_scm import TreeLayer
        from tabicl.prior._tree_scm import TreeSCM as TabICLTreeSCM

        params = self._params(n_rows, n_features, hp)

        def make() -> tuple[torch.Tensor, torch.Tensor]:
            scm = TabICLTreeSCM(**params)
            for layer in scm.modules():
                if isinstance(layer, TreeLayer):
                    # XGBoost 기본(n_jobs=-1)은 코어를 다 쓴다. DataLoader worker가 여럿이라 1개로
                    layer.model.set_params(n_jobs=1)
            x, y = scm()
            if (y == -100).all():
                return torch.zeros(n_rows, n_features), torch.full((n_rows,), -100.0)
            x, y = Reg2Cls(params)(x, y)
            if not (torch.isfinite(x).all() and torch.isfinite(y).all()):
                return torch.zeros(n_rows, n_features), torch.full((n_rows,), -100.0)
            return x, y

        return filtered(make, self, regression=self.regression, attempts=FILTER_ATTEMPTS)
