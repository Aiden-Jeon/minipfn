"""범주형 열 (TabPFN v1 `flexible_categorical.py`): 일부 그룹에서 일부 연속값 열을 범주로 바꾼다."""

from __future__ import annotations

from minipfn.priors.pipeline import Group, Rng
from minipfn.priors.transforms.classes import multiclass_rank


class Categorical:
    """그룹 비율 `feature_p`에서, 열마다 같은 확률 p ~ U(0, 1)로 범주형으로 바꾼다.

    범주 수는 max(round(Gamma(1, 10)), 2)이고, 값을 무작위 경계로 잘라 구간 번호를 쓴다
    (`multiclass_rank`). 번호를 섞지 않고 둘 확률이 `ordered_p`다.
    """

    def __init__(self, feature_p: float, ordered_p: float) -> None:
        self.feature_p = feature_p
        self.ordered_p = ordered_p

    def __call__(self, group: Group, rng: Rng) -> Group:
        r = rng.numpy
        if r.random() < self.feature_p:
            x = group.x
            p = r.random()
            for col in range(x.shape[-1]):
                n_unique = max(round(r.gamma(1, 10)), 2)
                if r.random() < p:
                    x[:, :, col] = multiclass_rank(
                        x[:, :, col], n_unique, self.ordered_p, r, rng.torch
                    )
        return group
