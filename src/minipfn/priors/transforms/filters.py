"""쓸 수 없는 그룹 거르기."""

from __future__ import annotations

from minipfn.priors.base import IGNORE_INDEX
from minipfn.priors.pipeline import Group, Rng


class DropInvalid:
    """target이 모두 무시 표시(-100)인 데이터셋이 있으면 그룹을 버린다(None, 다시 뽑는다).

    TabICL graph SCM은 NaN이 생긴 데이터셋의 y를 -100으로 채워 낸다.
    """

    def __call__(self, group: Group, rng: Rng) -> Group | None:
        if (group.y == IGNORE_INDEX).all(0).any():
            return None
        return group
