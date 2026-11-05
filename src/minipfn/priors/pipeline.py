"""prior 조립기: 하이퍼파라미터 → 생성기 → 후처리 → `PriorBatch`.

표 데이터 prior 논문들은 같은 단계를 밟는다. 표 크기를 정하고, 그룹(하이퍼파라미터를 공유하는
데이터셋 묶음)마다 하이퍼파라미터를 뽑고, 생성기로 연속값 표를 만들고, 후처리로 다듬는다. 새 논문은
대개 생성기 하나(`minipfn.priors.generators`)나 후처리 몇 개(`minipfn.priors.transforms`)만 더하고
나머지 단계는 있는 부품을 다시 쓴다.

- 표 크기(학습 행·전체 행·feature 수)는 조립한 prior가 정해 넘긴다(`minipfn.priors.table_size` 등).
- 하이퍼파라미터: `(rng, n_train, n_features) -> dict`. 그룹마다 한 번 뽑는다. 쓸 feature 수
  `num_features_used`를 정하면 생성기는 그 수로 만들고, 후처리(`PadFeatures`)가 채운다.
- 생성기(`Generator`): 그룹 하나의 (x, y).
- 후처리(`Transform`): 그룹을 받아 새 그룹을 낸다. None이면 그 그룹을 버리고 하이퍼파라미터부터
  다시 뽑는다(최대 `max_attempts`번).

그룹 안의 텐서는 (행, 데이터셋, feature)·(행, 데이터셋) 순서이고(TabPFN v1·PFN 원 코드의 순서),
`PriorBatch`로 내보낼 때 (데이터셋, 행, feature)로 바꾼다.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch

from minipfn.priors.base import PriorBatch


@dataclass
class Rng:
    """부품이 같이 쓰는 난수 생성기. 전역 난수는 쓰지 않는다(외부 패키지를 감싼 생성기는 예외)."""

    numpy: np.random.Generator
    torch: torch.Generator
    device: torch.device

    @classmethod
    def from_seed(cls, seed: int, device: torch.device | str = "cpu") -> Rng:
        device = torch.device(device)
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)
        return cls(np.random.default_rng(seed), gen, device)


@dataclass
class Group:
    x: torch.Tensor  # (행, 데이터셋, feature)
    y: torch.Tensor  # (행, 데이터셋)
    n_train: int  # 앞의 n_train행이 학습 행
    hp: dict  # 그룹이 공유하는 하이퍼파라미터


class Generator(Protocol):
    def __call__(
        self, n_rows: int, n_features: int, n_datasets: int, hp: dict, rng: Rng
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(x (n_rows, n_datasets, n_features), y (n_rows, n_datasets))."""
        ...


class Transform(Protocol):
    def __call__(self, group: Group, rng: Rng) -> Group | None: ...


Hyperparameters = Callable[[Rng, int, int], dict]


class Mixture:
    """그룹마다 생성기 하나를 고른다. `choose(hp)`가 `generators`의 이름을 낸다.

    어느 생성기를 쓸지도 하이퍼파라미터로 뽑는다(그룹마다 한 번, `Pipeline`의 하이퍼파라미터 단계).
    `choose`는 DataLoader worker로 넘길 수 있도록 모듈 수준 함수로 둔다.
    """

    def __init__(self, generators: Mapping[str, Generator], choose: Callable[[dict], str]) -> None:
        self.generators = dict(generators)
        self.choose = choose

    def __call__(
        self, n_rows: int, n_features: int, n_datasets: int, hp: dict, rng: Rng
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.generators[self.choose(hp)](n_rows, n_features, n_datasets, hp, rng)


class Pipeline:
    """부품을 묶은 prior. `sample_group`은 그룹 하나, `sample_batch`는 그룹을 이어 붙인 batch."""

    def __init__(
        self,
        generator: Generator,
        *,
        rng: Rng,
        hyperparameters: Hyperparameters | None = None,
        transforms: Sequence[Transform] = (),
        group_size: int = 1,
        max_attempts: int = 1,
    ) -> None:
        if group_size < 1 or max_attempts < 1:
            raise ValueError(
                f"group_size {group_size}, max_attempts {max_attempts}는 1 이상이어야 한다"
            )
        self.generator = generator
        self.rng = rng
        self.hyperparameters = hyperparameters
        self.transforms = list(transforms)
        self.group_size = group_size
        self.max_attempts = max_attempts

    def sample_group(self, n_rows: int, n_train: int, n_features: int) -> Group:
        for _ in range(self.max_attempts):
            hp = self.hyperparameters(self.rng, n_train, n_features) if self.hyperparameters else {}
            n_used = hp.get("num_features_used", n_features)
            x, y = self.generator(n_rows, n_used, self.group_size, hp, self.rng)
            group: Group | None = Group(x, y, n_train, hp)
            for transform in self.transforms:
                group = transform(group, self.rng)
                if group is None:
                    break
            if group is not None:
                return group
        raise RuntimeError(
            f"{self.max_attempts}번 시도해도 데이터셋을 만들지 못했다 ({n_train}, {n_features})"
        )

    def sample_batch(
        self, batch_size: int, n_rows: int, n_train: int, n_features: int
    ) -> PriorBatch:
        """데이터셋 batch_size개 (group_size의 배수여야 한다)."""
        n_groups = math.ceil(batch_size / self.group_size)
        if n_groups * self.group_size != batch_size:
            raise ValueError(
                f"batch_size {batch_size}가 group_size {self.group_size}의 배수가 아니다"
            )
        groups = [self.sample_group(n_rows, n_train, n_features) for _ in range(n_groups)]
        x = torch.cat([g.x.float() for g in groups], dim=1)
        y = torch.cat([g.y.float() for g in groups], dim=1)
        return PriorBatch(x.transpose(0, 1).contiguous(), y.transpose(0, 1).contiguous(), n_train)
