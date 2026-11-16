"""GPU끼리 표 크기 나누기 (학습 옵션 `TrainConfig.balance_tables`, issue #91. 논문에 없음, 여기서
정함).

끄면(기본) GPU마다 표 크기를 따로 뽑는다. 한 step의 표 계산 시간이 행 수의 제곱에 가까운 긴 표
학습(curriculum stage 4: 400~200,000행)에서는, 큰 표를 여럿 받은 GPU를 나머지가 DDP all-reduce에서
기다린다(32 GPU 측정에서 forward 최대/평균 2.5배).

켜면 한 step에 쓸 표 크기 world_size × (GPU 하나가 받는 batch 수)개를 모든 GPU가 같은 seed로 함께
뽑고(`step_shapes`), 비싼 표부터 지금까지 가장 덜 찬 GPU에 하나씩 준다(LPT, `assign`). GPU마다 받는
표 수는 같다. 표 내용은 지금처럼 GPU마다 자기 prior가 만든다.

학습 데이터 분포는 그대로다: 한 step에 쓰는 표 크기는 끈 것과 같이 prior의 크기 규칙
(`sample_shape`)에서 서로 독립으로 뽑은 world_size × batch 수개이고, 바뀌는 것은 그 표를 어느 GPU가
계산하는지뿐이다. 모든 표가 gradient 평균에 같은 무게로 들어가므로(GPU마다 표 수가 같다) step의
gradient 분포도 같다. task 순서(step마다 번갈아)는 학습 루프가 정하고 여기서 건드리지 않는다.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Any

import numpy as np

from minipfn.training.data import derive_seed

# 표 하나의 학습 step 시간 모델 (초): 행 ICL attention이 행 수 × 학습 행 수, 나머지는 셀 수에 비례.
# stage 4 크기의 1×H100 측정(activation checkpointing 켬, compile, forward·backward·Muon 갱신:
# 40,000×100 6.0초, 60,000×20 11.7초, 100,000×40 31.4초, 200,000×10·20 122초)에 맞춘 값이다.
# GPU마다 받는 표 수가 같아서 표마다 같은 상수 시간은 나누는 데 상관없어 넣지 않는다. 비율만 쓴다
COST_PER_ATTENTION = 3.77e-9  # 행 수 × 학습 행 수 하나당
COST_PER_CELL = 1.2e-7  # 행 수 × feature 수 하나당


def table_cost(n_train: int, n_features: int, n_eval: int) -> float:
    """표 하나(학습 행 `n_train`, feature `n_features`, 검증 행 `n_eval`)의 학습 step 시간 어림
    (초, `COST_PER_*`). 인자 순서는 prior의 `sample_shape`와 같다."""
    n_rows = n_train + n_eval
    return COST_PER_ATTENTION * n_rows * n_train + COST_PER_CELL * n_rows * n_features


def assign(costs: list[float], world_size: int, per_rank: int) -> list[list[int]]:
    """표 `world_size × per_rank`개를 GPU마다 `per_rank`개씩 나눈다 (LPT: 비싼 표부터 지금까지
    비용 합이 가장 작은, 아직 덜 찬 GPU에). GPU마다 받은 표 번호를 작은 것부터 돌려준다. 같은 비용·
    같은 합이면 번호가 작은 것이 먼저라, 모든 GPU가 같은 입력으로 같은 결과를 얻는다."""
    if len(costs) != world_size * per_rank:
        raise ValueError(
            f"표 수 {len(costs)}가 world_size × per_rank({world_size} × {per_rank})와 다르다"
        )
    order = sorted(range(len(costs)), key=lambda i: (-costs[i], i))
    heap = [(0.0, rank) for rank in range(world_size)]
    members: list[list[int]] = [[] for _ in range(world_size)]
    for i in order:
        load, rank = heapq.heappop(heap)
        members[rank].append(i)
        if len(members[rank]) < per_rank:
            heapq.heappush(heap, (load + costs[i], rank))
    return [sorted(m) for m in members]


@dataclass(frozen=True)
class TableBalance:
    """한 GPU의 data worker가 표 크기를 나눠 받는 데 쓰는 것 (`training.data.PriorBatches`).

    `seed`는 rank가 들어가지 않은 seed(모든 GPU가 같다), `per_rank`는 GPU 하나가 한 step에 받는
    batch 수(grad_accumulation)다. 이 GPU의 batch 순서에서 n번째 batch는 step 묶음 n // per_rank의
    n % per_rank번째 표다(DataLoader가 worker를 돌아가며 받으므로 worker 수와 상관없다)."""

    seed: tuple[int, ...]
    rank: int
    world_size: int
    per_rank: int

    def __post_init__(self) -> None:
        if not 0 <= self.rank < self.world_size or self.per_rank < 1:
            raise ValueError(f"rank·world_size·per_rank가 맞지 않다: {self}")

    def step_shapes(self, prior: Any, group: int) -> list[tuple[int, int, int]]:
        """step 묶음 `group`의 표 크기 전부 (모든 GPU가 같다): prior의 크기 규칙에서 독립으로."""
        rng = np.random.default_rng(derive_seed(*self.seed, group))
        return [tuple(prior.sample_shape(rng)) for _ in range(self.world_size * self.per_rank)]

    def rank_shapes(self, prior: Any, group: int) -> list[tuple[int, int, int]]:
        """step 묶음 `group`에서 이 GPU가 받는 표 크기 `per_rank`개 (`assign`)."""
        shapes = self.step_shapes(prior, group)
        members = assign([table_cost(*s) for s in shapes], self.world_size, self.per_rank)
        return [shapes[i] for i in members[self.rank]]
