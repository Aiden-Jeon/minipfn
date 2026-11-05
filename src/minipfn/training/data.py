"""학습 데이터를 DataLoader로 끝없이 내보내는 IterableDataset. `DataLoader(batch_size=None)`로
쓴다.

worker·GPU·이어 학습한 시점마다 서로 다른 데이터셋이 나오도록 seed를 나눠 주는 것은 학습 쪽 일이라
여기에 둔다. prior는 seed 하나를 받아 만들어지기만 하면 된다.

- `PriorBatches`: 공용 학습 루프(`minipfn.training.loop`). prior의 `sample_batch`가 낸 `PriorBatch`
  하나가 forward batch 하나다(학습 행 수는 prior가 정한다)
- `PriorGroups`: 데이터셋 묶음("그룹")을 내는 prior(`minipfn.priors.prior_bag`). batch마다 학습 행
  수(single_eval_pos)를 여기서 뽑는다 (TabPFN v1 recipe의 학습 루프)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info


def derive_seed(*keys: int) -> int:
    """정수 여러 개를 섞어 64비트 seed 하나로 만든다. 곱하고 더하는 방식과 달리 겹치지 않는다."""
    return int(np.random.SeedSequence(keys).generate_state(1, np.uint64)[0])


class PriorBatches(IterableDataset):
    """DataLoader worker마다 다른 시드로 prior batch를 끝없이 만든다.

    `config`는 `make_prior(seed)`와 `micro_batch`가 있는 학습 설정(`TrainConfig`)이다."""

    def __init__(self, config: Any, seed: tuple[int, ...]) -> None:
        self.config = config
        self.seed = seed

    def __iter__(self):
        info = get_worker_info()
        worker = 0 if info is None else info.id
        torch.set_num_threads(1)
        seed = hash((*self.seed, worker)) % (2**31)
        prior = self.config.make_prior(seed)
        while True:
            batch = prior.sample_batch(self.config.micro_batch)
            yield batch.x, batch.y, batch.n_train


class PriorGroups(IterableDataset):
    """forward batch `(x, y, single_eval_pos)`를 끝없이 내보낸다.

    `make_prior(seed=...)`는 `sample_group(single_eval_pos) -> (x, y, hp)`가 있는 prior를 만든다
    (DataLoader worker로 넘기므로 `functools.partial`처럼 pickle할 수 있어야 한다). forward batch
    하나는 그룹 `groups_per_batch`개를 이어 붙인 것이다. batch마다 single_eval_pos를
    Uniform{1, ..., max_eval_pos-1}에서 뽑는다. TabPFN v1은 0부터 뽑지만, 학습 행이 0개면 데이터셋
    전체가 무시되어(-100) gradient가 0이므로 빼도 학습은 같다. 0이 섞이면 그 GPU에서만 attention
    가중치가 쓰이지 않아 DDP의 gradient 동기화가 어긋난다(8×H100 medium 실행에서 NCCL timeout).
    `seed`(정수 또는 정수 tuple, 예: (config seed, 시작 step, rank))와 worker 번호를
    `SeedSequence`로 섞어 seed를 만든다. 그래서 worker·GPU·resume 시점마다 서로 다른 데이터셋이
    나오고, 정수 seed를 직접 쓰는 검증용 prior와도 겹치지 않는다.
    """

    def __init__(
        self,
        make_prior: Callable[..., Any],
        *,
        max_eval_pos: int,
        groups_per_batch: int = 1,
        seed: int | tuple[int, ...] = 0,
    ) -> None:
        super().__init__()
        self.make_prior = make_prior
        self.max_eval_pos = max_eval_pos
        self.groups_per_batch = groups_per_batch
        self.seed = seed

    def __iter__(self):
        info = get_worker_info()
        worker_id = info.id if info is not None else 0
        if info is not None:
            # worker가 많을 때 각자 CPU 스레드를 여러 개 쓰면 학습 프로세스까지 느려진다.
            torch.set_num_threads(1)
        keys = (self.seed,) if isinstance(self.seed, int) else tuple(self.seed)
        prior = self.make_prior(seed=derive_seed(*keys, worker_id, 0))
        rng = np.random.default_rng(derive_seed(*keys, worker_id, 1))  # single_eval_pos용
        while True:
            single_eval_pos = int(rng.integers(1, self.max_eval_pos))
            groups = [prior.sample_group(single_eval_pos) for _ in range(self.groups_per_batch)]
            x = torch.cat([g[0] for g in groups])
            y = torch.cat([g[1] for g in groups])
            yield x, y, single_eval_pos
