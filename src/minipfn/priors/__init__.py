"""학습 데이터 (prior 축).

학습 루프(`minipfn.training.loop`)가 쓰는 prior는 `PRIOR_TYPES`에서 이름
(`TrainConfig.prior_type`)으로 고른다. prior 모듈은 다음 두 함수를 둔다.

- `make_prior(config, seed)`: `sample_batch(batch_size)`로 `PriorBatch`를 내는 객체. 데이터셋 크기
  설정은 `config`(`TrainConfig`)에서 읽는다
- `regression_borders(n_buckets, n_datasets, seed, *, n_jobs, max_train_rows, n_eval_rows,
  max_features)`: 회귀 bar distribution 구간 경계

PFN(`gp`, `scm`)·TabPFN v1(`tabpfn_v1`) prior는 각자의 학습 루프가 직접 쓴다.
"""

from __future__ import annotations

import importlib
from types import ModuleType

from minipfn.priors.base import PriorBatch

# 이름 → 모듈 (쓸 때 import한다. tabicl prior는 tabicl 패키지가 필요하다)
PRIOR_TYPES: dict[str, str] = {
    "tabicl": "minipfn.priors.tabicl",
}
# prior_type을 남기지 않은 이전 학습 run은 모두 tabicl prior다
DEFAULT_PRIOR_TYPE = "tabicl"


def prior_module(prior_type: str) -> ModuleType:
    if prior_type not in PRIOR_TYPES:
        raise ValueError(f"prior_type은 {tuple(PRIOR_TYPES)} 가운데 하나여야 한다: {prior_type}")
    return importlib.import_module(PRIOR_TYPES[prior_type])


__all__ = ["DEFAULT_PRIOR_TYPE", "PRIOR_TYPES", "PriorBatch", "prior_module"]
