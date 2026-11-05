"""학습 데이터 (prior 축). 모듈은 무엇을 만드는지로 이름을 붙이고, 논문별 설정은
`minipfn.recipes`에 둔다.

표 데이터 prior는 단계별 부품을 `pipeline.Pipeline`으로 묶어 만든다. 새 논문의 prior는 대개 생성기
하나나 후처리 몇 개만 더하고 나머지 부품은 다시 쓴다.

- `base`: 공용 학습 루프에 내는 `PriorBatch`, 손실에서 무시할 target `IGNORE_INDEX`
- `pipeline`: 하이퍼파라미터 → 생성기 → 후처리 → `PriorBatch` 조립기, 생성기 섞기(`Mixture`)
- `table_size`: batch마다 학습 행·feature 수를 뽑는 표 크기 규칙
- `hyperparameters`: meta 분포 2단계 샘플링(TabPFN v1), 클래스 수
- `generators`: 그룹 하나의 연속값 표 (MLP SCM / BNN, GP 표, graph SCM)
- `transforms`: 그룹 다듬기 (범주형 열, 표준화, 클래스, 패딩, 결측값, 거르기)

부품을 묶은 prior:

- `prior_bag`: 그룹마다 GP 또는 MLP 생성기를 고르는 prior bag (TabPFN v1 recipe가 쓴다)
- `graph_scm`: graph SCM 생성기 + 표 크기 규칙 + 결측값 (TabPFN v2 recipe가 쓴다)
- `gp`: 1D GP 회귀 prior (`GPPriorSampler`)와 analytic posterior (PFN recipe가 쓴다. 표 prior가
  아니라 조립기를 쓰지 않는다)

공용 학습 루프(`minipfn.training.loop`)가 쓰는 prior는 `PRIOR_TYPES`에서 이름
(`TrainConfig.prior_type`)으로 고른다. 그 prior 모듈은 다음을 둔다.

- `make_prior(config, seed)`: `sample_batch(batch_size)`로 `PriorBatch`를 내는 객체. 표 크기
  설정은 `config`(`TrainConfig`)에서 읽는다
- `make_borders(config, n_buckets, seed, n_jobs)`: 회귀 bar distribution 구간 경계. 학습 prior와
  같은 표 크기 설정으로 뽑는다
- `check_config(config)`: 표 크기 설정을 그 prior로 만들 수 있는지 (틀리면 ValueError)
- `MAX_CLASSES`: 분류 데이터셋의 클래스 수 상한 (분류 모델의 출력 수)

PFN(`gp`)·TabPFN v1(`prior_bag`) recipe는 각자의 학습 루프에서 prior를 바로 쓴다.
"""

from __future__ import annotations

import importlib
from types import ModuleType

from minipfn.priors.base import PriorBatch

# 이름 → 모듈 (쓸 때 import한다. graph_scm은 tabicl 패키지가 필요하다)
PRIOR_TYPES: dict[str, str] = {
    "graph_scm": "minipfn.priors.graph_scm",
}
# 예전 이름 → 지금 이름. 예전 학습 run의 MLflow param에 남아 있어 읽을 때 받는다
PRIOR_TYPE_ALIASES: dict[str, str] = {
    "tabicl": "graph_scm",
}
# prior_type을 남기지 않은 이전 학습 run은 모두 graph_scm prior다
DEFAULT_PRIOR_TYPE = "graph_scm"


def canonical_prior_type(prior_type: str) -> str:
    """예전 이름이면 지금 이름으로. 없는 이름은 그대로 둔다(`prior_module`이 ValueError를 낸다)."""
    return PRIOR_TYPE_ALIASES.get(prior_type, prior_type)


def prior_module(prior_type: str) -> ModuleType:
    name = canonical_prior_type(prior_type)
    if name not in PRIOR_TYPES:
        raise ValueError(f"prior_type은 {tuple(PRIOR_TYPES)} 가운데 하나여야 한다: {prior_type}")
    return importlib.import_module(PRIOR_TYPES[name])


__all__ = [
    "DEFAULT_PRIOR_TYPE",
    "PRIOR_TYPES",
    "PRIOR_TYPE_ALIASES",
    "PriorBatch",
    "canonical_prior_type",
    "prior_module",
]
