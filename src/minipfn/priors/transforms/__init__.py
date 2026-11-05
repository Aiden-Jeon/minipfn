"""후처리: 생성기가 만든 그룹을 다듬는다. 논문마다 조합만 바꿔 다시 쓴다.

후처리는 `minipfn.priors.pipeline.Transform` 모양의 호출 가능한 객체다: `(group, rng) -> group`.
None을 내면 `Pipeline`이 그 그룹을 버리고 다시 뽑는다.

- `categorical.Categorical`: 일부 열을 범주형으로
- `scaling.Standardize`, `scaling.PadFeatures`: 이상치 누르기·표준화, feature 수 보정과 0 패딩
- `classes.ToClasses`, `classes.MatchClasses`, `classes.Relabel`, `classes.CoverClasses`: 연속값
  target을 클래스로, 학습·검증 행의 클래스 맞추기, 라벨 정리
- `missing.Missing`: 결측값 (MCAR)
- `filters.DropInvalid`: target이 모두 무시 표시인 데이터셋이 있는 그룹을 버린다

TabPFN v1 prior bag은 Categorical → Standardize → ToClasses → PadFeatures → MatchClasses → Relabel,
graph SCM prior는 DropInvalid → CoverClasses(분류) → Missing 순서로 쓴다.
"""

from minipfn.priors.transforms.categorical import Categorical
from minipfn.priors.transforms.classes import CoverClasses, MatchClasses, Relabel, ToClasses
from minipfn.priors.transforms.filters import DropInvalid
from minipfn.priors.transforms.missing import Missing
from minipfn.priors.transforms.scaling import PadFeatures, Standardize

__all__ = [
    "Categorical",
    "CoverClasses",
    "DropInvalid",
    "MatchClasses",
    "Missing",
    "PadFeatures",
    "Relabel",
    "Standardize",
    "ToClasses",
]
