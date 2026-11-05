"""TabArena-Lite (TabArena v0.1의 데이터셋 51개, 분류 38·회귀 13, 데이터셋마다 첫 split 하나).

TabArena(Erickson et al. 2025, arXiv 2506.16791)는 OpenML 데이터셋 51개를 데이터셋마다 여러 번
나눈 split으로 평가한다. Lite는 그 가운데 첫 split(repeat 0, fold 0)만 쓴다. 공개 leaderboard
(Hugging Face Space `TabArena/leaderboard`)의 "TabArena-Lite" 표와 같은 split이다. 비교 대상은
imputed 방법을 넣은 표(`imputation_yes`)다: 모델이 못 도는 데이터셋은 RandomForest (default)로
채워 지는 방식이 같아서다.

평가는 `minipfn.benchmarks.arena.run(SUITE, ...)`.
"""

from __future__ import annotations

from minipfn.benchmarks.arena import Suite

_PUBLIC = (
    "data/entrants_models/imputation_yes/splits_lite/tasks_{tasks}/datasets_all/"
    "website_leaderboard.csv"
)

SUITE = Suite(
    name="tabarena-lite",
    context="TabArenaContext",
    subset=("lite",),
    public_csv=_PUBLIC.format(tasks="all"),
    breakdown={
        task: (("lite", task), _PUBLIC.format(tasks=task))
        for task in ("classification", "regression", "binary", "multiclass")
    },
)
