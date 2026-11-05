"""BeyondArena (데이터셋 142개, 분류 98·회귀 44; arXiv 2606.30410 "Beyond IID").

TabArena는 IID split이 대부분이라 foundation model이 약한 경우가 잘 안 보인다. BeyondArena는 split
방식(IID·temporal·grouped), 크기(tiny~large), feature 종류(low-dim·high-dim·text·high-cardinality)가
다양한 데이터셋을 모았다. 데이터는 Hugging Face dataset `TabArena/BeyondArena`(큐레이션 도구
`TabArena/data-foundry`)이고 `tabarena` 패키지가 받는다.

기본 split 범위는 BeyondArena 공식 프로토콜 `core2k`(데이터셋마다 비용을 따져 나눈 split
2,000개)다. 공개 leaderboard(Space `TabArena/leaderboard`의 `data_beyondarena/`)도 이 범위다.
`lite`(데이터셋마다 첫 split)는 빠른 확인용이다.

subset별 점수(`SUITE.breakdown`): split 방식(random·temporal·grouped), 크기, feature 종류.
subset마다 Elo를 따로 낸다(공개 leaderboard처럼 subset 하나가 평가 하나다).
"""

from __future__ import annotations

from minipfn.benchmarks.arena import Suite

_PUBLIC = (
    "data_beyondarena/subsets/split_{split}/size_{size}/features_{features}/tasks_all/"
    "website_leaderboard.csv"
)


def _public(split: str = "all", size: str = "all", features: str = "all") -> str:
    return _PUBLIC.format(split=split, size=size, features=features)


BREAKDOWN = {
    **{f"split_{s}": ((s,), _public(split=s)) for s in ("random", "temporal", "grouped")},
    **{f"size_{s}": ((s,), _public(size=s)) for s in ("tiny", "small", "medium", "large")},
    **{
        f"features_{f}": ((f,), _public(features=f))
        for f in ("low-dim", "high-dim", "text", "high-cardinality")
    },
}


def suite(splits: str = "core2k") -> Suite:
    """`splits`: "core2k"(공식, 기본) 또는 "lite"(데이터셋마다 첫 split)."""
    base = ("core2k",) if splits == "core2k" else ("core2k", "lite")
    return Suite(
        name="beyondarena" if splits == "core2k" else "beyondarena-lite",
        context="BeyondArenaContext",
        subset=base,
        public_csv=_public(),
        breakdown={k: ((*base, *s), csv) for k, (s, csv) in BREAKDOWN.items()},
    )


SUITE = suite()
