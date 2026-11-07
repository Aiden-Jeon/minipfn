"""학습 속도 비교 (학습 축, issue #46): 목표 검증 손실에 닿는 순수 학습 시간.

nanoTabPFN speedrun(arXiv 2606.03681)은 "정해진 평가 점수에 닿는 시간"을 잰다. 여기서는 같은
생각을 학습 run의 검증 기록에 쓴다. 학습 루프가 검증마다 남기는 `train_seconds`(검증·checkpoint·
기록을 뺀 학습 시간)와 `val_loss`(multitask면 두 task 평균)로, 처음 목표 이하가 된 검증 시점의
학습 시간을 낸다. 검증 사이는 보간하지 않는다(검증 간격만큼 늦게 잡히지만, 같은 `val_every`로
견주면 공평하다).

    target = val_loss_at(baseline, step=5_000)   # 기준 run의 5k step 검증 손실
    seconds = time_to_target(candidate, target)  # 다른 설정이 거기 닿는 학습 시간
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class ValPoint:
    """검증 한 번: step, 그때까지의 순수 학습 시간(초), 검증 손실."""

    step: int
    train_seconds: float
    val_loss: float


def val_loss_at(points: Sequence[ValPoint], step: int) -> float:
    """`step`의 검증 손실 (그 step에 검증이 없으면 ValueError)."""
    for p in points:
        if p.step == step:
            return p.val_loss
    raise ValueError(f"step {step}에 검증 기록이 없다 (있는 step: {[p.step for p in points]})")


def time_to_target(points: Sequence[ValPoint], target: float) -> ValPoint | None:
    """검증 손실이 처음 `target` 이하가 된 검증 (끝까지 못 닿으면 None)."""
    for p in sorted(points, key=lambda p: p.step):
        if p.val_loss <= target:
            return p
    return None
