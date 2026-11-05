"""prior-fitting 손실 (학습 축): 분류 cross-entropy, 회귀 bar distribution NLL과 target 흩기."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from minipfn.models.bar_distribution import FullSupportBarDistribution
from minipfn.priors.base import IGNORE_INDEX


def prior_loss(
    logits: torch.Tensor,
    y_eval: torch.Tensor,
    task: str,
    bar: FullSupportBarDistribution | None,
) -> torch.Tensor:
    """검증 행의 평균 손실. 분류: cross-entropy, 회귀: bar distribution NLL."""
    logits = logits.float()
    if task == "classification":
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y_eval.reshape(-1).long())
    return bar(logits, y_eval.float()).mean()


def dequantize_targets(
    y: torch.Tensor, n_train: int, bar: FullSupportBarDistribution, n_buckets: float
) -> torch.Tensor:
    """데이터셋 안에서 반복되는 target 값의 검증 행 정답을 이웃 구간 `n_buckets`개에 흩는다.

    같은 값이 반복되는 이산 target은 늘 같은 구간에 떨어지므로, 그 구간에 확률을 몰수록 NLL이
    끝없이 낮아진다(최적 logit이 무한대). 그러면 출력층이 계속 커지고 gradient가 연속 target의
    수백 배가 되어 학습이 무너진다(논문 규모 회귀 run에서 겪음). 구간 번호 좌표에서 균일 noise를
    더해 정답을 이웃 구간에 퍼뜨리면 최적이 유한해진다. 연속 target, 양 끝 꼬리 구간의 값, 결측은
    그대로 둔다(여기서 정함). 반환은 검증 행 정답 `(batch, n_eval)`이다.
    """
    y_eval = y[:, n_train:]
    # 데이터셋마다 정렬해 이웃과 같은 값이면 반복 (GPU 동기화 없이 batch 전체를 한 번에)
    y_sorted, order = y.sort(dim=1)
    same = y_sorted[:, 1:] == y_sorted[:, :-1]
    dup = torch.zeros_like(y_sorted, dtype=torch.bool)
    dup[:, 1:] |= same
    dup[:, :-1] |= same
    repeated = torch.zeros_like(dup).scatter_(1, order, dup)[:, n_train:]
    borders, widths, last = bar.borders, bar.widths, bar.num_buckets - 1
    nan = torch.isnan(y_eval)
    y_safe = torch.where(nan, borders[1], y_eval)  # 결측은 바꾸지 않지만 index 계산은 유한하게
    idx = bar._bucket_index(y_safe.contiguous())
    interior = (idx >= 1) & (idx <= last - 1) & ~nan
    coord = idx + (y_safe - borders[idx]) / widths[idx]  # 구간 번호 좌표 (k ~ k+1이 구간 k)
    moved = coord + (torch.rand_like(y_eval) - 0.5) * n_buckets
    # 꼬리 구간으로는 보내지 않는다. 잘라내면 경계에 점 질량이 다시 생기므로 경계에서 반사한다
    lo, hi = 1.0, float(last)
    moved = torch.where(moved < lo, 2 * lo - moved, moved)
    moved = torch.where(moved > hi, 2 * hi - moved, moved).clamp(lo, hi)
    j = moved.floor().long().clamp(1, last - 1)
    shifted = borders[j] + (moved - j) * widths[j]
    # float 반올림으로 오른쪽 꼬리 경계(borders[-2])에 닿지 않게 그 바로 아래 값까지만
    shifted = torch.minimum(shifted, torch.nextafter(borders[-2], borders[-3]))
    return torch.where(repeated & interior, shifted, y_eval)


def classification_loss(logits: torch.Tensor, y_test: torch.Tensor) -> torch.Tensor:
    """테스트 행 cross-entropy. 무시할 행(-100)은 0으로 세고 전체 평균을 낸다 (TabPFN v1과 같은
    가중)."""
    losses = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(),
        y_test.reshape(-1).long(),
        ignore_index=IGNORE_INDEX,
        reduction="none",
    )
    return losses.mean()
