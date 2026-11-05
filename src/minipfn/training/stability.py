"""학습 안정화 (학습 축): QK-clip(attention logit 상한)과 붕괴 감지."""

from __future__ import annotations

import math

import torch
import torch.distributed as dist


def _qk_weights(model) -> list[tuple[str, torch.Tensor]]:
    # (종류, qkv weight). 어느 attention을 볼지는 모델이 정한다 (`qk_clip_weights`)
    return model.qk_clip_weights()


@torch.no_grad()
def qk_logit_bounds(model) -> torch.Tensor:
    """attention마다 head별 logit 상한 σ_max(Wq_hᵀ Wk_h) · E / √d. 모양 (attention 수, head 수).

    attention 순서는 `_qk_weights`와 같다. 층 입력은 affine 없는 LayerNorm 뒤라 토큰 크기가 √E로
    같으므로 |q·k| / √d ≤ σ_max · E / √d이다. 논문 규모 회귀 checkpoint에서 이 값은 실제 최대
    logit과 거의 같았다(678k vs 676k). Wq_hᵀ = Q_q R_q, Wk_hᵀ = Q_k R_k로 QR 분해하면
    σ_max(Wq_hᵀ Wk_h) = σ_max(R_q R_kᵀ)라 E×E 대신 d×d에서 계산한다.
    """
    weights = _qk_weights(model)
    if not weights:
        return torch.zeros(0, model.config.nhead)
    nhead = model.config.nhead
    e = weights[0][1].shape[1]
    d = e // nhead
    w = torch.stack([w for _, w in weights]).double()  # (A, 3E, E)
    q = w[:, :e].reshape(len(weights), nhead, d, e)
    k = w[:, e : 2 * e].reshape(len(weights), nhead, d, e)
    r_q = torch.linalg.qr(q.transpose(-1, -2)).R  # (A, H, d, d)
    r_k = torch.linalg.qr(k.transpose(-1, -2)).R
    sigma = torch.linalg.matrix_norm(r_q @ r_k.transpose(-1, -2), ord=2)
    return (sigma * e / math.sqrt(d)).float()


def _sample_rows(model) -> list[int]:
    # qk_logit_bounds의 행 가운데 sample attention인 것
    return [i for i, (kind, _) in enumerate(_qk_weights(model)) if kind == "sample_attention"]


@torch.no_grad()
def apply_qk_clip(model, tau: float, world_size: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    """logit 상한이 tau를 넘는 head의 Wq_h·Wk_h를 각각 √(tau / 상한)배로 줄인다.

    반환: (줄인 head 수, 줄이기 전 가장 큰 상한). 매 step 부르므로 GPU에 둔 채 돌려준다(값을 읽는
    것은 기록할 때만). GPU마다 계산이 아주 조금 다를 수 있어 배율은 모든 GPU의 최솟값으로
    맞춘다(가중치가 GPU마다 같게 남도록).
    """
    bounds = qk_logit_bounds(model)
    scale = (tau / bounds).clamp(max=1.0).sqrt()
    if world_size > 1:
        dist.all_reduce(scale, op=dist.ReduceOp.MIN)
    nhead = model.config.nhead
    for (_, w), s in zip(_qk_weights(model), scale, strict=True):
        e = w.shape[1]
        head = s.to(w.dtype).repeat_interleave(e // nhead)[:, None]  # (E, 1): 행마다 그 head의 배율
        w[:e].mul_(head)
        w[e : 2 * e].mul_(head)
    return (scale < 1).sum(), bounds.max()


class CollapseGuard:
    """train loss가 갑자기 뛰어 돌아오지 않는 붕괴를 알아챈다.

    논문 규모 run에서 분류·회귀 모두 grad spike 없이도 수백 step 안에 상수 출력으로 무너진 적이
    있다. 같은 시드로 다시 돌리면 무너지지 않아서(GPU 비결정성), 막기보다 알아채고 되돌린다.
    log_every마다 window 평균 loss를 받아, 최근 건강한 window들의 중앙값보다 `threshold` 넘게
    높은 window가 `patience`번 이어지면 붕괴로 본다. 모든 GPU가 같은 평균(all-reduce 뒤)을 보므로
    판단도 같다.
    """

    def __init__(self, threshold: float, patience: int, n_windows: int = 20, min_windows: int = 5):
        self.threshold, self.patience, self.min_windows = threshold, patience, min_windows
        self.healthy: list[float] = []  # 최근 건강한 window 평균
        self.n_windows = n_windows
        self.bad = 0  # 이어진 나쁜 window 수

    def update(self, window_mean: float) -> bool:
        """window 평균을 넣고, 붕괴로 판단하면 True.

        유한한 loss가 하나도 없던 window(inf)는 나쁜 window로 센다.
        """
        bad = not math.isfinite(window_mean)
        if not bad and len(self.healthy) >= self.min_windows:
            ref = sorted(self.healthy)[len(self.healthy) // 2]
            bad = window_mean > ref + self.threshold
        if bad:
            self.bad += 1
            return self.bad >= self.patience
        self.bad = 0
        self.healthy = (self.healthy + [window_mean])[-self.n_windows :]
        return False

    @property
    def suspect(self) -> bool:
        return self.bad > 0
