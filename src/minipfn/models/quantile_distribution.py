"""분위수 회귀 출력 — 회귀 예측분포를 고정된 분위수 수준 τ_k의 값 q_k로 나타낸다 (issue #110).

Kumo-Tabular(NVIDIA, 블로그와 공개 소스 NVIDIA/structured-data-models)의 회귀 출력은 분위수
999개(q001~q999)이고 학습 손실은 분위수(pinball) 손실이다. 여기서는 그 동작만 다시 구현했다.

- 수준: τ_k = k / (n + 1), k = 1..n (n = `N_QUANTILES` = 999이면 0.001 ~ 0.999)
- 손실(`pinball_loss`): 수준마다 ρ_τ(y − q) = max(τ (y − q), (τ − 1)(y − q))의 평균. 정답이 결측이면
  0 (bar distribution 손실과 같다)
- 처음 bias(`initial_bias`): 표준정규 분위수 Φ⁻¹(τ_k). 학습 target은 z-정규화한 값이라, 처음
  예측이 표준정규 분포가 된다 (여기서 정함)
- 추론(`quantiles_to_bucket_probs`): 분위수를 정렬하고(Kumo처럼 교차를 없앤다), 분위수 사이에서
  CDF가 선형이라고 보아 bar distribution 구간 경계 위의 구간 확률로 바꾼다. 그래서 bar distribution
  출력을 받는 추론 코드(평균, target 변환 멤버, 분포 평가)를 그대로 쓴다. 맨 아래·맨 위 분위수 밖의
  질량(각 1/(n + 1))은 그 분위수가 든 구간에 둔다 (여기서 정함)
"""

from __future__ import annotations

import torch

N_QUANTILES = 999
# 분위수를 구간 확률로 바꿀 때 한 번에 계산하는 (행 × 경계) 수 상한 (float32 256 MiB)
CHUNK_ELEMENTS = 1 << 26


def quantile_levels(
    n: int = N_QUANTILES, device: torch.device | str | None = None, dtype=torch.float32
) -> torch.Tensor:
    """수준 τ_k = k / (n + 1), k = 1..n."""
    return torch.arange(1, n + 1, device=device, dtype=dtype) / (n + 1)


def initial_bias(n: int = N_QUANTILES) -> torch.Tensor:
    """표준정규 분위수 Φ⁻¹(τ_k) (출력층 bias의 처음 값)."""
    return torch.special.ndtri(quantile_levels(n, dtype=torch.float64)).float()


def pinball_loss(q: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """분위수 손실: q (..., n), y (...) → (...). 수준 n개의 평균이다. y가 결측이면 0."""
    q = q.float()
    y = y.reshape(q.shape[:-1]).to(q.device).float()
    missing = torch.isnan(y)
    diff = torch.where(missing, 0.0, y)[..., None] - q  # (..., n)
    tau = quantile_levels(q.shape[-1], q.device)
    loss = torch.maximum(tau * diff, (tau - 1) * diff).mean(-1)
    return torch.where(missing, torch.zeros_like(loss), loss)


def quantiles_to_bucket_probs(q: torch.Tensor, borders: torch.Tensor) -> torch.Tensor:
    """분위수 q (..., n) → 구간 경계 `borders` (K + 1,) 위의 구간 확률 (..., K) (모듈 docstring).

    분위수를 정렬하고 경계 범위 안으로 자른 뒤, 경계 b에서의 CDF를 F(b) = 0 (b < q_1), 1 (b > q_n),
    그 사이는 (q_k, τ_k) 점들을 잇는 선형 보간으로 둔다. 구간 확률은 이웃 경계의 F 차이다. 행은
    서로 따로라 `CHUNK_ELEMENTS`씩 나눠 계산한다(결과는 같다). float32로 계산한다."""
    shape = q.shape
    q = q.float().reshape(-1, shape[-1])
    borders = borders.float().to(q.device)
    q = q.sort(-1).values.clamp(borders[0], borders[-1])
    n = q.shape[-1]
    tau = quantile_levels(n, q.device)
    step = max(1, CHUNK_ELEMENTS // borders.numel())
    out = []
    for i in range(0, q.shape[0], step):
        qi = q[i : i + step]
        b = borders.expand(qi.shape[0], -1).contiguous()
        # hi: b보다 크거나 같은 첫 분위수 (0이면 b ≤ q_1, n이면 b > q_n)
        hi = torch.searchsorted(qi.contiguous(), b, side="left")
        lo_i, hi_i = (hi - 1).clamp(0, n - 1), hi.clamp(0, n - 1)
        q_lo, q_hi = qi.gather(-1, lo_i), qi.gather(-1, hi_i)
        width = q_hi - q_lo
        frac = torch.where(width > 0, (b - q_lo) / width.clamp_min(1e-30), 1.0).clamp(0, 1)
        cdf = tau[lo_i] + frac * (tau[hi_i] - tau[lo_i])
        cdf = torch.where(hi == 0, 0.0, torch.where(hi == n, 1.0, cdf))
        # 마지막 경계는 맨 위 분위수 위에 있다 (분위수를 경계 안으로 잘랐다): 모든 질량을 넣는다
        cdf[:, -1] = 1.0
        out.append((cdf[:, 1:] - cdf[:, :-1]).clamp_min(0.0))
    return torch.cat(out).reshape(*shape[:-1], borders.numel() - 1)
