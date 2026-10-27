"""Riemann(bar) distribution — 회귀 출력을 구간 분류로 다루는 예측분포.

노트북 03 §6의 아이디어: 출력 범위를 bucket으로 잘라 각 bucket 확률(softmax)을 내고,
bucket 안에서는 밀도가 균일하다고 보아 계단형 밀도 `p_k / width_k`를 만든다. 학습 손실은
정답 y가 든 bucket의 밀도에 대한 NLL이다.

공식 라이브러리 `automl/PFNs`의 `BarDistribution` / `FullSupportBarDistribution` API를
미러링한다. `FullSupportBarDistribution`은 양 끝 bucket을 half-normal tail로 바꿔 bucket
범위 밖의 y에도 유한한 NLL을 준다(학습 기본값).
"""

from __future__ import annotations

import math

import torch
from torch import nn

_SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)


def get_bucket_borders(
    num_buckets: int,
    value_range: tuple[float, float] = (-4.0, 4.0),
) -> torch.Tensor:
    """등간격 bucket 경계 (num_buckets + 1개). §6의 linspace(-4, 4, B+1)."""
    return torch.linspace(value_range[0], value_range[1], num_buckets + 1)


class BarDistribution(nn.Module):
    """등간격 bucket 위의 계단형(piecewise-constant) 예측분포.

    logits (..., num_buckets) → bucket 확률(softmax). 밀도 = 확률 / bucket 폭.
    """

    def __init__(self, borders: torch.Tensor, ignore_nan_targets: bool = True) -> None:
        super().__init__()
        assert borders.dim() == 1 and borders.numel() >= 2
        self.num_buckets = borders.numel() - 1
        self.ignore_nan_targets = ignore_nan_targets
        widths = borders[1:] - borders[:-1]
        centers = 0.5 * (borders[1:] + borders[:-1])
        # bucket별 2차 모멘트 E[y^2 | bucket] = (r^2 + r*l + l^2) / 3 (균일분포)
        left, right = borders[:-1], borders[1:]
        second_moment = (right**2 + right * left + left**2) / 3.0
        self.register_buffer("borders", borders)
        self.register_buffer("widths", widths)
        self.register_buffer("bucket_centers", centers)
        self.register_buffer("bucket_second_moment", second_moment)

    def _bucket_index(self, y: torch.Tensor) -> torch.Tensor:
        return (torch.searchsorted(self.borders, y) - 1).clamp(0, self.num_buckets - 1)

    def forward(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """정답 y에 대한 NLL = -log(density) = -log p_k + log(width_k). 반환 shape = y.shape."""
        y = y.reshape(logits.shape[:-1]).to(logits.device)
        nan_mask = torch.isnan(y)
        y_safe = torch.where(nan_mask, self.borders[0], y)
        idx = self._bucket_index(y_safe)
        log_probs = torch.log_softmax(logits, dim=-1)
        log_p_at = log_probs.gather(-1, idx.unsqueeze(-1)).squeeze(-1)
        nll = -log_p_at + torch.log(self.widths[idx])
        if self.ignore_nan_targets:
            nll = torch.where(nan_mask, torch.zeros_like(nll), nll)
        return nll

    def mean(self, logits: torch.Tensor) -> torch.Tensor:
        """예측분포의 평균. 반환 shape = logits.shape[:-1]."""
        probs = torch.softmax(logits, dim=-1)
        return (probs * self.bucket_centers).sum(-1)

    def variance(self, logits: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=-1)
        mean = (probs * self.bucket_centers).sum(-1)
        second = (probs * self.bucket_second_moment).sum(-1)
        return (second - mean**2).clamp_min(0.0)

    def std(self, logits: torch.Tensor) -> torch.Tensor:
        return self.variance(logits).sqrt()

    def pdf(self, logits: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
        """grid ys에서의 밀도. logits (..., B), ys (G,) → (..., G). bucket 밖은 0."""
        probs = torch.softmax(logits, dim=-1)
        idx = self._bucket_index(ys)
        density = probs[..., idx] / self.widths[idx]
        inside = (ys >= self.borders[0]) & (ys <= self.borders[-1])
        return density * inside

    def quantile(
        self, logits: torch.Tensor, center_prob: float = 0.682, grid_size: int = 2000
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """중심 center_prob 구간의 (하한, 상한). grid 기반 수치 역-cdf."""
        lo, hi = self._grid_range()
        grid = torch.linspace(lo, hi, grid_size, device=logits.device)
        dens = self.pdf(logits, grid)  # (..., G)
        cdf = torch.cumsum(dens * (grid[1] - grid[0]), dim=-1)
        cdf = cdf / cdf[..., -1:].clamp_min(1e-12)
        tail = (1.0 - center_prob) / 2.0
        lower = self._invert_cdf(cdf, grid, tail)
        upper = self._invert_cdf(cdf, grid, 1.0 - tail)
        return lower, upper

    def _grid_range(self) -> tuple[float, float]:
        return float(self.borders[0]), float(self.borders[-1])

    @staticmethod
    def _invert_cdf(cdf: torch.Tensor, grid: torch.Tensor, q: float) -> torch.Tensor:
        idx = torch.searchsorted(cdf, torch.full_like(cdf[..., :1], q)).clamp(0, grid.numel() - 1)
        return grid[idx.squeeze(-1)]


class FullSupportBarDistribution(BarDistribution):
    """양 끝 bucket을 half-normal tail로 바꿔 무한 범위를 지지(support)하는 bar distribution.

    첫/마지막 bucket의 폭으로 half-normal 스케일을 정해(중앙값이 그 폭과 같도록), 경계 밖의
    y도 유한한 밀도·NLL을 갖는다. 학습 타깃 y가 [-4, 4]를 벗어나도 안정적으로 학습된다.
    """

    def __init__(self, borders: torch.Tensor, ignore_nan_targets: bool = True) -> None:
        super().__init__(borders, ignore_nan_targets)
        # half-normal 스케일: P(D <= width) = 0.5 가 되도록 (icdf(0.5) = 0.6745)
        icdf_half = torch.distributions.HalfNormal(1.0).icdf(torch.tensor(0.5))
        scale_left = (self.widths[0] / icdf_half).item()
        scale_right = (self.widths[-1] / icdf_half).item()
        self.scale_left = scale_left
        self.scale_right = scale_right
        # 평균/2차모멘트를 tail bucket에 맞게 보정
        centers = self.bucket_centers.clone()
        second = self.bucket_second_moment.clone()
        b1, b_last = self.borders[1], self.borders[-2]
        # 왼쪽 tail: y = b1 - D, D ~ HalfNormal(scale_left)
        centers[0] = b1 - scale_left * _SQRT_2_OVER_PI
        second[0] = b1**2 - 2 * b1 * (scale_left * _SQRT_2_OVER_PI) + scale_left**2
        # 오른쪽 tail: y = b_last + D
        centers[-1] = b_last + scale_right * _SQRT_2_OVER_PI
        second[-1] = b_last**2 + 2 * b_last * (scale_right * _SQRT_2_OVER_PI) + scale_right**2
        self.bucket_centers.copy_(centers)
        self.bucket_second_moment.copy_(second)

    def _halfnormal(self, scale: float, device: torch.device) -> torch.distributions.HalfNormal:
        return torch.distributions.HalfNormal(torch.tensor(scale, device=device))

    def forward(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y = y.reshape(logits.shape[:-1]).to(logits.device)
        nan_mask = torch.isnan(y)
        y_safe = torch.where(nan_mask, self.bucket_centers[0], y)
        idx = self._bucket_index(y_safe)
        log_probs = torch.log_softmax(logits, dim=-1)
        log_p_at = log_probs.gather(-1, idx.unsqueeze(-1)).squeeze(-1)

        b1, b_last = self.borders[1], self.borders[-2]
        left_mask = y_safe < b1
        right_mask = y_safe >= b_last
        interior = ~(left_mask | right_mask)

        hn_l = self._halfnormal(self.scale_left, logits.device)
        hn_r = self._halfnormal(self.scale_right, logits.device)
        # 각 영역의 밀도 로그 (tail은 half-normal, interior는 균일)
        log_dens = torch.zeros_like(log_p_at)
        log_dens = torch.where(interior, -torch.log(self.widths[idx]), log_dens)
        log_dens = torch.where(left_mask, hn_l.log_prob((b1 - y_safe).clamp_min(0.0)), log_dens)
        log_dens = torch.where(
            right_mask, hn_r.log_prob((y_safe - b_last).clamp_min(0.0)), log_dens
        )

        nll = -log_p_at - log_dens
        if self.ignore_nan_targets:
            nll = torch.where(nan_mask, torch.zeros_like(nll), nll)
        return nll

    def pdf(self, logits: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=-1)
        idx = self._bucket_index(ys)
        density = probs[..., idx] / self.widths[idx]  # interior 기준
        b1, b_last = self.borders[1], self.borders[-2]
        hn_l = self._halfnormal(self.scale_left, logits.device)
        hn_r = self._halfnormal(self.scale_right, logits.device)
        left_mask = ys < b1
        right_mask = ys >= b_last
        dens_l = probs[..., :1] * hn_l.log_prob((b1 - ys).clamp_min(0.0)).exp()
        dens_r = probs[..., -1:] * hn_r.log_prob((ys - b_last).clamp_min(0.0)).exp()
        density = torch.where(left_mask, dens_l, density)
        density = torch.where(right_mask, dens_r, density)
        return density

    def _grid_range(self) -> tuple[float, float]:
        # tail을 담도록 경계 밖으로 몇 스케일 확장
        lo = float(self.borders[0]) - 4.0 * self.scale_left
        hi = float(self.borders[-1]) + 4.0 * self.scale_right
        return lo, hi
