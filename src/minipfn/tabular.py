"""표 데이터 처리: 열별 표준화와 이상치 누르기 (TabPFN v1 `flexible_categorical.py`,
`tabpfn/utils.py`).

prior의 후처리(`minipfn.priors.transforms.scaling`)와 v1 추론 전처리
(`minipfn.inference.rotation_ensemble`)가 같이 쓴다. 텐서는 행 방향(dim 0)으로 통계를 내고, 결측값
(NaN)은 통계에서 뺀다.
"""

from __future__ import annotations

import torch


def _masked_mean_std(x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    num = mask.sum(0).clamp(min=1)
    xm = torch.where(mask, x, torch.zeros_like(x))
    mean = xm.sum(0) / num
    var = (torch.where(mask, (x - mean) ** 2, torch.zeros_like(x))).sum(0) / (num - 1).clamp(min=1)
    return mean, var.sqrt()


def normalization_stats(ref: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """normalize_data가 쓰는 열별 평균·표준편차 (결측값 제외)."""
    return _masked_mean_std(ref, ~torch.isnan(ref))


def apply_normalization(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return torch.clip((x - mean) / (std + 1e-6), -100, 100)


def normalize_data(x: torch.Tensor, n_rows: int = -1) -> torch.Tensor:
    """행 방향(dim 0)으로 표준화하고 ±100으로 자른다. n_rows > 0이면 앞 n_rows행의 통계를 쓴다."""
    ref = x[:n_rows] if n_rows > 0 else x
    return apply_normalization(x, *normalization_stats(ref))


def outlier_bounds(ref: torch.Tensor, n_sigma: float = 4) -> tuple[torch.Tensor, torch.Tensor]:
    """remove_outliers의 경계: 평균 ± nσ 안의 값만으로 평균·표준편차를 다시 구한 ± nσ."""
    mean, std = _masked_mean_std(ref, ~torch.isnan(ref))
    lower, upper = mean - n_sigma * std, mean + n_sigma * std
    mask = (ref <= upper) & (ref >= lower) & ~torch.isnan(ref)
    mean, std = _masked_mean_std(ref, mask)
    return mean - n_sigma * std, mean + n_sigma * std


def squash_outliers(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    x = torch.maximum(-torch.log(1 + torch.abs(x)) + lower, x)
    return torch.minimum(torch.log(1 + torch.abs(x)) + upper, x)


def remove_outliers(x: torch.Tensor, n_sigma: float = 4, n_rows: int = -1) -> torch.Tensor:
    """평균 ± 4σ 밖의 값을 log로 눌러 준다 (v1 remove_outliers)."""
    ref = x[:n_rows] if n_rows > 0 else x
    return squash_outliers(x, *outlier_bounds(ref, n_sigma))
