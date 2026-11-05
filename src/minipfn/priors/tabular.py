"""표 데이터 처리 (TabPFN v1 `priors/flexible_categorical.py`, `tabpfn/utils.py`): 정규화, 이상치
누르기, 연속값을 클래스로 바꾸기, 학습·테스트 행의 클래스 맞추기.

prior(`minipfn.priors.prior_bag`)가 데이터셋을 다듬을 때와 v1 추론
(`minipfn.inference.rotation_ensemble`)이 입력을 전처리할 때 같이 쓴다. 텐서는 (행, 데이터셋,
feature) 또는 (행, 데이터셋) 순서다.
"""

from __future__ import annotations

import numpy as np
import torch

IGNORE_INDEX = -100  # 손실에서 무시할 target (cross-entropy의 ignore_index)


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


def _randomize_classes(d: torch.Tensor, n_classes: int, gen: torch.Generator) -> torch.Tensor:
    perm = torch.randperm(n_classes, generator=gen, device=d.device).to(d.dtype)
    return perm[d.long()]


def _class_count(max_classes: int, rng: np.random.Generator) -> int:
    # v1 class_sampler_f: 50%는 round(U(2, max)), 50%는 2
    if rng.random() > 0.5:
        return int(round(rng.uniform(2, max_classes)))
    return 2


def multiclass_rank(
    v: torch.Tensor,
    max_classes: int,
    ordered_p: float,
    rng: np.random.Generator,
    gen: torch.Generator,
) -> torch.Tensor:
    """연속값 (T, B)를 클래스 번호로 바꾼다 (v1 MulticlassRank, 논문 4.5절).

    무작위 행의 값을 경계로 써서 구간 번호를 매긴다. 경계 행은 그룹 안 데이터셋이 공유한다.
    """
    n_classes = _class_count(max_classes, rng)
    idx = torch.randint(0, v.shape[0], (n_classes - 1,), generator=gen, device=v.device)
    bounds = v[idx].unsqueeze(1)  # (n-1, 1, B)
    d = (v.unsqueeze(0) > bounds).sum(0)  # (T, B)
    randomized = torch.rand(d.shape[1], generator=gen, device=v.device) > ordered_p
    if randomized.any():
        d[:, randomized] = _randomize_classes(d[:, randomized], n_classes, gen)
    reverse = torch.rand(d.shape[1], generator=gen, device=v.device) > 0.5
    d[:, reverse] = n_classes - 1 - d[:, reverse]
    return d.float()


def make_compatible(
    x: torch.Tensor, y: torch.Tensor, single_eval_pos: int, gen: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    """학습 행과 테스트 행의 클래스 집합이 같아질 때까지 행을 섞는다(최대 10번). 실패하면 -100."""
    for b in range(y.shape[1]):
        ok = False
        for _ in range(10):
            train_cls = torch.unique(y[:single_eval_pos, b])
            test_cls = torch.unique(y[single_eval_pos:, b])
            ok = len(train_cls) > 1 and torch.equal(train_cls, test_cls)
            if ok:
                break
            perm = torch.randperm(x.shape[0], generator=gen, device=x.device)
            x[:, b], y[:, b] = x[perm, b], y[perm, b]
        if not ok:
            y[:, b] = IGNORE_INDEX
    return x, y


def normalize_labels(y: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """클래스 번호를 0..n-1로 다시 매기고 무작위로 회전한다 (v1 normalize_labels)."""
    for b in range(y.shape[1]):
        valid = y[:, b] != IGNORE_INDEX
        if not valid.any():
            continue
        vals = y[valid, b]
        ranks = (vals.unsqueeze(0) > vals.unique().unsqueeze(1)).sum(0).float()
        n_classes = int(ranks.max().item()) + 1
        shift = int(torch.randint(0, n_classes, (1,), generator=gen, device=y.device).item())
        y[valid, b] = (ranks + shift) % n_classes
    return y
