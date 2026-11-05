"""GP 표 생성기 (TabPFN v1 `priors/fast_gp.py`): RBF 커널 GP에서 k차원 입력의 표.

v1 prior bag이 MLP 생성기와 섞어 쓴다. 1D GP 회귀 prior(PFN 논문)는 `minipfn.priors.gp`다.
"""

from __future__ import annotations

import torch

from minipfn.priors.pipeline import Rng


def sample_gp_tables(
    seq_len: int, n_datasets: int, k: int, hp: dict, gen: torch.Generator, device
) -> tuple[torch.Tensor, torch.Tensor]:
    """RBF 커널 GP에서 표 데이터셋 n_datasets개 (TabPFN v1 `priors/fast_gp.py`). x는 k차원
    표준정규분포(sampling='normal'), 하이퍼파라미터 `hp`의 outputscale·lengthscale·noise를 쓴다.
    반환은 (행, 데이터셋, k), (행, 데이터셋) 순서다."""
    x = torch.randn(n_datasets, seq_len, k, generator=gen, device=device)
    xd = x.double()
    sq_dist = torch.cdist(xd, xd) ** 2
    lengthscale = max(hp["lengthscale"], 1e-6)
    cov = hp["outputscale"] * torch.exp(-0.5 * sq_dist / lengthscale**2)
    cov = cov + (hp["noise"] + 1e-6) * torch.eye(seq_len, dtype=torch.float64, device=device)
    chol = torch.linalg.cholesky(cov)
    z = torch.randn(n_datasets, seq_len, 1, generator=gen, device=device, dtype=torch.float64)
    y = (chol @ z)[..., 0].float()
    return x.transpose(0, 1), y.transpose(0, 1)  # (T, B, k), (T, B)


class GPTables:
    """GP 표 생성기 (`Generator`). 하이퍼파라미터의 outputscale·lengthscale·noise를 쓴다."""

    def __call__(
        self, n_rows: int, n_features: int, n_datasets: int, hp: dict, rng: Rng
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return sample_gp_tables(n_rows, n_datasets, n_features, hp, rng.torch, rng.device)
