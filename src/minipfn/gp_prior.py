"""GP prior: 학습 데이터를 만드는 prior 샘플러와, 재현 타깃인 analytic posterior.

논문(arXiv:2112.10510) §5.1 / 부록 F.1의 GP 설정을 따른다: zero-mean GP, RBF 커널,
기본 하이퍼파라미터 lengthscale=0.6, variance=1.0, 관측 노이즈 1e-4.
`rbf_kernel`/`analytic_gp_posterior`는 노트북 03 §8과 동일한 식으로, 재현 결과를
대조할 "정답"을 계산한다. `GPPriorSampler`는 매 학습 스텝의 데이터셋을 batch로 뽑는다.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch

# 논문 GP 실험의 기본 하이퍼파라미터 (노트북 03 §8과 공유하는 상수)
DEFAULT_LENGTHSCALE = 0.6
DEFAULT_VARIANCE = 1.0
DEFAULT_NOISE = 1e-4


def rbf_kernel(
    a: np.ndarray,
    b: np.ndarray,
    lengthscale: float = DEFAULT_LENGTHSCALE,
    variance: float = DEFAULT_VARIANCE,
) -> np.ndarray:
    """RBF(제곱지수) 커널 행렬 K[i, j] = variance * exp(-0.5 * ((a_i - b_j) / ls)^2)."""
    d = a[:, None] - b[None, :]
    return variance * np.exp(-0.5 * (d / lengthscale) ** 2)


def analytic_gp_posterior(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    lengthscale: float = DEFAULT_LENGTHSCALE,
    variance: float = DEFAULT_VARIANCE,
    noise: float = DEFAULT_NOISE,
) -> tuple[np.ndarray, np.ndarray]:
    """관측 (x_train, y_train)에 대한 zero-mean GP posterior의 평균과 공분산.

    mu_*   = K_*,tr (K_tr,tr + noise^2 I)^{-1} y_tr
    Sigma_* = K_*,* - K_*,tr (K_tr,tr + noise^2 I)^{-1} K_tr,*

    이 값이 재현 타깃(정답)이며, 검증 오버레이에서 PFN 예측과 대조한다.
    """
    k_tr = rbf_kernel(x_train, x_train, lengthscale, variance)
    k_tr = k_tr + noise**2 * np.eye(x_train.shape[0])
    k_s = rbf_kernel(x_test, x_train, lengthscale, variance)
    # inv 대신 solve로 수치 안정성 확보 (noise가 작아 near-singular 위험)
    k_inv_y = np.linalg.solve(k_tr, y_train)
    k_inv_ks = np.linalg.solve(k_tr, k_s.T)
    mean = k_s @ k_inv_y
    cov = rbf_kernel(x_test, x_test, lengthscale, variance) - k_s @ k_inv_ks
    return mean, cov


class GPPriorSampler:
    """GP prior에서 학습용 데이터셋 batch를 뽑는다.

    한 데이터셋 = 입력 x ~ U(x_range)와, 그 x에서 zero-mean GP에서 뽑은 라벨
    y ~ N(0, K(x, x) + noise^2 I). context/query 분할은 학습 루프가 맡는다.

    `hyperparameter_prior`가 주어지면 batch 원소마다 (lengthscale, variance)를 새로
    뽑아 "논문 그대로"의 다양한 task를 만든다(보너스 모드). None이면 고정 하이퍼파라미터로,
    §8의 특정 posterior와 pointwise 대조가 가능한 재현 모드가 된다.
    """

    def __init__(
        self,
        x_range: tuple[float, float] = (-5.0, 5.0),
        lengthscale: float = DEFAULT_LENGTHSCALE,
        variance: float = DEFAULT_VARIANCE,
        noise: float = DEFAULT_NOISE,
        hyperparameter_prior: Callable[[int], tuple[torch.Tensor, torch.Tensor]] | None = None,
        jitter: float = 1e-6,
    ) -> None:
        self.x_range = x_range
        self.lengthscale = lengthscale
        self.variance = variance
        self.noise = noise
        self.hyperparameter_prior = hyperparameter_prior
        self.jitter = jitter

    def sample_batch(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device | str = "cpu",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """데이터셋 batch를 뽑는다. 반환 x: (B, S, 1), y: (B, S)."""
        device = torch.device(device)
        # MPS는 torch 2.11에서 cholesky 등 일부 linalg를 지원하지 않으므로 CPU에서 만들고 옮긴다.
        gen_device = torch.device("cpu") if device.type == "mps" else device
        lo, hi = self.x_range
        x = torch.rand(batch_size, seq_len, 1, device=gen_device) * (hi - lo) + lo

        if self.hyperparameter_prior is not None:
            ls, var = self.hyperparameter_prior(batch_size)
            ls = ls.to(gen_device).view(batch_size, 1, 1)
            var = var.to(gen_device).view(batch_size, 1, 1)
        else:
            ls = torch.full((batch_size, 1, 1), self.lengthscale, device=gen_device)
            var = torch.full((batch_size, 1, 1), self.variance, device=gen_device)

        # 커널·Cholesky는 float64로 계산한다. lengthscale이 짧으면 가까운 입력점들이 겹쳐
        # 커널이 near-singular가 되는데, float32에서는 특히 not-PD로 깨지기 쉽다.
        d = (x - x.transpose(1, 2)).double()  # (B, S, S)
        k = var.double() * torch.exp(-0.5 * (d / ls.double()) ** 2)
        eye = torch.eye(seq_len, dtype=torch.float64, device=gen_device)
        k = k + self.noise**2 * eye

        # 그래도 not-PD면 jitter를 키워가며 재시도한다 (표준 GP 안정화 트릭).
        jitter = self.jitter
        for _ in range(7):
            try:
                chol = torch.linalg.cholesky(k + jitter * eye)  # (B, S, S)
                break
            except RuntimeError:
                jitter *= 10.0
        else:
            chol = torch.linalg.cholesky(k + jitter * eye)

        eps = torch.randn(batch_size, seq_len, 1, dtype=torch.float64, device=gen_device)
        y = (chol @ eps).squeeze(-1).float()  # (B, S)
        return x.to(device), y.to(device)
