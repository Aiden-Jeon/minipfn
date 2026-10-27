"""학습된 PFN을 analytic GP posterior와 대조하는 재현 검증.

노트북 03 §8과 같은 setup(고정 5점 context, y = sin(x))에서 PFN 예측을 정답 GP posterior에
겹쳐 그리고, 평균·표준편차 오차와 query NLL을 지표로 낸다. GP posterior는 prior에서 뽑은
데이터셋들의 평균 NLL로 보면 가장 낮은(Bayes-optimal) 예측이라 기준으로 쓴다. 여기 query는
sin 함수 하나라서 PFN NLL이 GP보다 조금 낮게 나올 수도 있다(하한이 아니다).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from torch import nn

from minipfn.bar_distribution import FullSupportBarDistribution
from minipfn.gp_prior import DEFAULT_NOISE, analytic_gp_posterior
from minipfn.train import pfn_predict
from minipfn.utils import set_seed


def evaluate_against_gp(
    model: nn.Module,
    criterion: FullSupportBarDistribution,
    title: str | None = None,
) -> tuple[dict[str, float], Figure]:
    """PFN 예측을 analytic GP posterior와 비교해 (metrics, 오버레이 figure)를 반환한다.

    metrics: rmse_mean, mean_abs_dsigma, pfn_nll, gp_nll, nll_gap(= pfn_nll - gp_nll),
    zero_ctx_mean, zero_ctx_std(context 0개일 때의 평균 예측).
    """
    x_grid = np.linspace(-5, 5, 200)
    x_tr = np.array([-4.0, -2.5, -0.5, 1.0, 3.5])
    set_seed(0)
    y_tr = np.sin(x_tr) + DEFAULT_NOISE * np.random.randn(x_tr.size)

    mu, cov = analytic_gp_posterior(x_tr, y_tr, x_grid)
    sd = np.sqrt(np.clip(np.diag(cov), 0, None))

    xc = torch.tensor(x_tr, dtype=torch.float32)
    yc = torch.tensor(y_tr, dtype=torch.float32)
    pred = pfn_predict(model, criterion, xc, yc, torch.tensor(x_grid, dtype=torch.float32))
    pfn_mean = pred["mean"].cpu().numpy()
    pfn_sd = pred["std"].cpu().numpy()

    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.fill_between(x_grid, mu - 2 * sd, mu + 2 * sd, color="C0", alpha=0.15, label="analytic ±2σ")
    ax.plot(x_grid, mu, color="C0", lw=2, label="analytic mean")
    ax.plot(x_grid, pfn_mean, color="C3", lw=2, ls="--", label="PFN mean")
    ax.fill_between(
        x_grid,
        pfn_mean - 2 * pfn_sd,
        pfn_mean + 2 * pfn_sd,
        color="C3",
        alpha=0.15,
        label="PFN ±2σ",
    )
    ax.scatter(x_tr, y_tr, c="black", s=35, zorder=5, label="context D")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title or "PFN vs analytic GP posterior")
    ax.legend(loc="upper left", fontsize=8)

    # NLL 비교: query 점에서 PFN NLL vs Gaussian analytic-GP NLL
    xq = np.linspace(-4.5, 4.5, 40)
    yq = np.sin(xq)
    mu_q, cov_q = analytic_gp_posterior(x_tr, y_tr, xq)
    var_q = np.clip(np.diag(cov_q), 1e-9, None) + DEFAULT_NOISE**2
    gp_nll = float(np.mean(0.5 * np.log(2 * np.pi * var_q) + 0.5 * (yq - mu_q) ** 2 / var_q))
    predq = pfn_predict(model, criterion, xc, yc, torch.tensor(xq, dtype=torch.float32))
    yq_t = torch.tensor(yq, dtype=torch.float32, device=predq["logits"].device)
    pfn_nll = float(criterion(predq["logits"], yq_t).mean())

    # context 0개: prior 자체를 돌려줘야 한다 (평균 ≈ 0, 표준편차 ≈ √(variance + noise²) ≈ 1)
    empty = torch.zeros(0, dtype=torch.float32)
    pred0 = pfn_predict(model, criterion, empty, empty, torch.tensor(x_grid, dtype=torch.float32))

    metrics = {
        "rmse_mean": float(np.sqrt(np.mean((pfn_mean - mu) ** 2))),
        "mean_abs_dsigma": float(np.mean(np.abs(pfn_sd - sd))),
        "pfn_nll": pfn_nll,
        "gp_nll": gp_nll,
        "nll_gap": pfn_nll - gp_nll,
        "zero_ctx_mean": float(pred0["mean"].mean()),
        "zero_ctx_std": float(pred0["std"].mean()),
    }
    return metrics, fig
