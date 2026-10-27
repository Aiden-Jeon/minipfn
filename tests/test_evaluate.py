import math

import torch
from matplotlib.figure import Figure

from minipfn import PFNConfig, build_criterion, build_model, evaluate_against_gp, pfn_predict
from minipfn.utils import set_seed


def test_evaluate_against_gp_returns_finite_metrics_and_figure():
    set_seed(0)
    config = PFNConfig()
    metrics, fig = evaluate_against_gp(build_model(config), build_criterion(config), title="t")

    assert set(metrics) == {
        "rmse_mean",
        "mean_abs_dsigma",
        "pfn_nll",
        "gp_nll",
        "nll_gap",
        "zero_ctx_abs_mean",
        "zero_ctx_std",
    }
    assert all(math.isfinite(v) for v in metrics.values())
    assert metrics["nll_gap"] == metrics["pfn_nll"] - metrics["gp_nll"]
    assert isinstance(fig, Figure)
    assert fig.axes[0].get_title() == "t"


def test_zero_context_metrics_use_an_empty_context():
    set_seed(0)
    config = PFNConfig()
    model, criterion = build_model(config), build_criterion(config)
    metrics, _ = evaluate_against_gp(model, criterion)

    empty = torch.zeros(0)
    pred0 = pfn_predict(model, criterion, empty, empty, torch.linspace(-5, 5, 200))
    assert math.isclose(
        metrics["zero_ctx_abs_mean"], float(pred0["mean"].abs().mean()), rel_tol=1e-5
    )
    assert math.isclose(metrics["zero_ctx_std"], float(pred0["std"].mean()), rel_tol=1e-5)
    assert metrics["zero_ctx_abs_mean"] >= 0
    assert metrics["zero_ctx_std"] > 0
