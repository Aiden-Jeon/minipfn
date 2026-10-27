import math

from matplotlib.figure import Figure

from minipfn import PFNConfig, build_criterion, build_model, evaluate_against_gp
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
        "zero_ctx_mean",
        "zero_ctx_std",
    }
    assert all(math.isfinite(v) for v in metrics.values())
    assert metrics["nll_gap"] == metrics["pfn_nll"] - metrics["gp_nll"]
    assert isinstance(fig, Figure)
    assert fig.axes[0].get_title() == "t"
