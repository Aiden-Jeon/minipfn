import math

import torch

from minipfn import FullSupportBarDistribution, get_bucket_borders
from minipfn.bar_distribution import BarDistribution


def test_uniform_logits_give_log_width_nll():
    # borders [-4, 4], 균일 logits → interior 밀도 = (1/B)/(8/B) = 1/8, NLL = log(8)
    borders = get_bucket_borders(100, (-4.0, 4.0))
    dist = BarDistribution(borders)
    logits = torch.zeros(1, 100)
    y = torch.tensor([1.3])
    nll = dist(logits, y)
    assert torch.allclose(nll, torch.tensor([math.log(8.0)]), atol=1e-5)


def test_pdf_integrates_to_one():
    borders = get_bucket_borders(50, (-4.0, 4.0))
    dist = BarDistribution(borders)
    logits = torch.randn(50)
    grid = torch.linspace(-4.0, 4.0, 4000)
    integral = torch.trapz(dist.pdf(logits, grid), grid)
    assert abs(integral.item() - 1.0) < 1e-2


def test_symmetric_distribution_mean_near_zero():
    borders = get_bucket_borders(101, (-4.0, 4.0))  # 홀수 bucket → 중앙 대칭
    dist = BarDistribution(borders)
    logits = torch.zeros(1, 101)  # 균일 → 대칭
    assert torch.allclose(dist.mean(logits), torch.tensor([0.0]), atol=1e-5)


def test_full_support_finite_nll_outside_range():
    borders = get_bucket_borders(100, (-4.0, 4.0))
    dist = FullSupportBarDistribution(borders)
    logits = torch.zeros(1, 100)
    y = torch.tensor([9.0])  # border 밖
    nll = dist(logits, y)
    assert torch.isfinite(nll).all()


def test_nan_target_is_ignored():
    borders = get_bucket_borders(100, (-4.0, 4.0))
    dist = FullSupportBarDistribution(borders)
    logits = torch.zeros(1, 100)
    nll = dist(logits, torch.tensor([float("nan")]))
    assert torch.allclose(nll, torch.zeros(1))


def test_quantile_brackets_central_mass():
    borders = get_bucket_borders(200, (-4.0, 4.0))
    dist = FullSupportBarDistribution(borders)
    logits = torch.zeros(1, 200)
    lower, upper = dist.quantile(logits, center_prob=0.682)
    assert (lower < 0).all() and (upper > 0).all()
    assert torch.allclose(lower, -upper, atol=0.2)  # 대칭
