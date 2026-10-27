import numpy as np
import torch

from minipfn import GPPriorSampler, analytic_gp_posterior, rbf_kernel
from minipfn.gp_prior import DEFAULT_VARIANCE
from minipfn.utils import set_seed


def test_sample_batch_shapes():
    sampler = GPPriorSampler()
    x, y = sampler.sample_batch(batch_size=8, seq_len=30)
    assert x.shape == (8, 30, 1)
    assert y.shape == (8, 30)


def test_sample_batch_stable_short_lengthscale():
    # seq_len=200, ls=0.6는 near-singular라 순진한 float32 Cholesky가 깨졌던 케이스(A10 회귀).
    set_seed(0)
    sampler = GPPriorSampler(lengthscale=0.6, variance=1.0, noise=1e-4)
    x, y = sampler.sample_batch(batch_size=16, seq_len=200)
    assert x.shape == (16, 200, 1) and y.shape == (16, 200)
    assert torch.isfinite(y).all()


def test_rbf_kernel_basic():
    x = np.array([0.0, 1.0, 2.0])
    k = rbf_kernel(x, x, lengthscale=0.6, variance=1.0)
    assert np.allclose(np.diag(k), 1.0)  # 자기 자신 = variance
    assert np.allclose(k, k.T)  # 대칭
    assert k[0, 2] < k[0, 1]  # 멀수록 상관 작음


def test_prior_marginal_variance():
    set_seed(0)
    # 각 y 점은 주변적으로 N(0, variance + noise^2). 대량 샘플로 평균≈0, 분산≈1 확인.
    sampler = GPPriorSampler(lengthscale=0.6, variance=1.0, noise=1e-4)
    _, y = sampler.sample_batch(batch_size=20000, seq_len=4)
    y = y.flatten().numpy()
    assert abs(float(y.mean())) < 0.05
    assert abs(float(y.var()) - 1.0) < 0.05


def test_analytic_posterior_collapses_at_observations():
    x_tr = np.array([-4.0, -2.5, -0.5, 1.0, 3.5])
    y_tr = np.sin(x_tr)
    mean, cov = analytic_gp_posterior(x_tr, y_tr, x_tr, lengthscale=0.6, noise=1e-4)
    # 관측점에서 평균 ≈ 관측값, 분산 < prior 분산
    assert np.allclose(mean, y_tr, atol=1e-2)
    assert np.all(np.diag(cov) < DEFAULT_VARIANCE)
