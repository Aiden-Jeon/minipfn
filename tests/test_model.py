import torch

from minipfn import (
    GPPriorSampler,
    PFNConfig,
    PFNTransformer,
    build_criterion,
    build_model,
    train_pfn,
)
from minipfn.utils import set_seed


def _random_dataset(n_context, n_query, x_dim=1):
    x_ctx = torch.randn(1, n_context, x_dim)
    y_ctx = torch.randn(1, n_context)
    x_qry = torch.randn(1, n_query, x_dim)
    return x_ctx, y_ctx, x_qry


def test_output_shape():
    model = PFNTransformer(num_buckets=100)
    x_ctx, y_ctx, x_qry = _random_dataset(6, 4)
    logits = model(x_ctx, y_ctx, x_qry)
    assert logits.shape == (1, 4, 100)


def test_context_permutation_invariance():
    set_seed(0)
    model = PFNTransformer(num_buckets=50).eval()
    x_ctx, y_ctx, x_qry = _random_dataset(6, 3)
    with torch.no_grad():
        base = model(x_ctx, y_ctx, x_qry)
        perm = torch.randperm(6)
        shuffled = model(x_ctx[:, perm], y_ctx[:, perm], x_qry)
    assert torch.allclose(base, shuffled, atol=1e-5)


def test_query_independence():
    set_seed(0)
    model = PFNTransformer(num_buckets=50).eval()
    x_ctx = torch.randn(1, 5, 1)
    y_ctx = torch.randn(1, 5)
    x_qry = torch.randn(1, 3, 1)
    with torch.no_grad():
        full = model(x_ctx, y_ctx, x_qry)
        # 첫 query만 넣어도 같은 결과여야 (query끼리 독립)
        single = model(x_ctx, y_ctx, x_qry[:, :1])
    assert torch.allclose(full[:, :1], single, atol=1e-5)


def test_zero_context_runs():
    model = PFNTransformer(num_buckets=50).eval()
    x_ctx = torch.zeros(1, 0, 1)
    y_ctx = torch.zeros(1, 0)
    x_qry = torch.randn(1, 4, 1)
    with torch.no_grad():
        logits = model(x_ctx, y_ctx, x_qry)
    assert logits.shape == (1, 4, 50)
    assert torch.isfinite(logits).all()


def test_smoke_training_reduces_loss():
    set_seed(0)
    config = PFNConfig(steps=60, batch_size=8, seq_len=40, max_context=25)
    model = build_model(config)
    criterion = build_criterion(config)
    sampler = GPPriorSampler(
        lengthscale=config.lengthscale, variance=config.variance, noise=config.noise
    )
    history = train_pfn(model, sampler, criterion, config, device="cpu", progress=False)
    losses = history["losses"]
    initial = sum(losses[:10]) / 10
    final = sum(losses[-10:]) / 10
    assert final < 0.9 * initial
