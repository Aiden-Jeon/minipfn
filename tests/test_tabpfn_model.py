import torch

from minipfn.tabpfn import spec
from minipfn.tabpfn.model import TabPFNTransformer, classification_loss


def _tiny_model():
    torch.manual_seed(0)
    model = TabPFNTransformer(num_features=6, n_out=4, emsize=16, nhead=2, nhid=32, nlayers=2)
    # 0 초기화된 층은 처음에 항등이라 순열 불변 등을 확인하기 어려우므로 무작위로 다시 채운다.
    for p in model.parameters():
        torch.nn.init.normal_(p, std=0.2)
    return model.eval()


def test_paper_size_matches_reported_parameter_count():
    model = TabPFNTransformer()  # 논문 크기 (12층, emsize 512)
    assert abs(model.num_parameters() - spec.PAPER_N_PARAMETERS) / spec.PAPER_N_PARAMETERS < 0.001


def test_output_shape():
    model = _tiny_model()
    out = model(torch.randn(3, 10, 6), torch.randint(0, 4, (3, 10)).float(), torch.randn(3, 7, 6))
    assert out.shape == (3, 7, 4)


def test_train_row_permutation_invariance():
    model = _tiny_model()
    xtr, ytr, xte = (
        torch.randn(1, 12, 6),
        torch.randint(0, 4, (1, 12)).float(),
        torch.randn(1, 5, 6),
    )
    perm = torch.randperm(12)
    a = model(xtr, ytr, xte)
    b = model(xtr[:, perm], ytr[:, perm], xte)
    assert torch.allclose(a, b, atol=1e-5)


def test_test_rows_are_independent():
    # 테스트 행 하나의 예측은 다른 테스트 행에 영향을 받지 않는다.
    model = _tiny_model()
    xtr, ytr, xte = (
        torch.randn(1, 12, 6),
        torch.randint(0, 4, (1, 12)).float(),
        torch.randn(1, 5, 6),
    )
    full = model(xtr, ytr, xte)
    single = model(xtr, ytr, xte[:, 2:3])
    assert torch.allclose(full[:, 2:3], single, atol=1e-5)


def test_zero_train_rows_runs():
    model = _tiny_model()
    out = model(torch.randn(2, 0, 6), torch.zeros(2, 0), torch.randn(2, 4, 6))
    assert out.shape == (2, 4, 4) and torch.isfinite(out).all()


def test_loss_ignores_marked_rows():
    logits = torch.randn(2, 3, 4)
    y = torch.tensor([[0, 1, 2], [-100, -100, -100]])
    loss = classification_loss(logits, y)
    expected = torch.nn.functional.cross_entropy(logits[0], y[0], reduction="sum") / 6
    assert torch.allclose(loss, expected)


def test_one_step_reduces_loss_on_fixed_batch():
    torch.manual_seed(0)
    model = TabPFNTransformer(num_features=6, n_out=3, emsize=32, nhead=2, nhid=64, nlayers=2)
    xtr, xte = torch.randn(4, 20, 6), torch.randn(4, 10, 6)
    ytr, yte = (xtr[..., 0] > 0).float(), (xte[..., 0] > 0).long()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    first = None
    for _ in range(30):
        loss = classification_loss(model(xtr, ytr, xte), yte)
        first = loss.item() if first is None else first
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < first
