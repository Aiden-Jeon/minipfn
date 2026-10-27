import torch

from minipfn import get_device, set_seed


def test_set_seed_is_reproducible():
    set_seed(0)
    a = torch.randn(3)
    set_seed(0)
    b = torch.randn(3)
    assert torch.equal(a, b)


def test_get_device_respects_prefer():
    assert get_device("cpu").type == "cpu"


def test_get_device_returns_available_device():
    device = get_device()
    torch.zeros(1, device=device)
