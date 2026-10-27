import numpy as np
import torch

from minipfn.tabpfn import spec
from minipfn.tabpfn.prior import (
    TabPFNPrior,
    TabPFNPriorDataset,
    _sample_gp_group,
    sample_group_hyperparameters,
    sample_hyperparameter,
)


def _small_prior(seed=0, seq_len=128):
    return TabPFNPrior(seq_len=seq_len, num_features=20, max_classes=10, group_size=4, seed=seed)


def test_group_shapes_and_padding():
    prior = _small_prior()
    for _ in range(10):
        x, y, hp = prior.sample_group(single_eval_pos=64)
        k = hp["num_features_used"]
        assert x.shape == (4, 128, 20) and y.shape == (4, 128)
        assert torch.isfinite(x).all()
        assert torch.all(x[..., k:] == 0)  # 쓰지 않는 feature는 0으로 채운다


def test_labels_are_contiguous_and_train_test_compatible():
    prior = _small_prior(seed=1)
    sep = 64
    for _ in range(10):
        _, y, _ = prior.sample_group(single_eval_pos=sep)
        for b in range(y.shape[0]):
            valid = y[b] != spec.IGNORE_INDEX
            if not valid.any():
                continue
            assert valid.all()  # 데이터셋 단위로만 무시한다
            classes = torch.unique(y[b])
            assert torch.equal(classes, torch.arange(len(classes), dtype=y.dtype))
            assert torch.equal(torch.unique(y[b, :sep]), torch.unique(y[b, sep:]))


def test_zero_train_rows_marks_dataset_ignored():
    # 학습 행이 없으면 클래스를 맞출 수 없으므로 v1처럼 데이터셋 전체를 무시한다.
    _, y, _ = _small_prior(seed=2).sample_group(single_eval_pos=0)
    assert torch.all(y == spec.IGNORE_INDEX)


def test_same_seed_is_reproducible():
    a = _small_prior(seed=5).sample_group(single_eval_pos=50)
    b = _small_prior(seed=5).sample_group(single_eval_pos=50)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_feature_scaling_by_used_features():
    # 쓴 feature 열은 표준화 후 num_features / k배가 된다 (논문 E.2).
    prior = _small_prior(seed=3, seq_len=256)
    for _ in range(5):
        x, _, hp = prior.sample_group(single_eval_pos=128)
        k = hp["num_features_used"]
        std = x[..., :k].std(dim=1).median().item()
        assert 0.3 * 20 / k < std < 3 * 20 / k


def test_hyperparameter_bounds():
    rng = np.random.default_rng(0)
    for _ in range(200):
        hp = sample_group_hyperparameters(rng, max_classes=10, num_features=100)
        assert hp["num_layers"] >= 2 and hp["prior_mlp_hidden_dim"] >= 4
        assert hp["num_causes"] >= 2
        assert 0 <= hp["prior_mlp_dropout_prob"] <= 0.6
        assert hp["noise_std"] >= 0 and hp["init_std"] >= 0
        assert 1 <= hp["num_features_used"] <= 100
        assert 2 <= hp["num_classes"] <= 10
        assert hp["prior_mlp_activations"] in {"tanh", "identity", "relu"}


def test_meta_choice_mixes_both_values():
    rng = np.random.default_rng(0)
    draws = [sample_hyperparameter(spec.MLP_HYPERPARAMETERS["is_causal"], rng) for _ in range(500)]
    share = np.mean(draws)
    assert 0.3 < share < 0.7  # SCM(True)과 BNN(False)이 대략 반반


def test_prior_bag_mostly_mlp():
    rng = np.random.default_rng(0)
    use_mlp = [sample_group_hyperparameters(rng, 10, 100)["use_mlp"] for _ in range(1000)]
    assert 0.85 < np.mean(use_mlp) < 1.0


def test_gp_group_shapes():
    gen = torch.Generator().manual_seed(0)
    hp = {"outputscale": 1.0, "lengthscale": 1.0, "noise": 1e-4}
    x, y = _sample_gp_group(64, 3, 5, hp, gen, "cpu")
    assert x.shape == (64, 3, 5) and y.shape == (64, 3)
    assert torch.isfinite(y).all()


def test_dataset_yields_groups_with_eval_pos_in_range():
    ds = TabPFNPriorDataset(seq_len=64, max_eval_pos=40, num_features=10, group_size=2, seed=0)
    it = iter(ds)
    for _ in range(3):
        x, y, sep = next(it)
        assert x.shape == (2, 64, 10) and y.shape == (2, 64)
        assert 1 <= sep < 40  # 학습 행이 0개인 batch는 만들지 않는다 (DDP 동기화)


def test_dataset_seeds_do_not_collide_with_validation_seed():
    # 예전 방식((seed * 100_003 + step) * 1000 + rank) * 1000 + worker는 smoke resume(step 10)에서
    # 검증 seed 10_000_000과 같아졌다. 학습 prior의 첫 그룹이 검증 그룹과 달라야 한다.
    kwargs = dict(seq_len=64, max_eval_pos=40, num_features=10, group_size=2)
    x_train, _, _ = next(iter(TabPFNPriorDataset(**kwargs, seed=(0, 10, 0))))
    val_prior = TabPFNPrior(
        seq_len=64, num_features=10, max_classes=10, group_size=2, seed=10_000_000
    )
    x_val, _, _ = val_prior.sample_group(20)
    assert not torch.equal(x_train, x_val)

    # 같은 seed tuple이면 같은 데이터, resume 시작 step이 다르면 다른 데이터
    again, _, _ = next(iter(TabPFNPriorDataset(**kwargs, seed=(0, 10, 0))))
    other, _, _ = next(iter(TabPFNPriorDataset(**kwargs, seed=(0, 20, 0))))
    assert torch.equal(x_train, again) and not torch.equal(x_train, other)
