import pytest
import torch

from minipfn.priors.base import IGNORE_INDEX, PriorBatch
from minipfn.priors.generators.mlp import MLP
from minipfn.priors.hyperparameters import GroupHyperparameters
from minipfn.priors.pipeline import Group, Mixture, Pipeline, Rng
from minipfn.priors.transforms import (
    CoverClasses,
    DropInvalid,
    Missing,
    PadFeatures,
    Standardize,
    ToClasses,
)
from minipfn.recipes import tabpfn_v1 as spec


class Constant:
    """값이 정해진 생성기: x = value, y = 행 번호 % 2."""

    def __init__(self, value: float) -> None:
        self.value = value

    def __call__(self, n_rows, n_features, n_datasets, hp, rng):
        x = torch.full((n_rows, n_datasets, n_features), self.value)
        y = (torch.arange(n_rows) % 2).float().unsqueeze(1).expand(n_rows, n_datasets).clone()
        return x, y


class RejectFirst:
    def __init__(self, n: int) -> None:
        self.n = n
        self.calls = 0

    def __call__(self, group, rng):
        self.calls += 1
        return None if self.calls <= self.n else group


def _counter():
    draws = []

    def hyperparameters(rng, n_train, n_features):
        draws.append(1)
        return {"draw": len(draws)}

    return hyperparameters, draws


def test_rejected_groups_are_redrawn_from_the_hyperparameters():
    hyperparameters, draws = _counter()
    reject = RejectFirst(2)
    pipe = Pipeline(
        Constant(1.0),
        rng=Rng.from_seed(0),
        hyperparameters=hyperparameters,
        transforms=[reject],
        max_attempts=5,
    )
    group = pipe.sample_group(n_rows=6, n_train=4, n_features=3)
    assert reject.calls == 3 and len(draws) == 3
    assert group.hp == {"draw": 3}


def test_gives_up_after_max_attempts():
    pipe = Pipeline(
        Constant(1.0), rng=Rng.from_seed(0), transforms=[RejectFirst(10)], max_attempts=3
    )
    with pytest.raises(RuntimeError, match="3번 시도해도"):
        pipe.sample_group(6, 4, 3)


def test_mixture_uses_the_generator_the_hyperparameters_chose():
    mixture = Mixture({"one": Constant(1.0), "two": Constant(2.0)}, lambda hp: hp["pick"])
    for pick, value in [("one", 1.0), ("two", 2.0)]:
        pipe = Pipeline(
            mixture, rng=Rng.from_seed(0), hyperparameters=lambda r, n, f, p=pick: {"pick": p}
        )
        assert (pipe.sample_group(5, 3, 2).x == value).all()


def test_generator_gets_the_used_feature_count_and_batch_is_datasets_first():
    pipe = Pipeline(
        Constant(1.0),
        rng=Rng.from_seed(0),
        hyperparameters=lambda rng, n_train, n_features: {"num_features_used": 2},
        transforms=[PadFeatures(5, rescale=False)],
        group_size=2,
    )
    batch = pipe.sample_batch(4, n_rows=7, n_train=5, n_features=5)
    assert isinstance(batch, PriorBatch) and batch.n_train == 5
    assert batch.x.shape == (4, 7, 5) and batch.y.shape == (4, 7)
    assert (batch.x[..., :2] == 1).all() and (batch.x[..., 2:] == 0).all()
    assert torch.equal(batch.y[0], torch.arange(7.0) % 2)
    for bad in (3, 0, -2):
        with pytest.raises(ValueError, match="배수"):
            pipe.sample_batch(bad, 7, 5, 5)


def test_sample_group_can_run_only_some_transforms():
    pipe = Pipeline(
        Constant(1.0), rng=Rng.from_seed(0), transforms=[RejectFirst(10)], max_attempts=2
    )
    assert (pipe.sample_group(5, 3, 2, transforms=[]).x == 1).all()
    with pytest.raises(RuntimeError):
        pipe.sample_group(5, 3, 2)


def test_v1_generator_with_v2_missing_values():
    # 다른 논문의 부품끼리 묶기: v1 MLP 생성기·하이퍼파라미터·후처리에 v2 결측값
    pipe = Pipeline(
        MLP(scale_weights_sqrt=True, random_feature_rotation=True),
        rng=Rng.from_seed(0),
        hyperparameters=GroupHyperparameters(
            spec.MLP_HYPERPARAMETERS,
            spec.GP_HYPERPARAMETERS,
            mlp_weight=spec.PRIOR_BAG_MLP_WEIGHT,
            min_classes=2,
            max_classes=10,
        ),
        transforms=[
            Standardize(),
            ToClasses(0.0),
            PadFeatures(8, rescale=True),
            Missing(1.0, 0.5),
        ],
        group_size=4,
    )
    batch = pipe.sample_batch(8, n_rows=40, n_train=30, n_features=8)
    assert batch.x.shape == (8, 40, 8)
    assert torch.isnan(batch.x).any()
    assert ((batch.y >= 0) | (batch.y == IGNORE_INDEX)).all()


def test_filters_and_cover_classes_look_at_every_dataset():
    rng = Rng.from_seed(0)
    y = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])  # 데이터셋 2개
    group = Group(torch.arange(8.0).reshape(4, 2, 1), y, n_train=2, hp={})
    covered = CoverClasses()(group, rng)
    for d in range(2):
        assert set(covered.y[:2, d].tolist()) == {0.0, 1.0}
        # 행 순서만 바뀌고 x와 y는 같이 움직인다
        pairs = {
            (float(a), float(b)) for a, b in zip(covered.x[:, d, 0], covered.y[:, d], strict=True)
        }
        assert pairs == {
            (float(a), float(b)) for a, b in zip(group.x[:, d, 0], y[:, d], strict=True)
        }
    one_class = Group(group.x, torch.zeros(4, 2), 2, {})
    assert CoverClasses()(one_class, rng) is None
    invalid = Group(
        group.x, torch.stack([y[:, 0], torch.full((4,), float(IGNORE_INDEX))], 1), 2, {}
    )
    assert DropInvalid()(invalid, rng) is None and DropInvalid()(group, rng) is group


def test_missing_only_touches_x_of_chosen_datasets():
    group = Group(torch.ones(50, 3, 4), torch.zeros(50, 3), 40, {})
    assert not torch.isnan(Missing(0.0, 0.9)(group, Rng.from_seed(0)).x).any()
    out = Missing(1.0, 0.9)(group, Rng.from_seed(1))
    assert torch.isnan(out.x).any() and torch.equal(out.y, group.y)


def test_settings_baked_into_parts_are_read_only():
    from minipfn.priors.graph_scm import GraphSCMPrior
    from minipfn.recipes.tabpfn_v1 import TabPFNPrior

    graph = GraphSCMPrior("regression", seed=0, missing_dataset_prob=0.5, n_eval_rows=8)
    assert graph.missing_dataset_prob == 0.5 and graph.task == "regression"
    bag = TabPFNPrior(seq_len=16, num_features=4, group_size=2)
    assert bag.num_features == 4 and bag.group_size == 2
    for obj, name in [(graph, "missing_dataset_prob"), (graph, "task"), (bag, "num_features")]:
        with pytest.raises(AttributeError):
            setattr(obj, name, 0)


ACCELERATOR = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else None
)


@pytest.mark.skipif(ACCELERATOR is None, reason="GPU(cuda·mps)가 없다")
def test_transforms_keep_tensors_on_their_device():
    # 생성기가 GPU 텐서를 내고 Rng는 CPU인 경우 (device를 Rng가 아니라 텐서에서 읽는다)
    dev = torch.device(ACCELERATOR)
    y = torch.tensor([[0.0, 1.0]] * 5 + [[1.0, 0.0]] * 5, device=dev)
    group = Group(torch.randn(10, 2, 3, device=dev), y, 6, {})
    rng = Rng.from_seed(0)
    for transform in (Missing(1.0, 0.5), CoverClasses(), PadFeatures(5, rescale=True)):
        out = transform(group, rng)
        assert out.x.device.type == dev.type and out.y.device.type == dev.type
