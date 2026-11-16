"""graph_scm prior 변형 (issue #92): tree SCM 섞기, TabICL 보정 플래그, 군집(Gaussian mixture) root.
기본값이면 변형 전과 같은 부품·같은 난수를 쓴다."""

import importlib
import importlib.util
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from minipfn.priors import graph_scm as P
from minipfn.priors.generators import graph_scm as G
from minipfn.priors.pipeline import Mixture, Rng
from minipfn.training.config import TrainConfig


def _needs_xgboost():
    # graph SCM은 xgboost를 불러오지 못하면 sys.modules에 대체 모듈을 넣는다 (importorskip이 속는다)
    xgboost = pytest.importorskip("xgboost")
    if not hasattr(xgboost, "XGBModel"):
        pytest.skip("xgboost가 없다 (graph SCM의 대체 모듈만 있다)")


SMALL = dict(
    size_rule="train_fraction",
    min_rows=80,
    max_rows=80,
    max_features=8,
    max_cells=640,
)


def _prior(task="classification", seed=0, **options):
    return P.GraphSCMPrior(task, seed=seed, **SMALL, **options)


def _same(a, b):
    return torch.equal(torch.nan_to_num(a, nan=-7.0), torch.nan_to_num(b, nan=-7.0))


def test_defaults_build_the_same_parts_as_before():
    prior = _prior()
    generator = prior.pipeline.generator
    assert isinstance(generator, G.GraphSCM)  # Mixture로 감싸지 않는다
    assert generator.mixture_root_prob == 0.0
    assert not any(getattr(generator._config, flag) for flag in G.CORRECTED_FLAGS)
    # 같은 시드면 같은 데이터셋 (변형 옵션을 기본값으로 줘도 같다). TabICL은 전역 난수를 쓰므로
    # prior를 하나씩 만들어 다 뽑는다
    a = [prior.sample_batch(2) for _ in range(2)]
    explicit = _prior(tree_scm_prob=0.0, graph_scm_corrected=False, mixture_root_prob=0.0)
    b = [explicit.sample_batch(2) for _ in range(2)]
    for p, q in zip(a, b, strict=True):
        assert _same(p.x, q.x) and torch.equal(p.y, q.y)


def test_corrected_turns_on_the_tabicl_fix_flags():
    prior = _prior(graph_scm_corrected=True)
    config = prior.pipeline.generator._config
    assert all(getattr(config, flag) for flag in G.CORRECTED_FLAGS)
    batch = prior.sample_batch(2)
    assert batch.x.shape[:2] == (2, 80)


def test_mixture_roots_replace_only_the_root_points_and_restore_them():
    module = importlib.import_module("tabicl.prior.graph_lib._node_function")
    original = module.RandomPoints
    points = G.mixture_root_points_class()
    calls = []
    sample = points.sample

    def spy(self, n_batch, n):
        out = sample(self, n_batch, n)
        calls.append(out.shape)
        return out

    points.sample = spy
    try:
        for task in ("classification", "regression"):
            prior = _prior(task, mixture_root_prob=1.0)
            batch = prior.sample_batch(2)
            assert torch.isfinite(batch.y).all()
        assert calls and all(shape[0] == 80 for shape in calls)  # root 노드마다 행 수만큼
        assert module.RandomPoints is original  # 끝나면 되돌린다
        calls.clear()
        _prior(mixture_root_prob=0.0).sample_batch(2)
        assert not calls
    finally:
        points.sample = sample


def test_mixture_root_points_have_the_shape_of_tabicl_points():
    from tabicl.prior.graph_lib._base import Context

    np.random.seed(0)
    torch.manual_seed(0)
    points = G.mixture_root_points_class()(Context())
    out = points.sample(500, 4)
    assert out.shape == (500, 4) and torch.isfinite(out).all()


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_tree_scm_mixture_uses_the_tree_generator_with_its_probability(task):
    _needs_xgboost()
    from minipfn.priors.generators.tree_scm import TreeSCM

    prior = _prior(task, tree_scm_prob=1.0)
    generator = prior.pipeline.generator
    assert isinstance(generator, Mixture)
    assert isinstance(generator.generators["tree_scm"], TreeSCM)
    used = []
    tree = generator.generators["tree_scm"]
    original = tree._dataset
    tree._dataset = lambda *a: used.append(1) or original(*a)
    batch = prior.sample_batch(3)
    assert len(used) >= 3
    assert batch.x.shape[:2] == (3, 80) and torch.isfinite(batch.y).all()
    if task == "classification":
        for y in batch.y:
            assert y.min() >= 0 and y.max() < P.MAX_CLASSES
            assert len(torch.unique(y)) >= 2


def test_generator_choice_draws_tree_with_its_probability():
    choice = P.GeneratorChoice(None, 0.3)
    rng = Rng.from_seed(0)
    picks = [choice(rng, 10, 5)["generator"] for _ in range(4000)]
    assert set(picks) == {"graph_scm", "tree_scm"}
    assert abs(picks.count("tree_scm") / len(picks) - 0.3) < 0.03
    # 분류 클래스 수는 base가 정한다
    hp = P.GeneratorChoice(P.ClassCount(10), 0.5)(rng, 10, 5)
    assert set(hp) == {"num_classes", "generator"}


def test_tree_scm_filters_only_the_first_attempts(monkeypatch):
    _needs_xgboost()
    from minipfn.priors.generators import tree_scm as T

    np.random.seed(0)
    torch.manual_seed(0)
    tree = T.TreeSCM("regression", filter_unpredictable=True)
    calls = []
    monkeypatch.setattr(tree, "_should_filter", lambda *a, **k: calls.append(1) or True)
    x, y = tree._dataset(60, 4, {})
    assert len(calls) == T.FILTER_ATTEMPTS  # 그 뒤에는 받는다
    assert x.shape == (60, 4) and y.shape == (60,)


def test_tree_scm_needs_real_xgboost(monkeypatch):
    import sys
    import types

    from minipfn.priors.generators.tree_scm import TreeSCM

    monkeypatch.setitem(sys.modules, "xgboost", types.ModuleType("xgboost"))  # 대체 모듈
    with pytest.raises(ImportError, match="xgboost"):
        TreeSCM("classification")


def test_options_are_checked():
    with pytest.raises(ValueError, match="tree_scm_prob"):
        _prior(tree_scm_prob=1.5)
    with pytest.raises(ValueError, match="mixture_root_prob"):
        _prior(mixture_root_prob=-0.1)
    with pytest.raises(ValueError, match="tree_scm_prob"):
        TrainConfig(tree_scm_prob=2.0)


def test_training_prior_and_borders_get_the_options(monkeypatch):
    config = TrainConfig(
        task="regression",
        **SMALL,
        filter_unpredictable=True,
        graph_scm_corrected=True,
        mixture_root_prob=0.25,
    )
    prior = P.make_prior(config, seed=0)
    assert prior.pipeline.generator.mixture_root_prob == 0.25
    assert prior.pipeline.generator._config.use_corrected_cat_sizes
    seen = {}

    def capture(n_buckets, n_datasets, seed, **kwargs):
        seen.update(kwargs)
        return torch.zeros(n_buckets + 1)

    monkeypatch.setattr(P, "regression_borders", capture)
    P.make_borders(config, 10, seed=0)
    assert seen["graph_scm_corrected"] and seen["mixture_root_prob"] == 0.25
    assert seen["tree_scm_prob"] == 0.0 and seen["filter_unpredictable"]
    assert "binary_prob" not in seen  # 분류만 바꾸는 옵션


def test_regression_borders_run_with_the_variants():
    borders = P.regression_borders(
        20, 6, seed=1, graph_scm_corrected=True, mixture_root_prob=0.5, **SMALL
    )
    assert borders.shape == (21,) and (borders.diff() > 0).all()


def _train_script():
    path = Path(__file__).parents[1] / "scripts" / "tabpfn_v2" / "train.py"
    spec = importlib.util.spec_from_file_location("train_tabpfn_v2_script_variants", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_train_script_sets_the_prior_variants_seed_and_warmup_from_env(monkeypatch):
    for name in ("TABPFN_V2_ARCH", "TABPFN_V2_CONFIG_NAME", "TABPFN_V2_TASK"):
        monkeypatch.delenv(name, raising=False)
    args = ["--config-name", "stage1", "--arch", "v3.5", "--task", "multitask"]
    script = _train_script()
    base, _ = script.build_config(script.parse_args(args))
    env = {
        "TABPFN_V2_TREE_SCM_PROB": "0.3",
        "TABPFN_V2_MIXTURE_ROOT_PROB": "0.2",
        "TABPFN_V2_GRAPH_SCM_CORRECTED": "1",
        "TABPFN_V2_SEED": "1",
        "TABPFN_V2_WARMUP_STEPS": "200",
    }
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    script = _train_script()
    config, _ = script.build_config(script.parse_args(args))
    assert (config.tree_scm_prob, config.mixture_root_prob) == (0.3, 0.2)
    assert config.graph_scm_corrected and config.seed == 1 and config.warmup_steps == 200
    # 그 밖의 설정은 그대로
    changed = dict(tree_scm_prob=0.0, mixture_root_prob=0.0, graph_scm_corrected=False)
    assert replace(config, **changed, seed=base.seed, warmup_steps=base.warmup_steps) == base
