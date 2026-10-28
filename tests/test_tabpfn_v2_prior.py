"""TabPFN v2 학습 데이터(TabICL prior + 논문 크기 규칙) 테스트."""

import numpy as np
import torch

from minipfn.tabpfn_v2 import prior as P


def test_table_size_follows_paper_rules():
    rng = np.random.default_rng(0)
    sizes = np.array([P.sample_table_size(rng) for _ in range(5000)])
    n_train, n_features = sizes[:, 0], sizes[:, 1]
    assert n_train.min() >= 1 and n_train.max() <= P.MAX_TRAIN_ROWS
    assert n_features.min() >= 1 and n_features.max() <= P.MAX_FEATURES
    assert ((n_train + P.N_EVAL_ROWS) * n_features < P.MAX_CELLS).all()
    assert 10 < np.median(n_features) < 20  # Beta(0.95, 8)을 160으로 늘리면 중앙값은 13 근처


def test_classification_batch_shapes_and_label_split():
    prior = P.TabPFNv2Prior("classification", seed=0, max_train_rows=60, n_eval_rows=20)
    batch = prior.sample_batch(3, size=(50, 5))
    assert batch.x.shape == (3, 70, 5)
    assert batch.y.shape == (3, 70)
    assert batch.n_train == 50
    for y in batch.y:
        train, rest = set(y[:50].tolist()), set(y[50:].tolist())
        assert rest <= train and len(train) >= 2  # 가린 행의 클래스는 모두 학습 행에 있다
        assert max(train) < P.MAX_CLASSES


def test_missing_values_are_added_only_when_requested():
    clean = P.TabPFNv2Prior("regression", seed=1, missing_dataset_prob=0.0, n_eval_rows=10)
    assert not torch.isnan(clean.sample_batch(2, size=(30, 4)).x).any()
    noisy = P.TabPFNv2Prior("regression", seed=1, missing_dataset_prob=1.0, max_missing_rate=0.5)
    x = noisy.sample_batch(4, size=(30, 4)).x
    assert torch.isnan(x).any()


def test_regression_targets_are_standardized():
    prior = P.TabPFNv2Prior("regression", seed=2, n_eval_rows=50)
    y = prior.sample_batch(2, size=(200, 3)).y
    assert torch.allclose(y.mean(dim=1), torch.zeros(2), atol=1e-3)
    assert torch.allclose(y.std(dim=1, correction=0), torch.ones(2), atol=1e-2)


def test_regression_borders_are_increasing():
    borders = P.regression_borders(n_buckets=100, n_datasets=20, seed=0)
    assert borders.shape == (101,)
    assert (borders[1:] > borders[:-1]).all()


def test_classification_with_very_few_training_rows_does_not_hang():
    prior = P.TabPFNv2Prior("classification", seed=3, n_eval_rows=30, missing_dataset_prob=0.0)
    for n_train in [1, 2, 3]:  # 학습 행 1개는 2개로 올라간다
        batch = prior.sample_batch(4, size=(n_train, 3))
        for y in batch.y:
            train = set(y[: batch.n_train].tolist())
            assert len(train) >= 2 and set(y[batch.n_train :].tolist()) <= train


def test_cell_budget_uses_the_capped_feature_count():
    rng = np.random.default_rng(0)
    sizes = np.array([P.sample_table_size(rng, 64, 16, 8) for _ in range(2000)])
    assert sizes[:, 1].max() <= 8 and sizes[:, 0].max() <= 64
    # feature 8개면 셀 제한(75,000)에 걸리지 않으므로 학습 행이 64개까지 고르게 나온다
    assert (sizes[:, 0] > 48).mean() > 0.2
