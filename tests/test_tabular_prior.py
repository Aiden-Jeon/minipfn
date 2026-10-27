import numpy as np

from minipfn.tabular_prior import sample_random_scm, sample_scm_dataset, to_classes


def test_to_classes_range_and_count():
    rng = np.random.default_rng(0)
    y_cont = rng.standard_normal(500)
    y = to_classes(y_cont, max_classes=4, rng=rng)
    assert y.shape == (500,)
    assert 2 <= len(np.unique(y)) <= 4
    assert y.min() >= 0 and y.max() < 4


def test_to_classes_keeps_interval_structure():
    # 같은 구간에 든 값은 같은 클래스여야 한다 (번호만 섞인다).
    rng = np.random.default_rng(1)
    y_cont = np.linspace(-1, 1, 200)
    y = to_classes(y_cont, max_classes=3, rng=rng)
    n_changes = np.count_nonzero(np.diff(y))
    assert n_changes == len(np.unique(y)) - 1


def test_sample_random_scm_has_enough_nodes():
    rng = np.random.default_rng(0)
    for _ in range(50):
        scm = sample_random_scm(n_features=10, rng=rng)
        n_nodes = (scm["n_layers"] + 1) * scm["hidden"]
        nodes = np.append(scm["feature_nodes"], scm["target_node"])
        assert len(np.unique(nodes)) == 11
        assert nodes.max() < n_nodes


def test_sample_scm_dataset_shapes_and_standardization():
    rng = np.random.default_rng(0)
    scm = sample_random_scm(n_features=5, rng=rng)
    x, y = sample_scm_dataset(scm, n_samples=300, max_classes=4, rng=rng)
    assert x.shape == (300, 5)
    assert y.shape == (300,)
    assert np.all(np.isfinite(x))
    assert np.allclose(x.mean(0), 0, atol=1e-6)


def test_sample_scm_dataset_is_reproducible():
    def draw(seed):
        rng = np.random.default_rng(seed)
        scm = sample_random_scm(n_features=3, rng=rng)
        return sample_scm_dataset(scm, n_samples=50, max_classes=3, rng=rng)

    (x1, y1), (x2, y2) = draw(7), draw(7)
    assert np.array_equal(x1, x2) and np.array_equal(y1, y2)
