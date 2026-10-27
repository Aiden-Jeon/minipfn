import numpy as np
import torch

from minipfn.tabpfn.evaluate import (
    OpenMLDataset,
    evaluate_methods,
    generate_split,
    roc_auc,
    sklearn_baselines,
    tabpfn_predict_proba,
)
from minipfn.tabpfn.model import TabPFNTransformer


def _toy(n=300, n_classes=3, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 5)).astype(np.float32)
    y = np.digitize(x[:, 0], np.quantile(x[:, 0], np.linspace(0, 1, n_classes + 1)[1:-1]))
    return x, y


def test_split_is_half_half_for_small_datasets():
    x, y = _toy(n=300)
    x_tr, y_tr, x_te, y_te = generate_split(x, y, split_number=1)
    assert len(x_tr) == len(x_te) == 150
    assert np.array_equal(np.unique(y_tr), np.unique(y_te))


def test_split_caps_train_at_1000_rows():
    x, y = _toy(n=5000)
    x_tr, _, x_te, _ = generate_split(x, y, split_number=2)
    assert len(x_tr) == len(x_te) == 1000


def test_splits_differ_and_are_deterministic():
    x, y = _toy(n=400)
    a1, a2 = generate_split(x, y, 1)[0], generate_split(x, y, 1)[0]
    b = generate_split(x, y, 3)[0]
    assert np.array_equal(a1, a2) and not np.array_equal(a1, b)


def test_predict_proba_shape_and_normalization():
    torch.manual_seed(0)
    model = TabPFNTransformer(num_features=10, n_out=4, emsize=16, nhead=2, nhid=32, nlayers=1)
    x, y = _toy(n=120)
    x_tr, y_tr, x_te, _ = generate_split(x, y, 1)
    for n_ensemble in (1, 4):
        proba = tabpfn_predict_proba(model, x_tr, y_tr, x_te, n_ensemble=n_ensemble)
        assert proba.shape == (len(x_te), 3)
        assert np.allclose(proba.sum(1), 1, atol=1e-5)


def test_roc_auc_binary_and_multiclass():
    assert (
        roc_auc(np.array([0, 1, 0, 1]), np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.4, 0.6]]))
        == 1.0
    )
    proba = np.eye(3)[[0, 1, 2, 0, 1, 2]]
    assert roc_auc(np.array([0, 1, 2, 0, 1, 2]), proba) == 1.0


def test_evaluate_methods_rows():
    x, y = _toy(n=200)
    ds = OpenMLDataset(did=0, name="toy", x=x, y=y, categorical=[])

    def oracle(x_tr, y_tr, x_te):
        cuts = np.quantile(x_tr[:, 0], [1 / 3, 2 / 3])
        return np.eye(3)[np.digitize(x_te[:, 0], cuts)]

    rows = evaluate_methods([ds], {"oracle": oracle}, n_splits=2)
    assert len(rows) == 2 and all(r["roc_auc"] > 0.9 for r in rows)


def test_sklearn_baselines_accept_missing_values():
    # --datasets all에는 결측값이 있는 데이터셋(credit-approval 등)이 들어 있다.
    x, y = _toy(n=120)
    x[::7, 1] = np.nan
    for make in sklearn_baselines().values():
        proba = make().fit(x, y).predict_proba(x)
        assert proba.shape == (len(x), 3)
