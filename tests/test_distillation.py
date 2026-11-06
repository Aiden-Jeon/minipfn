"""TFM → GBDT 증류 (`minipfn.inference.distillation`) 테스트."""

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.datasets import load_breast_cancer, load_diabetes

from minipfn.inference.distillation import (
    METHODS,
    DistilledClassifier,
    DistilledRegressor,
    cutmix,
    resolve_method,
)
from minipfn.models.cell_transformer import CellTransformer, CellTransformerConfig
from minipfn.recipes.tabpfn_v2 import TabPFNv2Classifier


class FakeTeacher:
    """context와 query를 기록하는 teacher. 확률은 첫 열의 sigmoid (클래스 0·1)."""

    def __init__(self, calls: list) -> None:
        self.calls = calls
        self.categorical_features_indices = None

    def fit(self, X, y):
        self.X_ = np.asarray(X, dtype=float)
        self.classes_ = np.unique(y)
        self.calls.append({"context": self.X_, "query": []})
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        self.calls[-1]["query"].append(X)
        p = 1 / (1 + np.exp(-X[:, 0]))
        proba = np.stack([1 - p, p], axis=1)
        return proba[:, : len(self.classes_)]

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        self.calls[-1]["query"].append(X)
        return 2 * X[:, 0]


def _binary(n: int = 120, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    y = (X[:, 0] + 0.3 * rng.normal(size=n) > 0).astype(int)
    return X, y


def test_cutmix_takes_each_feature_from_one_of_two_rows():
    rng = np.random.default_rng(0)
    X = np.arange(40, dtype=float).reshape(10, 4) * 10 + np.arange(4)  # 열마다 끝자리가 열 번호
    out = cutmix(X, 50, rng)
    assert out.shape == (50, 4)
    for row in out:
        for j, value in enumerate(row):
            assert value in X[:, j]  # 같은 열의 학습 행 값
    assert not all((out[i] == X).all(axis=1).any() for i in range(len(out)))  # 섞인 행이 있다


def test_resolve_method_overrides_and_rejects_bad_values():
    assert resolve_method("oof") == METHODS["oof"]
    assert resolve_method("oof", synthetic_ratio=2.0, alpha=None)["synthetic_ratio"] == 2.0
    with pytest.raises(ValueError):
        resolve_method("nope")
    with pytest.raises(ValueError):
        resolve_method("oof", alpha=1.5)
    with pytest.raises(ValueError):
        resolve_method("oof", teacher_labels="half")


def test_oof_teacher_never_sees_the_row_it_labels():
    X, y = _binary()
    calls = []
    clf = DistilledClassifier(FakeTeacher(calls), method="oof", student="hist").fit(X, y)
    assert len(calls) == 5  # fold마다 teacher 하나, synthetic 행은 없다
    labelled = []
    for call in calls:
        query = np.concatenate(call["query"])
        context = {tuple(r) for r in call["context"]}
        assert not any(tuple(r) in context for r in query)
        labelled.append(query)
    assert len(np.concatenate(labelled)) == len(X)
    assert clf.fit_record_["synthetic_rows"] == 0 and clf.fit_record_["alpha"] == 0.7


def test_synthetic_teacher_uses_all_rows_as_context_and_labels_batches():
    X, y = _binary(n=60)
    calls = []
    clf = DistilledClassifier(
        FakeTeacher(calls), method="synthetic", student="hist", teacher_batch_rows=50
    ).fit(X, y)
    assert len(calls) == 1
    assert np.allclose(calls[0]["context"], X)
    sizes = [len(q) for q in calls[0]["query"]]
    assert sizes == [50, 50, 50, 30]  # 학습 행 60 + synthetic 120을 50행씩
    assert np.allclose(np.concatenate(calls[0]["query"])[:60], X)  # 학습 행 자체도 묻는다
    assert clf.fit_record_["synthetic_rows"] == 120


def test_labels_method_does_not_call_the_teacher():
    X, y = _binary()
    calls = []
    clf = DistilledClassifier(FakeTeacher(calls), method="labels", student="hist").fit(X, y)
    assert calls == []
    assert clf.predict_proba(X).shape == (len(X), 2)


def test_soft_targets_become_weighted_class_copies():
    X = np.arange(6, dtype=float).reshape(3, 2)
    targets = np.array([[0.2, 0.8], [1.0, 0.0], [0.5, 0.5]])
    weights = np.array([1.0, 2.0, 0.5])
    rows, classes, w = DistilledClassifier._expand(X, targets, weights, np.arange(3))
    assert classes.tolist() == [0, 1, 0, 0, 1]  # 확률 0인 복제는 뺀다
    assert np.allclose(w, [0.2, 0.8, 2.0, 0.25, 0.25])
    assert np.allclose(rows, X[[0, 0, 1, 2, 2]])


def test_classifier_keeps_string_labels_and_categorical_columns():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data[:, :5], columns=[f"c{i}" for i in range(5)])
    X["color"] = np.random.default_rng(0).choice(["red", "green", None], size=len(X))
    y = np.where(data.target == 1, "benign", "malignant")
    calls = []
    clf = DistilledClassifier(FakeTeacher(calls), method="synthetic", student="hist").fit(X, y)
    assert calls[0]["context"].shape[1] == 6
    assert clf.cat_ix_ == [5]
    proba = clf.predict_proba(X.iloc[:20])
    assert proba.shape == (20, 2) and np.allclose(proba.sum(axis=1), 1)
    assert set(clf.predict(X.iloc[:20])) <= {"benign", "malignant"}
    assert list(clf.classes_) == ["benign", "malignant"]


def test_rare_class_stays_in_student_training_rows():
    X, y = _binary(n=60)
    y = y.copy()
    y[7] = 2  # 행 하나뿐인 클래스. 검증 행으로 뽑혀도 학습 행으로 옮긴다
    for seed in range(5):
        clf = DistilledClassifier(None, method="labels", student="hist", random_state=seed)
        assert clf.fit(X, y).predict_proba(X).shape == (len(X), 3)


def test_regressor_learns_the_teacher_mean():
    X, _ = _binary(n=200)
    y = np.random.default_rng(1).normal(size=len(X))  # 정답은 잡음, teacher 평균은 2·x0
    reg = DistilledRegressor(FakeTeacher([]), method="synthetic", student="hist").fit(X, y)
    pred = reg.predict(X)
    assert np.corrcoef(pred, 2 * X[:, 0])[0, 1] > 0.95
    labels = DistilledRegressor(None, method="labels", student="hist").fit(X, y).predict(X)
    assert np.corrcoef(labels, 2 * X[:, 0])[0, 1] < 0.5


def test_regressor_with_real_ensemble_teacher():
    from minipfn.recipes.tabpfn_v2 import TabPFNv2Regressor

    data = load_diabetes()
    X, y = data.data[:150], data.target[:150]
    teacher = TabPFNv2Regressor(_tiny_model(n_out=8), torch.linspace(-3, 3, 9), n_estimators=2)
    reg = DistilledRegressor(teacher, method="oof", student="hist").fit(X, y)
    assert reg.predict(data.data[150:170]).shape == (20,)


def _tiny_model(n_out: int = 10) -> CellTransformer:
    torch.manual_seed(0)
    model = CellTransformer(CellTransformerConfig(emsize=16, nlayers=2, nhead=2, n_out=n_out))
    for p in model.parameters():
        torch.nn.init.normal_(p, std=0.3)
    return model.eval()


def test_classifier_with_real_ensemble_teacher():
    data = load_breast_cancer()
    X, y = data.data[:150, :6], data.target[:150]
    teacher = TabPFNv2Classifier(_tiny_model(), n_estimators=2)
    clf = DistilledClassifier(teacher, method="oof", synthetic_ratio=1.0, student="hist")
    proba = clf.fit(X, y).predict_proba(data.data[150:170, :6])
    assert proba.shape == (20, 2) and np.allclose(proba.sum(axis=1), 1)


@pytest.mark.parametrize("student", ["xgboost", "lightgbm"])
def test_gbdt_students(student):
    try:
        __import__(student)
    except Exception as e:  # macOS에서 libomp가 없으면 불러오지 못한다
        pytest.skip(f"{student}를 불러올 수 없다: {e}")
    data = load_breast_cancer()
    X = pd.DataFrame(data.data[:, :5])
    X["cat"] = np.random.default_rng(0).choice(["a", "b", "c", None], size=len(X))
    y = data.target
    clf = DistilledClassifier(FakeTeacher([]), method="synthetic", student=student).fit(X, y)
    assert clf.predict_proba(X.iloc[:10]).shape == (10, 2)
    assert clf.fit_record_["n_trees"] >= 1
    reg = DistilledRegressor(FakeTeacher([]), method="oof", student=student).fit(X, y * 1.0)
    assert reg.predict(X.iloc[:10]).shape == (10,)
