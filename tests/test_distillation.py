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


def _numbers(X) -> np.ndarray:
    """teacher가 받은 표의 숫자 열 (범주 문자열은 NaN)."""
    return pd.DataFrame(X).apply(pd.to_numeric, errors="coerce").to_numpy(float)


class FakeTeacher:
    """context와 query를 기록하는 teacher. 확률은 첫 열의 sigmoid (클래스 0·1)."""

    def __init__(self, calls: list) -> None:
        self.calls = calls
        self.categorical_features_indices = None

    def fit(self, X, y):
        self.frame_ = pd.DataFrame(X)
        self.X_ = _numbers(X)
        self.classes_ = np.unique(y)
        self.calls.append({"context": self.X_, "query": []})
        return self

    def predict_proba(self, X):
        X = _numbers(X)
        self.calls[-1]["query"].append(X)
        p = 1 / (1 + np.exp(-X[:, 0]))
        proba = np.stack([1 - p, p], axis=1)
        return proba[:, : len(self.classes_)]

    def predict(self, X):
        X = _numbers(X)
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
    proba = clf.predict_proba(X.iloc[:10])
    assert proba.shape == (10, 2) and np.allclose(proba.sum(axis=1), 1)
    assert clf.student_.task == "soft_binary"  # 이진 soft label은 행을 복제하지 않는다
    assert clf.fit_record_["n_trees"] >= 1
    y3 = np.digitize(data.data[:, 0], np.quantile(data.data[:, 0], [0.33, 0.66]))
    multi = DistilledClassifier(None, method="labels", student=student).fit(X, y3)
    assert multi.student_.task == "classification"
    assert multi.predict_proba(X.iloc[:10]).shape == (10, 3)
    reg = DistilledRegressor(FakeTeacher([]), method="oof", student=student).fit(X, y * 1.0)
    assert reg.predict(X.iloc[:10]).shape == (10,)


class RecordingTeacher(FakeTeacher):
    """받은 context 표를 그대로 기록하는 teacher."""

    def fit(self, X, y):
        super().fit(X, y)
        self.calls[-1]["frame"] = pd.DataFrame(X).copy()
        self.calls[-1]["y"] = np.asarray(y)
        return self


def test_teacher_gets_original_category_values_not_float_codes():
    # 범주 11개: 번호를 문자열로 넘기면 "10.0"이 "2.0"보다 앞에 와 teacher의 번호 순서가 바뀐다
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"x": rng.normal(size=60), "c": rng.choice([f"v{i}" for i in range(11)], 60)})
    X.loc[3, "c"] = None
    calls = []
    DistilledClassifier(RecordingTeacher(calls), method="synthetic", student="hist").fit(
        X, (X["x"] > 0).astype(int)
    )
    frame = calls[0]["frame"]
    assert frame[1].tolist()[:3] == X["c"].tolist()[:3]
    assert pd.isna(frame[1].iloc[3])
    assert set(frame[1].dropna()) == set(X["c"].dropna())


def test_category_count_does_not_include_missing():
    X = pd.DataFrame({"c": ["a", "b", None, "a"] * 10, "x": np.arange(40.0)})
    clf = DistilledClassifier(None, method="labels", student="hist").fit(X, np.arange(40) % 2)
    assert clf.n_categories_ == [2]


def test_oof_works_on_tiny_tables_and_rejects_one_row():
    X, y = _binary(n=4)
    y = np.array([0, 1, 0, 1])
    reg = DistilledRegressor(FakeTeacher([]), method="oof", student="hist")
    assert reg.fit(X, y * 1.0).predict(X).shape == (4,)
    with pytest.raises(ValueError, match="2개 이상"):
        DistilledRegressor(FakeTeacher([]), method="oof", student="hist").fit(X[:1], [1.0])


def test_single_class_fold_context_skips_the_teacher():
    X, _ = _binary(n=40)
    y = np.zeros(40, dtype=int)
    y[5] = 1  # 행 하나뿐인 클래스: 그 행을 예측하는 fold의 context는 클래스 0뿐이다
    calls = []
    clf = DistilledClassifier(RecordingTeacher(calls), method="oof", student="hist").fit(X, y)
    assert len(calls) == 4  # 5 fold 가운데 한 fold는 teacher를 부르지 않는다
    assert all(len(np.unique(c["y"])) == 2 for c in calls)
    assert clf.predict_proba(X).shape == (40, 2)


def test_class_copies_drop_tiny_targets_but_keep_the_largest():
    targets = np.array([[0.9995, 0.0004, 0.0001], [0.0002, 0.0003, 0.0004], [0.5, 0.5, 0.0]])
    kept = DistilledClassifier._copies(targets)
    assert kept.tolist() == [[True, False, False], [False, False, True], [True, True, False]]


def test_multiclass_hist_student():
    X, _ = _binary(n=150)
    y = np.digitize(X[:, 0], [-0.5, 0.5])  # 클래스 3개
    clf = DistilledClassifier(None, method="labels", student="hist").fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape == (150, 3) and (clf.predict(X) == y).mean() > 0.8


def _eval_script():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).parents[1] / "scripts" / "tabpfn_v2" / "eval.py"
    spec = importlib.util.spec_from_file_location("eval_tabpfn_v2_distill", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_recorded_estimator_writes_after_fit_or_after_the_latency_probe(tmp_path):
    import copy
    import json

    script = _eval_script()
    X, y = _binary(n=60)
    path = tmp_path / "fits.jsonl"
    inner = DistilledClassifier(FakeTeacher([]), method="oof", student="hist")
    record = {"suite": "tabarena-lite", "task": "classification", "dataset": None}
    est = script.RecordedEstimator(inner, record, path, latency=False)
    assert est.fit(X, y) is est
    assert est.classes_.tolist() == [0, 1] and est.predict_proba(X).shape == (60, 2)
    first = json.loads(path.read_text())
    assert first["suite"] == "tabarena-lite" and first["rows"] == 60 and first["method"] == "oof"
    assert "latency_ms" not in first

    timed = script.RecordedEstimator(inner, record, path, latency=True).fit(X, y)
    assert len(path.read_text().splitlines()) == 1  # 한 행 예측 시간을 잰 뒤에 쓴다
    timed.after_predict(pd.DataFrame(X[:1]))
    last = json.loads(path.read_text().splitlines()[-1])
    assert last["latency_ms"] > 0
    assert copy.deepcopy(timed).classes_.tolist() == [0, 1]  # 복사해도 감싼 estimator를 쓴다
    assert copy.copy(timed).estimator is inner


def test_eval_rejects_distilled_regression_uncertainty_and_amlb_latency():
    script = _eval_script()
    with pytest.raises(SystemExit, match="uncertainty"):
        script.main(["--suite", "regression", "--distill", "oof", "--uncertainty"])
    with pytest.raises(SystemExit, match="latency"):
        script.main(["--suite", "classification", "--latency"])
    with pytest.raises(SystemExit, match="test-time"):
        script.main(["--suite", "tabarena-lite", "--distill", "oof", "--test-time", "greedy"])


def test_fit_records_are_per_suite_and_cleared_each_run(tmp_path):
    script = _eval_script()
    args = script.parse_args(["--suite", "tabarena-lite", "--out-dir", str(tmp_path)])
    path = script.fits_path(args)
    assert path.name == "distill_tabarena-lite.jsonl"
    path.write_text("old\n")
    with pytest.raises(SystemExit):  # 체크포인트가 없어 멈추기 전에 예전 기록을 지운다
        script.main(["--suite", "tabarena-lite", "--out-dir", str(tmp_path)])
    assert not path.exists()
