import numpy as np
import pytest
import torch

from minipfn.tabpfn.inference import TabPFNClassifier, _Preprocessor, tabpfn_predict_proba
from minipfn.tabpfn.model import TabPFNTransformer


def _tiny_model():
    torch.manual_seed(0)
    model = TabPFNTransformer(num_features=10, n_out=4, emsize=16, nhead=2, nhid=32, nlayers=2)
    # 0 초기화된 층은 처음에 항등이라 캐시 경로 차이가 드러나지 않으므로 무작위로 다시 채운다.
    for p in model.parameters():
        torch.nn.init.normal_(p, std=0.2)
    return model.eval()


def _toy(n=90, n_classes=3, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.lognormal(size=(n, 6)).astype(np.float32)  # 치우친 분포라 power transform이 의미 있다
    x[:, 4] = 1.0  # 상수 열은 지운다
    y = np.digitize(x[:, 0], np.quantile(x[:, 0], np.linspace(0, 1, n_classes + 1)[1:-1]))
    return x[:60], y[:60], x[60:]


def test_fit_modes_give_same_proba():
    model = _tiny_model()
    x_tr, y_tr, x_te = _toy()
    probas = [
        TabPFNClassifier(model, n_ensemble=8, fit_mode=mode, batch_size=3)
        .fit(x_tr, y_tr)
        .predict_proba(x_te)
        for mode in ("fit_preprocessors", "fit_with_cache")
    ]
    assert probas[0].shape == (len(x_te), 3)
    assert np.allclose(probas[0].sum(1), 1, atol=1e-5)
    assert np.allclose(probas[0], probas[1], atol=1e-5)


def test_function_matches_classifier():
    model = _tiny_model()
    x_tr, y_tr, x_te = _toy()
    clf = TabPFNClassifier(model, n_ensemble=4, seed=3).fit(x_tr, y_tr)
    proba = tabpfn_predict_proba(model, x_tr, y_tr, x_te, n_ensemble=4, seed=3)
    assert np.array_equal(clf.predict_proba(x_te), proba)


@pytest.mark.parametrize("fit_mode", ["fit_preprocessors", "fit_with_cache"])
def test_one_row_at_a_time_matches_batch(fit_mode):
    # 실시간으로 한 행씩 예측해도 한 번에 예측한 값과 같다.
    model = _tiny_model()
    x_tr, y_tr, x_te = _toy()
    clf = TabPFNClassifier(model, n_ensemble=4, fit_mode=fit_mode).fit(x_tr, y_tr)
    batch = clf.predict_proba(x_te)
    rows = np.concatenate([clf.predict_proba(x_te[i : i + 1]) for i in range(len(x_te))])
    assert np.allclose(batch, rows, atol=1e-5)


def test_preprocessor_transforms_train_rows_like_fit():
    x_tr, _, _ = _toy()
    for transform in ("none", "power_all"):
        prep = _Preprocessor(transform, max_features=10)
        fitted = prep.fit_transform(torch.tensor(x_tr))
        assert fitted.shape == (len(x_tr), 5)  # 상수 열 하나가 빠진다
        assert torch.allclose(prep.transform_rows(torch.tensor(x_tr)), fitted, atol=1e-6)


def test_string_labels_and_predict():
    model = _tiny_model()
    x_tr, y_tr, x_te = _toy()
    names = np.array(["b", "c", "a"])[y_tr]
    clf = TabPFNClassifier(model, n_ensemble=2).fit(x_tr, names)
    assert list(clf.classes_) == ["a", "b", "c"]
    proba = clf.predict_proba(x_te)
    assert np.array_equal(clf.predict(x_te), clf.classes_[proba.argmax(1)])


def test_rejects_unknown_fit_mode_and_too_many_classes():
    model = _tiny_model()
    with pytest.raises(ValueError, match="fit_mode"):
        TabPFNClassifier(model, fit_mode="low_memory")
    x_tr, _, _ = _toy()
    with pytest.raises(ValueError, match="클래스"):
        TabPFNClassifier(model).fit(x_tr, np.arange(len(x_tr)) % 5)


def test_power_transform_overflow_on_test_rows_stays_finite():
    # v1은 학습·테스트 행을 함께 변환하다 넘치면 그 열을 변환하지 않았다. 이제는 학습 행으로만
    # 판단하므로, 테스트 행에서 넘친 값이 NaN 확률로 번지지 않아야 한다.
    model = _tiny_model()
    x_tr, y_tr, x_te = _toy()
    clf = TabPFNClassifier(model, n_ensemble=4).fit(x_tr, y_tr)
    for pt, _ in clf.preprocessors_["power_all"].power_.values():
        pt.lambdas_[:] = 30.0  # 101**30은 float32에서 넘친다
    x_te[0] = 1e4
    rows = clf.preprocessors_["power_all"].transform_rows(torch.tensor(x_te))
    assert torch.isfinite(rows).all()
    assert np.isfinite(clf.predict_proba(x_te)).all()


def test_rejects_feature_count_different_from_fit():
    model = _tiny_model()
    x_tr, y_tr, x_te = _toy()
    clf = TabPFNClassifier(model, n_ensemble=2).fit(x_tr, y_tr)
    with pytest.raises(ValueError, match="feature"):
        clf.predict_proba(x_te[:, :1])
    with pytest.raises(ValueError, match="2차원"):
        clf.predict_proba(x_te[0])


def test_uses_model_device_by_default_and_drops_rows_with_cache():
    model = _tiny_model()
    x_tr, y_tr, _ = _toy()
    clf = TabPFNClassifier(model, n_ensemble=2).fit(x_tr, y_tr)
    assert clf.device_ == next(model.parameters()).device
    assert clf.x_train_.device == clf.device_
    cached = TabPFNClassifier(model, n_ensemble=2, fit_mode="fit_with_cache").fit(x_tr, y_tr)
    assert cached.x_train_ is None and cached.y_train_ is None
