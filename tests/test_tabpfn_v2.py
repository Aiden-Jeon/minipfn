"""TabPFN v2 구조·추론 테스트."""

import numpy as np
import pandas as pd
import torch
from sklearn.datasets import load_breast_cancer, load_diabetes

from minipfn.tabpfn_v2.inference import TabPFNv2Classifier, TabPFNv2Regressor, _probs_on_borders
from minipfn.tabpfn_v2.model import TabPFNv2, TabPFNv2Config, load_checkpoint, save_checkpoint


def _tiny_model(n_out: int = 10) -> TabPFNv2:
    torch.manual_seed(0)
    model = TabPFNv2(TabPFNv2Config(emsize=16, nlayers=2, nhead=2, n_out=n_out))
    # 0으로 초기화한 출력층이 있으면 attention이 아무 일도 하지 않으므로 무작위로 채운다
    for p in model.parameters():
        torch.nn.init.normal_(p, std=0.3)
    return model.eval()


def test_default_size_is_paper_model():
    model = TabPFNv2()
    assert model.config.emsize == 192 and model.config.nlayers == 12
    assert 7_000_000 < model.num_parameters() < 7_500_000  # 논문 기본 모델은 약 7.2M


def test_train_row_order_and_test_row_independence():
    model = _tiny_model()
    x = torch.randn(1, 30, 5)
    y = torch.randint(0, 3, (1, 20)).float()
    with torch.no_grad():
        base = model(x, y)
        perm = torch.randperm(20)
        shuffled = model(torch.cat([x[:, perm], x[:, 20:]], dim=1), y[:, perm])
        one_by_one = torch.cat(
            [model(torch.cat([x[:, :20], x[:, j : j + 1]], dim=1), y) for j in range(20, 30)], dim=1
        )
    assert torch.allclose(base, shuffled, atol=1e-5)
    assert torch.allclose(base, one_by_one, atol=1e-5)


def test_test_rows_do_not_change_when_other_test_rows_change():
    model = _tiny_model()
    x = torch.randn(1, 25, 4)
    y = torch.randint(0, 2, (1, 20)).float()
    other = x.clone()
    other[:, 21:] = torch.randn(1, 4, 4)  # 첫 테스트 행만 그대로
    with torch.no_grad():
        assert torch.allclose(model(x, y)[:, 0], model(other, y)[:, 0], atol=1e-5)


def test_missing_values_constant_columns_and_odd_feature_counts():
    model = _tiny_model()
    x = torch.randn(2, 25, 7)
    x[x > 1.5] = float("nan")
    x[:, :, 3] = 5.0  # 상수 열
    y = torch.randint(0, 2, (2, 15)).float()
    with torch.no_grad():
        out = model(x, y)
    assert out.shape == (2, 10, 10)
    assert torch.isfinite(out).all()


def test_checkpoint_roundtrip(tmp_path):
    model = _tiny_model(n_out=8)
    borders = torch.linspace(-3, 3, 9)
    save_checkpoint(tmp_path / "m.pt", model, borders)
    loaded, loaded_borders = load_checkpoint(tmp_path / "m.pt")
    x, y = torch.randn(1, 12, 3), torch.randn(1, 8)
    with torch.no_grad():
        assert torch.allclose(model(x, y), loaded(x, y))
    assert torch.equal(borders, loaded_borders)


def test_classifier_runs_and_handles_categorical_dataframe():
    x, y = load_breast_cancer(return_X_y=True)
    clf = TabPFNv2Classifier(_tiny_model()).fit(x[:80], y[:80])
    proba = clf.predict_proba(x[80:100])
    assert proba.shape == (20, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)

    g = np.random.default_rng(0)
    df = pd.DataFrame({"a": g.normal(size=60), "color": g.choice(["red", "blue", None], 60)})
    labels = g.choice(["cat", "dog", "fish"], 60)
    clf = TabPFNv2Classifier(_tiny_model()).fit(df[:40], labels[:40])
    assert set(clf.predict(df[40:])) <= {"cat", "dog", "fish"}


def test_regressor_runs():
    x, y = load_diabetes(return_X_y=True)
    borders = torch.linspace(-4, 4, 33)
    reg = TabPFNv2Regressor(_tiny_model(n_out=32), borders).fit(x[:80], y[:80])
    pred = reg.predict(x[80:100])
    assert pred.shape == (20,)
    assert np.isfinite(pred).all()


def test_probs_on_same_borders_are_unchanged():
    borders = torch.linspace(-2, 2, 11)
    probs = torch.softmax(torch.randn(3, 10), dim=-1)
    assert torch.allclose(_probs_on_borders(probs, borders, borders), probs, atol=1e-6)


def test_nearly_constant_training_column_does_not_blow_up():
    model = _tiny_model()
    x = torch.randn(1, 30, 3)
    x[:, :20, 1] = 1.0 + 1e-7 * torch.randn(20)  # 학습 행에서는 거의 상수
    x[:, 20:, 1] = 50.0  # 검증 행에서는 크게 다르다
    y = torch.randint(0, 2, (1, 20)).float()
    out = model(x, y)
    out.sum().backward()
    assert torch.isfinite(out).all()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
