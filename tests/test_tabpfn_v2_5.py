"""TabPFN-2.5 구조 설정(thinking 행, 셀 3개 묶기, MLP encoder)과 robust 전처리 테스트."""

import numpy as np
import pytest
import torch
from sklearn.datasets import load_breast_cancer, load_diabetes

from minipfn.tabpfn_v2.inference import (
    MEMBER_KINDS_V2_5,
    RobustSoftClip,
    TabPFNv2Classifier,
    TabPFNv2Regressor,
)
from minipfn.tabpfn_v2.model import TabPFNv2, TabPFNv2Config
from minipfn.tabpfn_v2.train import ARCHITECTURES, TabPFNv2TrainConfig


def _tiny(n_out: int = 10, **kw) -> TabPFNv2:
    torch.manual_seed(0)
    model = TabPFNv2(TabPFNv2Config(emsize=16, nlayers=2, nhead=2, n_out=n_out, **kw))
    for p in model.parameters():  # 0으로 초기화한 출력층이 있으면 attention이 일을 안 한다
        torch.nn.init.normal_(p, std=0.3)
    return model.eval()


def test_default_config_keeps_v2_state_dict():
    # 2.5 설정이 꺼져 있으면 parameter 이름·모양이 v2와 같다 (v2 checkpoint를 그대로 읽는다)
    keys = {k: v.shape for k, v in TabPFNv2().state_dict().items()}
    assert "thinking_rows" not in keys
    assert keys["feature_embedder.weight"] == (192, 4)


def test_default_forward_is_unchanged_by_new_options():
    # thinking 행 0개·linear encoder는 이 변경 전 forward와 같은 계산이다: 같은 가중치면 같은 출력
    torch.manual_seed(0)
    a = TabPFNv2(TabPFNv2Config(emsize=16, nlayers=2, nhead=2))
    b = TabPFNv2(TabPFNv2Config(emsize=16, nlayers=2, nhead=2, n_thinking_rows=0))
    b.load_state_dict(a.state_dict())
    x, y = torch.randn(2, 20, 5), torch.randint(0, 2, (2, 12)).float()
    with torch.no_grad():
        assert torch.equal(a(x, y), b(x, y))


@pytest.mark.parametrize("encoder", ["linear", "mlp"])
@pytest.mark.parametrize("group", [2, 3])
def test_thinking_rows_keep_permutation_invariance_and_test_row_independence(encoder, group):
    model = _tiny(n_thinking_rows=4, encoder=encoder, features_per_group=group)
    x = torch.randn(1, 30, 7)
    y = torch.randint(0, 3, (1, 20)).float()
    with torch.no_grad():
        base = model(x, y)
        perm = torch.randperm(20)
        shuffled = model(torch.cat([x[:, perm], x[:, 20:]], dim=1), y[:, perm])
        one_by_one = torch.cat(
            [model(torch.cat([x[:, :20], x[:, j : j + 1]], dim=1), y) for j in range(20, 30)], dim=1
        )
    assert base.shape == (1, 10, 10)
    assert torch.allclose(base, shuffled, atol=1e-5)
    assert torch.allclose(base, one_by_one, atol=1e-5)


def test_thinking_rows_change_output_and_get_gradient():
    model = _tiny(n_thinking_rows=4)
    x, y = torch.randn(2, 20, 5), torch.randint(0, 2, (2, 12)).float()
    out = model(x, y)
    out.sum().backward()
    assert model.thinking_rows.grad is not None and model.thinking_rows.grad.abs().sum() > 0
    with torch.no_grad():
        model.thinking_rows.zero_()
        assert not torch.allclose(model(x, y), out)


def test_thinking_rows_work_with_a_single_train_row():
    model = _tiny(n_thinking_rows=4)
    with torch.no_grad():
        out = model(torch.randn(1, 5, 3), torch.zeros(1, 1))
    assert out.shape == (1, 4, 10) and torch.isfinite(out).all()


def test_bad_options_raise():
    with pytest.raises(ValueError):
        TabPFNv2Config(encoder="conv")
    with pytest.raises(ValueError):
        TabPFNv2Config(n_thinking_rows=-1)
    with pytest.raises(ValueError):
        TabPFNv2TrainConfig.paper("classification").with_architecture("v3")


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_v2_5_paper_architecture(task):
    config = TabPFNv2TrainConfig.paper(task).with_architecture("v2.5")
    expected = ARCHITECTURES["v2.5"][task]
    for key, value in expected.items():
        assert getattr(config, key) == value
    model_config = config.model_config()
    assert model_config.nlayers == (24 if task == "classification" else 18)
    assert model_config.features_per_group == 3 and model_config.n_thinking_rows == 64
    assert model_config.encoder == ("mlp" if task == "regression" else "linear")
    # 학습 설정은 v2 paper 그대로 (회귀 QK-clip 250 등)
    assert config.qk_clip == TabPFNv2TrainConfig.paper(task).qk_clip
    assert TabPFNv2TrainConfig.paper(task).with_architecture("v2") == TabPFNv2TrainConfig.paper(
        task
    )


def test_smoke_v2_5_stays_small():
    config = TabPFNv2TrainConfig.smoke("regression").with_architecture("v2.5")
    assert config.nlayers == 2 and config.n_thinking_rows == 4
    assert config.features_per_group == 3 and config.encoder == "mlp"


def test_robust_soft_clip():
    rng = np.random.default_rng(0)
    X = np.column_stack(
        [
            rng.normal(5, 2, 200),
            np.r_[np.zeros(190), rng.normal(0, 1, 10)],  # 사분위 폭 0 → 최대·최소로
            np.full(200, 3.0),  # 상수 → 0
            np.r_[np.full(200, np.nan)],  # 모두 결측
        ]
    )
    X[0, 0] = 1e6  # 극단값
    t = RobustSoftClip().fit(X[:, :3])
    out = t.transform(X[:, :3])
    assert np.isfinite(out).all()
    assert np.abs(out).max() <= 3.0  # smooth clipping은 ±3을 넘지 않는다
    assert abs(np.median(out[:, 0])) < 0.05
    assert np.all(out[:, 2] == 0)
    assert not np.all(out[:, 1] == 0)
    with_nan = RobustSoftClip().fit(X).transform(X)
    assert np.isnan(with_nan[:, 3]).all() and np.isfinite(with_nan[:, :3]).all()


def test_estimators_with_v2_5_model_and_robust_members():
    X, y = load_breast_cancer(return_X_y=True)
    X[::7, 3] = np.nan
    model = _tiny(n_thinking_rows=4, features_per_group=3)
    clf = TabPFNv2Classifier(model, member_kinds=MEMBER_KINDS_V2_5).fit(X[:100], y[:100])
    assert [p.member.kind for p in clf.preprocessors_] == ["quantile", "robust"] * 2
    proba = clf.predict_proba(X[100:130])
    assert proba.shape == (30, 2) and np.allclose(proba.sum(1), 1, atol=1e-5)

    Xr, yr = load_diabetes(return_X_y=True)
    reg_model = _tiny(n_out=20, n_thinking_rows=4, features_per_group=3, encoder="mlp")
    reg = TabPFNv2Regressor(reg_model, torch.linspace(-3, 3, 21), member_kinds=MEMBER_KINDS_V2_5)
    pred = reg.fit(Xr[:100], yr[:100]).predict(Xr[100:120])
    assert pred.shape == (20,) and np.isfinite(pred).all()


def test_default_member_kinds_are_v2():
    X, y = load_breast_cancer(return_X_y=True)
    clf = TabPFNv2Classifier(_tiny()).fit(X[:60], y[:60])
    assert [p.member.kind for p in clf.preprocessors_] == ["quantile", "power"] * 2
    with pytest.raises(ValueError):
        TabPFNv2Classifier(_tiny(), member_kinds=("quantile", "box-cox"))
