"""TabPFN v2 구조·추론 테스트."""

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.datasets import load_breast_cancer, load_diabetes

from minipfn.inference.ensemble import _probs_on_borders
from minipfn.models import load_checkpoint, save_checkpoint
from minipfn.models.cell_transformer import CellTransformer, CellTransformerConfig
from minipfn.recipes.tabpfn_v2 import TabPFNv2Classifier, TabPFNv2Regressor


def _tiny_model(n_out: int = 10) -> CellTransformer:
    torch.manual_seed(0)
    model = CellTransformer(CellTransformerConfig(emsize=16, nlayers=2, nhead=2, n_out=n_out))
    # 0으로 초기화한 출력층이 있으면 attention이 아무 일도 하지 않으므로 무작위로 채운다
    for p in model.parameters():
        torch.nn.init.normal_(p, std=0.3)
    return model.eval()


def test_default_size_is_paper_model():
    model = CellTransformer()
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


def test_regressor_tail_bucket_uses_training_target_range():
    from minipfn.inference.ensemble import _bucket_values

    # 맨 바깥 구간이 넓은 경계: half-normal 꼬리 평균이면 z 30 근처가 된다
    borders = torch.tensor([-30.0, -4.0, -1.0, 1.0, 4.0, 30.0])
    values = _bucket_values(borders, np.array([-1.0, 0.0, 6.0]))
    assert values[0] == -4.0  # 학습 target이 안쪽 경계 안이면 안쪽 경계
    assert values[-1] == 6.0  # 안쪽 경계를 넘으면 학습 target 최댓값
    assert np.allclose(values[1:-1], [-2.5, 0.0, 2.5])
    assert _bucket_values(borders, np.array([0.0, 60.0]))[-1] == 30.0  # 맨 바깥 경계 안으로


def test_clipped_inverse_stays_in_training_range_on_both_sides():
    from sklearn.preprocessing import PowerTransformer

    from minipfn.inference.ensemble import _clipped_inverse

    rng = np.random.default_rng(0)
    # λ > 2: 왼쪽 끝에 몰린 값 + 오른쪽 덩어리 (역변환이 왼쪽에서 끝난다)
    # λ < 0: 오른쪽으로 긴 꼬리 (역변환이 오른쪽에서 끝난다)
    for z in [
        np.concatenate([np.full(50, -4.5), rng.normal(0.5, 0.3, 500)]),
        rng.lognormal(0, 1.5, 2000),
    ]:
        z = (z - z.mean()) / z.std()
        transform = PowerTransformer(method="yeo-johnson", standardize=True)
        u = transform.fit_transform(z.reshape(-1, 1)).ravel()
        assert transform.lambdas_[0] > 2 or transform.lambdas_[0] < 0
        g = _clipped_inverse(transform, u)
        back = g(np.linspace(-30, 30, 601))
        assert np.isfinite(back).all() and (np.diff(back) >= 0).all()
        assert back.min() >= z.min() - 1e-6 and back.max() <= z.max() + 1e-6
        assert np.allclose(g(u), z, atol=1e-6)  # 학습 범위 안에서는 원래 역변환


def test_regressor_bucket_values_stay_in_training_target_range():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(300, 3))
    y = rng.lognormal(0, 1.5, 300)  # 오른쪽으로 치우친 target: power 멤버 λ < 0
    inner = torch.linspace(-4.7, 4.4, 31)
    borders = torch.cat([torch.tensor([-29.0]), inner, torch.tensor([32.0])])
    reg = TabPFNv2Regressor(_tiny_model(n_out=32), borders, n_estimators=4).fit(x, y)
    z = (y - reg.y_mean_) / reg.y_std_
    lo, hi = min(float(borders[1]), z.min()), max(float(borders[-2]), z.max())
    for values in reg.bucket_values_:
        assert lo - 1e-5 <= float(values.min()) and float(values.max()) <= hi + 1e-5
    assert np.isfinite(reg.predict(x[:20])).all()
    log_probs, _ = reg.predict_distribution(x[:20])
    assert torch.allclose(log_probs.exp().sum(-1), torch.ones(20), atol=1e-4)


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


def test_outlier_clip_leaves_categorical_codes_alone():
    from minipfn.inference.preprocessing import Member, MemberPreprocessor

    x = np.zeros((1000, 2))
    x[:3, 0] = 50.0  # 드문 범주 번호 50
    x[:, 1] = np.random.default_rng(0).normal(size=1000)
    member = Member(kind="quantile", class_permutation=None, target_power=False, seed=0)
    prep = MemberPreprocessor(member, cat_ix=[0], max_one_hot_width=160).fit(x)
    assert prep.upper[0] == np.inf and len(prep.cat_maps[0]) == 51
    codes = prep._base(prep._clip(x))[:, 0]
    assert len(np.unique(codes[:3])) == 1 and codes[0] != codes[3]  # 50은 다른 범주와 섞이지 않는다


def test_one_hot_expands_small_categorical_columns_only():
    from minipfn.inference.preprocessing import Member, MemberPreprocessor

    rng = np.random.default_rng(0)
    x = np.stack(
        [
            rng.integers(0, 3, 200),
            rng.integers(0, 2, 200),
            rng.integers(0, 30, 200),
            rng.normal(size=200),
        ],
        axis=1,
    ).astype(float)
    x[0, 0] = np.nan
    member = Member(kind="power", class_permutation=None, target_power=False, seed=0, one_hot=True)
    prep = MemberPreprocessor(member, cat_ix=[0, 1, 2], max_one_hot_width=160).fit(x)
    assert prep.one_hot_sizes == {0: 3, 1: 2}  # 범주 30개 열은 번호로 둔다
    base = prep._base(prep._clip(x))
    assert base.shape[1] == 2 + 3 + 1  # 번호 열 1개 + 수치 1개, 3범주 → 3열, 2범주 → 1열
    block = base[:, 2:5]
    assert np.isnan(block[0]).all()  # 결측은 NaN
    assert np.array_equal(block[1:].argmax(1), x[1:, 0]) and (block[1:].sum(1) == 1).all()
    assert np.array_equal(base[:, 5], x[:, 1])
    out = prep.transform(x, np.random.default_rng(1))
    assert out.shape == (200, base.shape[1] + 1)  # 샘플 식별자 열


MAX_ONE_HOT_WIDTH = 160  # v2 논문 prior의 feature 수 상한 (모델 설정 max_features의 기본값)


def test_one_hot_width_is_capped_by_reverting_the_widest_columns():
    from minipfn.inference.preprocessing import one_hot_sizes

    # 범주 20개 열 8개 + 범주 2개 열 10개 + 수치 열 2개: 펼치면 2 + 160 + 10 = 172열
    n_categories = {i: 20 for i in range(8)} | {i: 2 for i in range(8, 18)}
    sizes = one_hot_sizes(n_categories, n_features=20, max_width=MAX_ONE_HOT_WIDTH)
    assert sum(1 if n == 2 else n for n in sizes.values()) + 2 <= MAX_ONE_HOT_WIDTH
    assert len([n for n in sizes.values() if n == 20]) == 7 and 0 not in sizes  # 하나만 되돌린다
    assert all(i in sizes for i in range(8, 18))  # 열 수가 늘지 않는 2범주 열은 그대로
    # 원래 열이 이미 상한보다 많으면 열 수를 늘리는 펼침은 하지 않는다
    assert one_hot_sizes({0: 5, 1: 2}, MAX_ONE_HOT_WIDTH + 1, MAX_ONE_HOT_WIDTH) == {1: 2}


def test_one_hot_out_of_range_codes_are_missing():
    from minipfn.inference.preprocessing import MemberPreprocessor

    block = MemberPreprocessor._one_hot(np.array([0.0, 2.0, 3.0, np.nan, -1.0]), 3)
    assert np.array_equal(block[:2], [[1, 0, 0], [0, 0, 1]])
    assert np.isnan(block[2:]).all()  # 범위 밖과 결측은 번호 열처럼 결측


@pytest.mark.parametrize("one_hot", ["off", "power", "all"])
def test_estimators_run_with_one_hot(one_hot):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.choice(list("xyz"), 80), "b": rng.normal(size=80)})
    y = (df["b"] > 0).astype(int).to_numpy()
    clf = TabPFNv2Classifier(_tiny_model(), one_hot=one_hot).fit(df[:60], y[:60])
    assert clf.predict_proba(df[60:]).shape == (20, 2)
    reg = TabPFNv2Regressor(_tiny_model(n_out=32), torch.linspace(-4, 4, 33), one_hot=one_hot)
    assert reg.fit(df[:60], df["b"][:60]).predict(df[60:]).shape == (20,)
    with pytest.raises(ValueError, match="one_hot"):
        TabPFNv2Classifier(_tiny_model(), one_hot="half")


def test_predict_is_repeatable_and_regressor_keeps_borders_on_cpu():
    x, y = load_breast_cancer(return_X_y=True)
    clf = TabPFNv2Classifier(_tiny_model()).fit(x[:60], y[:60])
    assert np.array_equal(clf.predict_proba(x[60:70]), clf.predict_proba(x[60:70]))
    reg = TabPFNv2Regressor(_tiny_model(n_out=32), torch.linspace(-4, 4, 33).double())
    assert reg.borders.device.type == "cpu" and reg.borders.dtype == torch.float32


def test_one_hot_width_follows_the_feature_count_the_model_was_trained_on():
    # 긴 표로 학습한 모델(max_features 500)은 160열을 넘게 펼쳐도 된다
    from dataclasses import replace

    from minipfn.recipes import tabpfn_v2 as recipe
    from minipfn.training.config import TrainConfig

    wide = replace(TrainConfig.smoke(), max_features=500, feature_scale=500)
    assert wide.model_config().max_features == 500
    model = wide.build_model().eval()
    assert recipe.TabPFNv2Classifier(model).max_one_hot_width == 500
    assert recipe.TabPFNv2Classifier(_tiny_model()).max_one_hot_width == MAX_ONE_HOT_WIDTH
    assert recipe.TabPFNv2Classifier(model, max_one_hot_width=30).max_one_hot_width == 30


def test_max_one_hot_width_falls_back_to_the_default_without_config_max_features():
    # registry 규약만 지키는 모델은 config에 max_features가 없어도 추론이 돌아야 한다
    from types import SimpleNamespace

    from minipfn.inference.ensemble import max_features_of

    class NoMaxFeatures(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(n_out=10)

    model = NoMaxFeatures()
    assert max_features_of(_tiny_model()) == MAX_ONE_HOT_WIDTH
    assert max_features_of(model) == MAX_ONE_HOT_WIDTH  # MAX_FEATURES(160)
    assert TabPFNv2Classifier(model).max_one_hot_width == MAX_ONE_HOT_WIDTH
