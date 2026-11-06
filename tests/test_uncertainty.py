"""예측 분포·불확실성 지표 (minipfn.benchmarks.uncertainty) 테스트."""

import numpy as np
import pytest
import torch
from sklearn.datasets import load_diabetes

from minipfn.benchmarks import uncertainty as unc


def _random_histogram(n=5, buckets=12, seed=0):
    g = np.random.default_rng(seed)
    borders = np.sort(g.uniform(-3, 3, buckets + 1))
    probs = g.dirichlet(np.ones(buckets), size=n)
    return probs, borders


def _sample(probs, borders, size, g):
    """히스토그램(구간 안은 균일)에서 행마다 size개를 뽑는다."""
    out = np.empty((len(probs), size))
    for i, p in enumerate(probs):
        k = g.choice(len(p), size=size, p=p)
        out[i] = g.uniform(borders[k], borders[k + 1])
    return out


def test_crps_matches_numerical_integration_of_the_cdf():
    # CRPS = ∫ (F(t) − 1{t ≥ y})² dt를 촘촘한 격자에서 직접 적분한 값과 같다
    probs, borders = _random_histogram()
    y = np.array([-4.0, -1.0, 0.3, 2.9, 5.0])  # 경계 밖 두 개 포함
    t = np.linspace(-6, 8, 200_001)
    cdf = np.stack([unc.pit(np.repeat(p[None], len(t), 0), borders, t) for p in probs])
    step = (t[None, :] >= y[:, None]).astype(float)
    expected = np.trapezoid((cdf - step) ** 2, t, axis=1)
    assert np.allclose(unc.crps(probs, borders, y), expected, atol=1e-4)


def test_crps_of_a_narrow_bucket_is_the_absolute_error():
    borders = np.array([-10.0, 0.999, 1.001, 10.0])
    probs = np.array([[0.0, 1.0, 0.0]])
    assert unc.crps(probs, borders, [3.0])[0] == pytest.approx(2.0, abs=1e-3)


def test_quantile_inverts_the_cdf_and_pit_is_the_cdf():
    probs, borders = _random_histogram(n=3)
    for q in (0.05, 0.3, 0.5, 0.97):
        x = unc.quantile(probs, borders, q)
        assert np.allclose(unc.pit(probs, borders, x), q, atol=1e-9)
    assert np.allclose(unc.pit(probs, borders, [-9, -9, 9]), [0, 0, 1])


def test_zero_width_buckets_are_atoms():
    # float32로 옮긴 경계에서 폭이 0이 된 구간: 확률이 그 점에 몰린다
    borders = np.array([0.0, 1.0, 1.0, 2.0])
    probs = np.array([[0.25, 0.5, 0.25]])
    assert unc.pit(probs, borders, [1.0])[0] == pytest.approx(0.75)
    assert unc.quantile(probs, borders, 0.5)[0] == pytest.approx(1.0)
    assert np.isfinite(unc.crps(probs, borders, [1.0])).all()


def test_calibrated_forecasts_cover_at_the_nominal_level():
    # 정답을 예측 분포 자신에서 뽑으면 PIT는 균일하고 coverage는 level과 같다
    g = np.random.default_rng(0)
    probs, borders = _random_histogram(n=4000, seed=2)
    y = _sample(probs, borders, 1, g)[:, 0]
    out = unc.regression_metrics(probs, borders, y, y_scale=2.0)
    for level in unc.COVERAGE_LEVELS:
        name = round(level * 100)
        assert out[f"coverage_{name}"] == pytest.approx(level, abs=0.03)
        assert out[f"coverage_min_by_width_{name}"] == pytest.approx(level, abs=0.06)
    assert out["pit_ks"] < 0.03
    assert out["crps_scaled"] == pytest.approx(out["crps"] / 2.0)
    # 분포를 한쪽으로 밀면 PIT가 균일에서 멀어지고 CRPS가 커진다
    shifted = unc.regression_metrics(probs, borders + 1.0, y)
    assert shifted["pit_ks"] > 0.2 and shifted["crps"] > out["crps"]


def test_ece_is_zero_when_confidence_matches_accuracy():
    proba = np.tile([0.75, 0.25], (400, 1))
    labels = np.array([0] * 300 + [1] * 100)
    assert unc.ece(proba, labels) == pytest.approx(0.0)
    assert unc.ece(proba, np.array([0] * 200 + [1] * 200)) == pytest.approx(0.25)


def test_conformal_threshold_and_scores():
    assert unc.conformal_threshold(np.arange(9.0), 0.1) == 8.0  # ⌈10 × 0.9⌉ = 9번째
    assert unc.conformal_threshold(np.arange(5.0), 0.1) == float("inf")
    proba = np.array([[0.6, 0.3, 0.1]])
    assert np.allclose(unc.conformal_scores(proba, "lac"), [[0.4, 0.7, 0.9]])
    assert np.allclose(unc.conformal_scores(proba, "aps"), [[0.6, 0.9, 1.0]])


@pytest.mark.parametrize("kind", unc.CONFORMAL_SCORES)
def test_split_conformal_covers_at_least_the_target(kind):
    g = np.random.default_rng(0)
    proba = g.dirichlet(np.ones(4) * 0.7, size=2000)
    labels = np.array([g.choice(4, p=p) for p in proba])  # 확률이 맞는 모델
    out = unc.conformal(proba, labels, alpha=0.1, kind=kind)
    assert out["coverage"] == pytest.approx(0.9, abs=0.02)
    assert 1.0 <= out["set_size"] < 4.0
    assert out["sscs"] <= out["coverage"] + 1e-12
    assert out["class_coverage_gap"] >= 0.0


def test_classification_metrics_ignore_classes_unseen_in_training():
    proba = np.tile([0.8, 0.2], (40, 1))
    y = np.array(["a"] * 32 + ["b"] * 6 + ["z"] * 2)  # "z"는 학습 fold에 없던 클래스
    out = unc.classification_metrics(proba, y, np.array(["a", "b"]))
    assert out["ece"] == pytest.approx(abs(0.8 - 32 / 38))
    assert set(out) >= {"log_loss", "brier", "lac_coverage", "aps_sscs"}


def test_regressor_predict_full_matches_predict_and_distribution():
    from minipfn.models.cell_transformer import CellTransformer, CellTransformerConfig
    from minipfn.recipes.tabpfn_v2 import TabPFNv2Regressor

    torch.manual_seed(0)
    model = CellTransformer(CellTransformerConfig(emsize=16, nlayers=2, nhead=2, n_out=32))
    for p in model.parameters():
        torch.nn.init.normal_(p, std=0.3)
    x, y = load_diabetes(return_X_y=True)
    reg = TabPFNv2Regressor(model.eval(), torch.linspace(-4, 4, 33)).fit(x[:80], y[:80])
    mean, log_probs, borders = reg.predict_full(x[80:100])
    assert np.array_equal(mean, reg.predict(x[80:100]))
    log_probs2, borders2 = reg.predict_distribution(x[80:100])
    assert torch.equal(log_probs, log_probs2) and torch.equal(borders, borders2)
    out = unc.regression_metrics(log_probs.exp().numpy(), borders.numpy(), y[80:100])
    assert np.isfinite(out["crps"]) and 0.0 <= out["coverage_90"] <= 1.0


def test_evaluate_task_adds_uncertainty_columns():
    import pandas as pd

    from minipfn.benchmarks.amlb import OpenMLTask, evaluate_task

    class Folds:
        def get_train_test_split_indices(self, fold, repeat, sample):
            ix = np.arange(60)
            return ix[ix % 2 != fold], ix[ix % 2 == fold]

    class Gaussian:
        """평균은 학습 target 평균, 분포는 학습 target 히스토그램인 회귀 estimator."""

        def fit(self, X, y):
            self.counts, self.edges = np.histogram(y, bins=10)
            self.mean = float(np.mean(y))
            return self

        def predict(self, X):
            return np.full(len(X), self.mean)

        def predict_distribution(self, X):
            p = (self.counts + 1e-3) / (self.counts + 1e-3).sum()
            return np.log(np.tile(p, (len(X), 1))), self.edges

    g = np.random.default_rng(0)
    X = pd.DataFrame({"a": g.normal(size=60)})
    task = OpenMLTask(1, "toy", X, pd.Series(g.normal(size=60)), [], "regression", Folds())
    rows = evaluate_task(task, lambda t: Gaussian(), n_folds=2, uncertainty=True)
    assert len(rows) == 2
    assert {"crps", "crps_scaled", "coverage_90", "pit_ks", "rmse_scaled"} <= set(rows[0])
    assert rows[0]["score"] == pytest.approx(-rows[0]["rmse_scaled"] * np.std(task.y[1::2]))
