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


def _sample(probs, borders, size, g, tails):
    """히스토그램(안쪽 구간은 균일, 양 끝은 tails)에서 행마다 size개를 뽑는다."""
    h = unc.bar_histogram(probs, borders, tails)
    out = np.empty((len(probs), size))
    for i, p in enumerate(h.probs):
        k = g.choice(len(p), size=size, p=p)
        out[i] = g.uniform(borders[k], borders[k + 1])
        if tails == "half_normal":
            d = np.abs(g.normal(size=size))
            out[i] = np.where(k == 0, borders[1] - h.scale_left * d, out[i])
            out[i] = np.where(k == len(p) - 1, borders[-2] + h.scale_right * d, out[i])
    return out


@pytest.mark.parametrize("tails", unc.TAILS)
def test_crps_matches_numerical_integration_of_the_cdf(tails):
    # CRPS = ∫ (F(t) − 1{t ≥ y})² dt를 촘촘한 격자에서 직접 적분한 값과 같다
    probs, borders = _random_histogram()
    y = np.array([-4.0, -1.0, 0.3, 2.9, 5.0])  # 경계 밖 두 개 포함
    t = np.linspace(-25, 25, 500_001)
    rows = [unc.bar_histogram(np.repeat(p[None], len(t), 0), borders, tails) for p in probs]
    cdf = np.stack([unc.pit(row, t) for row in rows])
    step = (t[None, :] >= y[:, None]).astype(float)
    expected = np.trapezoid((cdf - step) ** 2, t, axis=1)
    h = unc.bar_histogram(probs, borders, tails)
    assert np.allclose(unc.crps(h, y), expected, atol=1e-4)


def test_crps_is_computed_in_row_chunks(monkeypatch):
    probs, borders = _random_histogram(n=7)
    y = np.linspace(-4, 4, 7)
    h = unc.bar_histogram(probs, borders)
    full = unc.crps(h, y)
    monkeypatch.setattr(unc, "CRPS_CHUNK_ROWS", 3)
    assert np.array_equal(unc.crps(h, y), full)


def test_crps_of_a_narrow_bucket_is_the_absolute_error():
    borders = np.array([-10.0, 0.999, 1.001, 10.0])
    h = unc.bar_histogram(np.array([[0.0, 1.0, 0.0]]), borders)
    assert unc.crps(h, [3.0])[0] == pytest.approx(2.0, abs=1e-3)


@pytest.mark.parametrize("tails", unc.TAILS)
def test_quantile_inverts_the_cdf_and_pit_is_the_cdf(tails):
    probs, borders = _random_histogram(n=3)
    probs[:, [0, -1]] = 0.1  # 양 끝 구간에도 확률을 둬서 꼬리 분위수를 확인한다
    h = unc.bar_histogram(probs, borders, tails)
    levels = np.array([0.01, 0.05, 0.3, 0.5, 0.97, 0.995])
    x = unc.quantile(h, levels)
    assert x.shape == (3, len(levels))
    for j, q in enumerate(levels):
        assert np.allclose(unc.pit(h, x[:, j]), q, atol=1e-9)
    edge = unc.pit(h, [borders[0], borders[0], borders[-1]])
    if tails == "uniform":
        assert np.allclose(edge, [0, 0, 1])  # 맨 바깥 경계 밖에는 확률이 없다
    else:
        # half-normal: 꼬리 확률의 절반이 맨 바깥 경계 밖에 있다 (중앙값 = 구간 폭)
        p_left, p_right = h.probs[:, 0], h.probs[:, -1]
        assert np.allclose(edge, [p_left[0] / 2, p_left[1] / 2, 1 - p_right[2] / 2])


def test_zero_width_buckets_are_atoms():
    # 경계가 같은 값인 구간: 확률이 그 점에 몰린다
    borders = np.array([0.0, 1.0, 1.0, 2.0])
    h = unc.bar_histogram(np.array([[0.25, 0.5, 0.25]]), borders)
    assert unc.pit(h, [1.0])[0] == pytest.approx(0.75)
    assert unc.quantile(h, 0.5)[0, 0] == pytest.approx(1.0)
    assert np.isfinite(unc.crps(h, [1.0])).all()


@pytest.mark.parametrize("tails", unc.TAILS)
def test_calibrated_forecasts_cover_at_the_nominal_level(tails):
    # 정답을 예측 분포 자신에서 뽑으면 PIT는 균일하고 coverage는 level과 같다
    g = np.random.default_rng(0)
    probs, borders = _random_histogram(n=4000, seed=2)
    y = _sample(probs, borders, 1, g, tails)[:, 0]
    out = unc.regression_metrics(probs, borders, y, y_scale=2.0, tails=tails)
    for level in unc.COVERAGE_LEVELS:
        name = round(level * 100)
        assert out[f"coverage_{name}"] == pytest.approx(level, abs=0.03)
        assert out[f"coverage_min_by_width_{name}"] == pytest.approx(level, abs=0.06)
    assert out["pit_ks"] < 0.03
    assert out["crps_scaled"] == pytest.approx(out["crps"] / 2.0)
    # 분포를 한쪽으로 밀면 PIT가 균일에서 멀어지고 CRPS가 커진다
    shifted = unc.regression_metrics(probs, borders + 1.0, y, tails=tails)
    assert shifted["pit_ks"] > 0.2 and shifted["crps"] > out["crps"]


def test_min_coverage_by_width_does_not_depend_on_row_order():
    # 폭이 모두 같으면 한 분위에 들어가 전체 coverage와 같다 (행 순서를 바꿔도)
    g = np.random.default_rng(0)
    covered = g.random(200) < 0.9
    width = np.ones(200)
    expected = covered.mean()
    for _ in range(3):
        perm = g.permutation(200)
        assert unc.min_coverage_by_width(covered[perm], width[perm]) == pytest.approx(expected)


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
    assert np.allclose(unc.conformal_scores(proba, "aps"), [[0.6, 0.9, 1.0]])  # u = 1
    assert np.allclose(unc.conformal_scores(proba, "aps", u=[0.5]), [[0.3, 0.75, 0.95]])


def test_sscs_puts_empty_and_singleton_sets_together():
    # 정확한 모델: 빈 집합(coverage 0)과 크기 1 집합이 섞여도 묶음 0~1의 coverage가 나온다
    sizes = np.array([0] * 10 + [1] * 90 + [2] * 20)
    covered = np.r_[np.zeros(10, bool), np.ones(90, bool), np.r_[np.ones(18), np.zeros(2)] > 0]
    assert unc.size_stratified_coverage(covered, sizes) == pytest.approx(0.9)
    assert np.isnan(unc.size_stratified_coverage(covered[:5], sizes[:5]))


def test_randomized_aps_does_not_return_every_class_on_hard_binary_data():
    # 이진, 정확도 75%: 무작위화하지 않은 APS는 threshold가 1이라 모든 집합이 클래스 둘 다였다
    g = np.random.default_rng(0)
    p1 = g.uniform(0.5, 1.0, 2000)
    proba = np.c_[p1, 1 - p1]
    labels = (g.random(2000) > p1).astype(int)
    out = unc.conformal(proba, labels, alpha=0.1, kind="aps")
    assert out["coverage"] == pytest.approx(0.9, abs=0.02)
    assert out["set_size"] < 1.9


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


def test_classification_metrics_are_nan_when_no_row_is_left():
    proba = np.tile([0.8, 0.2], (5, 1))
    out = unc.classification_metrics(proba, np.array(["z"] * 5), np.array(["a", "b"]))
    assert out and all(np.isnan(v) for v in out.values())
    assert np.isnan(unc.ece(np.zeros((0, 2)), np.zeros(0, dtype=int)))


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
    # 경계는 float64로 원래 단위로 옮긴다 (float32면 평균이 크고 퍼짐이 작을 때 뭉친다)
    assert borders.dtype == torch.float64
    assert torch.equal(borders, reg.borders.double() * reg.y_std_ + reg.y_mean_)
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


def test_eval_script_rejects_uncertainty_for_arena_suites(tmp_path):
    import importlib.util
    from pathlib import Path

    path = Path(__file__).parents[1] / "scripts" / "tabpfn_v2" / "eval.py"
    spec = importlib.util.spec_from_file_location("eval_tabpfn_v2", path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    out = tmp_path / "out"
    with pytest.raises(SystemExit, match="AMLB"):
        script.main(["--suite", "tabarena-lite", "--uncertainty", "--out-dir", str(out)])
    assert not out.exists()
