"""예측 분포 품질과 불확실성 지표 (issue #45): 회귀 bar distribution의 CRPS·구간·PIT, 분류 확률의
ECE와 split conformal prediction의 (조건부) coverage.

점 예측 지표(ROC AUC, RMSE)는 분포가 잘 맞는지, 불확실성을 믿을 수 있는지는 보지 않는다. 학습 없이
checkpoint의 예측만으로 계산한다.

회귀 분포는 구간 히스토그램이다: 경계 `borders` (구간 수 + 1,)와 구간 확률 `probs` (n, 구간 수).
구간 안의 밀도는 균일하다고 보므로 CDF는 경계 사이에서 직선이다. ScoringBench(arXiv 2603.29928)가
TabPFN의 bar distribution을 채점하는 방식과 같다. 학습 손실(`FullSupportBarDistribution`)은 양 끝
구간을 half-normal 꼬리로 보지만 여기서는 그 구간도 균일로 본다(맨 바깥 경계 밖의 확률은 0).
`EnsembleRegressor.predict_distribution`의 (log 확률, 경계)를 `np.exp`해서 그대로 넣으면 된다.

- `crps`: CRPS = ∫ (F(t) − 1{t ≥ y})² dt. CDF가 구간마다 직선이라 닫힌 식으로 정확히 계산한다.
- `interval_scores`: 가운데 `level` 구간 [Q((1−level)/2), Q((1+level)/2)]의 coverage, 폭, interval
  score(Gneiting & Raftery 2007: 폭 + (2/α)·벗어난 거리, α = 1 − level). 조건부 coverage의 대리로
  구간 폭 5분위마다의 coverage 가운데 가장 낮은 값(`coverage_min_by_width`)을 같이 낸다. 넓은
  구간과 좁은 구간을 같은 비율로 맞히는지 본다(분류의 SSCS와 같은 생각).
- `pit`: PIT F(y). 분포가 맞으면 Uniform(0, 1)이고, 그 거리를 KS 통계량(`pit_ks`)으로 낸다.

분류 확률 `proba` (n, 클래스 수)는 `classes`(estimator의 `classes_`) 순서다. 학습 fold에 없던
클래스의 테스트 행은 `amlb.score_fold`처럼 뺀다.

- `ece`: top-label expected calibration error (같은 폭의 확신도 구간 15개), log loss, Brier.
- `conformal`: split conformal prediction. 모델은 학습 fold로 이미 fit했으므로 테스트 fold 예측을
  calibration 절반과 평가 절반으로 무작위로 나눠(`n_repeats`번) 따로 학습하지 않는다(나눈 두 절반은
  교환 가능하다). 비순응 점수는 LAC(1 − p_y, Sadinle et al. 2019)와 APS(p_y 이상인 확률의 합,
  Romano et al. 2020, 무작위화 없이). 낸 값은 marginal coverage, 평균 집합 크기, SSCS
  (size-stratified coverage, Angelopoulos et al. 2021: 집합 크기마다의 coverage 가운데 가장 낮은 값,
  행이 `min_group`개 이상인 크기만), 클래스 조건부 coverage gap(클래스마다 |coverage − (1 − α)|의
  평균, Ding et al. 2023). 반복한 평가 절반을 모두 모아 크기·클래스로 나눈다.

`regression_metrics`, `classification_metrics`가 fold 하나의 지표를 dict로 모아 준다
(`amlb.evaluate_task(..., uncertainty=True)`).
"""

from __future__ import annotations

import numpy as np
from scipy import stats

COVERAGE_LEVELS = (0.5, 0.8, 0.9, 0.95)
CONFORMAL_ALPHA = 0.1
CONFORMAL_SCORES = ("lac", "aps")


# ---------------------------------------------------------------------------
# 회귀: 구간 히스토그램
# ---------------------------------------------------------------------------


def _histogram(probs, borders) -> tuple[np.ndarray, np.ndarray]:
    probs = np.asarray(probs, dtype=np.float64)
    borders = np.asarray(borders, dtype=np.float64)
    if borders.ndim != 1 or borders.shape[0] != probs.shape[-1] + 1:
        raise ValueError(f"경계는 (구간 수 + 1,)이어야 한다: {borders.shape}, {probs.shape}")
    # 경계를 float32로 원래 단위로 옮기면 아주 좁은 구간(z 폭 1e-6)이 폭 0이 될 수 있다.
    # 폭 0인 구간의 확률은 그 점에 놓인 원자로 본다
    if np.any(np.diff(borders) < 0):
        raise ValueError("경계는 줄어들면 안 된다")
    probs = np.atleast_2d(probs).clip(min=0.0)
    return probs / probs.sum(axis=1, keepdims=True), borders


def _cdf_nodes(probs: np.ndarray) -> np.ndarray:
    """경계 위의 CDF (n, 구간 수 + 1). 맨 앞은 0, 맨 끝은 1."""
    cdf = np.concatenate([np.zeros((len(probs), 1)), np.cumsum(probs, axis=1)], axis=1)
    cdf[:, -1] = 1.0
    return cdf


def _bucket_of(borders: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (np.searchsorted(borders, y, side="right") - 1).clip(0, len(borders) - 2)


def pit(probs, borders, y) -> np.ndarray:
    """F(y). 맨 바깥 경계 밖은 0 또는 1."""
    probs, borders = _histogram(probs, borders)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    cdf = _cdf_nodes(probs)
    rows = np.arange(len(y))
    k = _bucket_of(borders, y)
    return (cdf[rows, k] + probs[rows, k] * _fraction(borders, k, y)).clip(0.0, 1.0)


def _fraction(borders: np.ndarray, k: np.ndarray, y: np.ndarray) -> np.ndarray:
    """구간 k 안에서 y의 위치 (0~1). 폭 0인 구간은 y가 그 점 이상이면 1."""
    width = borders[k + 1] - borders[k]
    safe = np.where(width > 0, width, 1.0)
    return np.where(width > 0, (y - borders[k]) / safe, (y >= borders[k]).astype(float)).clip(0, 1)


def pit_ks(pit_values) -> float:
    """PIT와 Uniform(0, 1)의 KS 통계량 (0이면 분포가 맞다)."""
    return float(stats.kstest(np.asarray(pit_values, dtype=np.float64), "uniform").statistic)


def quantile(probs, borders, q) -> np.ndarray:
    """q 분위수 (n,). CDF가 구간마다 직선이므로 구간 안에서 직선으로 거꾸로 푼다."""
    probs, borders = _histogram(probs, borders)
    cdf = _cdf_nodes(probs)
    rows = np.arange(len(probs))
    # q가 들어 있는 구간 k: cdf[k] < q <= cdf[k + 1] (확률 0인 구간은 건너뛴다)
    k = (cdf[:, 1:] < q).sum(axis=1).clip(0, probs.shape[1] - 1)
    p = probs[rows, k]
    frac = np.where(p > 0, (q - cdf[rows, k]) / np.where(p > 0, p, 1.0), 0.0).clip(0.0, 1.0)
    return borders[k] + frac * (borders[k + 1] - borders[k])


def _segment_square_integral(a, b, h, width):
    """[l, r]에서 F가 a → b로 직선일 때 ∫ (F − h)² dt."""
    a, b = a - h, b - h
    return width * (a * a + a * b + b * b) / 3.0


def crps(probs, borders, y) -> np.ndarray:
    """행마다 CRPS (원래 단위). y가 맨 바깥 경계 밖이면 경계까지의 거리를 더한다."""
    probs, borders = _histogram(probs, borders)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    cdf = _cdf_nodes(probs)
    left, right = borders[:-1], borders[1:]
    width = right - left
    a, b = cdf[:, :-1], cdf[:, 1:]
    below = _segment_square_integral(a, b, 0.0, width)  # 구간 전체가 y보다 왼쪽 (H = 0)
    above = _segment_square_integral(a, b, 1.0, width)  # 구간 전체가 y보다 오른쪽 (H = 1)
    is_above = left[None, :] >= y[:, None]
    total = np.where(is_above, above, below)
    # y가 든 구간은 y에서 둘로 나눈다
    inside = (y > borders[0]) & (y < borders[-1])
    rows = np.flatnonzero(inside)
    k = _bucket_of(borders, y[rows])
    f_y = cdf[rows, k] + probs[rows, k] * _fraction(borders, k, y[rows])
    split = _segment_square_integral(cdf[rows, k], f_y, 0.0, y[rows] - left[k])
    split += _segment_square_integral(f_y, cdf[rows, k + 1], 1.0, right[k] - y[rows])
    total[rows, k] = split
    out = total.sum(axis=1)
    # 맨 바깥 경계 밖: F는 0 또는 1이고 H와 1만큼 다르다
    out += (borders[0] - y).clip(min=0.0) + (y - borders[-1]).clip(min=0.0)
    return out


def interval_scores(
    probs, borders, y, levels=COVERAGE_LEVELS, n_width_bins: int = 5
) -> dict[str, float]:
    """가운데 구간마다 coverage, 평균 폭, interval score, 폭 5분위별 coverage의 최솟값."""
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    out = {}
    for level in levels:
        alpha = 1.0 - level
        lo = quantile(probs, borders, alpha / 2)
        hi = quantile(probs, borders, 1.0 - alpha / 2)
        covered = (y >= lo) & (y <= hi)
        width = hi - lo
        score = width + (2 / alpha) * ((lo - y).clip(min=0) + (y - hi).clip(min=0))
        name = f"{round(level * 100)}"
        out[f"coverage_{name}"] = float(covered.mean())
        out[f"width_{name}"] = float(width.mean())
        out[f"interval_score_{name}"] = float(score.mean())
        out[f"coverage_min_by_width_{name}"] = _min_coverage_by_width(covered, width, n_width_bins)
    return out


def _min_coverage_by_width(covered: np.ndarray, width: np.ndarray, n_bins: int) -> float:
    if len(covered) < 2 * n_bins:
        return float("nan")
    # 폭 순서로 같은 행 수의 묶음 n_bins개 (같은 폭이 많아도 묶음 크기는 같다)
    order = np.argsort(width, kind="stable")
    return float(min(covered[ix].mean() for ix in np.array_split(order, n_bins)))


def regression_metrics(
    probs, borders, y, y_scale: float = 1.0, levels=COVERAGE_LEVELS
) -> dict[str, float]:
    """fold 하나의 회귀 분포 지표. `y_scale`(학습 target 표준편차 등)로 나눈 CRPS·폭·interval
    score를 `_scaled`로 같이 낸다 (데이터셋끼리 평균할 때 쓴다)."""
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    pit_values = pit(probs, borders, y)
    out = {"crps": float(crps(probs, borders, y).mean())}
    out.update(interval_scores(probs, borders, y, levels))
    out["pit_ks"] = pit_ks(pit_values)
    scaled = {
        f"{k}_scaled": v / y_scale
        for k, v in out.items()
        if k == "crps" or k.startswith(("width_", "interval_score_"))
    }
    return {**out, **scaled}


# ---------------------------------------------------------------------------
# 분류
# ---------------------------------------------------------------------------


def _seen_rows(proba, y_true, classes) -> tuple[np.ndarray, np.ndarray]:
    """학습 fold에 있던 클래스의 행만 남기고 정답을 `classes` 안의 번호로 바꾼다."""
    proba = np.asarray(proba, dtype=np.float64)
    y_true, classes = np.asarray(y_true), np.asarray(classes)
    seen = np.isin(y_true, classes)
    proba = proba[seen]
    proba = proba / proba.sum(axis=1, keepdims=True)
    index = {c: i for i, c in enumerate(classes.tolist())}
    return proba, np.array([index[c] for c in y_true[seen].tolist()], dtype=int)


def ece(proba, labels, n_bins: int = 15) -> float:
    """top-label ECE. `labels`는 `proba` 열 번호."""
    proba, labels = np.asarray(proba, dtype=np.float64), np.asarray(labels)
    conf = proba.max(axis=1)
    correct = proba.argmax(axis=1) == labels
    bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    total = 0.0
    for b in np.unique(bins):
        m = bins == b
        total += m.sum() * abs(correct[m].mean() - conf[m].mean())
    return float(total / len(labels))


def conformal_scores(proba: np.ndarray, kind: str) -> np.ndarray:
    """모든 클래스의 비순응 점수 (n, 클래스 수): 정답이 그 클래스일 때의 점수."""
    if kind == "lac":
        return 1.0 - proba
    if kind == "aps":
        # 클래스 k의 점수 = p_k 이상인 확률의 합 (자기 자신 포함, 같은 확률도 포함)
        return (proba[:, None, :] * (proba[:, None, :] >= proba[:, :, None])).sum(-1)
    raise ValueError(f"conformal 점수는 {CONFORMAL_SCORES} 가운데 하나: {kind}")


def conformal_threshold(cal_scores: np.ndarray, alpha: float) -> float:
    """calibration 점수의 ⌈(n + 1)(1 − α)⌉번째 값. 그 순위가 n을 넘으면 무한대(모든 클래스)."""
    n = len(cal_scores)
    rank = int(np.ceil((n + 1) * (1 - alpha)))
    if rank > n:
        return float("inf")
    return float(np.sort(cal_scores)[rank - 1])


def conformal(
    proba,
    labels,
    alpha: float = CONFORMAL_ALPHA,
    kind: str = "lac",
    n_repeats: int = 20,
    min_group: int = 10,
    seed: int = 0,
) -> dict[str, float]:
    """split conformal prediction: 행을 calibration·평가 절반으로 무작위로 나누기를 반복한다."""
    proba, labels = np.asarray(proba, dtype=np.float64), np.asarray(labels)
    n = len(labels)
    names = ("coverage", "set_size", "sscs", "class_coverage_gap")
    if n < 20:
        return {k: float("nan") for k in names}
    scores = conformal_scores(proba, kind)
    true_scores = scores[np.arange(n), labels]
    rng = np.random.default_rng(seed)
    covered, sizes, eval_labels = [], [], []
    for _ in range(n_repeats):
        perm = rng.permutation(n)
        cal, ev = perm[: n // 2], perm[n // 2 :]
        q = conformal_threshold(true_scores[cal], alpha)
        sets = scores[ev] <= q
        covered.append(sets[np.arange(len(ev)), labels[ev]])
        sizes.append(sets.sum(axis=1))
        eval_labels.append(labels[ev])
    covered, sizes = np.concatenate(covered), np.concatenate(sizes)
    eval_labels = np.concatenate(eval_labels)
    size_values, size_counts = np.unique(sizes, return_counts=True)
    by_size = [covered[sizes == s].mean() for s in size_values[size_counts >= min_group]]
    by_class = [covered[eval_labels == c].mean() for c in np.unique(eval_labels)]
    return {
        "coverage": float(covered.mean()),
        "set_size": float(sizes.mean()),
        "sscs": float(min(by_size)) if by_size else float("nan"),
        "class_coverage_gap": float(np.mean(np.abs(np.array(by_class) - (1 - alpha)))),
    }


def classification_metrics(
    proba, y_true, classes, alpha: float = CONFORMAL_ALPHA, seed: int = 0
) -> dict[str, float]:
    """fold 하나의 분류 확률 지표: ECE, log loss, Brier, conformal(LAC·APS)."""
    proba, labels = _seen_rows(proba, y_true, classes)
    rows = np.arange(len(labels))
    onehot = np.zeros_like(proba)
    onehot[rows, labels] = 1.0
    out = {
        "ece": ece(proba, labels),
        "log_loss": float(-np.log(proba[rows, labels].clip(min=1e-15)).mean()),
        "brier": float(((proba - onehot) ** 2).sum(axis=1).mean()),
    }
    for kind in CONFORMAL_SCORES:
        result = conformal(proba, labels, alpha=alpha, kind=kind, seed=seed)
        out.update({f"{kind}_{k}": v for k, v in result.items()})
    return out
