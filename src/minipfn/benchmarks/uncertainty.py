"""예측 분포 품질과 불확실성 지표 (issue #45): 회귀 bar distribution의 CRPS·구간·PIT, 분류 확률의
ECE와 split conformal prediction의 (조건부) coverage.

점 예측 지표(ROC AUC, RMSE)는 분포가 잘 맞는지, 불확실성을 믿을 수 있는지는 보지 않는다. 학습 없이
checkpoint의 예측만으로 계산한다.

회귀 분포는 `bar_histogram(probs, borders, tails)`로 한 번 만든다: 경계 `borders` (구간 수 + 1,)와
구간 확률 `probs` (n, 구간 수). 안쪽 구간은 밀도가 균일해 CDF가 경계 사이에서 직선이다. 양 끝 구간은
`tails`로 정한다.

- `"half_normal"`(기본): 모델이 학습한 분포(`FullSupportBarDistribution`)와 같다. 첫 구간의 확률은
  `borders[1]`에서 왼쪽으로, 마지막 구간의 확률은 `borders[-2]`에서 오른쪽으로 half-normal로 퍼지고,
  scale은 중앙값이 그 구간의 폭이 되도록 정한다(맨 바깥 경계 밖에도 확률이 있다).
  `EnsembleRegressor.predict_distribution`의 꼬리도 이것이다.
- `"uniform"`: 양 끝 구간도 균일로 본다(맨 바깥 경계 밖의 확률은 0). ScoringBench
  (arXiv 2603.29928)가 TabPFN의 bar distribution을 채점하는 방식이라, 그쪽 숫자와 견줄 때 쓴다.

`EnsembleRegressor.predict_distribution`의 (log 확률, 경계)를 `np.exp`해서 그대로 넣으면 된다. 폭이
0인 구간(경계가 같은 값)의 확률은 그 점의 원자로 본다.

- `crps`: CRPS = ∫ (F(t) − 1{t ≥ y})² dt. 안쪽 구간은 CDF가 직선이고 half-normal 꼬리도 적분이 닫힌
  식이라 정확히 계산한다.
- `interval_scores`: 가운데 `level` 구간 [Q((1−level)/2), Q((1+level)/2)]의 coverage, 폭, interval
  score(Gneiting & Raftery 2007: 폭 + (2/α)·벗어난 거리, α = 1 − level). 조건부 coverage의 대리로
  구간 폭 5분위마다의 coverage 가운데 가장 낮은 값(`coverage_min_by_width`)을 같이 낸다. 넓은
  구간과 좁은 구간을 같은 비율로 맞히는지 본다(분류의 SSCS와 같은 생각). 폭이 같은 행은 같은
  분위에 넣으므로 행 순서와 상관없다.
- `pit`: PIT F(y). 분포가 맞으면 Uniform(0, 1)이고, 그 거리를 KS 통계량(`pit_ks`)으로 낸다.

분류 확률 `proba` (n, 클래스 수)는 `classes`(estimator의 `classes_`) 순서다. 학습 fold에 없던
클래스의 테스트 행은 ROC AUC와 같은 규칙(`amlb.drop_unseen_classes`)으로 뺀다.

- `ece`: top-label expected calibration error (같은 폭의 확신도 구간 15개), log loss, Brier.
- `conformal`: split conformal prediction. 모델은 학습 fold로 이미 fit했으므로 테스트 fold 예측을
  calibration 절반과 평가 절반으로 무작위로 나눠(`n_repeats`번) 따로 학습하지 않는다(나눈 두 절반은
  교환 가능하다). 비순응 점수는 LAC(1 − p_y, Sadinle et al. 2019)와 APS(p_y 이상인 확률의 합,
  Romano et al. 2020, 무작위화 없이). 낸 값은 marginal coverage, 평균 집합 크기, SSCS
  (size-stratified coverage, Angelopoulos et al. 2021: 집합 크기마다의 coverage 가운데 가장 낮은 값,
  행이 `min_group`개 이상인 크기만), 클래스 조건부 coverage gap(클래스마다 |coverage − (1 − α)|의
  평균, Ding et al. 2023). 반복한 평가 절반을 모두 모아 크기·클래스로 나눈다.

`regression_metrics`, `classification_metrics`가 fold 하나의 지표를 dict로 모아 준다
(`amlb.evaluate_task(..., uncertainty=True)`). 남는 행이 없으면 모두 NaN이다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import special, stats

from minipfn.benchmarks.amlb import drop_unseen_classes

COVERAGE_LEVELS = (0.5, 0.8, 0.9, 0.95)
CONFORMAL_ALPHA = 0.1
CONFORMAL_SCORES = ("lac", "aps")
TAILS = ("half_normal", "uniform")
DEFAULT_TAILS = "half_normal"
CRPS_CHUNK_ROWS = 512  # crps가 한 번에 만드는 (행, 구간 수) 배열의 행 수

# half-normal(scale 1)의 중앙값 = Φ⁻¹(0.75). `FullSupportBarDistribution`과 같은 scale 규칙
_HALF_NORMAL_MEDIAN = float(special.ndtri(0.75))
_INV_SQRT_PI = 1.0 / np.sqrt(np.pi)


# ---------------------------------------------------------------------------
# 회귀: 구간 히스토그램
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BarHistogram:
    """정규화한 구간 확률과 경계 위의 CDF. `bar_histogram`으로 만든다."""

    probs: np.ndarray  # (n, B), 행마다 합 1
    borders: np.ndarray  # (B + 1,), float64
    cdf: np.ndarray  # (n, B + 1), 경계 위의 CDF (양 끝 0과 1). 꼬리 종류와 상관없이 같다
    tails: str

    @property
    def scale_left(self) -> float:
        return float(self.borders[1] - self.borders[0]) / _HALF_NORMAL_MEDIAN

    @property
    def scale_right(self) -> float:
        return float(self.borders[-1] - self.borders[-2]) / _HALF_NORMAL_MEDIAN


def bar_histogram(probs, borders, tails: str = DEFAULT_TAILS) -> BarHistogram:
    if tails not in TAILS:
        raise ValueError(f"tails는 {TAILS} 가운데 하나: {tails}")
    probs = np.atleast_2d(np.asarray(probs, dtype=np.float64))
    borders = np.asarray(borders, dtype=np.float64)
    if borders.ndim != 1 or borders.shape[0] != probs.shape[-1] + 1:
        raise ValueError(f"경계는 (구간 수 + 1,)이어야 한다: {borders.shape}, {probs.shape}")
    if np.any(np.diff(borders) < 0):
        raise ValueError("경계는 줄어들면 안 된다")
    if tails == "half_normal" and (
        len(borders) < 3 or borders[1] <= borders[0] or borders[-1] <= borders[-2]
    ):
        raise ValueError("half-normal 꼬리에는 구간이 2개 이상, 양 끝 구간의 폭이 0보다 커야 한다")
    probs = probs.clip(min=0.0)
    probs = probs / probs.sum(axis=1, keepdims=True)
    cdf = np.concatenate([np.zeros((len(probs), 1)), np.cumsum(probs, axis=1)], axis=1)
    cdf[:, -1] = 1.0
    return BarHistogram(probs, borders, cdf, tails)


def _upper_tail(u):
    """Q(u) = 1 − Φ(u)."""
    return special.ndtr(-u)


def _normal_pdf(u):
    return np.exp(-0.5 * u * u) / np.sqrt(2 * np.pi)


def _fraction(borders: np.ndarray, k: np.ndarray, y: np.ndarray) -> np.ndarray:
    """구간 k 안에서 y의 위치 (0~1). 폭 0인 구간은 y가 그 점 이상이면 1."""
    width = borders[k + 1] - borders[k]
    safe = np.where(width > 0, width, 1.0)
    return np.where(width > 0, (y - borders[k]) / safe, (y >= borders[k]).astype(float)).clip(0, 1)


def pit(h: BarHistogram, y) -> np.ndarray:
    """F(y) (n,)."""
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    b, rows = h.borders, np.arange(len(y))
    k = (np.searchsorted(b, y, side="right") - 1).clip(0, len(b) - 2)
    out = h.cdf[rows, k] + h.probs[rows, k] * _fraction(b, k, y)
    if h.tails == "half_normal":
        p_left, p_right = h.probs[:, 0], h.probs[:, -1]
        left = p_left * 2 * _upper_tail((b[1] - y) / h.scale_left)
        right = 1 - p_right * 2 * _upper_tail((y - b[-2]) / h.scale_right)
        out = np.where(y < b[1], left, np.where(y > b[-2], right, out))
    return out.clip(0.0, 1.0)


def pit_ks(pit_values) -> float:
    """PIT와 Uniform(0, 1)의 KS 통계량 (0이면 분포가 맞다)."""
    return float(stats.kstest(np.asarray(pit_values, dtype=np.float64), "uniform").statistic)


def quantile(h: BarHistogram, q) -> np.ndarray:
    """분위수 (n, len(q)). 안쪽 구간은 CDF를 직선으로, half-normal 꼬리는 정확히 거꾸로 푼다."""
    q = np.atleast_1d(np.asarray(q, dtype=np.float64))
    n, n_buckets = h.probs.shape
    # 행마다 cdf[k] < q <= cdf[k + 1]인 구간 k를 한 번에 찾는다: 행 i의 CDF(0~1)에 2i를 더해
    # 이어 붙이면 전체가 증가하므로 searchsorted 한 번으로 된다 (확률 0인 구간은 건너뛴다)
    offset = 2.0 * np.arange(n)[:, None]
    flat = (h.cdf + offset).ravel()
    pos = np.searchsorted(flat, (q[None, :] + offset).ravel(), side="left").reshape(n, len(q))
    k = (pos - np.arange(n)[:, None] * (n_buckets + 1) - 1).clip(0, n_buckets - 1)
    rows = np.arange(n)[:, None]
    p = h.probs[rows, k]
    frac = np.where(p > 0, (q[None, :] - h.cdf[rows, k]) / np.where(p > 0, p, 1.0), 0.0)
    b = h.borders
    out = b[k] + frac.clip(0.0, 1.0) * (b[k + 1] - b[k])
    if h.tails == "half_normal":
        p_left, p_right = h.probs[:, :1], h.probs[:, -1:]
        in_left, in_right = k == 0, k == n_buckets - 1
        # 왼쪽: p0 · 2Q(u) = q, 오른쪽: 1 − p_last · 2Q(u) = q. 꼬리 밖의 칸은 Φ⁻¹(0.5) = 0으로
        # 두고, q가 0이나 1이면 u가 무한대가 되지 않도록 Φ⁻¹의 인자를 1 − 1e-16 아래로 자른다
        with np.errstate(divide="ignore", invalid="ignore"):
            arg_left = np.where(in_left, 1 - q[None, :] / (2 * p_left), 0.5)
            arg_right = np.where(in_right, 1 - (1 - q[None, :]) / (2 * p_right), 0.5)
        u_left = special.ndtri(np.nan_to_num(arg_left, nan=0.5).clip(0.5, 1 - 1e-16))
        u_right = special.ndtri(np.nan_to_num(arg_right, nan=0.5).clip(0.5, 1 - 1e-16))
        out = np.where(in_left, b[1] - h.scale_left * u_left, out)
        out = np.where(in_right, b[-2] + h.scale_right * u_right, out)
    return out


def _segment_square_integral(a, b, h, width):
    """[l, r]에서 F가 a → b로 직선일 때 ∫ (F − h)² dt."""
    a, b = a - h, b - h
    return width * (a * a + a * b + b * b) / 3.0


def _sq_tail_integral(a):
    """∫_a^∞ Q(u)² du = −aQ(a)² + 2φ(a)Q(a) − Q(√2·a)/√π (a ≥ 0)."""
    q = _upper_tail(a)
    return -a * q * q + 2 * _normal_pdf(a) * q - _INV_SQRT_PI * _upper_tail(np.sqrt(2) * a)


def _right_tail_crps(p, start, scale, y):
    """c = start부터 오른쪽으로 퍼진 half-normal 꼬리(확률 p) 위의 ∫_c^∞ (F − H)² dt.
    꼬리에서 F(t) = 1 − 2p·Q(u), u = (t − c) / scale."""
    a = np.maximum((y - start) / scale, 0.0)  # y가 c보다 왼쪽이면 꼬리 전체에서 H = 1
    q_int = _normal_pdf(0.0) - _normal_pdf(a) + a * _upper_tail(a)  # ∫_0^a Q du
    tail_sq = _sq_tail_integral(a)
    below = a - 4 * p * q_int + 4 * p * p * (_sq_tail_integral(0.0) - tail_sq)  # [0, a]에서 H = 0
    return scale * (below + 4 * p * p * tail_sq)


def crps(h: BarHistogram, y) -> np.ndarray:
    """행마다 CRPS (원래 단위). 행을 `CRPS_CHUNK_ROWS`개씩 나눠 계산한다."""
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    return np.concatenate(
        [
            _crps_rows(h, slice(i, i + CRPS_CHUNK_ROWS), y[i : i + CRPS_CHUNK_ROWS])
            for i in range(0, len(y), CRPS_CHUNK_ROWS)
        ]
    )


def _crps_rows(h: BarHistogram, rows: slice, y: np.ndarray) -> np.ndarray:
    probs, cdf, b = h.probs[rows], h.cdf[rows], h.borders
    # 직선 CDF로 적분할 구간: uniform이면 전부, half-normal이면 양 끝을 뺀 안쪽
    lo, hi = (0, len(b) - 1) if h.tails == "uniform" else (1, len(b) - 2)
    left, right = b[lo:hi], b[lo + 1 : hi + 1]
    width = right - left
    a, c = cdf[:, lo:hi], cdf[:, lo + 1 : hi + 1]
    # 구간 전체가 y보다 오른쪽이면 H = 1, 아니면 H = 0으로 적분한다
    total = np.where(
        left[None, :] >= y[:, None],
        _segment_square_integral(a, c, 1.0, width),
        _segment_square_integral(a, c, 0.0, width),
    )
    # y가 든 구간은 y에서 둘로 나눈다
    inside = np.flatnonzero((y > b[lo]) & (y < b[hi]))
    if len(inside) and hi > lo:
        k = (np.searchsorted(b, y[inside], side="right") - 1).clip(lo, hi - 1)
        f_y = cdf[inside, k] + probs[inside, k] * _fraction(b, k, y[inside])
        split = _segment_square_integral(cdf[inside, k], f_y, 0.0, y[inside] - b[k])
        split += _segment_square_integral(f_y, cdf[inside, k + 1], 1.0, b[k + 1] - y[inside])
        total[inside, k - lo] = split
    out = total.sum(axis=1)
    if h.tails == "uniform":
        # 맨 바깥 경계 밖: F는 0 또는 1이고 H와 1만큼 다르다
        return out + (b[0] - y).clip(min=0.0) + (y - b[-1]).clip(min=0.0)
    # 왼쪽 꼬리는 t → −t로 뒤집으면 오른쪽 꼬리와 같은 식이다
    out += _right_tail_crps(probs[:, -1], b[-2], h.scale_right, y)
    out += _right_tail_crps(probs[:, 0], -b[1], h.scale_left, -y)
    return out


def interval_scores(
    h: BarHistogram, y, levels=COVERAGE_LEVELS, n_width_bins: int = 5
) -> dict[str, float]:
    """가운데 구간마다 coverage, 평균 폭, interval score, 폭 5분위별 coverage의 최솟값."""
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    alphas = 1.0 - np.asarray(levels, dtype=np.float64)
    bounds = quantile(h, np.concatenate([alphas / 2, 1 - alphas / 2]))  # 모든 level을 한 번에
    out = {}
    for j, (level, alpha) in enumerate(zip(levels, alphas, strict=True)):
        lo, hi = bounds[:, j], bounds[:, len(levels) + j]
        covered = (y >= lo) & (y <= hi)
        width = hi - lo
        score = width + (2 / alpha) * ((lo - y).clip(min=0) + (y - hi).clip(min=0))
        name = f"{round(level * 100)}"
        out[f"coverage_{name}"] = float(covered.mean())
        out[f"width_{name}"] = float(width.mean())
        out[f"interval_score_{name}"] = float(score.mean())
        out[f"coverage_min_by_width_{name}"] = min_coverage_by_width(covered, width, n_width_bins)
    return out


def min_coverage_by_width(
    covered: np.ndarray, width: np.ndarray, n_bins: int = 5, min_rows: int = 10
) -> float:
    """폭 n_bins분위마다의 coverage 가운데 최솟값 (행이 `min_rows`개 이상인 분위만).
    폭이 같은 행은 같은 평균 순위를 받아 같은 분위에 들어간다 (행 순서와 상관없다)."""
    if len(covered) < 2 * n_bins:
        return float("nan")
    rank = stats.rankdata(width, method="average")  # 1 ~ n
    bins = np.minimum(((rank - 0.5) / len(width) * n_bins).astype(int), n_bins - 1)
    per_bin = [covered[bins == j].mean() for j in range(n_bins) if (bins == j).sum() >= min_rows]
    return float(min(per_bin)) if per_bin else float("nan")


def regression_metrics(
    probs, borders, y, y_scale: float = 1.0, levels=COVERAGE_LEVELS, tails: str = DEFAULT_TAILS
) -> dict[str, float]:
    """fold 하나의 회귀 분포 지표. `y_scale`(학습 target 표준편차 등)로 나눈 CRPS·폭·interval
    score를 `_scaled`로 같이 낸다 (데이터셋끼리 평균할 때 쓴다)."""
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if len(y) == 0:
        return {}
    h = bar_histogram(probs, borders, tails)  # 정규화와 CDF는 한 번만
    out = {"crps": float(crps(h, y).mean())}
    out.update(interval_scores(h, y, levels))
    out["pit_ks"] = pit_ks(pit(h, y))
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
    """학습 fold에 있던 클래스의 행만 남기고(`amlb.drop_unseen_classes`) 정답을 `classes` 안의
    번호로 바꾼다."""
    y_true, proba = drop_unseen_classes(y_true, proba, classes)
    proba = np.asarray(proba, dtype=np.float64)
    proba = proba / proba.sum(axis=1, keepdims=True)
    index = {c: i for i, c in enumerate(np.asarray(classes).tolist())}
    return proba, np.array([index[c] for c in y_true.tolist()], dtype=int)


def ece(proba, labels, n_bins: int = 15) -> float:
    """top-label ECE. `labels`는 `proba` 열 번호."""
    proba, labels = np.asarray(proba, dtype=np.float64), np.asarray(labels)
    if len(labels) == 0:
        return float("nan")
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


_CONFORMAL_NAMES = ("coverage", "set_size", "sscs", "class_coverage_gap")


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
    if n < 20:
        return dict.fromkeys(_CONFORMAL_NAMES, float("nan"))
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
    if len(labels) == 0:  # 남는 행이 없으면 score_fold처럼 NaN
        names = ["ece", "log_loss", "brier"]
        names += [f"{kind}_{k}" for kind in CONFORMAL_SCORES for k in _CONFORMAL_NAMES]
        return dict.fromkeys(names, float("nan"))
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
