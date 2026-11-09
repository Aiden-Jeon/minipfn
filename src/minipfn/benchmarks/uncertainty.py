"""예측 분포 품질과 불확실성 지표 (issue #45): 회귀 bar distribution의 CRPS·구간·PIT, 분류 확률의
ECE와 split conformal prediction의 (조건부) coverage.

점 예측 지표(ROC AUC, RMSE)는 분포가 잘 맞는지, 불확실성을 믿을 수 있는지는 보지 않는다. 학습 없이
checkpoint의 예측만으로 계산한다.

회귀 분포는 `bar_histogram(probs, borders, tails)`로 한 번 만든다: 구간 확률 `probs` (n, 구간 수)와
경계 `borders`. 경계는 TabPFN의 bar distribution처럼 모든 행이 같으면 (구간 수 + 1,), 행마다 다르면
(n, 구간 수 + 1)이다. 분위수를 내는 모델(GBDT quantile 회귀, TabICL, 정규분포 모수를 분위수로 바꾼
것)은 `quantile_histogram(levels, values)`로 행마다 다른 경계의 히스토그램으로 바꿔 넣는다(issue
#48). 안쪽 구간은 밀도가 균일해 CDF가 경계 사이에서 직선이다. 양 끝 구간은 `tails`로 정한다.

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
  교환 가능하다). 비순응 점수는 LAC(1 − p_y, Sadinle et al. 2019)와 무작위화한 APS(p_y보다 큰
  확률의 합 + u · p_y, Romano et al. 2020). 낸 값은 marginal coverage, 평균 집합 크기, SSCS
  (size-stratified coverage, Angelopoulos et al. 2021: 집합 크기 묶음 0~1, 2~3, 4~10, …마다의
  coverage 가운데 가장 낮은 값, 행이 `min_group`개 이상인 묶음만), 클래스 조건부 coverage
  gap(클래스마다 |coverage − (1 − α)|의 평균, Ding et al. 2023). 반복한 평가 절반을 모두 모아
  크기·클래스로 나눈다.

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
    borders: np.ndarray  # (n, B + 1), float64. 모든 행이 같은 경계면 `np.broadcast_to`한 view
    cdf: np.ndarray  # (n, B + 1), 경계 위의 CDF (양 끝 0과 1). 꼬리 종류와 상관없이 같다
    tails: str
    shared_borders: np.ndarray | None = None  # 모든 행이 같은 경계 (B + 1,)면 그 경계

    @property
    def scale_left(self) -> np.ndarray:
        """행마다 왼쪽 half-normal 꼬리의 scale (n,)."""
        return (self.borders[:, 1] - self.borders[:, 0]) / _HALF_NORMAL_MEDIAN

    @property
    def scale_right(self) -> np.ndarray:
        """행마다 오른쪽 half-normal 꼬리의 scale (n,)."""
        return (self.borders[:, -1] - self.borders[:, -2]) / _HALF_NORMAL_MEDIAN


def bar_histogram(probs, borders, tails: str = DEFAULT_TAILS) -> BarHistogram:
    """구간 확률 (n, B)와 경계로 히스토그램을 만든다. 경계는 모든 행이 같으면 (B + 1,), 행마다
    다르면 (n, B + 1)이다 (분위수 예측은 `quantile_histogram`으로 바꿔서 넣는다)."""
    if tails not in TAILS:
        raise ValueError(f"tails는 {TAILS} 가운데 하나: {tails}")
    probs = np.atleast_2d(np.asarray(probs, dtype=np.float64))
    borders = np.asarray(borders, dtype=np.float64)
    n, n_buckets = probs.shape
    if borders.shape not in ((n_buckets + 1,), (n, n_buckets + 1)):
        raise ValueError(
            f"경계는 (구간 수 + 1,)이나 (행 수, 구간 수 + 1)이어야 한다: {borders.shape}, "
            f"{probs.shape}"
        )
    if np.any(np.diff(borders, axis=-1) < 0):
        raise ValueError("경계는 줄어들면 안 된다")
    if tails == "half_normal" and (
        n_buckets < 2
        or np.any(borders[..., 1] <= borders[..., 0])
        or np.any(borders[..., -1] <= borders[..., -2])
    ):
        raise ValueError("half-normal 꼬리에는 구간이 2개 이상, 양 끝 구간의 폭이 0보다 커야 한다")
    shared = borders if borders.ndim == 1 else None
    if shared is not None:
        borders = np.broadcast_to(shared, (n, n_buckets + 1))  # 복사하지 않는다
    probs = probs.clip(min=0.0)
    probs = probs / probs.sum(axis=1, keepdims=True)
    cdf = np.concatenate([np.zeros((n, 1)), np.cumsum(probs, axis=1)], axis=1)
    cdf[:, -1] = 1.0
    return BarHistogram(probs, borders, cdf, tails, shared)


def quantile_histogram(levels, values) -> tuple[np.ndarray, np.ndarray]:
    """분위수 예측을 행마다 경계가 다른 구간 히스토그램으로 바꾼다 (`bar_histogram`에 넣는다).

    `levels` (K,)는 0과 1 사이에서 커지는 확률 수준, `values` (n, K)는 행마다 그 분위수다. CDF가
    점 (values[:, k], levels[k])를 지나고 그 사이에서 직선이라고 보면 이웃한 두 분위수 사이 구간의
    확률은 levels의 차이다(ScoringBench가 quantile을 CDF 점으로 바꿔 채점하는 방식). 그러면
    `quantile(h, levels)`가 `values`를 그대로 돌려준다. 분위수가 엇갈리면(GBDT가 수준마다 따로
    학습하면 생긴다) 행마다 정렬해서 쓴다(rearrangement, Chernozhukov et al. 2010). 같은 값의
    분위수는 폭 0인 구간, 곧 원자가 된다.

    가장 낮은 분위수 아래 levels[0], 가장 높은 분위수 위 1 − levels[-1]의 확률은 양 끝 구간으로
    둔다. 그 폭(half-normal 꼬리의 중앙값)은 꼬리의 밀도가 맞닿은 안쪽 구간의 밀도와 이어지도록
    정한다. 그 구간의 폭이 0이면 안쪽 전체의 평균 밀도를, 모든 분위수가 같으면 아주 작은 폭을 쓴다.

    돌려주는 값은 (구간 확률 (n, K + 1), 경계 (n, K + 2))다.
    """
    levels = np.asarray(levels, dtype=np.float64)
    values = np.atleast_2d(np.asarray(values, dtype=np.float64))
    if levels.ndim != 1 or len(levels) < 2 or values.shape[1] != len(levels):
        raise ValueError(f"분위수는 (행 수, 수준 수 ≥ 2)여야 한다: {levels.shape}, {values.shape}")
    if levels[0] <= 0 or levels[-1] >= 1 or np.any(np.diff(levels) <= 0):
        raise ValueError("확률 수준은 0과 1 사이에서 커져야 한다")
    if not np.isfinite(values).all():
        raise ValueError("분위수에 NaN이나 무한대가 있다")
    q = np.sort(values, axis=1)
    p_buckets = np.concatenate([levels[:1], np.diff(levels), 1 - levels[-1:]])
    probs = np.broadcast_to(p_buckets, (len(q), len(p_buckets))).copy()
    # 확률 1당 폭 (밀도의 역수). 맞닿은 구간 → 안쪽 평균 → 아주 작은 폭 순서로 쓴다
    inner = (q[:, -1] - q[:, 0]) / (levels[-1] - levels[0])
    tiny = 1e-9 * np.maximum(1.0, np.abs(q).max(axis=1))

    def tail_width(p_tail: float, gap: np.ndarray, p_gap: float) -> np.ndarray:
        per_prob = np.where(gap > 0, gap / p_gap, inner)
        # half-normal(scale s, 확률 p)의 맞닿은 점 밀도 p·√(2/π)/s가 1/per_prob와 같다
        width = _HALF_NORMAL_MEDIAN * p_tail * np.sqrt(2 / np.pi) * per_prob
        return np.maximum(width, tiny)

    left = tail_width(levels[0], q[:, 1] - q[:, 0], levels[1] - levels[0])
    right = tail_width(1 - levels[-1], q[:, -1] - q[:, -2], levels[-1] - levels[-2])
    borders = np.concatenate([(q[:, 0] - left)[:, None], q, (q[:, -1] + right)[:, None]], axis=1)
    return probs, borders


def _bucket_index(borders: np.ndarray, y: np.ndarray, shared: np.ndarray | None) -> np.ndarray:
    """행마다 borders[k] ≤ y < borders[k + 1]인 k (0 ~ 구간 수 − 1로 자른다). borders (m, B + 1)."""
    if shared is not None:
        k = np.searchsorted(shared, y, side="right") - 1
    else:
        k = (borders <= y[:, None]).sum(axis=1) - 1
    return k.clip(0, borders.shape[1] - 2)


def _upper_tail(u):
    """Q(u) = 1 − Φ(u)."""
    return special.ndtr(-u)


def _normal_pdf(u):
    return np.exp(-0.5 * u * u) / np.sqrt(2 * np.pi)


def _fraction(lo: np.ndarray, hi: np.ndarray, y: np.ndarray) -> np.ndarray:
    """구간 [lo, hi] 안에서 y의 위치 (0~1). 폭 0인 구간은 y가 그 점 이상이면 1."""
    width = hi - lo
    safe = np.where(width > 0, width, 1.0)
    return np.where(width > 0, (y - lo) / safe, (y >= lo).astype(float)).clip(0, 1)


def pit(h: BarHistogram, y) -> np.ndarray:
    """F(y) (n,)."""
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    b, rows = h.borders, np.arange(len(y))
    k = _bucket_index(b, y, h.shared_borders)
    out = h.cdf[rows, k] + h.probs[rows, k] * _fraction(b[rows, k], b[rows, k + 1], y)
    if h.tails == "half_normal":
        p_left, p_right = h.probs[:, 0], h.probs[:, -1]
        left = p_left * 2 * _upper_tail((b[:, 1] - y) / h.scale_left)
        right = 1 - p_right * 2 * _upper_tail((y - b[:, -2]) / h.scale_right)
        out = np.where(y < b[:, 1], left, np.where(y > b[:, -2], right, out))
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
    out = b[rows, k] + frac.clip(0.0, 1.0) * (b[rows, k + 1] - b[rows, k])
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
        out = np.where(in_left, b[:, 1:2] - h.scale_left[:, None] * u_left, out)
        out = np.where(in_right, b[:, -2:-1] + h.scale_right[:, None] * u_right, out)
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
    probs, cdf, b = h.probs[rows], h.cdf[rows], h.borders[rows]
    # 직선 CDF로 적분할 구간: uniform이면 전부, half-normal이면 양 끝을 뺀 안쪽
    lo, hi = (0, b.shape[1] - 1) if h.tails == "uniform" else (1, b.shape[1] - 2)
    left, right = b[:, lo:hi], b[:, lo + 1 : hi + 1]
    width = right - left
    a, c = cdf[:, lo:hi], cdf[:, lo + 1 : hi + 1]
    # 구간 전체가 y보다 오른쪽이면 H = 1, 아니면 H = 0으로 적분한다
    total = np.where(
        left >= y[:, None],
        _segment_square_integral(a, c, 1.0, width),
        _segment_square_integral(a, c, 0.0, width),
    )
    # y가 든 구간은 y에서 둘로 나눈다
    inside = np.flatnonzero((y > b[:, lo]) & (y < b[:, hi]))
    if len(inside) and hi > lo:
        b_in, y_in, r = b[inside], y[inside], np.arange(len(inside))
        k = _bucket_index(b_in, y_in, h.shared_borders).clip(lo, hi - 1)
        b_k, b_k1 = b_in[r, k], b_in[r, k + 1]
        f_y = cdf[inside, k] + probs[inside, k] * _fraction(b_k, b_k1, y_in)
        split = _segment_square_integral(cdf[inside, k], f_y, 0.0, y_in - b_k)
        split += _segment_square_integral(f_y, cdf[inside, k + 1], 1.0, b_k1 - y_in)
        total[inside, k - lo] = split
    out = total.sum(axis=1)
    if h.tails == "uniform":
        # 맨 바깥 경계 밖: F는 0 또는 1이고 H와 1만큼 다르다
        return out + (b[:, 0] - y).clip(min=0.0) + (y - b[:, -1]).clip(min=0.0)
    # 왼쪽 꼬리는 t → −t로 뒤집으면 오른쪽 꼬리와 같은 식이다
    out += _right_tail_crps(probs[:, -1], b[:, -2], h.scale_right[rows], y)
    out += _right_tail_crps(probs[:, 0], -b[:, 1], h.scale_left[rows], -y)
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
    """fold 하나의 회귀 분포 지표. 경계는 (B + 1,)이나 행마다 (n, B + 1)이다. `y_scale`(학습
    target 표준편차 등)로 나눈 CRPS·폭·interval score를 `_scaled`로 같이 낸다 (데이터셋끼리
    평균할 때 쓴다)."""
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


def conformal_scores(proba: np.ndarray, kind: str, u: np.ndarray | None = None) -> np.ndarray:
    """모든 클래스의 비순응 점수 (n, 클래스 수): 정답이 그 클래스일 때의 점수.

    APS는 무작위화한 원래 정의다: 클래스 k의 점수 = p_k보다 큰 확률의 합 + u · p_k
    (u ~ Uniform(0, 1), 행마다 하나). 무작위화하지 않으면(u = 1) 이진 분류에서 정확도가 1 − α보다
    낮을 때 threshold가 1이 되어 모든 집합이 클래스 전체가 된다.
    """
    if kind == "lac":
        return 1.0 - proba
    if kind == "aps":
        u = np.ones(len(proba)) if u is None else np.asarray(u, dtype=np.float64)
        above = (proba[:, None, :] * (proba[:, None, :] > proba[:, :, None])).sum(-1)
        return above + u[:, None] * proba
    raise ValueError(f"conformal 점수는 {CONFORMAL_SCORES} 가운데 하나: {kind}")


# SSCS의 집합 크기 묶음 (Angelopoulos et al. 2021): 0~1, 2~3, 4~10, 11~100, 101 이상.
# 0과 1을 한 묶음으로 두는 이유: 정확도가 1 − α보다 높으면 coverage를 1 − α로 맞추려고 빈 집합이
# 생기는데, 빈 집합은 정의상 coverage가 0이라 따로 두면 SSCS가 늘 0이 된다
SSCS_SIZE_BINS = (0, 2, 4, 11, 101)


def size_stratified_coverage(covered: np.ndarray, sizes: np.ndarray, min_group: int = 10) -> float:
    """집합 크기 묶음(`SSCS_SIZE_BINS`)마다의 coverage 가운데 최솟값 (행이 `min_group`개 이상만)."""
    groups = np.searchsorted(SSCS_SIZE_BINS, sizes, side="right")
    per_group = [
        covered[groups == g].mean() for g in np.unique(groups) if (groups == g).sum() >= min_group
    ]
    return float(min(per_group)) if per_group else float("nan")


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
    rng = np.random.default_rng(seed)
    scores = conformal_scores(proba, kind, u=rng.random(n))
    true_scores = scores[np.arange(n), labels]
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
    by_class = [covered[eval_labels == c].mean() for c in np.unique(eval_labels)]
    return {
        "coverage": float(covered.mean()),
        "set_size": float(sizes.mean()),
        "sscs": size_stratified_coverage(covered, sizes, min_group),
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


# ---------------------------------------------------------------------------
# 집계
# ---------------------------------------------------------------------------

DATASET_KEYS = ["suite", "task_id", "dataset"]


def suite_means(per_dataset) -> dict[str, float]:
    """suite 하나의 데이터셋별 지표 표(`DATASET_KEYS` 열 + 지표 열)의 데이터셋 평균. 단위가 있는
    회귀 지표(CRPS, 폭, interval score)는 학습 target 표준편차로 나눈 `_scaled`만 평균하고, 모두
    NaN인 열은 뺀다."""
    values = per_dataset.drop(columns=[c for c in DATASET_KEYS if c in per_dataset.columns])
    values = values.dropna(axis=1, how="all")
    unscaled = {c.removesuffix("_scaled") for c in values.columns if c.endswith("_scaled")}
    means = values.drop(columns=[c for c in unscaled if c in values.columns]).mean()
    return {k: float(v) for k, v in means.items()}
