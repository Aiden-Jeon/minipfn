"""앙상블 멤버의 전처리 (TabPFN v2 논문 Methods "Inference details"의 전처리 목록).

멤버 하나는 학습 행에 전처리를 맞추고(`MemberPreprocessor.fit`), 학습 행과 테스트 행에 같은 변환을
쓴다. 멤버 종류(`MEMBER_KINDS`)는 네 가지다.

- `"quantile"`: 분위수 변환한 열을 원래 열 옆에 붙이고 SVD 성분을 덧붙인다. 범주형 열은 번호를
  섞는다
- `"power"`: Yeo-Johnson power transform
- `"robust"`: RealMLP(Holzmüller et al. 2024)의 robust scaling + smooth clipping
  (`RobustSoftClip`)
- `"standard"`: 열 값을 바꾸지 않는다(이상치 자르기만). 표준화는 모델이 학습 행으로 한다. TabPFN-3.5
  추론이 이것 하나만 쓴다

모든 멤버에서 수치형 열의 평균으로부터 표준편차 `outlier_sigma`배(기본 `OUTLIER_SIGMA`) 밖의 값은
그 경계로 자르고, 열
순서를 섞고, 표준정규 난수로 된 샘플 식별자 열을 붙인다. `one_hot`인 멤버는 범주형 열을 one-hot으로
펼친다(`one_hot_sizes`). 어느 멤버를 어떻게 섞는지는 `minipfn.inference.ensemble`에 있다.

feature가 멤버 상한보다 많으면 멤버마다 일부 열만 쓴다(`feature_subset`, 공식 TabPFN 추론 코드의
`max_features_per_estimator`와 v2·2.5의 방법 "random". `balanced_feature_subsets`는 3.5의
"balanced").

공식 TabPFN-3.5 추론(PriorLabs/TabPFN v9 이후, checkpoint에 든 inference config)에 맞추는 옵션
(`PreprocessOptions`, 기본은 모두 꺼져 있어 예전과 같다). 공식 코드의 동작을 읽고 다시 구현했다.

- `shuffle_categories`: 분위수가 아닌 멤버도 범주 번호를 멤버마다 섞는다 (공식 3.5의
  `ordinal_shuffled`. 분위수 멤버는 원래 섞는다)
- `fingerprint="hash"`: 샘플 식별자 열을 난수 대신 행 값의 hash로 만든다 (공식
  `FINGERPRINT_FEATURE`). 0~1 사이 값이고, 학습 행에서 같은 행이 여럿이면 둘째부터 번호를 더해 다시
  hash하고, 테스트 행은 그대로 hash해서 학습 행과 같은 행이면 그 학습 행(처음 것)과 같은 값이 된다
- `outlier_clip="soft"`: 이상치를 경계로 자르지 않고 부드럽게 누른다 (공식 3.5). 경계는 두 번
  구한다: 평균 ± σ배 밖을 빼고 평균·표준편차를 다시 구한다. 경계 밖 값 x는 경계에서 log(1 + |x|)만큼
  더 간 곳으로 (그쪽이 x보다 안쪽일 때만) 옮긴다. 공식은 분포 변환 뒤에 하지만 3.5의 standard
  멤버는 변환이 없어 자리가 같다
- `low_cardinality_values`: 학습 행이 `MIN_ROWS_INFER_CATEGORICAL`보다 많을 때 서로 다른 값(결측
  포함)이 `MIN_UNIQUE_NUMERICAL`개보다 적은 수치 열을 범주형으로 본다 (공식 type 추론)
- `constant_columns`: 학습 행에서 값이 하나뿐인(결측만인 것 포함) 열을 뺀다 (공식
  `RemoveConstantFeaturesStep`. 범주형으로 준 열은 빼지 않는다)
"""

from __future__ import annotations

import hashlib
import warnings
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import OrdinalEncoder, PowerTransformer, QuantileTransformer

OUTLIER_SIGMA = 12.0
MAX_ONE_HOT_CATEGORIES = 20  # benchmark의 socmob(17개)은 펼치고 colleges(수천 개)는 번호로 둔다
ONE_HOT_MODES = ("off", "power", "all")
MEMBER_KINDS = ("quantile", "power", "robust", "standard")
SOFT_CLIP = 3.0  # RealMLP smooth clipping의 폭
FINGERPRINTS = ("random", "hash")
OUTLIER_CLIPS = ("hard", "soft")
FEATURE_SUBSAMPLINGS = ("random", "balanced")
# 학습 행에 없던 범주: "nan"(기본, 결측과 같게) 또는 "minus1"(번호 -1이라는 보통 값. TabICLv2 방식)
UNSEEN_CATEGORIES = ("nan", "minus1")
UNSEEN_CODE = -1.0
# 공식 type 추론: 학습 행이 100개보다 많으면 서로 다른 값이 4개보다 적은 수치 열은 범주형
MIN_UNIQUE_NUMERICAL = 4
MIN_ROWS_INFER_CATEGORICAL = 100
_HASH_DECIMALS = 12  # hash 전에 반올림하는 자리 (계산 오차로 같은 행이 다른 값이 되지 않도록)
_HASH_MAX = 2**64 - 1
_HASH_RETRIES = 100


@dataclass(frozen=True)
class PreprocessOptions:
    """멤버 전처리의 공식 3.5 옵션 (모듈 docstring). 기본값은 예전 동작이다."""

    shuffle_categories: bool = False
    fingerprint: str = "random"
    outlier_clip: str = "hard"
    unseen_category: str = "nan"  # "minus1"이면 one-hot 블록에서 -1(학습에 없던 범주)은 모두 0

    def __post_init__(self) -> None:
        if self.fingerprint not in FINGERPRINTS:
            raise ValueError(f"fingerprint는 {FINGERPRINTS} 가운데 하나다: {self.fingerprint}")
        if self.outlier_clip not in OUTLIER_CLIPS:
            raise ValueError(f"outlier_clip은 {OUTLIER_CLIPS} 가운데 하나다: {self.outlier_clip}")


def unseen_value(unseen_category: str) -> float:
    """학습 행에 없던 범주의 번호 (`UNSEEN_CATEGORIES`)."""
    if unseen_category not in UNSEEN_CATEGORIES:
        raise ValueError(f"unseen_category는 {UNSEEN_CATEGORIES} 가운데 하나다: {unseen_category}")
    return np.nan if unseen_category == "nan" else UNSEEN_CODE


def to_numeric(
    X: Any, categorical: list[int] | None, unseen_category: str = "nan"
) -> tuple[np.ndarray, list[int], Any]:
    """범주형·문자열 열을 순서 번호로 바꾼다. 결측은 NaN으로 남긴다. 학습 행에 없던 범주(다른
    행에 쓸 때)는 기본으로 NaN, `unseen_category="minus1"`이면 -1이다."""
    df = pd.DataFrame(X).reset_index(drop=True)
    cat = set(categorical or [])
    cat |= {i for i, dtype in enumerate(df.dtypes) if not pd.api.types.is_numeric_dtype(dtype)}
    cat_ix = sorted(cat)
    encoder = None
    if cat_ix:
        encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            # sklearn은 정수나 np.nan만 받는다
            unknown_value=np.nan if unseen_category == "nan" else int(UNSEEN_CODE),
        )
        unseen_value(unseen_category)  # 값 확인
        encoder.fit(df.iloc[:, cat_ix].astype(str).where(df.iloc[:, cat_ix].notna()))
    return apply_encoder(df, cat_ix, encoder), cat_ix, encoder


def apply_encoder(df: pd.DataFrame, cat_ix: list[int], encoder: Any) -> np.ndarray:
    df = pd.DataFrame(df).reset_index(drop=True)
    out = np.empty(df.shape, dtype=np.float64)
    num_ix = [i for i in range(df.shape[1]) if i not in cat_ix]
    if num_ix:
        out[:, num_ix] = df.iloc[:, num_ix].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if cat_ix:
        cats = df.iloc[:, cat_ix]
        out[:, cat_ix] = encoder.transform(cats.astype(str).where(cats.notna()))
    return out


def feature_subset(n_features: int, max_features: int | None, seed: int) -> np.ndarray | None:
    """멤버 하나가 쓸 원래 열 번호(정렬한 것). 열이 `max_features`보다 적거나 같으면(또는 None이면)
    None(전부)이다.

    공식 TabPFN 추론 코드가 v2·2.5에 쓰는 "random" 방법(`_subsample_features_random`)처럼 멤버마다
    따로 뽑는다. `seed`는 멤버의 것(`Member.seed`)에서 정해서, 같은 멤버는 어느 fit에서나(greedy의
    fold와 마지막 fit) 같은 열을 쓴다. 상한 안의 표에는 난수를 쓰지 않아 예전과 같다."""
    if max_features is None or n_features <= max_features:
        return None
    rng = np.random.default_rng(seed)
    return np.sort(rng.permutation(n_features)[:max_features])


def balanced_feature_subsets(
    n_features: int, max_features: int | None, n_members: int, rng: np.random.Generator
) -> list[np.ndarray | None]:
    """멤버마다 쓸 원래 열 번호를 고르게 나눠 뽑는다 (공식 3.5의 "balanced").

    열을 한 번 섞어 둔 순서에서 멤버마다 차례로 `max_features`개씩 가져가고, 다 쓰면 다시 섞어
    채운다(그 멤버가 이미 가진 열은 빼고). 그래서 열마다 쓰이는 횟수가 거의 같다. 상한 안의 표는
    모두 None(전부)이고 난수를 쓰지 않는다."""
    if max_features is None or n_features <= max_features:
        return [None] * n_members
    order = rng.permutation(n_features)
    pool: list[int] = []
    out: list[np.ndarray | None] = []
    for _ in range(n_members):
        slots: list[int] = []
        while len(slots) < max_features:
            if not pool:
                taken = set(slots)
                pool = [i for i in range(n_features) if i not in taken]
                rng.shuffle(pool)
            take = min(max_features - len(slots), len(pool))
            slots += pool[:take]
            pool = pool[take:]
        out.append(np.sort(order[np.array(slots)]))
    return out


def low_cardinality_values(X: np.ndarray, cat_ix: list[int]) -> dict[int, np.ndarray]:
    """범주형으로 볼 수치 열과 그 열의 값(정렬, 결측 제외): 학습 행이
    `MIN_ROWS_INFER_CATEGORICAL`개보다 많을 때 서로 다른 값(결측도 하나로 센다)이
    `MIN_UNIQUE_NUMERICAL`개보다 적은 열. 값이 하나뿐인 열은 상수 열이라 빼지 않는다."""
    if X.shape[0] <= MIN_ROWS_INFER_CATEGORICAL:
        return {}
    cat = set(cat_ix)
    out = {}
    for i in range(X.shape[1]):
        if i in cat:
            continue
        col = X[:, i]
        values = np.unique(col[~np.isnan(col)])
        n_distinct = len(values) + int(np.isnan(col).any())
        if 1 < n_distinct < MIN_UNIQUE_NUMERICAL:
            out[i] = values
    return out


def encode_low_cardinality(
    X: np.ndarray, values: dict[int, np.ndarray], unseen: float = np.nan
) -> np.ndarray:
    """`low_cardinality_values`의 열을 값의 순서 번호(0부터)로 바꾼다. 결측은 결측, 학습 행에 없던
    값은 `unseen`(기본 결측, `unseen_value`)."""
    X = X.copy()
    for i, v in values.items():  # v에는 값이 하나 이상 있다
        col = X[:, i]
        ix = np.clip(np.searchsorted(v, col), 0, len(v) - 1)
        other = np.where(np.isnan(col), np.nan, unseen)
        X[:, i] = np.where(v[ix] == col, ix, other)  # 결측은 v[ix] == NaN이 거짓
    return X


def constant_columns(X: np.ndarray, keep: list[int] | None = None) -> list[int]:
    """학습 행에서 서로 다른 값(결측도 하나로 센다)이 하나뿐인 열. `keep`(범주형으로 준 열)은
    뺀다."""
    keep_set = set(keep or [])
    out = []
    for i in range(X.shape[1]):
        if i in keep_set:
            continue
        col = X[:, i]
        nan = np.isnan(col)
        if len(np.unique(col[~nan])) + int(nan.any()) <= 1:
            out.append(i)
    return out


def row_fingerprint(X: np.ndarray, salt: int, unique: bool) -> np.ndarray:
    """행마다 0~1 사이의 hash 값 (`PreprocessOptions.fingerprint="hash"`). 행 값을 반올림해 salt와
    함께 hash한다. `unique`(학습 행)면 같은 행이 다시 나올 때 번호를 더해 다른 값으로 만든다(값이
    모두 결측인 행은 그대로 둔다). 테스트 행은 `unique=False`: 학습 행과 같은 행은 그 행과 같은
    값이다."""
    rows = np.ascontiguousarray(np.round(np.asarray(X, dtype=np.float64), _HASH_DECIMALS))

    def h(data: bytes, s: int) -> float:
        digest = hashlib.sha256(data + s.to_bytes(8, "little", signed=False)).digest()
        return int.from_bytes(digest[-8:], "big") / _HASH_MAX

    out = np.empty(len(rows))
    seen: set[float] = set()
    count: dict[float, int] = defaultdict(int)
    for i, row in enumerate(rows):
        data = row.tobytes()
        base = h(data, salt)
        if not unique:
            out[i] = base
            continue
        k = count[base]
        value = base if k == 0 else h(data, salt + k)
        all_missing = bool(np.isnan(row).all())
        tries = 0
        while value in seen and not all_missing:
            k += 1
            tries += 1
            if tries > _HASH_RETRIES:
                raise RuntimeError(f"행 {i}의 hash 충돌이 {_HASH_RETRIES}번 안에 풀리지 않는다")
            value = h(data, salt + k)
        out[i] = value
        seen.add(value)
        count[base] = k + 1
    return out


def select_features(
    X: np.ndarray, cat_ix: list[int], features: np.ndarray | None
) -> tuple[np.ndarray, list[int]]:
    """X의 열 일부(`features`, None이면 전부)와 그 안에서의 범주형 열 번호."""
    if features is None:
        return X, cat_ix
    cat = set(cat_ix)
    return X[:, features], [j for j, i in enumerate(features) if int(i) in cat]


def _one_hot_width(n: int) -> int:
    return 1 if n == 2 else n  # 범주 2개는 0/1 열 하나


def one_hot_sizes(n_categories: dict[int, int], n_features: int, max_width: int) -> dict[int, int]:
    """one-hot으로 펼칠 열과 그 범주 수.

    범주가 `MAX_ONE_HOT_CATEGORIES`개 이하인 열을 펼친다. 펼친 뒤 열 수(원래 열 기준)가
    `max_width`(모델이 학습 때 본 feature 수)를 넘으면 범주가 많은 열부터 번호로 되돌린다 (열 수가
    늘지 않는 범주 2개 이하 열은 그대로 펼친다).
    """
    sizes = {i: n for i, n in n_categories.items() if n <= MAX_ONE_HOT_CATEGORIES}
    width = n_features + sum(_one_hot_width(n) - 1 for n in sizes.values())
    wide = [i for i in sizes if _one_hot_width(sizes[i]) > 1]
    for i in sorted(wide, key=lambda i: -sizes[i]):
        if width <= max_width:
            break
        width -= _one_hot_width(sizes.pop(i)) - 1
    return sizes


class RobustSoftClip:
    """RealMLP의 robust scaling + smooth clipping (열마다, 결측은 NaN 그대로)."""

    def fit(self, X: np.ndarray) -> RobustSoftClip:
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # 모두 결측인 열
            low, q1, median, q3, high = np.nanquantile(X, [0, 0.25, 0.5, 0.75, 1], axis=0)
            span = high - low
        iqr = q3 - q1
        self.scale_ = np.zeros(X.shape[1])
        ok = iqr > 0
        self.scale_[ok] = 1.0 / iqr[ok]
        flat = ~ok & (span > 0)  # 사분위 폭이 0인 열(대부분 같은 값)
        self.scale_[flat] = 2.0 / span[flat]
        self.median_ = np.nan_to_num(median)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        z = (X - self.median_) * self.scale_
        return z / np.sqrt(1.0 + (z / SOFT_CLIP) ** 2)


def one_hot_kinds(one_hot: str, member_kinds: tuple[str, ...]) -> frozenset[str]:
    """`one_hot` 모드에서 범주형 열을 펼치는 멤버 종류. 이 집합이 같은 두 모드는 결과도 같다."""
    if one_hot == "all":
        return frozenset(member_kinds)
    if one_hot == "power":  # 분위수가 아닌 멤버(power·robust·standard)만
        return frozenset(k for k in member_kinds if k != "quantile")
    return frozenset()


@dataclass
class Member:
    kind: str  # "quantile" | "power" | "robust" | "standard"
    class_permutation: np.ndarray | None
    target_power: bool
    seed: int
    one_hot: bool = False
    outlier_sigma: float = OUTLIER_SIGMA  # 평균에서 표준편차 몇 배 밖을 자르는지


class MemberPreprocessor:
    """멤버 하나의 전처리. fit은 학습 행으로만 한다."""

    def __init__(
        self,
        member: Member,
        cat_ix: list[int],
        max_one_hot_width: int,
        options: PreprocessOptions | None = None,
    ) -> None:
        self.member = member
        self.cat_ix = cat_ix
        self.max_one_hot_width = max_one_hot_width
        self.options = options or PreprocessOptions()

    def _bounds(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # 평균 ± σ배. soft면 그 밖의 값을 빼고 한 번 더 구한다 (모듈 docstring)
        sigma = self.member.outlier_sigma
        mean, std = np.nanmean(X, axis=0), np.nanstd(X, axis=0)
        lower, upper = mean - sigma * std, mean + sigma * std
        if self.options.outlier_clip == "hard":
            return lower, upper
        if X.shape[0] <= 1:
            return np.full(X.shape[1], -np.inf), np.full(X.shape[1], np.inf)
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # 모두 결측인 열
            inside = np.where((X < lower) | (X > upper), np.nan, X)
            mean, std = np.nanmean(inside, axis=0), np.nanstd(inside, axis=0)
        lower, upper = mean - sigma * std, mean + sigma * std
        # 모두 결측인 열은 경계가 결측이다: 자르지 않는다 (np.maximum이 결측을 퍼뜨리지 않도록)
        return np.nan_to_num(lower, nan=-np.inf), np.nan_to_num(upper, nan=np.inf)

    def fit(self, X: np.ndarray) -> MemberPreprocessor:
        rng = np.random.default_rng(self.member.seed)
        self.lower, self.upper = self._bounds(X)
        # 범주형 열은 자르지 않는다 (번호를 자르면 드문 범주가 다른 범주와 합쳐진다)
        self.lower[self.cat_ix], self.upper[self.cat_ix] = -np.inf, np.inf
        X = self._clip(X)
        num_ix = [i for i in range(X.shape[1]) if i not in self.cat_ix]
        self.num_ix = num_ix
        # 범주형 열의 번호는 0부터 (학습 행의 범주 수 - 1)까지다
        n_categories = {
            i: int(np.nanmax(X[:, i])) + 1 if np.isfinite(X[:, i]).any() else 1 for i in self.cat_ix
        }
        self.one_hot_sizes = (
            one_hot_sizes(n_categories, X.shape[1], self.max_one_hot_width)
            if self.member.one_hot
            else {}
        )
        self.cat_maps = {}
        if self.member.kind == "quantile" or self.options.shuffle_categories:
            # 범주형 열: 번호를 섞는다 (분위수 멤버, 또는 shuffle_categories면 모든 멤버.
            # one-hot으로 펼치는 열은 빼고)
            for i in self.cat_ix:
                if i not in self.one_hot_sizes:
                    self.cat_maps[i] = rng.permutation(n_categories[i])
        if self.member.kind == "quantile":
            n_quantiles = max(min(X.shape[0] // 10, 1000), 2)
            self.transformer = QuantileTransformer(
                n_quantiles=n_quantiles, random_state=self.member.seed
            )
        elif self.member.kind == "power":
            self.transformer = PowerTransformer(method="yeo-johnson", standardize=True)
        elif self.member.kind == "robust":
            self.transformer = RobustSoftClip()
        else:  # standard: 값은 그대로
            self.transformer = None
        # 학습 행에 값이 하나도 없는 수치 열은 변환기에 넣지 않고 결측으로 둔다(Yeo-Johnson은 빈
        # 열에서 실패한다. 분위수·robust 변환은 그 열을 원래도 NaN으로 냈다)
        self.fit_ix = [i for i in num_ix if np.isfinite(X[:, i]).any()]
        if self.fit_ix and self.transformer is not None:
            self.transformer.fit(X[:, self.fit_ix])
        self.svd = None
        base = self._base(X)
        n_components = 0
        if self.member.kind == "quantile" and X.shape[1] >= 2:
            # SVD 성분 min(n/10 + 1, d/2)개. 표준화하고 결측은 0으로 채워 넣는다
            n_components = max(1, min(X.shape[0] // 10 + 1, X.shape[1] // 2))
            self.svd_mean = np.nanmean(base, axis=0)
            self.svd_std = np.nanstd(base, axis=0) + 1e-8
            self.svd = TruncatedSVD(n_components=n_components, random_state=self.member.seed)
            self.svd.fit(self._standardize(base))
        # 열 순서: 변환한 열 + SVD 성분 + 샘플 식별자 1개를 한꺼번에 섞는다
        self.order = rng.permutation(base.shape[1] + n_components + 1)
        # fingerprint="hash"의 salt: 학습 행의 셀 수 (식별자 열 앞까지)
        self.n_cells = X.shape[0] * (base.shape[1] + n_components)
        return self

    def _standardize(self, base: np.ndarray) -> np.ndarray:
        return np.nan_to_num((base - self.svd_mean) / self.svd_std, nan=0.0)

    def _clip(self, X: np.ndarray) -> np.ndarray:
        if self.options.outlier_clip == "hard":
            return np.clip(X, self.lower, self.upper)
        # 경계 밖은 경계에서 log(1 + |x|)만큼 더 간 곳과 x 가운데 안쪽 (모듈 docstring)
        with np.errstate(invalid="ignore"):
            soft = np.log1p(np.abs(X))
            X = np.maximum(self.lower - soft, X)
            return np.minimum(self.upper + np.log1p(np.abs(X)), X)

    def _base(self, X: np.ndarray) -> np.ndarray:
        X = X.copy()
        for i, perm in self.cat_maps.items():
            # 학습 행에 없던 범주(-1, unseen_category="minus1")는 섞지 않고 -1로 둔다
            ok = np.isfinite(X[:, i]) & (X[:, i] >= 0) & (X[:, i] < len(perm))
            X[ok, i] = perm[X[ok, i].astype(int)]
        transformed = X.copy()
        empty = [i for i in self.num_ix if i not in self.fit_ix]
        transformed[:, empty] = np.nan
        if self.fit_ix and self.transformer is not None:
            with np.errstate(all="ignore"):
                transformed[:, self.fit_ix] = self.transformer.transform(X[:, self.fit_ix])
        if self.member.kind == "quantile":
            out = np.concatenate([X, transformed[:, self.num_ix]], axis=1)  # 원래 열 + 분위수 열
        else:
            out = transformed
        if not self.one_hot_sizes:
            return out
        # 번호 열을 빼고 one-hot 열을 뒤에 붙인다 (앞 d개 열이 원래 열 순서라 위치가 같다)
        keep = [j for j in range(out.shape[1]) if j not in self.one_hot_sizes]
        unseen_zero = self.options.unseen_category == "minus1"
        blocks = [self._one_hot(X[:, i], n, unseen_zero) for i, n in self.one_hot_sizes.items()]
        return np.concatenate([out[:, keep], *blocks], axis=1)

    @staticmethod
    def _one_hot(codes: np.ndarray, n: int, unseen_zero: bool = False) -> np.ndarray:
        # 결측과 범위 밖 번호는 NaN: 번호 열과 똑같이 결측으로 본다. `unseen_zero`
        # (unseen_category="minus1")면 학습 행에 없던 범주(-1)는 어느 범주도 아니므로 모두 0
        known = np.isfinite(codes) & (codes >= 0) & (codes < n)
        block = np.full((len(codes), n), np.nan)
        block[known] = 0.0
        if unseen_zero:
            block[codes == UNSEEN_CODE] = 0.0
        block[np.flatnonzero(known), codes[known].astype(int)] = 1.0
        return block[:, 1:] if n == 2 else block

    def transform(
        self, X: np.ndarray, rng: np.random.Generator, is_test: bool = False
    ) -> np.ndarray:
        """`is_test`는 fingerprint="hash"에서만 쓴다 (학습 행만 같은 행을 다른 값으로 만든다)."""
        base = self._base(self._clip(X))
        parts = [base]
        if self.svd is not None:
            parts.append(self.svd.transform(self._standardize(base)))
        # 샘플 식별자 열
        if self.options.fingerprint == "hash":
            features = np.concatenate(parts, axis=1)
            ident = row_fingerprint(features, self.n_cells, unique=not is_test)[:, None]
        else:
            ident = rng.standard_normal((X.shape[0], 1))
        parts.append(ident)
        return np.concatenate(parts, axis=1)[:, self.order]
