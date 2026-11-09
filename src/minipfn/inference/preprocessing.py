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
`max_features_per_estimator`와 v2·2.5의 방법 "random").
"""

from __future__ import annotations

import warnings
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


def to_numeric(X: Any, categorical: list[int] | None) -> tuple[np.ndarray, list[int], Any]:
    """범주형·문자열 열을 순서 번호로 바꾼다. 결측은 NaN으로 남긴다."""
    df = pd.DataFrame(X).reset_index(drop=True)
    cat = set(categorical or [])
    cat |= {i for i, dtype in enumerate(df.dtypes) if not pd.api.types.is_numeric_dtype(dtype)}
    cat_ix = sorted(cat)
    encoder = None
    if cat_ix:
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan)
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

    def __init__(self, member: Member, cat_ix: list[int], max_one_hot_width: int) -> None:
        self.member = member
        self.cat_ix = cat_ix
        self.max_one_hot_width = max_one_hot_width

    def fit(self, X: np.ndarray) -> MemberPreprocessor:
        rng = np.random.default_rng(self.member.seed)
        mean, std = np.nanmean(X, axis=0), np.nanstd(X, axis=0)
        sigma = self.member.outlier_sigma
        self.lower, self.upper = mean - sigma * std, mean + sigma * std
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
        if self.member.kind == "quantile":
            # 범주형 열: 번호를 섞는다 (분위수 멤버만, one-hot으로 펼치는 열은 빼고)
            for i in self.cat_ix:
                if i not in self.one_hot_sizes:
                    self.cat_maps[i] = rng.permutation(n_categories[i])
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
        return self

    def _standardize(self, base: np.ndarray) -> np.ndarray:
        return np.nan_to_num((base - self.svd_mean) / self.svd_std, nan=0.0)

    def _clip(self, X: np.ndarray) -> np.ndarray:
        return np.clip(X, self.lower, self.upper)

    def _base(self, X: np.ndarray) -> np.ndarray:
        X = X.copy()
        for i, perm in self.cat_maps.items():
            ok = np.isfinite(X[:, i]) & (X[:, i] < len(perm))
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
        blocks = [self._one_hot(X[:, i], n) for i, n in self.one_hot_sizes.items()]
        return np.concatenate([out[:, keep], *blocks], axis=1)

    @staticmethod
    def _one_hot(codes: np.ndarray, n: int) -> np.ndarray:
        # 결측과 범위 밖 번호는 NaN: 번호 열과 똑같이 결측으로 본다
        known = np.isfinite(codes) & (codes >= 0) & (codes < n)
        block = np.full((len(codes), n), np.nan)
        block[known] = 0.0
        block[np.flatnonzero(known), codes[known].astype(int)] = 1.0
        return block[:, 1:] if n == 2 else block

    def transform(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        base = self._base(self._clip(X))
        parts = [base]
        if self.svd is not None:
            parts.append(self.svd.transform(self._standardize(base)))
        parts.append(rng.standard_normal((X.shape[0], 1)))  # 샘플 식별자 열
        return np.concatenate(parts, axis=1)[:, self.order]
