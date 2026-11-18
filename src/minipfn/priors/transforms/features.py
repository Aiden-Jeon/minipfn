"""열 값 거칠게 만들기 (issue #103, Kumo-Tabular 블로그의 "현실의 지저분함"을 다시 구현).

- `HighCardinality`: 수치 열 몇 개를 범주 값이 많은 열(우편번호·상점 id 같은)로 바꾼다
- `CoarseFeatures`: 수치 열 일부를 격자로 반올림한다(측정 정밀도). y는 원래 값으로 이미 만들어져
  있어, 같은 x에 다른 y가 생긴다

graph SCM prior에서는 데이터셋을 받은 뒤(`DropInvalid`·`CoverClasses` 다음), 결측값을 넣기 전에
쓴다. 바꾼 열은 데이터셋마다 `hp["high_cardinality_columns"]`(데이터셋 번호 → 열 번호 목록)에 적어
`CoarseFeatures`가 건드리지 않게 한다.
"""

from __future__ import annotations

import numpy as np
import torch

from minipfn.priors.pipeline import Group, Rng

HIGH_CARDINALITY_KEY = "high_cardinality_columns"


def _unique_counts(x: np.ndarray) -> np.ndarray:
    return np.array([len(np.unique(x[:, c])) for c in range(x.shape[1])])


class HighCardinality:
    """데이터셋마다 확률 `dataset_prob`로, 수치 열 1~`max_columns`개를 범주 K개의 열로 바꾼다.

    K ~ logU(`min_categories`, min(`max_categories`, 행 수 // `rows_per_category`))이고, 상한이
    하한보다 작으면 그 데이터셋은 그대로 둔다. 고유값이 K개 이상인 열만 고른다. 행을 값 순서로
    정렬해 이어진 K 구간으로 나누고, 구간 크기는 Zipf(지수 s ~ U(0, `max_zipf`)) 비율을 무작위
    순서로 쓴다(빈도가 치우친 범주. 행이 모자라는 작은 구간은 빈다). 구간 번호는 무작위로 섞어 값의
    순서 정보를 없애고, 열은 표준화한다(graph SCM의 다른 열처럼).
    """

    def __init__(
        self,
        dataset_prob: float,
        *,
        min_categories: int = 50,
        max_categories: int = 500,
        max_columns: int = 3,
        rows_per_category: int = 4,
        max_zipf: float = 1.2,
    ) -> None:
        self.dataset_prob = dataset_prob
        self.min_categories = min_categories
        self.max_categories = max_categories
        self.max_columns = max_columns
        self.rows_per_category = rows_per_category
        self.max_zipf = max_zipf

    def __call__(self, group: Group, rng: Rng) -> Group:
        r = rng.numpy
        xs, changed = [], {}
        for d in range(group.x.shape[1]):
            x = group.x[:, d]
            if r.random() < self.dataset_prob:
                values = x.cpu().numpy().copy()
                cols = self.convert(values, r)
                if cols:
                    changed[d] = cols
                    x = torch.from_numpy(values).to(x.device, x.dtype)
            xs.append(x)
        hp = {**group.hp, HIGH_CARDINALITY_KEY: changed} if changed else group.hp
        return Group(torch.stack(xs, 1), group.y, group.n_train, hp)

    def convert(self, x: np.ndarray, r: np.random.Generator) -> list[int]:
        """데이터셋 하나 x (행, feature)의 열을 그 자리에서 바꾸고, 바꾼 열 번호를 낸다."""
        n_rows, n_features = x.shape
        upper = min(self.max_categories, n_rows // self.rows_per_category)
        if upper < self.min_categories:
            return []
        k = int(round(np.exp(r.uniform(np.log(self.min_categories), np.log(upper)))))
        candidates = np.flatnonzero(_unique_counts(x) >= k)
        if len(candidates) == 0:
            return []
        n_cols = min(int(r.integers(1, self.max_columns + 1)), len(candidates))
        cols = sorted(int(c) for c in r.choice(candidates, n_cols, replace=False))
        for c in cols:
            x[:, c] = self.categories(x[:, c], k, r)
        return cols

    def categories(self, v: np.ndarray, k: int, r: np.random.Generator) -> np.ndarray:
        """열 하나를 범주 번호(표준화한 값)로."""
        s = r.uniform(0.0, self.max_zipf)
        sizes = r.permutation(np.arange(1, k + 1, dtype=np.float64) ** -s)
        edges = np.round(np.cumsum(sizes / sizes.sum()) * len(v)).astype(int)
        order = np.argsort(v, kind="stable")
        bins = np.empty(len(v), dtype=np.int64)
        bins[order] = np.searchsorted(edges, np.arange(len(v)), side="right")
        codes = r.permutation(k)[np.minimum(bins, k - 1)].astype(np.float64)
        return ((codes - codes.mean()) / max(codes.std(), 1e-6)).astype(v.dtype)


class CoarseFeatures:
    """데이터셋마다 확률 `dataset_prob`로, 고유값이 `min_unique`개 이상인 수치 열의
    U(`min_fraction`, `max_fraction`) 비율(최소 1개)을 격자로 반올림한다.

    격자 폭은 열 표준편차 × logU(`min_width`, `max_width`)이고 격자 시작점은 U(0, 폭)이다. 폭이
    표준편차 0.1~2배면 graph SCM 열(±4σ 안)이 약 4~80 단계 값으로 줄어든다. `HighCardinality`가
    바꾼 열은 건드리지 않는다.
    """

    def __init__(
        self,
        dataset_prob: float,
        *,
        min_fraction: float = 0.1,
        max_fraction: float = 0.5,
        min_width: float = 0.1,
        max_width: float = 2.0,
        min_unique: int = 20,
    ) -> None:
        self.dataset_prob = dataset_prob
        self.min_fraction = min_fraction
        self.max_fraction = max_fraction
        self.min_width = min_width
        self.max_width = max_width
        self.min_unique = min_unique

    def __call__(self, group: Group, rng: Rng) -> Group:
        r = rng.numpy
        skip = group.hp.get(HIGH_CARDINALITY_KEY, {})
        xs = []
        for d in range(group.x.shape[1]):
            x = group.x[:, d]
            if r.random() < self.dataset_prob:
                values = x.cpu().numpy().copy()
                if self.coarsen(values, r, skip=skip.get(d, ())):
                    x = torch.from_numpy(values).to(x.device, x.dtype)
            xs.append(x)
        return Group(torch.stack(xs, 1), group.y, group.n_train, group.hp)

    def coarsen(self, x: np.ndarray, r: np.random.Generator, skip=()) -> list[int]:
        """데이터셋 하나 x (행, feature)의 열을 그 자리에서 반올림하고, 바꾼 열 번호를 낸다."""
        counts = _unique_counts(x)
        candidates = np.array(
            [c for c in np.flatnonzero(counts >= self.min_unique) if c not in set(skip)], dtype=int
        )
        if len(candidates) == 0:
            return []
        fraction = r.uniform(self.min_fraction, self.max_fraction)
        n_cols = max(1, round(fraction * len(candidates)))
        cols = sorted(int(c) for c in r.choice(candidates, n_cols, replace=False))
        for c in cols:
            v = x[:, c].astype(np.float64)
            width = max(v.std(), 1e-6) * np.exp(
                r.uniform(np.log(self.min_width), np.log(self.max_width))
            )
            offset = r.uniform(0.0, width)
            x[:, c] = (np.round((v - offset) / width) * width + offset).astype(x.dtype)
        return cols
