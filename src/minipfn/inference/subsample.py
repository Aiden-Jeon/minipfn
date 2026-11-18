"""추론 때 학습 행(문맥)을 줄이는 subsample (issue #91: 큰 표에서 긴 문맥을 쓰는지 보는 실험).

`SubsampledEstimator`는 추론 estimator를 감싸 fit에 들어온 학습 행이 `max_rows`개보다 많으면
`max_rows`개만 무작위로 골라 넘긴다. 앙상블 멤버의 전처리·문맥은 모두 고른 행으로 맞추므로 멤버는
같은 subsample을 쓴다. 분류는 클래스 층화로 고르고 클래스마다 최소 1행을 남긴다
(`subsample_indices`).
"""

from __future__ import annotations

from typing import Any

import numpy as np


def subsample_indices(y: Any, n: int, *, stratify: bool, seed: int) -> np.ndarray:
    """학습 행 가운데 고를 `n`개의 위치 (오름차순). 행이 `n`개 이하면 모두.

    `stratify`면 클래스마다 `n · 클래스 비율`에 가깝게 고른다. 내림한 몫에 클래스마다 최소 1행을
    보장하고, 모자라면 몫의 소수부가 큰 클래스부터, 넘치면 몫보다 많이 받은 클래스부터 하나씩
    맞춘다. 클래스 수가 `n`보다 많으면 클래스마다 1행이라 `n`개보다 많을 수 있다. 같은 `y`·`n`·
    `seed`면 결과가 같다.
    """
    if n < 1:
        raise ValueError(f"n은 1 이상이어야 한다: {n}")
    y = np.asarray(y)
    total = len(y)
    if total <= n:
        return np.arange(total)
    rng = np.random.default_rng(seed)
    if not stratify:
        return np.sort(rng.choice(total, n, replace=False))
    _, inverse = np.unique(y, return_inverse=True)
    inverse = inverse.reshape(-1)
    counts = np.bincount(inverse)
    quota = n * counts / total
    take = np.minimum(np.maximum(np.floor(quota).astype(int), 1), counts)
    while take.sum() < n:  # 내림으로 모자란 몫: 소수부가 큰 클래스부터 (행이 남은 클래스만)
        room = np.flatnonzero(take < counts)
        take[room[np.argmax((quota - take)[room])]] += 1
    while take.sum() > n:  # 최소 1행으로 넘친 몫: 몫보다 많이 받은 클래스부터 (1행은 남긴다)
        spare = np.flatnonzero(take > 1)
        if not len(spare):
            break
        take[spare[np.argmax((take - quota)[spare])]] -= 1
    # 클래스마다 행 위치 (원래 순서) 가운데 take개를 무작위로
    by_class = np.split(np.argsort(inverse, kind="stable"), np.cumsum(counts)[:-1])
    parts = [rng.choice(rows, k, replace=False) for rows, k in zip(by_class, take, strict=True)]
    return np.sort(np.concatenate(parts))


ROW_ALIGNED_FIT_KWARGS = ("groups",)  # 행과 나란한 fit 인자: 같은 행으로 줄인다


def _take(value: Any, ix: np.ndarray) -> Any:
    return value.iloc[ix] if hasattr(value, "iloc") else np.asarray(value)[ix]


class SubsampledEstimator:
    """estimator를 감싸 fit의 학습 행을 `max_rows`개로 줄인다 (`subsample_indices`).

    행과 나란한 fit 인자(`ROW_ALIGNED_FIT_KWARGS`, 지금은 `groups`)도 같은 행으로 줄인다.
    원래 행 수와 고른 행 수는 `train_rows_`(원래, 고른 수)에 남는다. 나머지 속성(`predict`,
    `predict_proba`, `classes_`, `after_predict` 등)은 감싼 estimator의 것이다.
    """

    def __init__(self, estimator, max_rows: int, *, stratify: bool, seed: int = 0) -> None:
        if max_rows < 1:
            raise ValueError(f"max_rows는 1 이상이어야 한다: {max_rows}")
        self.estimator = estimator
        self.max_rows = max_rows
        self.stratify = stratify
        self.seed = seed

    def __getattr__(self, name: str):
        # 복사·pickle 도중(estimator가 아직 없을 때)에는 감싼 estimator를 찾지 않는다
        if name.startswith("__") or name == "estimator":
            raise AttributeError(name)
        return getattr(self.estimator, name)

    def fit(self, X, y, **fit_kwargs) -> SubsampledEstimator:
        total = len(y)
        ix = subsample_indices(y, self.max_rows, stratify=self.stratify, seed=self.seed)
        self.train_rows_ = (total, len(ix))
        if len(ix) < total:
            X, y = _take(X, ix), _take(y, ix)
            fit_kwargs = {
                k: _take(v, ix) if k in ROW_ALIGNED_FIT_KWARGS and v is not None else v
                for k, v in fit_kwargs.items()
            }
        self.estimator.fit(X, y, **fit_kwargs)
        return self
