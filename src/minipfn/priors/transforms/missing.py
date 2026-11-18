"""결측값 (TabPFN v2 논문 "Post-processing"): 일부 데이터셋에서 셀마다 같은 확률로 값을
지운다(MCAR, `Missing`). `PatternMissing`은 결측이 다른 열·자기 값·target에 따라 생기는 패턴을
섞는다 (issue #103)."""

from __future__ import annotations

import numpy as np
import torch

from minipfn.priors.pipeline import Group, Rng

# `PatternMissing`이 고르는 결측 패턴 (데이터셋마다 같은 확률). 무엇이 결측을 정하는지:
# mcar 잡음만, mar 다른 열의 값, mnar 자기 열의 값, mnar_y target
MISSING_PATTERNS = ("mcar", "mar", "mnar", "mnar_y")


class Missing:
    """데이터셋마다 확률 `dataset_prob`로, 비율 r ~ U(0, max_rate)만큼 x를 NaN으로 바꾼다.

    두 값은 v2 논문에 없어서 graph SCM prior에서 정했다(`missing_dataset_prob`, `max_missing_rate`).
    """

    def __init__(self, dataset_prob: float, max_rate: float) -> None:
        self.dataset_prob = dataset_prob
        self.max_rate = max_rate

    def __call__(self, group: Group, rng: Rng) -> Group:
        r = rng.numpy
        xs = []
        for d in range(group.x.shape[1]):
            x = group.x[:, d]
            if r.random() < self.dataset_prob:
                rate = r.uniform(0.0, self.max_rate)
                mask = torch.from_numpy(r.random(tuple(x.shape)) < rate).to(x.device)
                x = x.masked_fill(mask, float("nan"))
            xs.append(x)
        return Group(torch.stack(xs, 1), group.y, group.n_train, group.hp)


class PatternMissing:
    """결측 패턴을 섞은 결측값 (issue #103, Kumo-Tabular 블로그의 "여러 결측 패턴"을 다시 구현).

    데이터셋마다 확률 `dataset_prob`로 결측을 넣는다. 넣는 데이터셋은 패턴 하나를
    `MISSING_PATTERNS`에서 같은 확률로 고르고, 열의 U(`min_column_fraction`, 1) 비율만 결측이
    생긴다.
    그 열마다 결측률 rate ~ U(0, `max_column_rate`)를 뽑아 score = a·driver + Logistic 잡음이 큰
    rate × 행 수 개의 행을 NaN으로 바꾼다. driver는 패턴에 따라 없음(mcar), 다른 열 하나(mar), 자기
    열(mnar), target(mnar_y, 분류면 클래스마다 N(0, 1) 값)이고 표준화해 쓴다. 세기 a는
    ±logU(`min_strength`, `max_strength`)다. 열이 하나뿐이면 mar는 mcar로 둔다. 학습 행·검증 행에
    같은 규칙을 쓴다.
    """

    def __init__(
        self,
        dataset_prob: float,
        *,
        classification: bool,
        max_column_rate: float = 0.5,
        min_column_fraction: float = 0.1,
        min_strength: float = 0.5,
        max_strength: float = 4.0,
    ) -> None:
        self.dataset_prob = dataset_prob
        self.classification = classification
        self.max_column_rate = max_column_rate
        self.min_column_fraction = min_column_fraction
        self.min_strength = min_strength
        self.max_strength = max_strength

    def __call__(self, group: Group, rng: Rng) -> Group:
        r = rng.numpy
        xs = []
        for d in range(group.x.shape[1]):
            x = group.x[:, d]
            if r.random() < self.dataset_prob:
                mask = self.mask(x.cpu().numpy(), group.y[:, d].cpu().numpy(), r)
                x = x.masked_fill(torch.from_numpy(mask).to(x.device), float("nan"))
            xs.append(x)
        return Group(torch.stack(xs, 1), group.y, group.n_train, group.hp)

    def mask(self, x: np.ndarray, y: np.ndarray, r: np.random.Generator) -> np.ndarray:
        """데이터셋 하나 x (행, feature), y (행)의 결측 위치 (행, feature) bool."""
        n_rows, n_features = x.shape
        pattern = MISSING_PATTERNS[r.integers(len(MISSING_PATTERNS))]
        if pattern == "mar" and n_features < 2:
            pattern = "mcar"
        n_cols = max(1, round(r.uniform(self.min_column_fraction, 1.0) * n_features))
        cols = r.choice(n_features, n_cols, replace=False)
        if pattern == "mnar_y":
            target = r.normal(size=int(y.max()) + 1)[y.astype(int)] if self.classification else y
        mask = np.zeros((n_rows, n_features), dtype=bool)
        for c in cols:
            k = round(r.uniform(0.0, self.max_column_rate) * n_rows)
            score = r.logistic(size=n_rows)
            if pattern != "mcar":
                if pattern == "mar":
                    driver = x[:, r.choice(np.setdiff1d(np.arange(n_features), [c]))]
                elif pattern == "mnar":
                    driver = x[:, c]
                else:
                    driver = target
                sign = 1.0 if r.random() < 0.5 else -1.0
                a = sign * np.exp(r.uniform(np.log(self.min_strength), np.log(self.max_strength)))
                score = score + a * _standardized(driver)
            if k:
                mask[np.argsort(-score, kind="stable")[:k], c] = True
        return mask


def _standardized(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.float64)
    return (v - v.mean()) / max(v.std(), 1e-6)
