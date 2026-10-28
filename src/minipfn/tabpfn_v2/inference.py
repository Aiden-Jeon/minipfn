"""TabPFN v2 추론: 앙상블과 전처리 (Methods "Inference details", "Regression details").

    clf = TabPFNv2Classifier(model).fit(x_train, y_train)
    proba = clf.predict_proba(x_test)

`fit`은 가중치를 학습하지 않는다. 멤버마다 전처리를 학습 행에 맞춰 두고, 예측할 때 학습 행과
테스트 행을 함께 forward한다. 논문이 기본 설정으로 적은 것 가운데 다음을 쓴다.

- 멤버 수: 분류 4개, 회귀 8개
- 멤버마다: feature 순서 섞기, (분류) 클래스 번호 섞기, 표준정규 난수로 된 샘플 식별자 열
- 전처리 두 가지를 번갈아: (1) 분위수 변환한 열을 원래 열 옆에 붙이고 SVD 성분을 덧붙임,
  범주형 열은 번호를 섞음 (2) Yeo-Johnson power transform
- 수치형 열에서 평균으로부터 표준편차 12배 밖의 값은 그 경계로 자른다
- 분류: logit을 온도 0.9로 나눠 softmax한 뒤 멤버 평균
- 회귀: y를 z-정규화하고, 멤버 절반은 y에도 power transform을 한다. 예측 분포는 구간 경계만
  역변환해 공통 경계로 옮긴 뒤 평균한다
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import (
    LabelEncoder,
    OrdinalEncoder,
    PowerTransformer,
    QuantileTransformer,
)

from minipfn.bar_distribution import FullSupportBarDistribution
from minipfn.tabpfn_v2.model import TabPFNv2

OUTLIER_SIGMA = 12.0
SOFTMAX_TEMPERATURE = 0.9


def _to_numeric(X: Any, categorical: list[int] | None) -> tuple[np.ndarray, list[int], Any]:
    """범주형·문자열 열을 순서 번호로 바꾼다. 결측은 NaN으로 남긴다."""
    df = pd.DataFrame(X).reset_index(drop=True)
    cat = set(categorical or [])
    cat |= {i for i, dtype in enumerate(df.dtypes) if not pd.api.types.is_numeric_dtype(dtype)}
    cat_ix = sorted(cat)
    encoder = None
    if cat_ix:
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan)
        encoder.fit(df.iloc[:, cat_ix].astype(str).where(df.iloc[:, cat_ix].notna()))
    return _apply_encoder(df, cat_ix, encoder), cat_ix, encoder


def _apply_encoder(df: pd.DataFrame, cat_ix: list[int], encoder: Any) -> np.ndarray:
    df = pd.DataFrame(df).reset_index(drop=True)
    out = np.empty(df.shape, dtype=np.float64)
    num_ix = [i for i in range(df.shape[1]) if i not in cat_ix]
    if num_ix:
        out[:, num_ix] = df.iloc[:, num_ix].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if cat_ix:
        cats = df.iloc[:, cat_ix]
        out[:, cat_ix] = encoder.transform(cats.astype(str).where(cats.notna()))
    return out


@dataclass
class _Member:
    kind: str  # "quantile" | "power"
    class_permutation: np.ndarray | None
    target_power: bool
    seed: int


class _MemberPreprocessor:
    """멤버 하나의 전처리. fit은 학습 행으로만 한다."""

    def __init__(self, member: _Member, cat_ix: list[int]) -> None:
        self.member = member
        self.cat_ix = cat_ix

    def fit(self, X: np.ndarray) -> _MemberPreprocessor:
        rng = np.random.default_rng(self.member.seed)
        mean, std = np.nanmean(X, axis=0), np.nanstd(X, axis=0)
        self.lower, self.upper = mean - OUTLIER_SIGMA * std, mean + OUTLIER_SIGMA * std
        # 범주형 열은 자르지 않는다 (번호를 자르면 드문 범주가 다른 범주와 합쳐진다)
        self.lower[self.cat_ix], self.upper[self.cat_ix] = -np.inf, np.inf
        X = self._clip(X)
        num_ix = [i for i in range(X.shape[1]) if i not in self.cat_ix]
        self.num_ix = num_ix
        self.cat_maps = {}
        if self.member.kind == "quantile":
            # 범주형 열: 번호를 섞는다 (분위수 멤버만)
            for i in self.cat_ix:
                n = int(np.nanmax(X[:, i])) + 1 if np.isfinite(X[:, i]).any() else 1
                self.cat_maps[i] = rng.permutation(n)
            n_quantiles = max(min(X.shape[0] // 10, 1000), 2)
            self.transformer = QuantileTransformer(
                n_quantiles=n_quantiles, random_state=self.member.seed
            )
        else:
            self.transformer = PowerTransformer(method="yeo-johnson", standardize=True)
        if num_ix:
            self.transformer.fit(X[:, num_ix])
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
        if self.num_ix:
            with np.errstate(all="ignore"):
                transformed[:, self.num_ix] = self.transformer.transform(X[:, self.num_ix])
        if self.member.kind == "quantile":
            return np.concatenate([X, transformed[:, self.num_ix]], axis=1)  # 원래 열 + 분위수 열
        return transformed

    def transform(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        base = self._base(self._clip(X))
        parts = [base]
        if self.svd is not None:
            parts.append(self.svd.transform(self._standardize(base)))
        parts.append(rng.standard_normal((X.shape[0], 1)))  # 샘플 식별자 열
        return np.concatenate(parts, axis=1)[:, self.order]


class _Base:
    def __init__(
        self,
        model: TabPFNv2,
        *,
        n_estimators: int,
        categorical_features_indices: list[int] | None = None,
        device: str = "cpu",
        random_state: int = 0,
    ) -> None:
        self.model = model
        self.n_estimators = n_estimators
        self.categorical_features_indices = categorical_features_indices
        self.device = torch.device(device)
        self.random_state = random_state

    def _fit_members(self, X: Any, members: list[_Member]) -> None:
        X_num, self.cat_ix_, self.encoder_ = _to_numeric(X, self.categorical_features_indices)
        self.preprocessors_ = [_MemberPreprocessor(m, self.cat_ix_).fit(X_num) for m in members]
        # 학습 행의 변환 결과는 바뀌지 않으므로 멤버마다 한 번만 계산해 둔다
        self.X_trains_ = [
            prep.transform(X_num, np.random.default_rng(prep.member.seed + 2))
            for prep in self.preprocessors_
        ]

    def _iter_logits(self, X: Any, y_trains: list[np.ndarray]):
        X_test = _apply_encoder(pd.DataFrame(X), self.cat_ix_, self.encoder_)
        model = self.model.to(self.device).eval()
        autocast = self.device.type == "cuda"  # 학습과 같은 bfloat16
        for i, (prep, y_train) in enumerate(zip(self.preprocessors_, y_trains, strict=True)):
            x_test = prep.transform(X_test, np.random.default_rng(prep.member.seed + 3))
            x = np.concatenate([self.X_trains_[i], x_test])
            x = torch.as_tensor(x, dtype=torch.float32, device=self.device).unsqueeze(0)
            y = torch.as_tensor(y_train, dtype=torch.float32, device=self.device).unsqueeze(0)
            with (
                torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=autocast),
                torch.inference_mode(),
            ):
                yield i, model(x, y)[0].float()


class TabPFNv2Classifier(_Base):
    def __init__(self, model: TabPFNv2, *, n_estimators: int = 4, **kwargs: Any) -> None:
        super().__init__(model, n_estimators=n_estimators, **kwargs)

    def fit(self, X: Any, y: Any) -> TabPFNv2Classifier:
        rng = np.random.default_rng(self.random_state)
        self.label_encoder_ = LabelEncoder()
        y = self.label_encoder_.fit_transform(np.asarray(y))
        self.classes_ = self.label_encoder_.classes_
        n_classes = len(self.classes_)
        if n_classes > self.model.config.n_out:
            raise ValueError(
                f"클래스 {n_classes}개는 모델 출력({self.model.config.n_out}개)보다 많다."
            )
        members = [
            _Member(
                kind="quantile" if i % 2 == 0 else "power",
                class_permutation=rng.permutation(n_classes),
                target_power=False,
                seed=int(rng.integers(0, 2**31 - 1)),
            )
            for i in range(self.n_estimators)
        ]
        self._fit_members(X, members)
        # 멤버마다 학습 행 정답을 섞은 번호로 넣는다: 원래 클래스 c → perm[c]
        self.y_trains_ = [m.class_permutation[y] for m in members]
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        n_classes = len(self.classes_)
        probs = []
        for i, logits in self._iter_logits(X, self.y_trains_):
            perm = self.preprocessors_[i].member.class_permutation
            logits = logits[:, :n_classes][:, perm]  # 섞은 번호의 logit을 원래 클래스 순서로
            probs.append(torch.softmax(logits / SOFTMAX_TEMPERATURE, dim=-1))
        return torch.stack(probs).mean(0).cpu().numpy()

    def predict(self, X: Any) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


def _probs_on_borders(probs: torch.Tensor, frm: torch.Tensor, to: torch.Tensor) -> torch.Tensor:
    """경계 frm 위의 구간 확률을 경계 to의 구간 확률로 옮긴다 (구간 안에서는 균일하다고 본다)."""
    cdf = torch.cat([torch.zeros_like(probs[:, :1]), probs.cumsum(-1)], dim=-1)  # frm 위의 CDF
    idx = (torch.searchsorted(frm, to) - 1).clamp(0, len(frm) - 2)
    width = (frm[idx + 1] - frm[idx]).clamp_min(1e-12)
    share = ((to - frm[idx]) / width).clamp(0, 1)
    at_to = cdf[:, idx] + probs[:, idx] * share
    at_to[:, to <= frm[0]] = 0.0
    at_to[:, to >= frm[-1]] = 1.0
    at_to[:, 0], at_to[:, -1] = 0.0, 1.0
    return (at_to[:, 1:] - at_to[:, :-1]).clamp_min(0)


class TabPFNv2Regressor(_Base):
    def __init__(
        self, model: TabPFNv2, borders: torch.Tensor, *, n_estimators: int = 8, **kwargs: Any
    ) -> None:
        super().__init__(model, n_estimators=n_estimators, **kwargs)
        self.borders = borders.float().cpu()  # 계산할 때만 장치로 옮기고, 결과는 CPU로 돌려준다

    def fit(self, X: Any, y: Any) -> TabPFNv2Regressor:
        rng = np.random.default_rng(self.random_state)
        y = np.asarray(y, dtype=np.float64)
        self.y_mean_, self.y_std_ = float(y.mean()), float(y.std()) + 1e-12
        z = (y - self.y_mean_) / self.y_std_
        members = [
            _Member(
                kind="quantile" if i % 2 == 0 else "power",
                class_permutation=None,
                target_power=(i // 2) % 2 == 1,
                seed=int(rng.integers(0, 2**31 - 1)),
            )
            for i in range(self.n_estimators)
        ]
        self._fit_members(X, members)
        self.target_transforms_, self.y_trains_ = [], []
        for m in members:
            if m.target_power:
                transform = PowerTransformer(method="yeo-johnson", standardize=True)
                self.y_trains_.append(transform.fit_transform(z.reshape(-1, 1)).ravel())
                self.target_transforms_.append(transform)
            else:
                self.y_trains_.append(z)
                self.target_transforms_.append(None)
        return self

    def predict_distribution(self, X: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """(구간 log 확률 (n, 구간 수), 원래 단위의 경계)."""
        borders = self.borders.to(self.device)
        probs = []
        for i, logits in self._iter_logits(X, self.y_trains_):
            p = torch.softmax(logits / SOFTMAX_TEMPERATURE, dim=-1)
            transform = self.target_transforms_[i]
            if transform is not None:
                # power transform한 공간의 경계를 z 공간으로 되돌려 확률을 옮긴다
                with np.errstate(all="ignore"):
                    back = transform.inverse_transform(borders.cpu().numpy().reshape(-1, 1)).ravel()
                # 역변환이 정의되지 않는 양 끝(NaN)은 가장 가까운 값으로 채우고, 순서를 지킨다
                back = pd.Series(np.where(np.isfinite(back), back, np.nan)).ffill().bfill()
                back = np.maximum.accumulate(back.to_numpy())
                frm = torch.as_tensor(back, dtype=torch.float32, device=self.device)
                p = _probs_on_borders(p, frm.clamp(borders[0], borders[-1]), borders)
            probs.append(p)
        log_probs = torch.stack(probs).mean(0).clamp_min(1e-12).log().cpu()
        return log_probs, (self.borders * self.y_std_ + self.y_mean_)

    def predict(self, X: Any) -> np.ndarray:
        log_probs, borders = self.predict_distribution(X)
        return FullSupportBarDistribution(borders).mean(log_probs).numpy()
