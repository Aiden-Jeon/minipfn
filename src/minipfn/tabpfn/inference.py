"""TabPFN v1 추론: sklearn 스타일 `TabPFNClassifier`와 stateless `tabpfn_predict_proba`.

    clf = TabPFNClassifier(model).fit(x_train, y_train)
    proba = clf.predict_proba(x_test)

`fit`은 가중치를 학습하지 않고 학습 행을 context로 준비한다. `fit_mode`는 그 준비를 `fit`에서
어디까지 해 둘지 정한다. 결과는 같고 비용만 다르다(공식 TabPFN v2의 `fit_mode`와 같은 이름).

- "fit_preprocessors"(기본): 전처리를 학습 행에 맞추고 변환한 학습 행을 캐시한다. `predict`마다
  학습 행 attention(n²)을 다시 계산한다. 테스트 행이 많은 batch 예측용.
- "fit_with_cache": 층별 학습 행 key·value까지 캐시한다. `predict`는 테스트 행 → 학습 행
  attention(n·m)만 계산해 한 행 예측이 빠르다. 캐시가 크다(논문 크기, 학습 1,000행, 앙상블 32개에서
  fp32 약 1.6GB). context를 고정하고 한 행씩 예측하는 서빙용.

학습 행끼리는 양방향으로 attention하므로 학습 행을 하나 더하면 캐시 전체가 바뀐다. context가
바뀌면 `fit`을 다시 부르고, 요청마다 context가 다르면 `tabpfn_predict_proba`를 쓴다.

전처리와 앙상블은 v1 `transformer_predict`를 따른다(`evaluate` 모듈 docstring 참고). v1은
power transform이 경고를 내면 그 열을 변환하지 않는데, 이 판단을 학습 행으로만 해서(v1은 학습·테스트
행을 함께 변환하며 판단한다) 테스트 행의 예측이 함께 넣은 다른 테스트 행에 영향받지 않게 했다.
"""

from __future__ import annotations

import itertools
import random
import warnings
from typing import Any, Literal

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder, PowerTransformer, StandardScaler

from minipfn.tabpfn import spec
from minipfn.tabpfn.model import TabPFNTransformer
from minipfn.tabpfn.prior import (
    apply_normalization,
    normalization_stats,
    outlier_bounds,
    squash_outliers,
)

FitMode = Literal["fit_preprocessors", "fit_with_cache"]
TEST_ROWS_PER_FORWARD = 1024  # fit_with_cache에서 한 forward에 넣는 테스트 행 수 (메모리 상한)


class _Preprocessor:
    """v1 preprocess_input을 학습 행에 fit하고 아무 행에나 적용하도록 나눈 것.

    순서: 학습 행 기준 표준화 → 학습 행에서 상수인 열 제거 → (power_all) 열별 Yeo-Johnson →
    이상치 누르기 → feature 수 보정(× max_features / 남은 열 수).
    """

    def __init__(self, transform: str, max_features: int) -> None:
        self.transform = transform
        self.max_features = max_features

    def fit_transform(self, x_train: torch.Tensor) -> torch.Tensor:
        """x_train: (n, F) -> (n, F'). 변환한 학습 행을 돌려준다."""
        x = x_train.unsqueeze(1)
        self.norm_stats_ = normalization_stats(x)
        x = apply_normalization(x, *self.norm_stats_)[:, 0]
        self.keep_ = [len(torch.unique(x[:, c])) > 1 for c in range(x.shape[1])]
        x = x[:, self.keep_]
        # PowerTransformer(standardize=True)와 같은 계산을 Yeo-Johnson과 표준화 두 단계로 나눠 둔다.
        # 테스트 행에서 넘친 값을 두 단계 사이에서 처리하기 위해서다(transform_rows).
        self.power_: dict[int, tuple[PowerTransformer, StandardScaler]] = {}
        if self.transform == "power_all":
            x = x.numpy().copy()
            for c in range(x.shape[1]):
                # v1은 경고도 예외로 바꿔서, 변환이 매끄럽지 않은 열은 그대로 둔다.
                with warnings.catch_warnings():
                    warnings.simplefilter("error")
                    try:
                        pt = PowerTransformer(standardize=False).fit(x[:, c : c + 1])
                        t = pt.transform(x[:, c : c + 1])
                        scaler = StandardScaler().fit(t)
                        x[:, c : c + 1] = scaler.transform(t)
                        self.power_[c] = (pt, scaler)
                    except Exception:
                        pass
            x = torch.tensor(x).float()
        self.outlier_bounds_ = outlier_bounds(x.unsqueeze(1))
        return self._finish(x)

    def transform_rows(self, x: torch.Tensor) -> torch.Tensor:
        """학습 행으로 맞춘 변환을 다른 행 (m, F)에 적용한다 -> (m, F')."""
        x = apply_normalization(x.unsqueeze(1), *self.norm_stats_)[:, 0][:, self.keep_]
        if self.power_:
            x = x.numpy().copy()
            for c, (pt, scaler) in self.power_.items():
                # 학습 범위 밖의 값은 Yeo-Johnson에서 inf로 넘칠 수 있다. 그대로 두면 표준화가
                # 에러를 내고, 지나가도 이상치 누르기 뒤에 inf로 남는다. float32 최댓값으로 바꾸면
                # 이상치 누르기가 log로 눌러 준다.
                with np.errstate(over="ignore"):
                    t = _finite(pt.transform(x[:, c : c + 1]))
                    x[:, c : c + 1] = _finite(scaler.transform(t))
            x = torch.tensor(x).float()
        return self._finish(x)

    def _finish(self, x: torch.Tensor) -> torch.Tensor:
        x = squash_outliers(x.unsqueeze(1), *self.outlier_bounds_)[:, 0]
        return x / (x.shape[-1] / self.max_features)


class TabPFNClassifier:
    """학습한 v1 TabPFN을 sklearn처럼 쓰는 분류기. 레이블은 아무 값이나 된다(클래스 10개까지)."""

    def __init__(
        self,
        model: TabPFNTransformer,
        *,
        n_ensemble: int = spec.N_ENSEMBLE,
        fit_mode: FitMode = "fit_preprocessors",
        seed: int = 0,
        batch_size: int = 16,
        device: torch.device | str | None = None,
    ) -> None:
        if fit_mode not in ("fit_preprocessors", "fit_with_cache"):
            raise ValueError(f"fit_mode는 'fit_preprocessors' 또는 'fit_with_cache': {fit_mode!r}")
        self.model = model
        self.n_ensemble = n_ensemble
        self.fit_mode = fit_mode
        self.seed = seed
        self.batch_size = batch_size  # 한 forward에 넣는 앙상블 멤버 수
        self.device = device  # None이면 모델이 있는 장치를 쓴다. 주면 fit에서 모델을 그리로 옮긴다.

    @torch.no_grad()
    def fit(self, X: Any, y: Any) -> TabPFNClassifier:
        x = _as_rows(X)
        self.n_features_in_ = x.shape[1]
        self.label_encoder_ = LabelEncoder()
        y = self.label_encoder_.fit_transform(np.asarray(y))
        self.classes_ = self.label_encoder_.classes_
        n_classes = len(self.classes_)
        if n_classes > self.model.n_out:
            raise ValueError(f"클래스 {n_classes}개는 모델 출력({self.model.n_out}개)보다 많다.")
        if x.shape[1] > self.model.num_features:
            raise ValueError(
                f"feature {x.shape[1]}개는 모델 입력({self.model.num_features}개)보다 많다."
            )
        if self.device is None:
            self.device_ = next(self.model.parameters()).device
        else:
            self.device_ = torch.device(self.device)
            self.model.to(self.device_)
        self.model.eval()

        # 앙상블 멤버 = (클래스 회전, feature 회전) × 전처리 2가지, 앞에서부터 n_ensemble개 (v1)
        gen = torch.Generator().manual_seed(self.seed)
        feature_shifts = torch.randperm(x.shape[1], generator=gen).tolist()
        class_shifts = torch.randperm(n_classes, generator=gen).tolist()
        configs = list(itertools.product(class_shifts, feature_shifts))
        random.Random(self.seed).shuffle(configs)
        self.members_ = list(itertools.product(configs, ["none", "power_all"]))[: self.n_ensemble]

        self.preprocessors_: dict[str, _Preprocessor] = {}
        x_trains = {}
        for _, transform in self.members_:
            if transform not in self.preprocessors_:
                prep = _Preprocessor(transform, self.model.num_features)
                x_trains[transform] = prep.fit_transform(x)
                self.preprocessors_[transform] = prep
        # predict마다 다시 옮기지 않도록 학습 행을 장치에 올려 둔다.
        x_train = self._member_inputs(x_trains).to(self.device_)  # (N, n, F)
        y = torch.tensor(y, dtype=torch.float32)
        y_train = torch.stack([(y + cs) % n_classes for (cs, _), _ in self.members_])
        y_train = y_train.to(self.device_)

        if self.fit_mode == "fit_with_cache":
            # 캐시만 있으면 되므로 학습 행은 들고 있지 않는다.
            self.x_train_ = self.y_train_ = None
            self.kv_caches_ = [
                self.model.build_kv_cache(xb, yb)
                for xb, yb in zip(self._batches(x_train), self._batches(y_train), strict=True)
            ]
        else:
            self.x_train_, self.y_train_, self.kv_caches_ = x_train, y_train, None
        return self

    @torch.no_grad()
    def predict_proba(self, X: Any) -> np.ndarray:
        """(m, F) -> 클래스 확률 (m, C). 열 순서는 classes_."""
        x = _as_rows(X)
        if x.shape[1] != self.n_features_in_:
            raise ValueError(f"feature {x.shape[1]}개: fit 때는 {self.n_features_in_}개였다.")
        x_test = self._member_inputs(
            {t: prep.transform_rows(x) for t, prep in self.preprocessors_.items()}
        )
        n_classes = len(self.classes_)
        logits = []
        for i, xb in enumerate(self._batches(x_test)):
            xb = xb.to(self.device_)
            if self.kv_caches_ is None:
                j = i * self.batch_size
                x_tr = self.x_train_[j : j + self.batch_size]
                y_tr = self.y_train_[j : j + self.batch_size]
                out = self.model(x_tr, y_tr, xb)
            else:
                out = torch.cat(
                    [
                        self.model.forward_cached(self.kv_caches_[i], chunk)
                        for chunk in xb.split(TEST_ROWS_PER_FORWARD, dim=1)
                    ],
                    dim=1,
                )
            logits.append(out[..., :n_classes].float().cpu() / spec.SOFTMAX_TEMPERATURE)
        logits = torch.cat(logits)  # (N, m, C)

        total = torch.zeros(logits.shape[1:])
        for i, ((class_shift, _), _) in enumerate(self.members_):
            out = logits[i]
            total += torch.cat(
                [out[:, class_shift:], out[:, :class_shift]], dim=1
            )  # 클래스 회전 되돌리기
        return torch.softmax(total / len(self.members_), dim=-1).numpy()

    def predict(self, X: Any) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    def _member_inputs(self, xs: dict[str, torch.Tensor]) -> torch.Tensor:
        """전처리별 행 (rows, F') -> 멤버마다 feature 회전 후 0으로 채운 (N, rows, max_features)."""
        out = []
        for (_, feature_shift), transform in self.members_:
            x = xs[transform]
            x = torch.cat([x[:, feature_shift:], x[:, :feature_shift]], dim=1)
            out.append(torch.cat([x, torch.zeros(len(x), self.model.num_features - x.shape[1])], 1))
        return torch.stack(out)

    def _batches(self, t: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return t.split(self.batch_size)


def _finite(a: np.ndarray) -> np.ndarray:
    """±inf를 float32 최댓값으로 바꾼다. 결측값(nan)은 그대로 둔다."""
    big = np.finfo(np.float32).max
    return np.nan_to_num(a, nan=np.nan, posinf=big, neginf=-big)


def _as_rows(X: Any) -> torch.Tensor:
    x = np.asarray(X, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"X는 (행 수, feature 수) 2차원이어야 한다: shape {x.shape}")
    return torch.tensor(x)


def tabpfn_predict_proba(
    model: TabPFNTransformer,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    n_ensemble: int = spec.N_ENSEMBLE,
    seed: int = 0,
    batch_size: int = 16,
    device: torch.device | str | None = None,
) -> np.ndarray:
    """테스트 행의 클래스 확률 (m, C). context(학습 행)가 호출마다 바뀔 때 쓰는 stateless 버전.

    v1 TabPFNClassifier.predict_proba와 같은 앙상블. 열 순서는 정렬한 레이블 순서다.
    device를 주지 않으면 모델이 있는 장치에서 돌리고 모델을 옮기지 않는다."""
    clf = TabPFNClassifier(
        model, n_ensemble=n_ensemble, seed=seed, batch_size=batch_size, device=device
    )
    return clf.fit(x_train, y_train).predict_proba(x_test)
