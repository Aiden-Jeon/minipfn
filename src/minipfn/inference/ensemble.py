"""전처리 멤버 앙상블 (TabPFN v2 논문 Methods "Inference details", "Regression details"의 추론):
sklearn 스타일 `EnsembleClassifier`·`EnsembleRegressor`. `forward(x, y_train)` 모델이면 클래스와
상관없이 쓴다 (`minipfn.models` registry의 모델).

    clf = EnsembleClassifier(model, n_estimators=4).fit(x_train, y_train)
    proba = clf.predict_proba(x_test)

논문 설정(멤버 수, 멤버 종류 묶음 "v2"·"v2.5")을 기본값으로 둔 것은
`minipfn.recipes.tabpfn_v2.TabPFNv2Classifier`·`TabPFNv2Regressor`다.

`fit`은 가중치를 학습하지 않는다. 멤버마다 전처리를 학습 행에 맞춰 두고, 예측할 때 학습 행과
테스트 행을 함께 forward한다. 논문이 기본 설정으로 적은 것 가운데 다음을 쓴다 (멤버 전처리는
`minipfn.inference.preprocessing`).

- 멤버 수(`n_estimators`): 논문 기본값은 분류 4개, 회귀 8개
- 멤버마다: feature 순서 섞기, (분류) 클래스 번호 섞기, 표준정규 난수로 된 샘플 식별자 열
- 전처리 두 가지를 번갈아: (1) 분위수 변환한 열을 원래 열 옆에 붙이고 SVD 성분을 덧붙임,
  범주형 열은 번호를 섞음 (2) Yeo-Johnson power transform
- 수치형 열에서 평균으로부터 표준편차 12배 밖의 값은 그 경계로 자른다
- (`one_hot`을 주면) 범주형 열을 one-hot으로 펼친다. 논문 전처리 목록의 6번인데, 어느 멤버에 어떤
  범주까지 쓰는지는 적혀 있지 않아 여기서 정한다: `"power"`는 분위수가 아닌 멤버(power·robust)만
  (분위수 멤버는 번호 섞기를 그대로), `"all"`은 모든 멤버.
  범주가 `MAX_ONE_HOT_CATEGORIES`개 넘는 열은 번호로 둔다.
  펼친 뒤 열 수가 `max_one_hot_width`(기본: 모델 설정의 `max_features`, 학습 때 본 feature 수.
  설정에 없으면 `MAX_FEATURES`)를 넘으면 범주가 많은 열부터 번호로 되돌린다.
  범주가 2개면 열 하나(0/1)로 둔다. 결측과 학습에 없던
  범주(번호로 바꿀 때 결측이 된다)는 NaN. 기본값 `DEFAULT_ONE_HOT`은 분류·회귀 모두 `"all"`이다.
  세 모드(off·power·all)를 benchmark에서 견줬을 때 분류는 suite 평균 ROC AUC(500k 0.9235 →
  0.9238), 회귀는 one-hot으로 결과가 바뀌는 11개에서 off 대비 RMSE 비율의 기하평균(QK-clip 200k,
  power 0.964, all 0.958)이 가장 좋았다.
  회귀의 neg RMSE 단순 평균은 RMSE가 수만인 MIP-2016 하나가 정해서 기준으로 쓰지 않았다
- (`member_kinds`) 멤버 전처리 종류를 차례로 돌려 쓴다. 기본 `("quantile", "power")`는 위의 v2
  두 가지다. TabPFN-2.5(arXiv 2511.08667)는 "robust scaling과 soft clipping을 분위수·표준 변환과
  함께 쓴다"고만 적었다. `"robust"` 멤버는 RealMLP(Holzmüller et al. 2024)의 robust scaling +
  smooth clipping이다: 열마다 (x − 중앙값) / (3사분위 − 1사분위)
  (사분위 폭이 0이면 2 / (최대 − 최소)), 그다음 x / √(1 + (x/3)²). 표준 변환은 모델 안에서
  학습 행으로 z-정규화하므로 따로 하지 않는다.
  범주형 열은 power 멤버처럼 번호 그대로다. recipe의 `MEMBER_KINDS_V2_5 = ("quantile", "robust")`
  (2.5가 Yeo-Johnson을 계속 쓰는지는 적혀 있지 않다)
- 분류: logit을 온도(`softmax_temperature`, 기본 0.9)로 나눠 softmax한 뒤 멤버 평균
- 회귀: y를 z-정규화하고, 멤버 절반은 y에도 power transform을 한다. 멤버 `max(2, 종류 수)`개마다
  켜고 끄므로, 종류가 1~2개면 예전처럼 2개씩(끔, 끔, 켬, 켬, …)이고 3개면 종류를 한 바퀴 돌 때마다
  바뀐다. 멤버 수가 `2 × max(2, 종류 수)`의 배수일 때 전체와 종류마다 정확히 절반이다(기본 8개는
  종류 1~2개에서 그렇다). 예측 분포는 구간 경계만
  역변환해 공통 경계로 옮긴 뒤 평균한다

회귀 예측값(평균)에는 논문에 없는 것을 정했다 (여기서 정함).

- 양 끝 꼬리 구간의 값: prior에서 만든 경계는 맨 바깥 구간이 아주 넓어서(z 폭 24~28)
  half-normal 꼬리의 평균이 z ±34 근처가 된다. 학습 target이 안쪽 경계를 넘는 데이터(치우친 y)에서
  꼬리에 놓인 1~2% 확률이 그 먼 값으로 계산돼 예측이 크게 밀리므로, 꼬리 구간의 값은 학습 target
  범위의 끝(안쪽 경계보다 바깥일 때만, 맨 바깥 경계 안으로)으로 본다.
- 평균은 공통 경계로 옮기지 않고 멤버마다 자기 공간에서 낸다. target power 멤버는 자기 구간 값을
  역변환한다. 공통 경계로 옮기면 구간 안을 균일하다고 보므로, 값이 몰린 곳이 공통 경계 가까이에
  있으면(cpu_activity의 y=0, 안쪽 경계 바로 위) 확률 일부가 꼬리 구간으로 넘어간다.
- Yeo-Johnson 역변환은 λ에 따라 한쪽 끝에서 정의가 끝나고(NaN) 그 가까이에서 끝없이 가파르다.
  학습 target 범위 밖은 그 끝값으로 둔다. 그래서 target power 멤버의 구간 값은 학습 target 범위
  안에 있다.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, PowerTransformer
from torch import nn

from minipfn.inference.preprocessing import (
    MEMBER_KINDS,
    ONE_HOT_MODES,
    Member,
    MemberPreprocessor,
    apply_encoder,
    one_hot_kinds,
    to_numeric,
)
from minipfn.priors.table_size import MAX_FEATURES

DEFAULT_ONE_HOT = "all"
SOFTMAX_TEMPERATURE = 0.9
DEFAULT_MEMBER_KINDS = ("quantile", "power")  # v2 논문의 두 가지


def resolve_member_kinds(
    value: str | tuple[str, ...] | list[str], presets: dict[str, tuple[str, ...]] | None = None
) -> tuple[str, ...]:
    """멤버 전처리 종류. `presets`의 이름(recipe의 `MEMBER_KIND_PRESETS` 등), 쉼표로 구분한 문자열,
    종류 목록을 받는다."""
    presets = presets or {}
    if isinstance(value, str):
        if value in presets:
            return presets[value]
        value = [k.strip() for k in value.split(",") if k.strip()]
    kinds = tuple(value)
    if not kinds or set(kinds) - set(MEMBER_KINDS):
        names = f" 또는 {tuple(presets)} 가운데 하나" if presets else ""
        raise ValueError(f"member_kinds는 {MEMBER_KINDS}에서 고른다{names}: {value}")
    return kinds


def max_features_of(model: nn.Module) -> int:
    """모델이 학습 때 본 feature 수 상한 (`config.max_features`).
    registry 규약에는 없는 값이라 없으면 `MAX_FEATURES`(논문 prior의 feature 수 상한)를 쓴다.
    추론 기본값과 평가 스크립트가 이 함수를 같이 쓴다."""
    return getattr(model.config, "max_features", MAX_FEATURES)


class _Base:
    def __init__(
        self,
        model: nn.Module,
        *,
        n_estimators: int,
        categorical_features_indices: list[int] | None = None,
        one_hot: str = DEFAULT_ONE_HOT,
        member_kinds: tuple[str, ...] | str = DEFAULT_MEMBER_KINDS,
        softmax_temperature: float = SOFTMAX_TEMPERATURE,
        max_one_hot_width: int | None = None,
        device: str = "cpu",
        random_state: int = 0,
    ) -> None:
        if one_hot not in ONE_HOT_MODES:
            raise ValueError(f"one_hot은 {ONE_HOT_MODES} 가운데 하나여야 한다: {one_hot}")
        self.one_hot = one_hot
        self.member_kinds = resolve_member_kinds(member_kinds)
        self.softmax_temperature = softmax_temperature
        # one-hot으로 펼친 뒤 열 수의 상한: 모델이 학습 때 본 feature 수 (모델 설정의 max_features)
        if max_one_hot_width is None:
            max_one_hot_width = max_features_of(model)
        self.max_one_hot_width = max_one_hot_width
        self.model = model
        self.n_estimators = n_estimators
        self.categorical_features_indices = categorical_features_indices
        self.device = torch.device(device)
        self.random_state = random_state

    def _fit_members(self, X: Any, members: list[Member]) -> None:
        expanded = one_hot_kinds(self.one_hot, self.member_kinds)
        for m in members:
            m.one_hot = m.kind in expanded
        X_num, self.cat_ix_, self.encoder_ = to_numeric(X, self.categorical_features_indices)
        self.preprocessors_ = [
            MemberPreprocessor(m, self.cat_ix_, self.max_one_hot_width).fit(X_num) for m in members
        ]
        # 학습 행의 변환 결과는 바뀌지 않으므로 멤버마다 한 번만 계산해 둔다
        self.X_trains_ = [
            prep.transform(X_num, np.random.default_rng(prep.member.seed + 2))
            for prep in self.preprocessors_
        ]

    def _kind(self, i: int) -> str:
        return self.member_kinds[i % len(self.member_kinds)]

    def _iter_logits(self, X: Any, y_trains: list[np.ndarray]):
        X_test = apply_encoder(pd.DataFrame(X), self.cat_ix_, self.encoder_)
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
                logits = model(x, y)[0].float()
            # autocast 밖에서 넘긴다: 받는 쪽의 계산(회귀 평균의 행렬곱 등)은 float32로 한다
            yield i, logits


class EnsembleClassifier(_Base):
    """분류: 멤버마다 클래스 번호를 섞고, logit을 온도로 나눠 softmax한 뒤 평균한다."""

    def fit(self, X: Any, y: Any) -> EnsembleClassifier:
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
            Member(
                kind=self._kind(i),
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
            probs.append(torch.softmax(logits / self.softmax_temperature, dim=-1))
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


def _clipped_inverse(transform: PowerTransformer, u_train: np.ndarray):
    """power transform의 역변환. 학습 target 범위 [lo, hi] 밖은 그 끝값으로 둔다.

    역변환은 λ < 0이면 오른쪽, λ > 2이면 왼쪽 끝에서 정의가 끝나고 그 가까이에서 끝없이 가팔라서,
    범위 밖으로 연장하면 (끝 기울기로 직선 연장해도) z가 수천까지 커진다.
    """
    lo, hi = float(u_train.min()), float(u_train.max())

    def g(u: np.ndarray) -> np.ndarray:
        u = np.clip(np.asarray(u, dtype=np.float64).reshape(-1, 1), lo, hi)
        with np.errstate(all="ignore"):
            return transform.inverse_transform(u).ravel()

    return g


def _bucket_values(borders: torch.Tensor, u_train: np.ndarray) -> np.ndarray:
    """평균에 쓸 구간마다의 값: 안쪽 구간은 가운데, 양 끝 꼬리 구간은 학습 target 범위의 끝.

    꼬리 구간의 값은 맨 바깥 경계 안으로 자른다 (극단값 행 하나가 꼬리 값을 정하지 않도록).
    """
    b = borders.double().numpy()
    values = (b[1:] + b[:-1]) / 2
    values[0] = max(b[0], min(b[1], float(u_train.min())))
    values[-1] = min(b[-1], max(b[-2], float(u_train.max())))
    return values


class EnsembleRegressor(_Base):
    """회귀: 구간 확률을 멤버마다 자기 공간에서 계산해 평균한다 (모듈 docstring)."""

    def __init__(self, model: nn.Module, borders: torch.Tensor, **kwargs: Any) -> None:
        super().__init__(model, **kwargs)
        self.borders = borders.float().cpu()  # 계산할 때만 장치로 옮기고, 결과는 CPU로 돌려준다

    def fit(self, X: Any, y: Any) -> EnsembleRegressor:
        rng = np.random.default_rng(self.random_state)
        y = np.asarray(y, dtype=np.float64)
        self.y_mean_, self.y_std_ = float(y.mean()), float(y.std()) + 1e-12
        z = (y - self.y_mean_) / self.y_std_
        members = [
            Member(
                kind=self._kind(i),
                class_permutation=None,
                # max(2, 종류 수)개마다 켜고 끈다 (종류 1~2개는 예전과 같은 끔, 끔, 켬, 켬)
                target_power=(i // max(2, len(self.member_kinds))) % 2 == 1,
                seed=int(rng.integers(0, 2**31 - 1)),
            )
            for i in range(self.n_estimators)
        ]
        self._fit_members(X, members)
        b = self.borders.double().numpy()
        self.y_trains_, self.bucket_values_, self.border_maps_ = [], [], []
        for m in members:
            u, values, border_map = z, _bucket_values(self.borders, z), None
            if m.target_power:
                transform = PowerTransformer(method="yeo-johnson", standardize=True)
                u = transform.fit_transform(z.reshape(-1, 1)).ravel()
                inverse = _clipped_inverse(transform, u)
                # 멤버 공간의 구간 값과 경계를 z 공간으로 (경계는 predict_distribution에서 쓴다)
                values = np.clip(inverse(_bucket_values(self.borders, u)), b[0], b[-1])
                back = np.clip(np.maximum.accumulate(inverse(b)), b[0], b[-1])
                border_map = torch.as_tensor(back, dtype=torch.float32)
            self.y_trains_.append(u)
            self.bucket_values_.append(torch.as_tensor(values, dtype=torch.float32))
            self.border_maps_.append(border_map)
        return self

    def predict_distribution(self, X: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """(구간 log 확률 (n, 구간 수), 원래 단위의 경계).

        멤버를 공통 경계로 옮겨 섞은 분포다. 꼬리는 공통 경계의 half-normal 그대로라, 이 분포로 낸
        평균은 `predict`와 다르다 (평균은 `predict`를 쓴다).
        """
        borders = self.borders.to(self.device)
        probs = []
        for i, logits in self._iter_logits(X, self.y_trains_):
            p = torch.softmax(logits / self.softmax_temperature, dim=-1)
            if self.border_maps_[i] is not None:
                # power transform한 공간의 경계를 z 공간으로 되돌린 것 위에서 확률을 옮긴다
                p = _probs_on_borders(p, self.border_maps_[i].to(self.device), borders)
            probs.append(p)
        log_probs = torch.stack(probs).mean(0).clamp_min(1e-12).log().cpu()
        return log_probs, (self.borders * self.y_std_ + self.y_mean_)

    def predict(self, X: Any) -> np.ndarray:
        """예측분포의 평균. 멤버를 섞은 분포의 평균은 멤버 평균의 평균이라, 공통 경계로 옮기지 않고
        멤버마다 자기 구간 값(`bucket_values_`)으로 계산한다."""
        means = []
        for i, logits in self._iter_logits(X, self.y_trains_):
            p = torch.softmax(logits / self.softmax_temperature, dim=-1)
            means.append(p @ self.bucket_values_[i].to(p.device))
        return torch.stack(means).mean(0).cpu().numpy() * self.y_std_ + self.y_mean_
