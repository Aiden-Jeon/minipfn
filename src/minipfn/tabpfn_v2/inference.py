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
- (`one_hot`을 주면) 범주형 열을 one-hot으로 펼친다. 논문 전처리 목록의 6번인데, 어느 멤버에 어떤
  범주까지 쓰는지는 적혀 있지 않아 여기서 정한다: `"power"`는 power 멤버만(분위수 멤버는 번호 섞기를
  그대로), `"all"`은 모든 멤버. 범주가 `MAX_ONE_HOT_CATEGORIES`개 넘는 열은 번호로 둔다.
  펼친 뒤 열 수가 `MAX_ONE_HOT_WIDTH`(prior의 최대 feature 수)를 넘으면 범주가 많은 열부터 번호로
  되돌린다. 범주가 2개면 열 하나(0/1)로 둔다. 결측과 학습에 없던 범주(번호로 바꿀 때 결측이 된다)는
  NaN. 기본값 `DEFAULT_ONE_HOT`은 분류·회귀 모두 `"all"`이다. 세 모드(off·power·all)를 benchmark에서
  견줬을 때 분류는 suite 평균 ROC AUC(500k 0.9235 → 0.9238), 회귀는 one-hot으로 결과가 바뀌는
  11개에서 off 대비 RMSE 비율의 기하평균(QK-clip 200k, power 0.964, all 0.958)이 가장 좋았다.
  회귀의 neg RMSE 단순 평균은 RMSE가 수만인 MIP-2016 하나가 정해서 기준으로 쓰지 않았다
- (`member_kinds`) 멤버 전처리 종류를 차례로 돌려 쓴다. 기본 `("quantile", "power")`는 위의 v2
  두 가지다. TabPFN-2.5(arXiv 2511.08667)는 "robust scaling과 soft clipping을 분위수·표준 변환과
  함께 쓴다"고만 적었다. `"robust"` 멤버는 RealMLP(Holzmüller et al. 2024)의 robust scaling +
  smooth clipping이다: 열마다 (x − 중앙값) / (3사분위 − 1사분위)
  (사분위 폭이 0이면 2 / (최대 − 최소)), 그다음 x / √(1 + (x/3)²). 표준 변환은 모델 안에서
  학습 행으로 z-정규화하므로 따로 하지 않는다.
  범주형 열은 power 멤버처럼 번호 그대로다. `MEMBER_KINDS_V2_5 = ("quantile", "robust")`
  (2.5가 Yeo-Johnson을 계속 쓰는지는 적혀 있지 않다)
- 분류: logit을 온도 0.9로 나눠 softmax한 뒤 멤버 평균
- 회귀: y를 z-정규화하고, 멤버 절반은 y에도 power transform을 한다(전처리 종류마다 절반씩:
  종류를 한 바퀴 돌 때마다 켜고 끈다). 예측 분포는 구간 경계만
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

import warnings
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

from minipfn.tabpfn_v2.model import TabPFNv2
from minipfn.tabpfn_v2.prior import MAX_FEATURES

OUTLIER_SIGMA = 12.0
MAX_ONE_HOT_CATEGORIES = 20  # benchmark의 socmob(17개)은 펼치고 colleges(수천 개)는 번호로 둔다
MAX_ONE_HOT_WIDTH = MAX_FEATURES  # 펼친 뒤 열 수의 상한: 학습 때 본 feature 수를 넘지 않게
ONE_HOT_MODES = ("off", "power", "all")
DEFAULT_ONE_HOT = "all"
SOFTMAX_TEMPERATURE = 0.9
MEMBER_KINDS = ("quantile", "power", "robust")
DEFAULT_MEMBER_KINDS = ("quantile", "power")  # v2
MEMBER_KINDS_V2_5 = ("quantile", "robust")
MEMBER_KIND_PRESETS = {"v2": DEFAULT_MEMBER_KINDS, "v2.5": MEMBER_KINDS_V2_5}
SOFT_CLIP = 3.0  # RealMLP smooth clipping의 폭


def resolve_member_kinds(value: str | tuple[str, ...] | list[str]) -> tuple[str, ...]:
    """멤버 전처리 종류. 쉼표로 이은 문자열, 프리셋 이름(`MEMBER_KIND_PRESETS`), 종류의 목록을
    받는다. 모르는 종류가 있거나 비어 있으면 ValueError."""
    if isinstance(value, str):
        if value in MEMBER_KIND_PRESETS:
            return MEMBER_KIND_PRESETS[value]
        value = [k.strip() for k in value.split(",") if k.strip()]
    kinds = tuple(value)
    if not kinds or set(kinds) - set(MEMBER_KINDS):
        raise ValueError(
            f"멤버 종류는 {MEMBER_KINDS}(또는 {tuple(MEMBER_KIND_PRESETS)})에서: {value}"
        )
    return kinds


def one_hot_kinds(one_hot: str, member_kinds: tuple[str, ...]) -> frozenset[str]:
    """`one_hot` 모드에서 범주형 열을 펼치는 멤버 종류. 이 집합이 같은 두 모드는 결과도 같다."""
    if one_hot == "all":
        return frozenset(member_kinds)
    if one_hot == "power":  # 분위수가 아닌 멤버(power·robust)만
        return frozenset(k for k in member_kinds if k != "quantile")
    return frozenset()


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


def _one_hot_width(n: int) -> int:
    return 1 if n == 2 else n  # 범주 2개는 0/1 열 하나


def _one_hot_sizes(n_categories: dict[int, int], n_features: int) -> dict[int, int]:
    """one-hot으로 펼칠 열과 그 범주 수.

    범주가 `MAX_ONE_HOT_CATEGORIES`개 이하인 열을 펼친다. 펼친 뒤 열 수(원래 열 기준)가
    `MAX_ONE_HOT_WIDTH`를 넘으면 범주가 많은 열부터 번호로 되돌린다 (열 수가 늘지 않는 범주 2개 이하
    열은 그대로 펼친다).
    """
    sizes = {i: n for i, n in n_categories.items() if n <= MAX_ONE_HOT_CATEGORIES}
    width = n_features + sum(_one_hot_width(n) - 1 for n in sizes.values())
    wide = [i for i in sizes if _one_hot_width(sizes[i]) > 1]
    for i in sorted(wide, key=lambda i: -sizes[i]):
        if width <= MAX_ONE_HOT_WIDTH:
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


@dataclass
class _Member:
    kind: str  # "quantile" | "power" | "robust"
    class_permutation: np.ndarray | None
    target_power: bool
    seed: int
    one_hot: bool = False


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
        # 범주형 열의 번호는 0부터 (학습 행의 범주 수 - 1)까지다
        n_categories = {
            i: int(np.nanmax(X[:, i])) + 1 if np.isfinite(X[:, i]).any() else 1 for i in self.cat_ix
        }
        self.one_hot_sizes = _one_hot_sizes(n_categories, X.shape[1]) if self.member.one_hot else {}
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
        else:
            self.transformer = RobustSoftClip()
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


class _Base:
    def __init__(
        self,
        model: TabPFNv2,
        *,
        n_estimators: int,
        categorical_features_indices: list[int] | None = None,
        one_hot: str = DEFAULT_ONE_HOT,
        member_kinds: tuple[str, ...] | str = DEFAULT_MEMBER_KINDS,
        device: str = "cpu",
        random_state: int = 0,
    ) -> None:
        if one_hot not in ONE_HOT_MODES:
            raise ValueError(f"one_hot은 {ONE_HOT_MODES} 가운데 하나여야 한다: {one_hot}")
        self.one_hot = one_hot
        self.member_kinds = resolve_member_kinds(member_kinds)
        self.model = model
        self.n_estimators = n_estimators
        self.categorical_features_indices = categorical_features_indices
        self.device = torch.device(device)
        self.random_state = random_state

    def _fit_members(self, X: Any, members: list[_Member]) -> None:
        expanded = one_hot_kinds(self.one_hot, self.member_kinds)
        for m in members:
            m.one_hot = m.kind in expanded
        X_num, self.cat_ix_, self.encoder_ = _to_numeric(X, self.categorical_features_indices)
        self.preprocessors_ = [_MemberPreprocessor(m, self.cat_ix_).fit(X_num) for m in members]
        # 학습 행의 변환 결과는 바뀌지 않으므로 멤버마다 한 번만 계산해 둔다
        self.X_trains_ = [
            prep.transform(X_num, np.random.default_rng(prep.member.seed + 2))
            for prep in self.preprocessors_
        ]

    def _kind(self, i: int) -> str:
        return self.member_kinds[i % len(self.member_kinds)]

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
                logits = model(x, y)[0].float()
            # autocast 밖에서 넘긴다: 받는 쪽의 계산(회귀 평균의 행렬곱 등)은 float32로 한다
            yield i, logits


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


class TabPFNv2Regressor(_Base):
    def __init__(
        self,
        model: TabPFNv2,
        borders: torch.Tensor,
        *,
        n_estimators: int = 8,
        **kwargs: Any,
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
                kind=self._kind(i),
                class_permutation=None,
                # 종류를 한 바퀴 돌 때마다 켜고 끈다: 종류마다 y power 변환이 절반씩
                target_power=(i // len(self.member_kinds)) % 2 == 1,
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
            p = torch.softmax(logits / SOFTMAX_TEMPERATURE, dim=-1)
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
            p = torch.softmax(logits / SOFTMAX_TEMPERATURE, dim=-1)
            means.append(p @ self.bucket_values_[i].to(p.device))
        return torch.stack(means).mean(0).cpu().numpy() * self.y_std_ + self.y_mean_
