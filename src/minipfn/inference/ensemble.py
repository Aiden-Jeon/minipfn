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
  범주까지 쓰는지는 적혀 있지 않아 여기서 정한다: `"power"`는 분위수가 아닌 멤버
  (power·robust·standard)만
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
- (`max_features_per_member`를 주면) feature가 그보다 많은 표에서 멤버마다 그만큼의 원래 열을
  무작위로 고른다(`minipfn.inference.preprocessing.feature_subset`, 공식 TabPFN 추론 코드의
  `max_features_per_estimator`와 v2·2.5의 "random" 방법). 기본은 None(모든 열).
  `feature_subsampling="balanced"`는 열이 고르게 쓰이도록 나눈다(공식 3.5의 "balanced")
- 공식 TabPFN-3.5 추론에 맞추는 옵션(기본은 모두 끔, 예전과 비트 동일): 멤버 전처리
  (`shuffle_categories`·`fingerprint`·`outlier_clip`, `minipfn.inference.preprocessing`), 모든
  멤버가 같이 쓰는 열 단계(`infer_categorical`: 값이 몇 개뿐인 수치 열을 범주형으로,
  `drop_constant`: 상수 열 빼기), 회귀 `target_transform`·`regression_mean`(`EnsembleRegressor`).
  `unseen_category="minus1"`은 학습 행에 없던 범주를 결측(NaN, 기본. 공식 3.5도 멤버의
  `ordinal_shuffled` 단계에서 NaN이 된다) 대신 번호 -1로 둔다(TabICLv2 방식, 진단용)
- 범주를 아는 셀 encoder 모델(`categorical_encoder`, issue #110)에는 멤버마다 입력 열의 범주 표시
  (`MemberPreprocessor.categorical_columns`: 번호로 둔 범주형 열만 True, one-hot으로 펼친 열은
  수치 열)를 forward의 `categorical`로 넘긴다
- 분류: logit을 온도(`softmax_temperature`, 기본 0.9)로 나눠 softmax한 뒤 멤버 평균
- 회귀: y를 z-정규화하고, 멤버 절반은 y에도 power transform을 한다. 멤버 `max(2, 종류 수)`개마다
  켜고 끄므로, 종류가 1~2개면 예전처럼 2개씩(끔, 끔, 켬, 켬, …)이고 3개면 종류를 한 바퀴 돌 때마다
  바뀐다. 멤버 수가 `2 × max(2, 종류 수)`의 배수일 때 전체와 종류마다 정확히 절반이다(기본 8개는
  종류 1~2개에서 그렇다). 예측 분포는 구간 경계만
  역변환해 공통 경계로 옮긴 뒤 평균한다. 회귀 분위수 출력 모델(`regression_head="quantile"`,
  issue #110)은 멤버마다 분위수를 구간 경계 위의 구간 확률로 바꿔(`minipfn.models.
  quantile_distribution.quantiles_to_bucket_probs`) 같은 계산을 한다. 온도는 쓰지 않는다

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
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, PowerTransformer
from torch import nn

from minipfn.inference.preprocessing import (
    FEATURE_SUBSAMPLINGS,
    MEMBER_KINDS,
    ONE_HOT_MODES,
    Member,
    MemberPreprocessor,
    PreprocessOptions,
    apply_encoder,
    balanced_feature_subsets,
    constant_columns,
    encode_low_cardinality,
    feature_subset,
    low_cardinality_values,
    one_hot_kinds,
    select_features,
    to_numeric,
    unseen_value,
)
from minipfn.models.bar_distribution import FullSupportBarDistribution
from minipfn.models.quantile_distribution import quantiles_to_bucket_probs
from minipfn.priors.table_size import MAX_FEATURES

DEFAULT_ONE_HOT = "all"
SOFTMAX_TEMPERATURE = 0.9
DEFAULT_MEMBER_KINDS = ("quantile", "power")  # v2 논문의 두 가지
TARGET_TRANSFORMS = ("power", "safepower", "none")
REGRESSION_MEANS = ("member", "mixture")
# 공식 3.5의 회귀 경계 되돌리기: z가 이 범위 밖이거나 유한하지 않은 경계는 망가진 것으로 본다
BROKEN_BORDER_LIMIT = 1e3


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


def max_classes_of(model: nn.Module) -> int:
    """분류에서 받는 클래스 수 상한: 모델의 `max_classes`(column_row_transformer의 retrieval
    decoder는 정답 embedding을 묶어 출력 수보다 많이 받는다), 없으면 출력 수(`config.n_out`).
    `EnsembleClassifier`와 평가 스크립트의 TabArena·BeyondArena 상한이 같이 쓴다."""
    return int(getattr(model, "max_classes", model.config.n_out))


def uses_categorical(model: nn.Module) -> bool:
    """forward에 범주 열 표시(`categorical`)를 받는 모델인지 (범주를 아는 셀 encoder,
    issue #110)."""
    return bool(getattr(getattr(model, "config", None), "categorical_encoder", False))


def quantile_regressor(model: nn.Module) -> bool:
    """회귀 출력이 분위수인 모델인지 (`regression_head="quantile"`, issue #110)."""
    return getattr(getattr(model, "config", None), "regression_head", "bar") == "quantile"


def check_basic_ensemble_only(model: nn.Module, task: str, method: str) -> None:
    """범주 열 표시·회귀 분위수 출력 모델(issue #110)은 기본 앙상블(`EnsembleClassifier`·
    `EnsembleRegressor`)만 맞췄다. `method`(greedy 선택, DiagScale 등)에 그런 모델을 주면 멈춘다
    (DiagScale은 범주 표시 없이 모델을 부르고, greedy pool은 분위수를 구간 logit으로 읽는다)."""
    if uses_categorical(model) or (task == "regression" and quantile_regressor(model)):
        raise ValueError(f"{method}는 범주 표시·회귀 분위수 출력 모델을 받지 않는다 (issue #110)")


def forward_kwargs(model: nn.Module, task: str) -> dict:
    """모델 forward에 줄 인자. task는 여러 task를 내는 모델(`model.tasks`, 예: TabPFN-3.5의
    multitask column_row_transformer)만 받고, 한 task 모델은 예전처럼 `model(x, y_train)`이다.
    추론용 인자를 받는 모델(`supports_inference_options`)에는 `inference=True`(큰 표의 메모리를
    줄이는 계산)를 준다."""
    kwargs = {"task": task} if len(getattr(model, "tasks", ())) > 1 else {}
    if getattr(model, "supports_inference_options", False):
        kwargs["inference"] = True
    return kwargs


class _Base:
    # 이 앙상블이 내는 task (여러 task 모델에 forward로 넘긴다)
    task = ""

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
        max_features_per_member: int | None = None,
        shuffle_categories: bool = False,
        fingerprint: str = "random",
        outlier_clip: str = "hard",
        infer_categorical: bool = False,
        drop_constant: bool = False,
        feature_subsampling: str = "random",
        unseen_category: str = "nan",
        device: str = "cpu",
        random_state: int = 0,
    ) -> None:
        if one_hot not in ONE_HOT_MODES:
            raise ValueError(f"one_hot은 {ONE_HOT_MODES} 가운데 하나여야 한다: {one_hot}")
        if feature_subsampling not in FEATURE_SUBSAMPLINGS:
            raise ValueError(
                f"feature_subsampling은 {FEATURE_SUBSAMPLINGS} 가운데 하나다: {feature_subsampling}"
            )
        # 공식 3.5 추론 옵션 (모듈 docstring. 기본은 모두 예전 동작)
        self.preprocess_options = PreprocessOptions(
            shuffle_categories=shuffle_categories,
            fingerprint=fingerprint,
            outlier_clip=outlier_clip,
            unseen_category=unseen_category,
        )
        self.infer_categorical = infer_categorical
        self.drop_constant = drop_constant
        self.feature_subsampling = feature_subsampling
        unseen_value(unseen_category)  # 값 확인
        self.unseen_category = unseen_category
        self.one_hot = one_hot
        self.member_kinds = resolve_member_kinds(member_kinds)
        self.softmax_temperature = softmax_temperature
        # one-hot으로 펼친 뒤 열 수의 상한: 모델이 학습 때 본 feature 수 (모델 설정의 max_features)
        if max_one_hot_width is None:
            max_one_hot_width = max_features_of(model)
        self.max_one_hot_width = max_one_hot_width
        # 멤버 하나가 쓰는 원래 feature 수 상한 (None이면 전부, `feature_subsets`)
        self.max_features_per_member = max_features_per_member
        self.model = model
        self.n_estimators = n_estimators
        self.categorical_features_indices = categorical_features_indices
        self.device = torch.device(device)
        self.random_state = random_state

    def _expands(self, kind: str) -> bool:
        # `one_hot` 모드에서 이 종류의 멤버가 범주형 열을 one-hot으로 펼치는지
        return kind in one_hot_kinds(self.one_hot, self.member_kinds)

    def _fit_columns(self, X: Any) -> np.ndarray:
        """학습 행을 수치 배열로 (모든 멤버가 같이 쓰는 열 단계). `cat_ix_`는 그 배열의 범주형 열,
        `encoded_ix_`는 문자열·범주를 번호로 바꾼 원래 열이다. 옵션을 끄면 둘은 같다."""
        X_num, self.encoded_ix_, self.encoder_ = to_numeric(
            X, self.categorical_features_indices, self.unseen_category
        )
        cat_ix = list(self.encoded_ix_)
        # 수치 열 가운데 값이 몇 개뿐인 것을 범주형으로 (공식 3.5의 type 추론)
        self.low_cardinality_ = (
            low_cardinality_values(X_num, cat_ix) if self.infer_categorical else {}
        )
        cat_ix = sorted({*cat_ix, *self.low_cardinality_})
        # 학습 행: 미지 값이 없으므로 unseen 값과 상관없다
        if self.low_cardinality_:
            X_num = encode_low_cardinality(X_num, self.low_cardinality_)
        # 학습 행에서 값이 하나뿐인 열을 뺀다 (범주형으로 준 열은 남긴다)
        self.kept_columns_ = None
        if self.drop_constant:
            dropped = set(constant_columns(X_num, self.categorical_features_indices))
            if dropped:
                self.kept_columns_ = np.array(
                    [i for i in range(X_num.shape[1]) if i not in dropped], dtype=int
                )
        if self.kept_columns_ is None:
            self.cat_ix_ = cat_ix
            return X_num
        X_num, self.cat_ix_ = select_features(X_num, cat_ix, self.kept_columns_)
        return X_num

    def _columns(self, X_num: np.ndarray) -> np.ndarray:
        # `_fit_columns`에서 정한 열 단계를 다른 행에도
        if self.low_cardinality_:
            unseen = unseen_value(self.unseen_category)
            X_num = encode_low_cardinality(X_num, self.low_cardinality_, unseen)
        if self.kept_columns_ is not None:
            X_num = X_num[:, self.kept_columns_]
        return X_num

    def _fit_members(self, X: Any, members: list[Member]) -> None:
        X_num = self._fit_columns(X)
        # 멤버마다 쓸 열 (feature가 상한보다 많을 때만. 멤버 seed로 정해 같은 멤버는 늘 같다)
        if self.feature_subsampling == "balanced":
            rng = np.random.default_rng([m.seed for m in members])
            self.feature_subsets_ = balanced_feature_subsets(
                X_num.shape[1], self.max_features_per_member, len(members), rng
            )
        else:
            self.feature_subsets_ = [
                feature_subset(X_num.shape[1], self.max_features_per_member, m.seed + 4)
                for m in members
            ]
        self.preprocessors_ = []
        self.X_trains_ = []
        for m, cols in zip(members, self.feature_subsets_, strict=True):
            x, cat_ix = select_features(X_num, self.cat_ix_, cols)
            prep = MemberPreprocessor(
                m, cat_ix, self.max_one_hot_width, self.preprocess_options
            ).fit(x)
            self.preprocessors_.append(prep)
            # 학습 행의 변환 결과는 바뀌지 않으므로 멤버마다 한 번만 계산해 둔다
            self.X_trains_.append(prep.transform(x, np.random.default_rng(m.seed + 2)))

    def _model_targets(self, y: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """모델에 줄 학습 행 정답 (1, 학습 행)과 forward에 더 줄 인자."""
        return y, {}

    def _kind(self, i: int) -> str:
        return self.member_kinds[i % len(self.member_kinds)]

    def _iter_inputs(self, X: Any, y_trains: list[np.ndarray]):
        """멤버마다 (번호, 모델 입력 x (1, 학습 행 + 테스트 행, F), y_train (1, 학습 행))."""
        X_test = self._columns(apply_encoder(pd.DataFrame(X), self.encoded_ix_, self.encoder_))
        for i, (prep, y_train) in enumerate(zip(self.preprocessors_, y_trains, strict=True)):
            x_test, _ = select_features(X_test, self.cat_ix_, self.feature_subsets_[i])
            x_test = prep.transform(
                x_test, np.random.default_rng(prep.member.seed + 3), is_test=True
            )
            x = np.concatenate([self.X_trains_[i], x_test])
            x = torch.as_tensor(x, dtype=torch.float32, device=self.device).unsqueeze(0)
            y = torch.as_tensor(y_train, dtype=torch.float32, device=self.device).unsqueeze(0)
            yield i, x, y

    def _iter_logits(self, X: Any, y_trains: list[np.ndarray]):
        model = self.model.to(self.device).eval()
        autocast = self.device.type == "cuda"  # 학습과 같은 bfloat16
        kwargs = forward_kwargs(model, self.task)
        categorical = uses_categorical(model)
        for i, x, y in self._iter_inputs(X, y_trains):
            y, extra = self._model_targets(y)
            if categorical:  # 멤버 입력 열의 범주 표시 (모듈 docstring)
                mask = self.preprocessors_[i].categorical_columns()
                extra = {**extra, "categorical": torch.as_tensor(mask, device=self.device)[None]}
            with (
                torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=autocast),
                torch.inference_mode(),
            ):
                logits = model(x, y, **kwargs, **extra)[0].float()
            # autocast 밖에서 넘긴다: 받는 쪽의 계산(회귀 평균의 행렬곱 등)은 float32로 한다
            yield i, logits


class EnsembleClassifier(_Base):
    """분류: 멤버마다 클래스 번호를 섞고, logit을 온도로 나눠 softmax한 뒤 평균한다.

    클래스가 모델 출력 수(`config.n_out`, 정답 embedding 수)보다 많으면 retrieval decoder 모델
    (`max_classes_of`가 더 큰 모델, column_row_transformer)만 받는다. 정답 embedding에는 섞은 번호를
    `n_out`으로 나눈 나머지(묶음)를, decoder에는 섞은 번호 그대로를 준다. decoder는 attention
    가중치로 학습 행의 원래 클래스 one-hot을 평균하므로 클래스마다 확률이 나온다. 멤버마다 번호를
    섞으므로 같은 embedding을 쓰는 클래스 묶음도 멤버마다 다르다. 클래스 11개면 한 묶음만 2개다.
    (여기서 정함. 공식 TabPFN-3.5는 정답 embedding 160개를 학습해 묶지 않는다)
    """

    task = "classification"

    def fit(self, X: Any, y: Any) -> EnsembleClassifier:
        rng = np.random.default_rng(self.random_state)
        y = self._fit_labels(y)
        return self.fit_members(X, y, self._default_members(rng, len(self.classes_)))

    def _default_members(self, rng: np.random.Generator, n_classes: int) -> list[Member]:
        return [
            Member(
                kind=self._kind(i),
                class_permutation=rng.permutation(n_classes),
                target_power=False,
                seed=int(rng.integers(0, 2**31 - 1)),
                one_hot=self._expands(self._kind(i)),
            )
            for i in range(self.n_estimators)
        ]

    def _fit_labels(self, y: Any) -> np.ndarray:
        """클래스(`classes_`)를 정하고 정답을 0부터 매긴 번호로 돌려준다."""
        self.label_encoder_ = LabelEncoder()
        y = self.label_encoder_.fit_transform(np.asarray(y))
        self.classes_ = self.label_encoder_.classes_
        n_classes = len(self.classes_)
        if n_classes > max_classes_of(self.model):
            raise ValueError(
                f"클래스 {n_classes}개는 모델이 받는 수({max_classes_of(self.model)}개)보다 많다."
            )
        return y

    def _model_targets(self, y: torch.Tensor) -> tuple[torch.Tensor, dict]:
        # 클래스가 정답 embedding 수보다 많으면 embedding은 묶음 번호, decoder는 원래 번호
        # (클래스 docstring)
        n_out, n_classes = self.model.config.n_out, len(self.classes_)
        if n_classes <= n_out:
            return y, {}
        return torch.remainder(y, n_out), {"y_classes": y, "n_classes": n_classes}

    def fit_members(self, X: Any, y: np.ndarray, members: list[Member]) -> EnsembleClassifier:
        """정한 멤버로 맞춘다. `y`는 `classes_`의 번호다 (`_fit_labels` 뒤)."""
        self._fit_members(X, members)
        # 멤버마다 학습 행 정답을 섞은 번호로 넣는다: 원래 클래스 c → perm[c]
        self.y_trains_ = [m.class_permutation[y] for m in members]
        return self

    def member_logits(self, X: Any):
        """멤버마다 (번호, 원래 클래스 순서의 logit (n, 클래스 수))."""
        n_classes = len(self.classes_)
        for i, logits in self._iter_logits(X, self.y_trains_):
            perm = self.preprocessors_[i].member.class_permutation
            yield i, logits[:, :n_classes][:, perm]  # 섞은 번호의 logit을 원래 클래스 순서로

    def predict_proba(self, X: Any) -> np.ndarray:
        probs = [
            torch.softmax(logits / self.softmax_temperature, dim=-1)
            for _, logits in self.member_logits(X)
        ]
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


_HALF_NORMAL_MEDIAN = float(torch.distributions.HalfNormal(1.0).icdf(torch.tensor(0.5)))


def _probs_on_full_support_borders(
    probs: torch.Tensor, frm: torch.Tensor, to: torch.Tensor
) -> torch.Tensor:
    """`_probs_on_borders`와 같지만 frm의 양 끝 구간을 half-normal 꼬리로 본다 (공식 3.5가 멤버를
    공통 경계로 옮기는 방식). 첫 구간의 확률은 frm[1]에서 왼쪽으로, 마지막 구간의 확률은 frm[-2]에서
    오른쪽으로 퍼진다(scale은 그 구간 폭이 중앙값이 되도록, `FullSupportBarDistribution`과 같다).
    to의 첫·마지막 구간은 to 밖의 확률을 모두 받는다. 위치 계산은 float64로 한다."""
    f, t = frm.double().cpu(), to.double().cpu()
    n = len(f) - 1
    idx = (torch.searchsorted(f, t) - 1).clamp(0, n - 1)
    width = (f[idx + 1] - f[idx]).clamp_min(1e-12)
    share = ((t - f[idx]) / width).clamp(0, 1)
    scale_low = float((f[1] - f[0]).clamp_min(1e-12)) / _HALF_NORMAL_MEDIAN
    scale_high = float((f[-1] - f[-2]).clamp_min(1e-12)) / _HALF_NORMAL_MEDIAN
    # half-normal 꼬리에서 그 위치보다 더 바깥에 있을 확률
    beyond_low = torch.erfc((f[1] - t).clamp_min(0) / (scale_low * 2**0.5))
    beyond_high = torch.erfc((t - f[-2]).clamp_min(0) / (scale_high * 2**0.5))
    dev, dtype = probs.device, probs.dtype
    low, high = (t < f[1]).to(dev), (t >= f[-2]).to(dev)
    idx, share = idx.to(dev), share.to(dev, dtype)
    cdf = torch.cat([torch.zeros_like(probs[:, :1]), probs.cumsum(-1)], dim=-1)
    at_to = cdf[:, idx] + probs[:, idx] * share
    at_to = torch.where(low, probs[:, :1] * beyond_low.to(dev, dtype), at_to)
    at_to = torch.where(high, 1.0 - probs[:, -1:] * beyond_high.to(dev, dtype), at_to)
    at_to[:, 0], at_to[:, -1] = 0.0, 1.0
    return (at_to[:, 1:] - at_to[:, :-1]).clamp_min(0)


def _official_border_map(transform: Any, borders: np.ndarray):
    """target 변환 멤버의 경계를 z 공간으로 되돌린다 (공식 3.5 방식, `regression_mean="mixture"`).

    학습 target 범위로 자르지 않고 역변환한다. 결측·무한이거나 |z| > `BROKEN_BORDER_LIMIT`인 경계는
    망가진 것으로 보고, 양 끝에 몰린 망가진 경계는 가장 가까운 성한 경계로 채운 뒤(맨 끝은 1만큼
    바깥) 그 경계에 닿는 구간의 logit을 지운다. 경계가 내림차순이면 뒤집는다. 끝 구간 폭이 0이면
    조금 넓힌다. 돌려주는 값: (z 경계, 지울 구간 mask 또는 None, 뒤집었는지)."""
    # 정의역 밖의 경계는 결측이 된다 (아래에서 고친다. sklearn의 경고는 끈다)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        z = np.asarray(transform.inverse_transform(borders.reshape(-1, 1)), dtype=np.float64)
    z = z.ravel()
    broken = ~np.isfinite(z) | (np.abs(z) > BROKEN_BORDER_LIMIT)
    usable = z[~broken]
    descending = len(usable) > 1 and bool((np.diff(usable) < 0).all())
    if descending:
        z, broken = z[::-1].copy(), broken[::-1].copy()
    cancel = None
    if broken.any():
        if broken.all():
            raise ValueError("target 변환을 되돌린 경계가 모두 망가졌다")
        ok = np.flatnonzero(~broken)
        first, last = ok[0], ok[-1]
        z[:first] = z[first]
        z[last + 1 :] = z[last]
        if first > 0:
            z[0] = z[1] - 1.0
        if last < len(z) - 1:
            z[-1] = z[-2] + 1.0
        # 가운데의 망가진 경계(드물다)는 앞 경계로 채워 폭 0 구간으로 둔다
        for i in np.flatnonzero(broken[first : last + 1]) + first:
            z[i] = z[i - 1]
        cancel = broken[1:] | broken[:-1]
    if z[-1] - z[-2] < 1e-6:
        z[-1] += abs(z[-1] * 0.1)
    if z[0] == z[1]:
        z[0] -= abs(z[0] * 0.1)
    return torch.as_tensor(z, dtype=torch.float32), cancel, descending


def _yeo_johnson_inverse(t: np.ndarray, lam: float) -> np.ndarray:
    """Yeo-Johnson 역변환. 지수의 인자를 float64 최댓값 안으로 잘라 넘침을 막는다. 정의역 밖은
    결측."""
    t = np.asarray(t, dtype=np.float64)
    out = np.empty_like(t)
    pos = t >= 0
    cap = np.log(np.finfo(np.float64).max) - 1e-6
    eps = np.spacing(1.0)
    with np.errstate(all="ignore"):
        if abs(lam) < eps:
            out[pos] = np.expm1(np.clip(t[pos], -cap, cap))
        else:
            out[pos] = np.expm1(np.clip(np.log1p(t[pos] * lam) / lam, -cap, cap))
        if abs(lam - 2) < eps:
            out[~pos] = -np.expm1(np.clip(-t[~pos], -cap, cap))
        else:
            out[~pos] = -np.expm1(np.clip(np.log1p(-(2 - lam) * t[~pos]) / (2 - lam), -cap, cap))
    return out


class SafePowerTarget:
    """공식 3.5의 회귀 target "safepower" (`target_transform="safepower"`): z를 받지만 원래 단위의
    y에 Yeo-Johnson을 하고 표준화한다. λ는 scipy의 범위를 둔 최대우도(`yeojohnson_normmax`,
    넘침을 막는 범위)이고, 거의 상수인 target은 λ = 1. 역변환도 넘치지 않게 한다
    (`_yeo_johnson_inverse`). `PowerTransformer`(z에 맞추고 λ 범위가 없다)와 같은
    `fit_transform`·`inverse_transform`을 준다."""

    def __init__(self, y_mean: float, y_std: float) -> None:
        self.y_mean, self.y_std = y_mean, y_std

    def fit_transform(self, z: np.ndarray) -> np.ndarray:
        from scipy import stats

        y = np.asarray(z, dtype=np.float64).ravel() * self.y_std + self.y_mean
        eps = np.finfo(np.float64).eps
        var, mean = y.var(), y.mean()
        if var <= len(y) * eps * var + (len(y) * mean * eps) ** 2:
            self.lam = 1.0
        else:
            self.lam = float(stats.yeojohnson_normmax(y))
        t = stats.yeojohnson(y, lmbda=self.lam)
        self.t_mean, self.t_std = float(t.mean()), float(t.std())
        if not (np.isfinite(self.t_std) and self.t_std > 0):
            self.t_std = 1.0
        return ((t - self.t_mean) / self.t_std).reshape(-1, 1)

    def inverse_transform(self, u: np.ndarray) -> np.ndarray:
        t = np.asarray(u, dtype=np.float64) * self.t_std + self.t_mean
        return (_yeo_johnson_inverse(t, self.lam) - self.y_mean) / self.y_std


class EnsembleRegressor(_Base):
    """회귀: 구간 확률을 멤버마다 자기 공간에서 계산해 평균한다 (모듈 docstring).

    `target_transform`: 멤버 절반의 target 변환. `"power"`(기본)는 z에 맞춘 Yeo-Johnson,
    `"safepower"`는 공식 3.5의 것(`SafePowerTarget`), `"none"`은 모든 멤버가 변환 없이.
    `regression_mean`: `"member"`(기본)는 위의 멤버별 평균(모듈 docstring), `"mixture"`는 공식
    3.5처럼 멤버 경계를 학습 범위로 자르지 않고 되돌려(`_official_border_map`) half-normal 꼬리째
    공통 경계로 옮긴(`_probs_on_full_support_borders`) 분포를 평균하고, 그 분포의 평균(꼬리 구간은
    half-normal의 평균, `FullSupportBarDistribution`)을 낸다. 분포(`predict_distribution`)도 그 섞은
    분포다.
    """

    task = "regression"

    def __init__(
        self,
        model: nn.Module,
        borders: torch.Tensor,
        target_transform: str = "power",
        regression_mean: str = "member",
        **kwargs: Any,
    ) -> None:
        super().__init__(model, **kwargs)
        if target_transform not in TARGET_TRANSFORMS:
            raise ValueError(
                f"target_transform은 {TARGET_TRANSFORMS} 가운데 하나다: {target_transform}"
            )
        if regression_mean not in REGRESSION_MEANS:
            raise ValueError(
                f"regression_mean은 {REGRESSION_MEANS} 가운데 하나다: {regression_mean}"
            )
        self.target_transform = target_transform
        self.regression_mean = regression_mean
        self.borders = borders.float().cpu()  # 계산할 때만 장치로 옮기고, 결과는 CPU로 돌려준다

    def fit(self, X: Any, y: Any) -> EnsembleRegressor:
        rng = np.random.default_rng(self.random_state)
        return self.fit_members(X, y, self._default_members(rng))

    def _default_members(self, rng: np.random.Generator) -> list[Member]:
        return [
            Member(
                kind=self._kind(i),
                class_permutation=None,
                # max(2, 종류 수)개마다 켜고 끈다 (종류 1~2개는 예전과 같은 끔, 끔, 켬, 켬)
                target_power=self.target_transform != "none"
                and (i // max(2, len(self.member_kinds))) % 2 == 1,
                seed=int(rng.integers(0, 2**31 - 1)),
                one_hot=self._expands(self._kind(i)),
            )
            for i in range(self.n_estimators)
        ]

    def fit_members(self, X: Any, y: Any, members: list[Member]) -> EnsembleRegressor:
        """정한 멤버로 맞춘다. `y`는 원래 단위다."""
        y = np.asarray(y, dtype=np.float64)
        self.y_mean_, self.y_std_ = float(y.mean()), float(y.std()) + 1e-12
        z = (y - self.y_mean_) / self.y_std_
        self._fit_members(X, members)
        b = self.borders.double().numpy()
        self.y_trains_, self.bucket_values_, self.border_maps_ = [], [], []
        self.official_maps_ = []  # regression_mean="mixture"에서 쓴다
        for m in members:
            u, values, border_map, official = z, _bucket_values(self.borders, z), None, None
            if m.target_power:
                transform = self._make_target_transform()
                u = np.asarray(transform.fit_transform(z.reshape(-1, 1))).ravel()
                if self.regression_mean == "mixture":
                    official = _official_border_map(transform, b)
                inverse = _clipped_inverse(transform, u)
                # 멤버 공간의 구간 값과 경계를 z 공간으로 (경계는 predict_distribution에서 쓴다)
                values = np.clip(inverse(_bucket_values(self.borders, u)), b[0], b[-1])
                back = np.clip(np.maximum.accumulate(inverse(b)), b[0], b[-1])
                border_map = torch.as_tensor(back, dtype=torch.float32)
            self.y_trains_.append(u)
            self.bucket_values_.append(torch.as_tensor(values, dtype=torch.float32))
            self.border_maps_.append(border_map)
            self.official_maps_.append(official)
        return self

    def _make_target_transform(self):
        if self.target_transform == "safepower":
            return SafePowerTarget(self.y_mean_, self.y_std_)
        return PowerTransformer(method="yeo-johnson", standardize=True)

    def predict_distribution(self, X: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """(구간 log 확률 (n, 구간 수), 원래 단위의 경계 (float64)).

        멤버를 공통 경계로 옮겨 섞은 분포다. 꼬리는 공통 경계의 half-normal 그대로라, 이 분포로 낸
        평균은 `predict`와 다르다 (평균은 `predict`를 쓴다). 경계는 float64로 원래 단위로 옮긴다:
        안쪽 구간은 z 폭이 1e-6까지 좁아서, target의 평균이 크고 퍼짐이 작으면 float32에서는 여러
        경계가 한 값으로 뭉친다.
        """
        if self.regression_mean == "mixture":
            return self._mixture(self._official_probs(X))
        probs = [self._on_common_borders(i, p) for i, p in self._member_probs(X)]
        return self._mixture(probs)

    def predict(self, X: Any) -> np.ndarray:
        """예측분포의 평균. 멤버를 섞은 분포의 평균은 멤버 평균의 평균이라, 공통 경계로 옮기지 않고
        멤버마다 자기 구간 값(`bucket_values_`)으로 계산한다 (`regression_mean="mixture"`면 클래스
        docstring)."""
        if self.regression_mean == "mixture":
            return self._mixture_mean(self._official_probs(X))
        return self._mean([self._member_mean(i, p) for i, p in self._member_probs(X)])

    def predict_full(self, X: Any) -> tuple[np.ndarray, torch.Tensor, torch.Tensor]:
        """(`predict`의 평균, `predict_distribution`의 log 확률과 경계)를 forward 한 번으로 낸다.
        분포 품질을 평가할 때(`minipfn.benchmarks.uncertainty`) 점 예측과 분포를 같이 쓴다."""
        if self.regression_mean == "mixture":
            probs = self._official_probs(X)
            return (self._mixture_mean(probs), *self._mixture(probs))
        means, probs = [], []
        for i, p in self._member_probs(X):
            means.append(self._member_mean(i, p))
            probs.append(self._on_common_borders(i, p))
        return (self._mean(means), *self._mixture(probs))

    def _bucket_logits(self, X: Any):
        """멤버마다 (번호, 자기 공간의 구간 logit을 온도로 나눈 것). 분위수 출력 모델은 분위수를
        구간 확률로 바꾼 log 확률이고 온도로 나누지 않는다 (모듈 docstring)."""
        quantile = quantile_regressor(self.model)
        for i, logits in self._iter_logits(X, self.y_trains_):
            if quantile:
                p = quantiles_to_bucket_probs(logits, self.borders)
                yield i, p.clamp_min(1e-30).log()
            else:
                yield i, logits / self.softmax_temperature

    def _member_probs(self, X: Any):
        """멤버마다 (번호, 자기 공간의 구간 확률)."""
        for i, logits in self._bucket_logits(X):
            yield i, torch.softmax(logits, dim=-1)

    def _official_probs(self, X: Any) -> list[torch.Tensor]:
        """`regression_mean="mixture"`: 멤버마다 공통 경계 위의 구간 확률 (클래스 docstring)."""
        probs = []
        for i, logits in self._bucket_logits(X):
            official = self.official_maps_[i]
            if official is None:  # target 변환이 없는 멤버는 경계가 공통 경계 그대로다
                probs.append(torch.softmax(logits, dim=-1))
                continue
            frm, cancel, descending = official
            if descending:
                logits = logits.flip(-1)
            if cancel is not None:
                mask = torch.as_tensor(cancel, device=logits.device)
                logits = logits.masked_fill(mask, -torch.inf)
            p = torch.softmax(logits, dim=-1)
            probs.append(_probs_on_full_support_borders(p, frm, self.borders))
        return probs

    def _mixture_mean(self, probs: list[torch.Tensor]) -> np.ndarray:
        # 섞은 분포의 평균: 꼬리 구간은 half-normal의 평균 (`FullSupportBarDistribution`).
        # 경계는 fit 뒤에 바뀌지 않으므로 한 번만 만든다
        centers = getattr(self, "mixture_centers_", None)
        if centers is None:
            centers = FullSupportBarDistribution(self.borders).bucket_centers
            self.mixture_centers_ = centers
        p = torch.stack(probs).mean(0)
        return (p @ centers.to(p.device, p.dtype)).cpu().numpy() * self.y_std_ + self.y_mean_

    def _member_mean(self, i: int, p: torch.Tensor) -> torch.Tensor:
        return p @ self.bucket_values_[i].to(p.device)

    def _mean(self, means: list[torch.Tensor]) -> np.ndarray:
        return torch.stack(means).mean(0).cpu().numpy() * self.y_std_ + self.y_mean_

    def _on_common_borders(self, i: int, p: torch.Tensor) -> torch.Tensor:
        if self.border_maps_[i] is None:
            return p
        # power transform한 공간의 경계를 z 공간으로 되돌린 것 위에서 확률을 옮긴다
        borders = self.borders.to(p.device)
        return _probs_on_borders(p, self.border_maps_[i].to(p.device), borders)

    def _mixture(self, probs: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        log_probs = torch.stack(probs).mean(0).clamp_min(1e-12).log().cpu()
        return log_probs, self.borders.double() * self.y_std_ + self.y_mean_
