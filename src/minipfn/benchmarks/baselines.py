"""예측 분포 baseline (issue #48): GBDT와 공개 TFM을 `amlb.evaluate_task(..., uncertainty=True)`에
그대로 넣는 estimator.

모두 기본 설정(튜닝 없음)이다. 분류는 `fit`, `predict_proba`, `classes_`를, 회귀는 `fit`,
`predict`(점 예측), `predict_full`(점 예측, 구간 log 확률, 경계)을 낸다. 경계는 TabPFN처럼 모든 행이
같으면 (B + 1,), 분위수를 내는 방법은 `uncertainty.quantile_histogram`으로 바꾼 행마다의
(n, B + 1)이다.

| 이름 | 분류 | 회귀 분포 | 회귀 점 예측 |
| --- | --- | --- | --- |
| `lightgbm` | 기본 확률 | 수준마다 quantile 모델 (`GBDT_QUANTILE_LEVELS`) | 따로 학습한 L2 |
| `xgboost` | 기본 확률 | 모델 하나가 모든 수준 (`reg:quantileerror`) | 따로 학습한 L2 |
| `catboost` | 기본 확률 | 모델 하나가 모든 수준 (`MultiQuantile`) | 따로 학습한 RMSE |
| `catboost_gaussian` | - | `RMSEWithUncertainty`의 정규분포 (분위수 999개로) | 그 평균 |
| `tabpfn_v2` | 공식 `tabpfn` 패키지 v2 가중치 | `output_type="full"`의 bar distribution | 그 평균 |
| `tabicl` | `tabicl` 패키지 기본 checkpoint | 분위수 999개 (`DENSE_QUANTILE_LEVELS`) | 그 평균 |

GBDT의 범주형 열은 학습 fold의 범주로 고정해 각 패키지의 범주형 처리(LightGBM·XGBoost는 pandas
category, CatBoost는 `cat_features` 문자열)에 넘긴다. 열 이름은 패키지가 받지 않는 문자가 있어
`f0, f1, …`로 바꾼다. 공개 TFM에는 DataFrame을 그대로 넘긴다(전처리는 패키지가 한다).

`catboost`·`lightgbm`·`xgboost`·`tabpfn`은 extra `baselines`에 있다(macOS에서는 OpenMP를 따로
설치한다: brew install libomp). `tabicl`은 기본 의존성이다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from scipy import special
from sklearn.preprocessing import LabelEncoder

from minipfn.benchmarks import uncertainty as unc

CLASSIFIERS = ("lightgbm", "xgboost", "catboost", "tabpfn_v2", "tabicl")
REGRESSORS = ("lightgbm", "xgboost", "catboost", "catboost_gaussian", "tabpfn_v2", "tabicl")
GBDTS = ("lightgbm", "xgboost", "catboost")

# GBDT quantile 회귀의 수준: 가운데 구간(50·80·90·95%)의 양 끝이 모두 들어 있고, 0.1~0.9는 0.05
# 간격이다 (23개)
GBDT_QUANTILE_LEVELS = (
    0.01,
    0.025,
    0.05,
    *(round(0.1 + 0.05 * i, 2) for i in range(17)),
    0.95,
    0.975,
    0.99,
)
# 분포 전체를 내는 방법(TabICL, 정규분포)을 분위수로 바꿀 때의 수준: TabICL이 직접 내는 수준과 같다
DENSE_QUANTILE_LEVELS = np.linspace(0.0, 1.0, 1001)[1:-1]


def methods(task_type: str) -> tuple[str, ...]:
    if task_type == "classification":
        return CLASSIFIERS
    if task_type == "regression":
        return REGRESSORS
    raise ValueError(f"task 종류는 classification이나 regression: {task_type}")


def make_baseline(method: str, task_type: str, device: str = "cpu", seed: int = 0):
    """baseline estimator 하나 (fold마다 새로 만든다)."""
    if method not in methods(task_type):
        raise ValueError(f"{task_type} baseline은 {methods(task_type)} 가운데 하나: {method}")
    if method in GBDTS:
        cls = GBDTClassifier if task_type == "classification" else GBDTQuantileRegressor
        return cls(method, seed=seed)
    if method == "catboost_gaussian":
        return CatBoostGaussianRegressor(seed=seed)
    if method == "tabpfn_v2":
        return OfficialTabPFNv2(task_type, device=device, seed=seed)
    return TabICLBaseline(task_type, device=device, seed=seed)


# ---------------------------------------------------------------------------
# GBDT
# ---------------------------------------------------------------------------


class FrameEncoder:
    """GBDT 입력: 학습 fold의 범주로 범주형 열을 고정하고 열 이름을 `f0, f1, …`로 바꾼다."""

    def fit(self, X) -> FrameEncoder:
        X = pd.DataFrame(X)
        self.columns_ = list(X.columns)
        self.categories_ = {}
        for i, column in enumerate(self.columns_):
            values = X[column]
            # 숫자(bool 포함)가 아닌 열: category, object, pandas 3의 str
            if isinstance(values.dtype, pd.CategoricalDtype) or not is_numeric_dtype(values):
                self.categories_[i] = pd.Index(values.dropna().astype(str).unique()).sort_values()
        return self

    @property
    def categorical_columns(self) -> list[str]:
        return [f"f{i}" for i in self.categories_]

    def transform(self, X, backend: str) -> pd.DataFrame:
        """LightGBM·XGBoost: 범주형 열은 학습 fold의 범주를 가진 category(처음 보는 범주는 NaN).
        CatBoost: 범주형 열은 문자열(NaN은 "nan")."""
        X = pd.DataFrame(X)
        if list(X.columns) != self.columns_:
            raise ValueError("학습 때와 열이 다르다")
        out = {}
        for i, column in enumerate(self.columns_):
            values = X[column]
            if i in self.categories_:
                text = values.astype(object).where(values.notna(), None)
                text = text.map(lambda v: v if v is None else str(v))
                if backend == "catboost":
                    out[f"f{i}"] = text.fillna("nan").astype(str)
                else:
                    out[f"f{i}"] = pd.Categorical(text, categories=self.categories_[i])
            else:
                out[f"f{i}"] = pd.to_numeric(values, errors="coerce").astype(float)
        return pd.DataFrame(out, index=X.index)


def _gbdt(backend: str, objective: str, seed: int, categorical: list[str], **params):
    """기본 설정의 GBDT 하나. objective: classifier, regressor, quantile(LightGBM은 수준 하나,
    XGBoost·CatBoost는 `levels`의 모든 수준)."""
    if backend == "lightgbm":
        import lightgbm as lgb

        common = {"random_state": seed, "verbose": -1, "n_jobs": -1}
        if objective == "classifier":
            return lgb.LGBMClassifier(**common)
        if objective == "regressor":
            return lgb.LGBMRegressor(**common)
        return lgb.LGBMRegressor(objective="quantile", alpha=params["alpha"], **common)
    if backend == "xgboost":
        import xgboost as xgb

        common = {"random_state": seed, "enable_categorical": True, "tree_method": "hist"}
        common["n_jobs"] = -1
        if objective == "classifier":
            return xgb.XGBClassifier(**common)
        if objective == "regressor":
            return xgb.XGBRegressor(**common)
        levels = np.asarray(params["levels"], dtype=float)
        return xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=levels, **common)
    if backend == "catboost":
        import catboost as cb

        common = {
            "random_seed": seed,
            "verbose": 0,
            "thread_count": -1,
            "allow_writing_files": False,
            "cat_features": categorical,
        }
        if objective == "classifier":
            return cb.CatBoostClassifier(**common)
        if objective == "regressor":
            return cb.CatBoostRegressor(**common)
        if objective == "gaussian":
            return cb.CatBoostRegressor(loss_function="RMSEWithUncertainty", **common)
        alphas = ",".join(f"{a:g}" for a in params["levels"])
        return cb.CatBoostRegressor(loss_function=f"MultiQuantile:alpha={alphas}", **common)
    raise ValueError(f"GBDT는 {GBDTS} 가운데 하나: {backend}")


class GBDTClassifier:
    """기본 설정 GBDT의 확률. 클래스는 학습 fold에 있는 것만 (`classes_` 순서)."""

    def __init__(self, backend: str, seed: int = 0) -> None:
        self.backend, self.seed = backend, seed

    def fit(self, X, y) -> GBDTClassifier:
        self.encoder_ = FrameEncoder().fit(X)
        self.labels_ = LabelEncoder().fit(np.asarray(y))
        self.classes_ = self.labels_.classes_
        categorical = self.encoder_.categorical_columns
        self.model_ = _gbdt(self.backend, "classifier", self.seed, categorical)
        self.model_.fit(self.encoder_.transform(X, self.backend), self.labels_.transform(y))
        return self

    def predict_proba(self, X) -> np.ndarray:
        return np.asarray(self.model_.predict_proba(self.encoder_.transform(X, self.backend)))


class GBDTQuantileRegressor:
    """점 예측은 L2(RMSE) 모델, 분포는 `GBDT_QUANTILE_LEVELS`의 quantile 회귀."""

    levels = np.asarray(GBDT_QUANTILE_LEVELS)

    def __init__(self, backend: str, seed: int = 0) -> None:
        self.backend, self.seed = backend, seed

    def fit(self, X, y) -> GBDTQuantileRegressor:
        self.encoder_ = FrameEncoder().fit(X)
        frame, y = self.encoder_.transform(X, self.backend), np.asarray(y, dtype=float)
        categorical = self.encoder_.categorical_columns
        self.mean_model_ = _gbdt(self.backend, "regressor", self.seed, categorical).fit(frame, y)
        if self.backend == "lightgbm":  # LightGBM quantile 목적함수는 수준 하나씩
            self.quantile_models_ = [
                _gbdt(self.backend, "quantile", self.seed, categorical, alpha=a).fit(frame, y)
                for a in self.levels
            ]
        else:
            model = _gbdt(self.backend, "quantile", self.seed, categorical, levels=self.levels)
            self.quantile_models_ = [model.fit(frame, y)]
        return self

    def predict(self, X) -> np.ndarray:
        return np.asarray(self.mean_model_.predict(self.encoder_.transform(X, self.backend)))

    def predict_quantiles(self, X) -> np.ndarray:
        """(n, 수준 수). 엇갈린 분위수는 `quantile_histogram`이 정렬한다."""
        frame = self.encoder_.transform(X, self.backend)
        out = [np.asarray(m.predict(frame), dtype=float) for m in self.quantile_models_]
        return np.stack(out, axis=1) if len(out) > 1 else out[0].reshape(len(frame), -1)

    def predict_full(self, X) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        probs, borders = unc.quantile_histogram(self.levels, self.predict_quantiles(X))
        return self.predict(X), np.log(probs), borders


class CatBoostGaussianRegressor:
    """CatBoost `RMSEWithUncertainty`: 행마다 정규분포 (평균, 분산)."""

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def fit(self, X, y) -> CatBoostGaussianRegressor:
        self.encoder_ = FrameEncoder().fit(X)
        frame = self.encoder_.transform(X, "catboost")
        categorical = self.encoder_.categorical_columns
        self.model_ = _gbdt("catboost", "gaussian", self.seed, categorical)
        self.model_.fit(frame, np.asarray(y, dtype=float))
        return self

    def predict_mean_var(self, X) -> tuple[np.ndarray, np.ndarray]:
        out = np.asarray(self.model_.predict(self.encoder_.transform(X, "catboost")), dtype=float)
        return out[:, 0], out[:, 1]

    def predict(self, X) -> np.ndarray:
        return self.predict_mean_var(X)[0]

    def predict_full(self, X) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mean, var = self.predict_mean_var(X)
        z = special.ndtri(DENSE_QUANTILE_LEVELS)
        values = mean[:, None] + np.sqrt(np.maximum(var, 0.0))[:, None] * z[None, :]
        probs, borders = unc.quantile_histogram(DENSE_QUANTILE_LEVELS, values)
        return mean, np.log(probs), borders


# ---------------------------------------------------------------------------
# 공개 TFM
# ---------------------------------------------------------------------------


class OfficialTabPFNv2:
    """공식 `tabpfn` 패키지의 v2 가중치와 그 버전의 기본 설정
    (`create_default_for_version(ModelVersion.V2)`)."""

    def __init__(self, task_type: str, device: str = "cpu", seed: int = 0) -> None:
        self.task_type, self.device, self.seed = task_type, device, seed

    def fit(self, X, y) -> OfficialTabPFNv2:
        from tabpfn import TabPFNClassifier, TabPFNRegressor
        from tabpfn.constants import ModelVersion

        cls = TabPFNClassifier if self.task_type == "classification" else TabPFNRegressor
        self.model_ = cls.create_default_for_version(
            ModelVersion.V2, device=self.device, random_state=self.seed
        )
        self.model_.fit(X, y if self.task_type == "classification" else np.asarray(y, float))
        if self.task_type == "classification":
            self.classes_ = np.asarray(self.model_.classes_)
        return self

    def predict_proba(self, X) -> np.ndarray:
        return np.asarray(self.model_.predict_proba(X))

    def predict(self, X) -> np.ndarray:
        return np.asarray(self.model_.predict(X))

    def predict_full(self, X) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        import torch

        out = self.model_.predict(X, output_type="full")
        log_probs = torch.log_softmax(out["logits"].double(), dim=-1).cpu().numpy()
        borders = out["criterion"].borders.double().cpu().numpy()
        return np.asarray(out["mean"]), log_probs, borders


class TabICLBaseline:
    """`tabicl` 패키지의 기본 checkpoint와 기본 설정. 회귀 분포는 분위수 999개."""

    def __init__(self, task_type: str, device: str = "cpu", seed: int = 0) -> None:
        self.task_type, self.device, self.seed = task_type, device, seed

    def fit(self, X, y) -> TabICLBaseline:
        from tabicl import TabICLClassifier, TabICLRegressor

        if self.task_type == "classification":
            self.model_ = TabICLClassifier(device=self.device, random_state=self.seed).fit(X, y)
            self.classes_ = np.asarray(self.model_.classes_)
        else:
            self.model_ = TabICLRegressor(device=self.device, random_state=self.seed)
            self.model_.fit(X, np.asarray(y, dtype=float))
        return self

    def predict_proba(self, X) -> np.ndarray:
        return np.asarray(self.model_.predict_proba(X))

    def predict(self, X) -> np.ndarray:
        return np.asarray(self.model_.predict(X))

    def predict_full(self, X) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        out = self.model_.predict(
            X, output_type=["mean", "quantiles"], alphas=DENSE_QUANTILE_LEVELS.tolist()
        )
        probs, borders = unc.quantile_histogram(DENSE_QUANTILE_LEVELS, out["quantiles"])
        return np.asarray(out["mean"]), np.log(probs), borders
