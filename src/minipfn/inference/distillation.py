"""TFM을 데이터셋마다 GBDT 학생 모델로 증류한다: sklearn 스타일 `DistilledClassifier`·
`DistilledRegressor`. `fit`에서 teacher(학습한 TFM의 추론 estimator, 예: `EnsembleClassifier`)가
학습 행과 synthetic query 행에 낸 예측을 학생(XGBoost, LightGBM, sklearn HistGradientBoosting)이
배운다. `predict`는 학생만 쓰므로 GPU 없이 CPU에서 ms 단위로 예측한다.

    teacher = EnsembleClassifier(model, n_estimators=4, device="cuda")
    clf = DistilledClassifier(teacher, method="synthetic", student="xgboost").fit(x_train, y_train)
    proba = clf.predict_proba(x_test)  # 학생(CPU)만 쓴다

teacher 라벨을 만드는 방식 두 가지 (`method`로 고르는 묶음 `METHODS`):

- `"oof"`: "Pocket Foundation Models"(arXiv 2605.18654). teacher가 자기 학습 행을 context에 넣고
  그 행을 예측하면 정답이 새어 soft target이 거의 one-hot이 된다. 그래서 학습 행을 stratified
  5-fold로 나누고, fold마다 나머지 4개 fold를 context로 그 fold를 예측한다(out-of-fold). 논문은
  fold 수 3·5·10의 결과가 같았다고 적었다. 손실은 Hinton식으로 soft target(α = 0.7)과 정답
  라벨(1 − α)을 섞는다. TabICLv2 → XGBoost가 분류 153개에서 macro AUC 0.882(teacher의 96.5%),
  CPU 1.9ms
- `"synthetic"`: "Distillation of Tabular Foundation Models into Efficient Predictors"
  (arXiv 2610.01435). teacher는 학습 행 전체를 context로 학습 행 자체(정답이 context에 있다)와
  synthetic query 행을 예측하고, 학생은 teacher 예측만 배운다(α = 1). synthetic 행은 CutMix로
  학습 행의 2배를 만든다: 학습 행 두 개를 뽑아 feature마다 확률 0.5로 둘 중 하나의 값을 쓴다.
  손실은 학습 행 묶음과 synthetic 행 묶음의 평균 손실을 더한 것이다(행 가중치로 맞춘다).
  TabArena에서 TabPFN v3 → XGBoost가 Elo 1458로 XGBoost (tuned + ensembled) 1360보다 높다
- `"labels"`: teacher 없이 정답 라벨만 배운 같은 학생 (비교용 기본 GBDT)

두 방식의 부품(`teacher_labels`, `synthetic_ratio`, `alpha`)은 따로 줄 수도 있다
(`METHODS`의 값을 덮어쓴다). 예: OOF 라벨 + synthetic 행이면
`DistilledClassifier(teacher, method="oof", synthetic_ratio=2.0)`. 이때 synthetic 행은 학습 행
전체를 context로 둔 teacher가 예측한다.

여기서 정한 것 (논문에 없거나 논문과 다르다):

- 분류 손실은 soft cross-entropy다("Pocket"은 tree 학생에 클래스마다 MSE 회귀를 썼다고 적었고,
  2610.01435는 soft cross-entropy). 이진 분류의 XGBoost·LightGBM 학생은 soft label을 그대로 받는
  목적함수(`binary:logistic`, `cross_entropy`: label이 0~1인 log loss)로 배운다. 다중 분류와
  HistGradientBoosting 학생은 행을 클래스 수만큼 복제해 클래스 k의 복제에 가중치 p_k를 준다
  (multiclass log loss 그대로 soft cross-entropy와 기울기·hessian이 같다). 학생 학습 행이 클래스
  수배로 늘지 않도록 target이 `MIN_COPY_TARGET`보다 작은 복제는 뺀다(행마다 target이 가장 큰
  클래스는 남긴다). 뺀 복제의 기울기는 크기가 그 target 정도라 무시한다
- 회귀 target은 teacher 예측분포의 평균(`predict`)이고 손실은 제곱 오차다. α가 1보다 작으면
  α·teacher 평균 + (1 − α)·정답을 배운다(제곱 오차에서는 두 손실을 섞은 것과 같다)
- "Pocket"의 행마다의 온도(teacher entropy로 1~5)와 confidence 가중치는 쓰지 않는다 (식이 다
  적혀 있지 않다)
- 학생 설정은 라이브러리마다 AutoGluon·TabArena 기본값에 가깝게 둔다(`STUDENT_DEFAULTS`). 나무
  수 상한 10,000에 학생 학습 행(synthetic 포함)의 10%를 떼어 50 round early stopping한다(검증
  손실도 teacher target에 대한 손실이다). 학생 bagging은 하지 않는다(2610.01435는 8-fold)
- 범주형 열은 순서 번호로 바꿔 학생의 범주형 지원으로 넘긴다(XGBoost `enable_categorical`,
  LightGBM `categorical_feature`, HistGradientBoosting은 범주가 255개 이하인 열만). teacher에는
  번호를 원래 범주 값(문자열)으로 되돌려 넘긴다. 그래서 teacher는 증류하지 않을 때와 같은 범주
  순서로 번호를 매긴다(번호 자체를 넘기면 문자열 "10.0"이 "2.0"보다 앞에 온다)
- OOF는 학습 행이 `n_folds`보다 적으면 fold 수를 행 수로 줄인다. fold의 context에 클래스가
  하나뿐이면 teacher를 부르지 않고 그 클래스의 확률 1을 쓴다(학습 행이 하나인 클래스가 그 fold에
  있을 때)
- teacher 예측은 `TEACHER_BATCH_ROWS`행씩 나눠 낸다(synthetic 행까지 한 번에 넣으면 GPU 메모리가
  모자랄 수 있다. 5,000행은 v2 상한인 학습 행 10,000의 TabArena 테스트 행 수 정도다). 행마다의
  예측은 같은 테스트 행 묶음과 상관없다
"""

from __future__ import annotations

import copy
import time
import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from minipfn.inference.preprocessing import apply_encoder, to_numeric

STUDENTS = ("xgboost", "lightgbm", "hist")
TEACHER_LABELS = ("oof", "full")
METHODS = {
    "oof": {"teacher_labels": "oof", "synthetic_ratio": 0.0, "alpha": 0.7},
    "synthetic": {"teacher_labels": "full", "synthetic_ratio": 2.0, "alpha": 1.0},
    "labels": {"teacher_labels": "full", "synthetic_ratio": 0.0, "alpha": 0.0},
}
N_FOLDS = 5
VALIDATION_FRACTION = 0.1
EARLY_STOPPING_ROUNDS = 50
MIN_EARLY_STOPPING_ROWS = 20  # 이보다 적으면 검증 행을 떼지 않고 나무 수를 고정한다
FIXED_ROUNDS = 100
TEACHER_BATCH_ROWS = 5_000
MIN_COPY_TARGET = 1e-3  # 분류 soft target 복제에서 뺄 target의 상한
SOFT_BINARY_STUDENTS = ("xgboost", "lightgbm")  # 이진 soft label을 그대로 받는 학생
MAX_HIST_CATEGORIES = 255  # HistGradientBoosting 범주형 열의 범주 수 상한 (max_bins)
STUDENT_DEFAULTS = {
    "xgboost": {
        "n_estimators": 10_000,
        "learning_rate": 0.1,
        "max_depth": 6,
        "tree_method": "hist",
        "n_jobs": -1,
    },
    "lightgbm": {"n_estimators": 10_000, "learning_rate": 0.05, "n_jobs": -1, "verbose": -1},
    "hist": {"max_iter": 10_000, "learning_rate": 0.1},
}


def cutmix(X: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """학습 행 두 개를 뽑아 feature마다 확률 0.5로 둘 중 하나의 값을 쓴 synthetic 행 `n`개."""
    a = X[rng.integers(len(X), size=n)]
    b = X[rng.integers(len(X), size=n)]
    return np.where(rng.random(a.shape) < 0.5, a, b)


def resolve_method(method: str, **overrides: Any) -> dict:
    """`METHODS`의 묶음에 None이 아닌 `overrides`를 덮어쓴 설정."""
    if method not in METHODS:
        raise ValueError(f"method는 {tuple(METHODS)} 가운데 하나여야 한다: {method}")
    config = {**METHODS[method], **{k: v for k, v in overrides.items() if v is not None}}
    if config["teacher_labels"] not in TEACHER_LABELS:
        raise ValueError(f"teacher_labels는 {TEACHER_LABELS} 가운데 하나: {config}")
    if not 0.0 <= config["alpha"] <= 1.0 or config["synthetic_ratio"] < 0:
        raise ValueError(f"alpha는 0~1, synthetic_ratio는 0 이상: {config}")
    return config


class _Student:
    """GBDT 학생 하나: 행 가중치가 있는 학습 행과 검증 행으로 early stopping해 fit한다.
    `task`는 "regression"(실수 target), "classification"(정수 라벨), "soft_binary"(0~1 label의
    이진 log loss, `SOFT_BINARY_STUDENTS`만)이다."""

    def __init__(self, backend: str, params: dict, task: str) -> None:
        self.backend = backend
        self.params = {**STUDENT_DEFAULTS[backend], **params}
        self.task = task

    def fit(self, X, y, w, X_val, y_val, w_val, cat_ix: list[int], n_categories: list[int]):
        self.cat_ix, self.n_categories = cat_ix, n_categories
        early = X_val is not None
        if self.backend == "xgboost":
            self._fit_xgboost(X, y, w, X_val, y_val, w_val, early)
        elif self.backend == "lightgbm":
            self._fit_lightgbm(X, y, w, X_val, y_val, w_val, early)
        else:
            self._fit_hist(X, y, w, X_val, y_val, w_val, early)
        return self

    def _frame(self, X: np.ndarray) -> pd.DataFrame:
        # XGBoost 범주형 열: 학습 때 정한 범주(번호 0..n-1)의 pandas Categorical. 결측·새 범주는 NaN
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
        for i, n in zip(self.cat_ix, self.n_categories, strict=True):
            codes = np.where(np.isfinite(X[:, i]), X[:, i], -1).astype(int)
            df[f"f{i}"] = pd.Categorical.from_codes(np.where(codes < n, codes, -1), range(n))
        return df

    def _fit_xgboost(self, X, y, w, X_val, y_val, w_val, early: bool) -> None:
        import xgboost as xgb

        params = dict(self.params, enable_categorical=True)
        cls = xgb.XGBClassifier if self.task == "classification" else xgb.XGBRegressor
        if self.task == "soft_binary":
            params.update(objective="binary:logistic", eval_metric="logloss")
        if early:
            self.model = cls(early_stopping_rounds=EARLY_STOPPING_ROUNDS, **params)
            self.model.fit(
                self._frame(X),
                y,
                sample_weight=w,
                eval_set=[(self._frame(X_val), y_val)],
                sample_weight_eval_set=[w_val],
                verbose=False,
            )
        else:
            self.model = cls(**{**params, "n_estimators": FIXED_ROUNDS})
            self.model.fit(self._frame(X), y, sample_weight=w, verbose=False)

    def _fit_lightgbm(self, X, y, w, X_val, y_val, w_val, early: bool) -> None:
        import lightgbm as lgb

        cls = lgb.LGBMClassifier if self.task == "classification" else lgb.LGBMRegressor
        params = dict(self.params)
        if self.task == "soft_binary":
            params["objective"] = "cross_entropy"
        cat = self.cat_ix or "auto"
        if early:
            self.model = cls(**params)
            self.model.fit(
                X,
                y,
                sample_weight=w,
                eval_X=X_val,
                eval_y=y_val,
                eval_sample_weight=[w_val],
                categorical_feature=cat,
                callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
            )
        else:
            self.model = cls(**{**params, "n_estimators": FIXED_ROUNDS})
            self.model.fit(X, y, sample_weight=w, categorical_feature=cat)

    def _fit_hist(self, X, y, w, X_val, y_val, w_val, early: bool) -> None:
        from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

        if self.task == "soft_binary":
            raise ValueError("HistGradientBoosting 학생은 soft label을 받지 않는다")
        regression = self.task == "regression"
        cls = HistGradientBoostingRegressor if regression else HistGradientBoostingClassifier
        mask = np.zeros(X.shape[1], dtype=bool)
        for i, n in zip(self.cat_ix, self.n_categories, strict=True):
            mask[i] = n <= MAX_HIST_CATEGORIES
        params = dict(self.params, categorical_features=mask if mask.any() else None)
        if early:
            self.model = cls(early_stopping=True, n_iter_no_change=EARLY_STOPPING_ROUNDS, **params)
            self.model.fit(X, y, sample_weight=w, X_val=X_val, y_val=y_val, sample_weight_val=w_val)
        else:
            self.model = cls(**{**params, "max_iter": FIXED_ROUNDS, "early_stopping": False})
            self.model.fit(X, y, sample_weight=w)

    def _input(self, X: np.ndarray):
        return self._frame(X) if self.backend == "xgboost" else X

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.task == "soft_binary":
            p = np.clip(self.model.predict(self._input(X)), 0.0, 1.0)
            return np.stack([1 - p, p], axis=1)
        return self.model.predict_proba(self._input(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(self._input(X))

    @property
    def n_trees(self) -> int:
        """early stopping으로 고른 boosting round 수."""
        m = self.model
        if self.backend == "xgboost":
            best = getattr(m, "best_iteration", None)
            return int(best) + 1 if best is not None else int(m.n_estimators)
        if self.backend == "lightgbm":
            return int(m.best_iteration_ or m.n_estimators_)
        return int(m.n_iter_)


class _Distilled:
    """teacher 라벨·synthetic 행·학생 학습의 공통 부분. 분류·회귀가 정답과 teacher 예측의 형식
    (`_prepare_y`·`_teacher_predict`·`_observed_targets`)과 학생 학습(`_fit_student`)을 정한다."""

    def __init__(
        self,
        teacher: Any,
        *,
        method: str = "synthetic",
        student: str = "xgboost",
        teacher_labels: str | None = None,
        synthetic_ratio: float | None = None,
        alpha: float | None = None,
        n_folds: int = N_FOLDS,
        student_params: dict | None = None,
        categorical_features_indices: list[int] | None = None,
        teacher_batch_rows: int = TEACHER_BATCH_ROWS,
        random_state: int = 0,
    ) -> None:
        self.config = resolve_method(
            method, teacher_labels=teacher_labels, synthetic_ratio=synthetic_ratio, alpha=alpha
        )
        if student not in STUDENTS:
            raise ValueError(f"student는 {STUDENTS} 가운데 하나여야 한다: {student}")
        self.teacher = teacher
        self.method = method
        self.student = student
        self.n_folds = n_folds
        self.student_params = dict(student_params or {})
        if categorical_features_indices is None:
            categorical_features_indices = getattr(teacher, "categorical_features_indices", None)
        self.categorical_features_indices = categorical_features_indices
        self.teacher_batch_rows = teacher_batch_rows
        self.random_state = random_state

    @property
    def needs_teacher(self) -> bool:
        return self.config["alpha"] > 0 or self.config["synthetic_ratio"] > 0

    def _encode(self, X: Any) -> np.ndarray:
        X_num, self.cat_ix_, self.encoder_ = to_numeric(X, self.categorical_features_indices)
        # 범주 수: 결측이 있는 열은 OrdinalEncoder가 NaN도 범주로 두지만 번호는 NaN이라 세지 않는다
        if self.encoder_ is None:
            self.n_categories_ = []
        else:
            self.n_categories_ = [int(pd.notna(c).sum()) for c in self.encoder_.categories_]
        return X_num

    def _transform(self, X: Any) -> np.ndarray:
        return apply_encoder(pd.DataFrame(X), self.cat_ix_, self.encoder_)

    def _teacher(self):
        # 범주형 열을 이 표의 열 번호로 알려 준다 (모델은 같이 쓴다)
        teacher = copy.copy(self.teacher)
        teacher.categorical_features_indices = list(self.cat_ix_)
        return teacher

    def _teacher_frame(self, X: np.ndarray) -> pd.DataFrame:
        """teacher에 넘길 표: 범주형 열의 번호를 원래 범주 값(`to_numeric`이 쓴 문자열)으로
        되돌린다. teacher가 다시 번호를 매길 때 증류하지 않을 때와 같은 순서가 된다. 결측은 NaN."""
        df = pd.DataFrame(X)
        for j, i in enumerate(self.cat_ix_):
            categories = np.asarray(self.encoder_.categories_[j], dtype=object)
            codes = X[:, i]
            seen = np.isfinite(codes)
            column = np.full(len(X), np.nan, dtype=object)
            column[seen] = categories[codes[seen].astype(int)]
            df[i] = column
        return df

    def _fit_teacher_predict(self, X_ctx, y_ctx, X_query) -> np.ndarray:
        constant = self._constant_prediction(y_ctx, len(X_query))
        if constant is not None:
            return constant
        teacher = self._teacher().fit(self._teacher_frame(X_ctx), y_ctx)
        parts = [
            self._teacher_predict(
                teacher, self._teacher_frame(X_query[i : i + self.teacher_batch_rows])
            )
            for i in range(0, len(X_query), self.teacher_batch_rows)
        ]
        return np.concatenate(parts) if parts else self._empty_prediction(0)

    def _constant_prediction(self, y_ctx: np.ndarray, n: int) -> np.ndarray | None:
        """teacher를 부르지 않아도 되는 context의 예측 (분류: 클래스가 하나뿐일 때)."""
        return None

    def _folds(self, y: np.ndarray) -> list:
        """OOF fold (학습 행 번호, 예측할 행 번호). 행이 `n_folds`보다 적으면 fold 수를 행 수로."""
        if len(y) < 2:
            raise ValueError("teacher_labels='oof'에는 학습 행이 2개 이상 필요하다")
        k = min(self.n_folds, len(y))
        index = np.zeros(len(y))
        if self._stratify and np.bincount(y).max() >= k:
            # 행이 k개보다 적은 클래스가 있으면 sklearn이 경고만 하고 나눈다
            splitter = StratifiedKFold(k, shuffle=True, random_state=self.random_state)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                return list(splitter.split(index, y))
        return list(KFold(k, shuffle=True, random_state=self.random_state).split(index, y))

    def _teacher_targets(self, X: np.ndarray, y: np.ndarray, X_syn: np.ndarray):
        """(학습 행의 teacher 예측 또는 None, synthetic 행의 teacher 예측)."""
        cfg = self.config
        observed = None
        if cfg["alpha"] > 0 and cfg["teacher_labels"] == "oof":
            observed = self._empty_prediction(len(X))
            for train_ix, held_ix in self._folds(y):
                observed[held_ix] = self._fit_teacher_predict(X[train_ix], y[train_ix], X[held_ix])
        if cfg["alpha"] > 0 and cfg["teacher_labels"] == "full":
            query = np.concatenate([X, X_syn])
            both = self._fit_teacher_predict(X, y, query)
            return both[: len(X)], both[len(X) :]
        synthetic = self._fit_teacher_predict(X, y, X_syn) if len(X_syn) else None
        return observed, synthetic

    def fit(self, X: Any, y: Any):
        start = time.perf_counter()
        rng = np.random.default_rng(self.random_state)
        X_num = self._encode(X)
        y = self._prepare_y(np.asarray(y))
        n_syn = int(round(self.config["synthetic_ratio"] * len(X_num)))
        X_syn = cutmix(X_num, n_syn, rng) if n_syn else X_num[:0]
        observed, synthetic = (None, None)
        if self.needs_teacher:
            observed, synthetic = self._teacher_targets(X_num, y, X_syn)
        teacher_seconds = time.perf_counter() - start

        # 학습 행 묶음과 synthetic 행 묶음의 가중치 합을 같게 (묶음마다의 평균 손실을 더한다)
        X_all = np.concatenate([X_num, X_syn])
        targets = np.concatenate([self._observed_targets(y, observed), self._as_targets(synthetic)])
        weights = np.concatenate([np.ones(len(X_num)), np.full(n_syn, len(X_num) / max(n_syn, 1))])
        self._fit_student(X_all, targets, weights, rng)
        self.fit_record_ = {
            "method": self.method,
            "student": self.student,
            **self.config,
            "rows": len(X_num),
            "synthetic_rows": n_syn,
            "n_trees": self.student_.n_trees if self.student_ is not None else 0,
            "teacher_seconds": teacher_seconds,
            "student_seconds": time.perf_counter() - start - teacher_seconds,
        }
        return self

    def _split(self, n: int, rng: np.random.Generator):
        """(학습 행 번호, 검증 행 번호 또는 None). 행이 적으면 검증 행을 떼지 않는다."""
        order = rng.permutation(n)
        if n < MIN_EARLY_STOPPING_ROWS:
            return order, None
        n_val = max(1, int(round(VALIDATION_FRACTION * n)))
        return order[n_val:], order[:n_val]


class DistilledClassifier(_Distilled):
    """분류: 학생이 teacher의 클래스 확률(soft target)을 배운다 (모듈 docstring)."""

    _stratify = True

    def _prepare_y(self, y: np.ndarray) -> np.ndarray:
        self.label_encoder_ = LabelEncoder()
        y = self.label_encoder_.fit_transform(y)
        self.classes_ = self.label_encoder_.classes_
        return y

    def _empty_prediction(self, n: int) -> np.ndarray:
        return np.zeros((n, len(self.classes_)))

    def _constant_prediction(self, y_ctx: np.ndarray, n: int) -> np.ndarray | None:
        classes = np.unique(y_ctx)
        if len(classes) != 1:
            return None
        out = self._empty_prediction(n)
        out[:, classes[0]] = 1.0
        return out

    def _teacher_predict(self, teacher, X) -> np.ndarray:
        # fold의 학습 행에 없던 클래스는 확률 0 (teacher의 classes_는 번호의 일부다)
        out = self._empty_prediction(len(X))
        if len(X):
            out[:, np.asarray(teacher.classes_, dtype=int)] = teacher.predict_proba(X)
        return out

    def _as_targets(self, proba: np.ndarray | None) -> np.ndarray:
        return self._empty_prediction(0) if proba is None else proba

    def _observed_targets(self, y: np.ndarray, proba: np.ndarray | None) -> np.ndarray:
        alpha = self.config["alpha"]
        hard = np.eye(len(self.classes_))[y]
        return hard if proba is None else alpha * proba + (1 - alpha) * hard

    def _fit_student(self, X, targets, weights, rng) -> None:
        n_classes = len(self.classes_)
        self.student_ = None
        if n_classes == 1:
            return
        train_ix, val_ix = self._split(len(X), rng)
        if n_classes == 2 and self.student in SOFT_BINARY_STUDENTS:
            # 이진 soft label을 그대로 (행을 복제하지 않는다)
            val = (None, None, None)
            if val_ix is not None:
                val = (X[val_ix], targets[val_ix, 1], weights[val_ix])
            student = _Student(self.student, self.student_params, "soft_binary")
            self.student_ = student.fit(
                X[train_ix],
                targets[train_ix, 1],
                weights[train_ix],
                *val,
                self.cat_ix_,
                self.n_categories_,
            )
            return
        # 학습 행에 남는 복제가 없는 클래스가 있으면 그 클래스의 검증 행을 학습으로 옮긴다
        # (학생이 모든 클래스를 알아야 predict_proba의 열이 classes_와 맞는다)
        kept = self._copies(targets)
        missing = kept[train_ix].sum(axis=0) == 0
        if val_ix is not None and missing.any():
            move = kept[val_ix][:, missing].any(axis=1)
            train_ix = np.concatenate([train_ix, val_ix[move]])
            val_ix = val_ix[~move] if (~move).any() else None
        val = (None, None, None) if val_ix is None else self._expand(X, targets, weights, val_ix)
        student = _Student(self.student, self.student_params, "classification")
        self.student_ = student.fit(
            *self._expand(X, targets, weights, train_ix), *val, self.cat_ix_, self.n_categories_
        )

    @staticmethod
    def _copies(targets: np.ndarray) -> np.ndarray:
        """남길 클래스 복제 (행, 클래스): target이 `MIN_COPY_TARGET` 이상이거나 행에서 가장 큰
        것."""
        kept = targets >= MIN_COPY_TARGET
        kept[np.arange(len(targets)), targets.argmax(axis=1)] = True
        return kept & (targets > 0)

    @classmethod
    def _expand(cls, X, targets, weights, ix):
        """soft target을 클래스마다 복제한 행으로: 행 i의 클래스 k 복제에 가중치 w_i · p_ik.
        `_copies`가 남기지 않는 복제는 뺀다."""
        rows, classes = np.nonzero(cls._copies(targets[ix]))
        w = weights[ix][rows] * targets[ix][rows, classes]
        return X[ix][rows], classes, w

    def predict_proba(self, X: Any) -> np.ndarray:
        X_num = self._transform(X)
        if self.student_ is None:
            return np.ones((len(X_num), 1))
        proba = self.student_.predict_proba(X_num)
        return proba / proba.sum(axis=1, keepdims=True)

    def predict(self, X: Any) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


class DistilledRegressor(_Distilled):
    """회귀: 학생이 teacher 예측분포의 평균을 제곱 오차로 배운다 (모듈 docstring)."""

    _stratify = False

    def _prepare_y(self, y: np.ndarray) -> np.ndarray:
        return y.astype(np.float64)

    def _empty_prediction(self, n: int) -> np.ndarray:
        return np.zeros(n)

    def _teacher_predict(self, teacher, X) -> np.ndarray:
        return np.asarray(teacher.predict(X), dtype=np.float64) if len(X) else np.zeros(0)

    def _as_targets(self, mean: np.ndarray | None) -> np.ndarray:
        return np.zeros(0) if mean is None else mean

    def _observed_targets(self, y: np.ndarray, mean: np.ndarray | None) -> np.ndarray:
        alpha = self.config["alpha"]
        return y if mean is None else alpha * mean + (1 - alpha) * y

    def _fit_student(self, X, targets, weights, rng) -> None:
        train_ix, val_ix = self._split(len(X), rng)
        val = (None, None, None)
        if val_ix is not None:
            val = (X[val_ix], targets[val_ix], weights[val_ix])
        self.student_ = _Student(self.student, self.student_params, "regression").fit(
            X[train_ix],
            targets[train_ix],
            weights[train_ix],
            *val,
            self.cat_ix_,
            self.n_categories_,
        )

    def predict(self, X: Any) -> np.ndarray:
        return np.asarray(self.student_.predict(self._transform(X)), dtype=np.float64)
