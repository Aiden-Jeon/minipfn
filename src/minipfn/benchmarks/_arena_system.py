"""sklearn 형식 estimator를 `tabarena`의 system(`ExternalSystemModel`)으로 감싼다.

`tabarena`를 import하므로 `minipfn.benchmarks.arena`가 쓸 때만 불러온다. system 설정은
yaml로 저장될 수 있어서 estimator를 만드는 함수는 이 모듈의 registry에 두고 그 이름만 넘긴다
(`debug_mode=True`로 같은 프로세스에서 돌리므로 registry가 그대로 보인다).

estimator를 만드는 함수에 `fit_groups = True`를 달면 fit에 `groups`(행마다의 group 번호)도
넘긴다. task의 split 정보(`validation_metadata`의 group·시간 열)로 만든다
(`minipfn.benchmarks.arena.split_groups`).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from tabarena.benchmark.exec_models import ExternalSystemModel

from minipfn.benchmarks.arena import split_groups

_FACTORIES: dict[str, Callable[[str, list[int]], Any]] = {}


def register_factory(make_estimator: Callable[[str, list[int]], Any]) -> str:
    key = f"factory{len(_FACTORIES)}"
    _FACTORIES[key] = make_estimator
    return key


def experiments(suite, method: str, key: str, bagging: int) -> list:
    from tabarena.benchmark.experiment import (
        BeyondArenaExperimentBundle,
        TabArenaV0pt1ExperimentBundle,
    )
    from tabarena.utils.config_utils import SystemConfigGenerator

    generator = SystemConfigGenerator(
        model_cls=EstimatorSystem,
        name=method,
        manual_configs=[{"factory": key, "bagging": bagging}],
    )
    bundle = (
        BeyondArenaExperimentBundle
        if suite.context == "BeyondArenaContext"
        else TabArenaV0pt1ExperimentBundle
    )
    return bundle(models=[(generator, 0)], system_experiments=True).build_experiments()


def prepare_frame(X: pd.DataFrame) -> tuple[pd.DataFrame, list[int]]:
    """날짜 열은 초 단위 수로 바꾸고, 숫자가 아닌 열(범주·텍스트·bool이 아닌 object)은 범주형
    번호로 돌려준다. 열 순서는 그대로다."""
    X = X.copy()
    categorical = []
    for i, col in enumerate(X.columns):
        s = X[col]
        if pd.api.types.is_datetime64_any_dtype(s):
            s = s.dt.tz_localize(None) if getattr(s.dt, "tz", None) is not None else s
            # 단위(datetime64[ns]·[us]·[s])와 상관없이 1970-01-01부터의 초. NaT는 NaN
            X[col] = (s - pd.Timestamp(0)) / pd.Timedelta(seconds=1)
        elif pd.api.types.is_bool_dtype(s):
            X[col] = s.astype(float)
        elif not pd.api.types.is_numeric_dtype(s):
            categorical.append(i)
    return X, categorical


class EstimatorSystem(ExternalSystemModel):
    """registry의 함수로 만든 estimator 하나(`bagging`이 2 이상이면 k개)를 학습 split에 fit한다."""

    warmup_torch_device = True

    def __init__(self, *, factory: str, bagging: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.factory = factory
        self.bagging = bagging
        self.estimators_: list = []

    def _fit_system(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        problem_type: str,
        random_state,
        validation_metadata=None,
        **_,
    ):
        make = _FACTORIES[self.factory]
        X = X.reset_index(drop=True)
        groups = (
            split_groups(X, validation_metadata) if getattr(make, "fit_groups", False) else None
        )
        self.grouped_ = groups is not None
        X, categorical = prepare_frame(X)
        y = y.reset_index(drop=True)
        self.problem_type_ = problem_type
        self.categorical_ = categorical
        if self.bagging and self.bagging > 1:
            parts = self._bags(X, y, problem_type, random_state)
        else:
            parts = [np.arange(len(X))]
        self.estimators_ = [
            make(problem_type, categorical).fit(
                X.iloc[ix],
                y.iloc[ix].to_numpy(),
                **({} if groups is None else {"groups": groups[ix]}),
            )
            for ix in parts
        ]
        return self

    def post_fit(self, X: pd.DataFrame, y: pd.Series, X_test: pd.DataFrame):
        # 예측이 끝난 뒤(post_predict) 쓸 테스트 행 하나
        self.test_row_ = X_test.iloc[:1]

    def post_predict(self):
        """학습·추론 시간을 재지 않는 hook. estimator에 `after_predict(X_row)`가 있으면 테스트 행
        하나(`_predict`와 같은 전처리)로 부른다 (한 행 예측 시간 재기 등)."""
        row = getattr(self, "test_row_", None)
        hooks = [getattr(e, "after_predict", None) for e in self.estimators_]
        if row is None or not any(hooks):
            return
        Xp, _ = prepare_frame(row)
        Xp = Xp.reset_index(drop=True)
        for hook in hooks:
            if hook is not None:
                hook(Xp)

    def _bags(self, X, y, problem_type: str, random_state) -> list[np.ndarray]:
        seed = 0 if random_state is None else int(random_state)
        splitter = KFold(self.bagging, shuffle=True, random_state=seed)
        if problem_type != "regression" and y.value_counts().min() >= self.bagging:
            splitter = StratifiedKFold(self.bagging, shuffle=True, random_state=seed)
        return [train_ix for train_ix, _ in splitter.split(X, y)]

    def _predict(self, X: pd.DataFrame) -> pd.Series:
        Xp, _ = prepare_frame(X)
        Xp = Xp.reset_index(drop=True)
        pred = np.mean([e.predict(Xp) for e in self.estimators_], axis=0)
        return pd.Series(pred, index=X.index)

    def _predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        Xp, _ = prepare_frame(X)
        Xp = Xp.reset_index(drop=True)
        # bag마다 본 클래스가 다를 수 있다: 모든 bag의 클래스로 열을 맞추고 없는 클래스는 0.
        # 열 순서는 라벨의 자연 순서(np.unique, estimator의 LabelEncoder와 같다)여야 한다.
        # TabArena가 이진 분류는 둘째 열을, 다중 분류는 열을 위치로 읽기 때문이다
        classes = list(
            np.unique(np.concatenate([np.asarray(e.classes_) for e in self.estimators_]))
        )
        total = pd.DataFrame(0.0, index=range(len(Xp)), columns=classes)
        for e in self.estimators_:
            proba = pd.DataFrame(np.asarray(e.predict_proba(Xp)), columns=list(e.classes_))
            total = total.add(proba.reindex(columns=classes, fill_value=0.0))
        total = total / len(self.estimators_)
        total.index = X.index
        return total

    def cleanup(self):
        self.estimators_ = []
