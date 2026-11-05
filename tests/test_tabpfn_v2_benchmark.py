"""benchmark 점수 계산과 평가 스크립트의 task 선택 테스트."""

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from minipfn.benchmarks.amlb import score_fold


def test_test_rows_of_classes_unseen_in_training_are_ignored():
    classes = np.array([0, 1, 2])
    y_true = np.array([0, 1, 2, 0, 1, 2, 3])  # 클래스 3은 학습 fold에 없었다
    proba = np.full((7, 3), 1 / 3)
    proba[np.arange(6), y_true[:6]] = 0.8
    proba = proba / proba.sum(axis=1, keepdims=True)
    assert score_fold("classification", y_true, proba, classes) == pytest.approx(1.0)


def test_single_class_fold_scores_nan():
    proba = np.array([[0.7, 0.3], [0.6, 0.4]])
    assert np.isnan(score_fold("classification", np.array([0, 0]), proba, np.array([0, 1])))


def _eval_script():
    path = Path(__file__).parents[1] / "scripts" / "tabpfn_v2" / "eval.py"
    spec = importlib.util.spec_from_file_location("eval_tabpfn_v2", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_unknown_task_ids_are_reported():
    script = _eval_script()
    with pytest.raises(SystemExit, match="35995"):
        script.selected_tasks(argparse.Namespace(suite="all", tasks="359955,35995"))
    assert script.selected_tasks(argparse.Namespace(suite="all", tasks="359955")) == [
        ("classification", 359955)
    ]


def test_one_hot_modes_are_parsed_and_headline_is_the_default_mode():
    script = _eval_script()
    assert script.parse_args([]).one_hot == ["all"]
    assert script.parse_args(["--one-hot", "all,off,all"]).one_hot == ["all", "off"]
    with pytest.raises(SystemExit):
        script.parse_args(["--one-hot", "half"])
    assert script.headline_mode(["off", "power", "all"]) == "all"  # 고르지 않고 기본값
    assert script.headline_mode(["power", "off"]) == "power"


def test_expands_one_hot_needs_a_small_categorical_column():
    import types

    import pandas as pd

    script = _eval_script()

    def task(X, cat):
        df = pd.DataFrame(X)

        def folds(n_folds=10):  # fold 하나: 행 전체가 학습 행
            return [(0, np.arange(len(df)), np.arange(0))]

        return types.SimpleNamespace(X=df, categorical_indices=cat, folds=folds)

    many = [f"c{i}" for i in range(30)]
    assert not script.expands_one_hot(task({"a": [1.0] * 30}, []))
    assert script.expands_one_hot(task({"a": [0, 1, 2] * 10}, [0]))
    assert script.expands_one_hot(task({"a": list("xyz") * 10}, []))  # 문자열 열도 범주형
    assert not script.expands_one_hot(task({"a": many}, []))  # 범주가 많으면 번호로 둔다
    wide = {f"c{i}": [0, 1, 2] * 10 for i in range(200)}
    assert not script.expands_one_hot(task(wide, list(range(200))))  # 폭 상한에 걸려 펼치지 않는다


def test_expands_one_hot_decides_per_fold_on_the_training_rows():
    import types

    import pandas as pd

    script = _eval_script()
    # 전체로는 범주 33개(>20)라 펼치지 않지만, 앞 fold의 학습 행(범주 3개)은 펼친다
    values = list("xyz") * 10 + [f"c{i}" for i in range(30)]
    X = pd.DataFrame({"a": values})
    folds = [
        (0, np.arange(30), np.arange(30, 60)),  # 학습 행: xyz만 → 펼친다
        (1, np.arange(30, 60), np.arange(30)),  # 학습 행: 범주 30개 → 번호로 둔다
    ]

    def task(fold_list):
        return types.SimpleNamespace(
            X=X, categorical_indices=[0], folds=lambda n_folds=10: fold_list
        )

    assert script.expands_one_hot(task(folds))
    assert not script.expands_one_hot(task(folds[1:]))


def test_modes_with_the_same_result_are_evaluated_once():
    script = _eval_script()
    modes = ["off", "power", "all"]
    # 분위수 멤버만 쓰면 power는 아무 열도 펼치지 않아 off와 같다
    assert script.same_result_modes(modes, ("quantile",), True) == [["all"], ["off", "power"]]
    assert script.same_result_modes(modes, ("quantile", "power"), True) == [
        ["all"],
        ["off"],
        ["power"],
    ]
    # 펼칠 열이 없으면 모두 같다 (headline이 맨 앞)
    assert script.same_result_modes(modes, ("quantile", "power"), False) == [
        ["all", "off", "power"]
    ]
    with pytest.raises(SystemExit):
        script.parse_args(["--member-kinds", "quantile,box-cox"])
