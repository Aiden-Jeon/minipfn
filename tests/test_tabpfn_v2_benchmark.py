"""benchmark 점수 계산과 평가 스크립트의 task 선택 테스트."""

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from minipfn.tabpfn_v2.benchmark import score_fold


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


def test_one_hot_modes_are_parsed_and_best_mode_is_picked_per_suite():
    import pandas as pd

    script = _eval_script()
    assert script.parse_args([]).one_hot == ["off", "power", "all"]
    assert script.parse_args(["--one-hot", "all,off,all"]).one_hot == ["all", "off"]
    with pytest.raises(SystemExit):
        script.parse_args(["--one-hot", "half"])
    by_mode = pd.DataFrame(
        {
            "suite": ["classification"] * 2 + ["regression"] * 2,
            "one_hot": ["off", "all", "off", "power"],
            "score": [0.90, 0.91, -12.0, -10.0],
        }
    )
    assert script.best_one_hot(by_mode) == {"classification": "all", "regression": "power"}
