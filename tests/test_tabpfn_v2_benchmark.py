"""benchmark 점수 계산과 평가 스크립트의 task 선택 테스트."""

import argparse
import importlib.util
import types
from pathlib import Path

import numpy as np
import pytest
import torch

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
    path = Path(__file__).parents[1] / "scripts" / "eval_tabpfn_v2.py"
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


def _training_state(tmp_path, config):
    from minipfn.tabpfn_v2.model import TabPFNv2

    model = TabPFNv2(config.model_config())
    path = tmp_path / "step_0000100.pt"
    torch.save({"model": model.state_dict(), "step": 100, "borders": None}, path)
    return model, path


def test_training_checkpoint_needs_a_run_uri(tmp_path):
    from minipfn.tabpfn_v2.train import TabPFNv2TrainConfig

    _, path = _training_state(tmp_path, TabPFNv2TrainConfig.smoke())
    with pytest.raises(SystemExit, match="runs:/"):
        _eval_script().load_model(str(path), "classification", tmp_path)


def test_training_checkpoint_from_run_uses_run_params(tmp_path, monkeypatch):
    from minipfn.tabpfn_v2.train import TabPFNv2TrainConfig

    config = TabPFNv2TrainConfig.smoke()
    model, path = _training_state(tmp_path, config)
    script = _eval_script()
    params = {"task": "classification", "emsize": str(config.emsize)}
    params |= {"nlayers": str(config.nlayers), "nhead": str(config.nhead)}
    run = types.SimpleNamespace(data=types.SimpleNamespace(params=params))
    seen = {}

    def download(artifact_uri, dst_path):
        seen["uri"] = artifact_uri
        return str(path)

    monkeypatch.setattr(script.mlflow.artifacts, "download_artifacts", download)
    monkeypatch.setattr(
        script.mlflow, "MlflowClient", lambda: types.SimpleNamespace(get_run=lambda run_id: run)
    )
    uri = "runs:/abc123/checkpoints/step_0000100.pt"
    loaded, borders = script.load_model(uri, "classification", tmp_path)
    assert seen["uri"] == uri and borders is None and not loaded.training
    for name, tensor in model.state_dict().items():
        assert torch.equal(loaded.state_dict()[name], tensor)
    with pytest.raises(SystemExit, match="regression"):
        script.load_model(uri, "regression", tmp_path)
