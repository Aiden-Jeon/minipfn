"""TabPFN v2 체크포인트 읽기·Hub 형식 테스트 (MLflow·Hub 서버 없이)."""

import json
import types

import pytest
import torch

from minipfn.tabpfn_v2 import hub
from minipfn.tabpfn_v2.model import TabPFNv2, save_checkpoint
from minipfn.tabpfn_v2.train import TabPFNv2TrainConfig


def _model(task="classification"):
    torch.manual_seed(0)
    return TabPFNv2(TabPFNv2TrainConfig.smoke(task).model_config()).eval()


def _same_weights(a, b):
    for name, tensor in a.state_dict().items():
        assert torch.equal(b.state_dict()[name], tensor), name


def _training_state(tmp_path, model):
    path = tmp_path / "step_0000100.pt"
    torch.save({"model": model.state_dict(), "step": 100, "borders": None}, path)
    return path


def test_save_and_load_pretrained_keeps_weights_and_borders(tmp_path):
    clf = _model()
    out = hub.save_pretrained(clf, None, tmp_path / "clf", source="runs:/abc/x.pt")
    assert json.loads((out / hub.CONFIG_NAME).read_text())["source"] == "runs:/abc/x.pt"
    loaded, borders = hub.from_pretrained(out)
    assert borders is None and not loaded.training
    _same_weights(clf, loaded)

    reg, edges = _model("regression"), torch.linspace(-3, 3, 5001)
    loaded, borders = hub.from_pretrained(hub.save_pretrained(reg, edges, tmp_path / "reg"))
    assert torch.equal(borders, edges)
    _same_weights(reg, loaded)


def test_load_model_reads_every_kind_of_checkpoint(tmp_path):
    clf = _model()
    save_checkpoint(tmp_path / "final.pt", clf)
    hub.save_pretrained(clf, None, tmp_path / "pretrained")
    for spec in (tmp_path / "final.pt", tmp_path / "pretrained"):
        loaded, borders = hub.load_model(str(spec), "classification", tmp_path)
        assert borders is None
        _same_weights(clf, loaded)
    with pytest.raises(SystemExit, match="regression 모델이 필요"):
        hub.load_model(str(tmp_path / "pretrained"), "regression", tmp_path)


def test_training_checkpoint_needs_a_run_uri(tmp_path):
    path = _training_state(tmp_path, _model())
    with pytest.raises(SystemExit, match="runs:/"):
        hub.load_model(str(path), "classification", tmp_path)


def test_training_checkpoint_from_run_uses_run_params(tmp_path, monkeypatch):
    config = TabPFNv2TrainConfig.smoke()
    model = _model()
    path = _training_state(tmp_path, model)
    params = {"task": "classification", "emsize": str(config.emsize)}
    params |= {"nlayers": str(config.nlayers), "nhead": str(config.nhead)}
    run = types.SimpleNamespace(data=types.SimpleNamespace(params=params))
    seen = {}

    def download(artifact_uri, dst_path):
        seen["uri"] = artifact_uri
        return str(path)

    monkeypatch.setattr(hub.mlflow.artifacts, "download_artifacts", download)
    monkeypatch.setattr(
        hub.mlflow, "MlflowClient", lambda: types.SimpleNamespace(get_run=lambda run_id: run)
    )
    uri = "runs:/abc123/checkpoints/step_0000100.pt"
    loaded, borders = hub.load_model(uri, "classification", tmp_path)
    assert seen["uri"] == uri and borders is None and not loaded.training
    _same_weights(model, loaded)
    with pytest.raises(SystemExit, match="regression 모델이 필요"):
        hub.load_model(uri, "regression", tmp_path)
