"""TabPFN v2 체크포인트 읽기·Hub 형식 테스트 (MLflow·Hub 서버 없이)."""

import json
import types

import pytest
import torch

from minipfn import hub
from minipfn.models import save_checkpoint
from minipfn.models.tabpfn_v2 import TabPFNv2
from minipfn.training.config import TrainConfig


@pytest.fixture
def chdir_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # 상대 경로가 저장소 안의 진짜 폴더와 겹치지 않도록


def _model(task="classification"):
    torch.manual_seed(0)
    return TabPFNv2(TrainConfig.smoke(task).model_config()).eval()


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
    config = TrainConfig.smoke()
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


def test_hub_needs_the_hf_prefix_and_missing_local_paths_fail(tmp_path, monkeypatch, chdir_tmp):
    seen = []
    monkeypatch.setattr(
        hub,
        "from_pretrained",
        lambda spec, revision=None: seen.append((spec, revision)) or (_model(), None),
    )
    hub.load_model("hf://owner/name", "classification", tmp_path)
    assert seen == [("owner/name", None)]
    hub.load_model(
        "hf://owner/name@step-100k", "classification", tmp_path
    )  # 중간 checkpoint branch
    assert seen[-1] == ("owner/name", "step-100k")
    with pytest.raises(SystemExit, match="hf://"):
        hub.load_model("models/clf", "classification", tmp_path)  # 오타 난 로컬 경로
    assert len(seen) == 2


def test_borders_keep_their_dtype(tmp_path):
    edges = torch.linspace(-3, 3, 5001, dtype=torch.float64)
    _, borders = hub.from_pretrained(hub.save_pretrained(_model("regression"), edges, tmp_path))
    assert borders.dtype == torch.float64 and torch.equal(borders, edges)


def test_run_uri_to_a_pretrained_folder(tmp_path, monkeypatch):
    clf = _model()
    folder = hub.save_pretrained(clf, None, tmp_path / "pretrained")
    monkeypatch.setattr(
        hub.mlflow.artifacts, "download_artifacts", lambda artifact_uri, dst_path: str(folder)
    )
    loaded, _ = hub.load_model("runs:/abc/model", "classification", tmp_path)
    _same_weights(clf, loaded)
