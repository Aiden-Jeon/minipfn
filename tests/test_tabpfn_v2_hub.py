"""TabPFN v2 체크포인트 읽기·Hub 형식 테스트 (MLflow·Hub 서버 없이)."""

import json
import types

import mlflow
import pytest
import torch

from minipfn import hub
from minipfn.models import config_dict, save_checkpoint
from minipfn.models.cell_transformer import CellTransformer
from minipfn.training.config import TrainConfig


@pytest.fixture
def chdir_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # 상대 경로가 저장소 안의 진짜 폴더와 겹치지 않도록


def _model(task="classification"):
    torch.manual_seed(0)
    return CellTransformer(TrainConfig.smoke(task).model_config()).eval()


def _same_weights(a, b):
    for name, tensor in a.state_dict().items():
        assert torch.equal(b.state_dict()[name], tensor), name


def _training_state(tmp_path, model):
    path = tmp_path / "step_0000100.pt"
    torch.save({"model": model.state_dict(), "step": 100, "borders": None}, path)
    return path


def test_save_and_load_pretrained_keeps_weights_and_borders(tmp_path):
    clf = _model()
    out = hub.save_pretrained(clf, None, tmp_path / "clf", inference={"n_estimators": 2})
    assert json.loads((out / hub.CONFIG_NAME).read_text())["inference"] == {"n_estimators": 2}
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

    monkeypatch.setattr(mlflow.artifacts, "download_artifacts", download)
    monkeypatch.setattr(
        mlflow, "MlflowClient", lambda: types.SimpleNamespace(get_run=lambda run_id: run)
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
        mlflow.artifacts, "download_artifacts", lambda artifact_uri, dst_path: str(folder)
    )
    loaded, _ = hub.load_model("runs:/abc/model", "classification", tmp_path)
    _same_weights(clf, loaded)


# v2 첫 공개(Hub에 올린 v2 가중치) 때의 모델 설정 필드. 예전 minipfn의 from_pretrained는
# `TabPFNv2Config(**config["model"])`로 읽으므로 이 밖의 키가 있으면 깨진다
V2_RELEASE_FIELDS = {
    "emsize",
    "nlayers",
    "nhead",
    "nhid_factor",
    "features_per_group",
    "n_out",
    "column_embedding_seed",
}


def _paper_width_model():
    # 학습 때 본 feature 수가 v2 논문 그대로(160)인 모델. smoke 설정은 8이라 config.json에 적힌다
    from dataclasses import replace

    torch.manual_seed(0)
    config = replace(TrainConfig.smoke().model_config(), max_features=160)
    return CellTransformer(config).eval()


def test_v2_config_json_has_only_fields_older_releases_know(tmp_path):
    out = hub.save_pretrained(_paper_width_model(), None, tmp_path / "v2")
    model = json.loads((out / hub.CONFIG_NAME).read_text())["model"]
    assert set(model) == V2_RELEASE_FIELDS
    # 학습 checkpoint도 같다 (예전 load_checkpoint도 같은 방식으로 읽는다)
    save_checkpoint(tmp_path / "v2.pt", _paper_width_model())
    assert set(torch.load(tmp_path / "v2.pt", weights_only=True)["config"]) == V2_RELEASE_FIELDS


def test_v2_5_config_json_keeps_the_added_fields(tmp_path):
    from dataclasses import replace

    from minipfn.recipes import tabpfn_v2 as recipe

    config = recipe.with_architecture(TrainConfig.smoke("regression"), "v2.5", keep_size=True)
    torch.manual_seed(0)
    model = config.build_model().eval()
    out = hub.save_pretrained(model, torch.linspace(-3, 3, 5001), tmp_path / "v25")
    saved = json.loads((out / hub.CONFIG_NAME).read_text())["model"]
    assert saved["n_thinking_rows"] == 4 and saved["encoder"] == "mlp"
    loaded, _ = hub.from_pretrained(out)
    assert loaded.config == model.config
    _same_weights(model, loaded)
    # 하나만 기본값이 아니면 그것만 적는다
    only_rows = replace(model.config, encoder="linear", max_features=160)
    assert set(config_dict(only_rows)) == V2_RELEASE_FIELDS | {"n_thinking_rows"}


def test_inference_classes_load_weights_without_from_pretrained(tmp_path, monkeypatch):
    # 사용자는 공개 가중치를 쓰므로 추론 클래스가 직접 받는다: 인자 없으면 그 논문의 공개 repo,
    # 문자열이면 Hub repo id(`owner/name@branch`)나 로컬 폴더
    from minipfn import TabPFNv1Classifier, TabPFNv2Classifier, TabPFNv2Regressor
    from minipfn.hub.tabpfn_v1 import save_pretrained as save_v1
    from minipfn.recipes import tabpfn_v1, tabpfn_v2

    cls_dir, reg_dir, v1_dir = tmp_path / "cls", tmp_path / "reg", tmp_path / "v1"
    cls_model, reg_model = _model(), _model("regression")
    borders = torch.linspace(-3, 3, reg_model.config.n_out + 1)
    hub.save_pretrained(cls_model, None, cls_dir)
    hub.save_pretrained(reg_model, borders, reg_dir)
    v1_config = tabpfn_v1.TabPFNConfig.smoke()
    save_v1(tabpfn_v1.build_tabpfn_model(v1_config), v1_config, v1_dir)

    # Hub 대신 로컬 폴더를 돌려준다. 공개 repo 이름으로 불렸는지 기록한다
    asked = []
    real = hub.from_pretrained
    folders = {
        tabpfn_v1.HUB_REPO: v1_dir,
        tabpfn_v2.HUB_REPOS["classification"]: cls_dir,
        tabpfn_v2.HUB_REPOS["regression"]: reg_dir,
    }

    def fake(spec, revision=None):
        asked.append(str(spec))
        return real(folders.get(str(spec).partition("@")[0], spec), revision)

    monkeypatch.setattr(hub, "from_pretrained", fake)

    _same_weights(cls_model, TabPFNv2Classifier().model)
    reg = TabPFNv2Regressor()
    _same_weights(reg_model, reg.model)
    assert torch.equal(reg.borders.cpu(), borders)
    assert isinstance(TabPFNv1Classifier().model, type(tabpfn_v1.build_tabpfn_model(v1_config)))
    assert asked == [
        tabpfn_v2.HUB_REPOS["classification"],
        tabpfn_v2.HUB_REPOS["regression"],
        tabpfn_v1.HUB_REPO,
    ]
    # branch·로컬 폴더·모델을 직접 줘도 된다
    TabPFNv2Classifier(tabpfn_v2.HUB_REPOS["classification"] + "@step-100k")
    assert asked[-1].endswith("@step-100k")
    _same_weights(cls_model, TabPFNv2Classifier(str(cls_dir)).model)
    assert TabPFNv2Regressor(reg_model, borders).model is reg_model

    # 잘못 준 것은 알려 준다
    with pytest.raises(ValueError, match="regression 모델"):
        TabPFNv2Classifier(str(reg_dir))
    with pytest.raises(ValueError, match="classification 모델"):
        TabPFNv2Regressor(str(cls_dir))
    with pytest.raises(ValueError, match="v2·2.5 모델이 아니다"):
        TabPFNv2Classifier(str(v1_dir))
    with pytest.raises(ValueError, match="v1 모델이 아니다"):
        TabPFNv1Classifier(str(cls_dir))
    with pytest.raises(ValueError, match="borders"):
        TabPFNv2Regressor(reg_model)
    with pytest.raises(ValueError, match="borders"):
        TabPFNv2Regressor(str(reg_dir), borders)


def test_v2_5_classes_default_to_the_v2_5_repos(tmp_path, monkeypatch):
    # 2.5 클래스는 v2 클래스와 같고 인자 없이 만들 때 받는 공개 repo만 다르다
    from minipfn import TabPFNv2_5Classifier, TabPFNv2_5Regressor, TabPFNv2Classifier
    from minipfn.recipes import tabpfn_v2

    cls_dir, reg_dir = tmp_path / "cls", tmp_path / "reg"
    cls_model, reg_model = _model(), _model("regression")
    borders = torch.linspace(-3, 3, reg_model.config.n_out + 1)
    hub.save_pretrained(cls_model, None, cls_dir)
    hub.save_pretrained(reg_model, borders, reg_dir)
    folders = {
        tabpfn_v2.HUB_REPOS_V2_5["classification"]: cls_dir,
        tabpfn_v2.HUB_REPOS_V2_5["regression"]: reg_dir,
    }
    asked = []
    real = hub.from_pretrained

    def fake(spec, revision=None):
        asked.append(str(spec))
        return real(folders.get(str(spec).partition("@")[0], spec), revision)

    monkeypatch.setattr(hub, "from_pretrained", fake)
    _same_weights(cls_model, TabPFNv2_5Classifier().model)
    reg = TabPFNv2_5Regressor()
    _same_weights(reg_model, reg.model)
    assert torch.equal(reg.borders.cpu(), borders)
    assert asked == [
        tabpfn_v2.HUB_REPOS_V2_5["classification"],
        tabpfn_v2.HUB_REPOS_V2_5["regression"],
    ]
    assert issubclass(TabPFNv2_5Classifier, TabPFNv2Classifier)
    assert TabPFNv2_5Regressor(reg_model, borders).model is reg_model


def test_from_pretrained_reads_the_branch_after_at(monkeypatch, tmp_path):
    model = _model()
    hub.save_pretrained(model, None, tmp_path)
    calls = []

    def download(repo_id, filename, revision=None):
        calls.append((repo_id, revision))
        return str(tmp_path / filename)

    monkeypatch.setattr(hub, "hf_hub_download", download)
    _same_weights(model, hub.from_pretrained("owner/name@step-100k")[0])
    hub.from_pretrained("owner/name", revision="step-200k")
    assert set(calls) == {("owner/name", "step-100k"), ("owner/name", "step-200k")}
    with pytest.raises(ValueError, match="revision"):
        hub.from_pretrained("owner/name@step-100k", revision="step-200k")
