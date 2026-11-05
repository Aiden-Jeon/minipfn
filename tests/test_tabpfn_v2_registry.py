"""모델 종류 목록(registry) 테스트: v2가 그대로 읽히고, 다른 모델도 학습·저장·Hub를 지난다."""

import json
import types
from dataclasses import asdict, dataclass, replace

import mlflow
import pytest
import torch
from torch import nn

from minipfn import hub
from minipfn import models as registry
from minipfn.models import save_checkpoint
from minipfn.models.tabpfn_v2 import TabPFNv2, TabPFNv2Config
from minipfn.recipes import tabpfn_v2 as recipe
from minipfn.training.config import TrainConfig
from minipfn.training.loop import train


@dataclass
class TinyConfig:
    emsize: int = 8
    nhead: int = 2
    n_out: int = 10

    def to_dict(self) -> dict:
        return asdict(self)


class TinyModel(nn.Module):
    """registry 규약만 지키는 작은 모델: 행마다 feature 평균 → Linear."""

    model_type = "tiny"
    config_class = TinyConfig

    def __init__(self, config: TinyConfig) -> None:
        super().__init__()
        self.config = config
        self.hidden = nn.Linear(2, config.emsize)
        self.head = nn.Linear(config.emsize, config.n_out)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def qk_clip_weights(self) -> list:
        return []

    def decoder_weight(self) -> None:
        return None

    def forward(self, x: torch.Tensor, y_train: torch.Tensor) -> torch.Tensor:
        n_train = y_train.shape[1]
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        row = x.mean(-1, keepdim=True)
        y_mean = y_train.mean(1, keepdim=True)[:, None].expand(-1, x.shape[1], 1)
        h = torch.relu(self.hidden(torch.cat([row, y_mean.to(row.dtype)], dim=-1)))
        return self.head(h[:, n_train:])


@pytest.fixture
def tiny(monkeypatch):
    monkeypatch.setitem(registry.MODEL_TYPES, "tiny", TinyModel)


def _same_weights(a, b):
    for name, tensor in a.state_dict().items():
        assert torch.equal(b.state_dict()[name], tensor), name


def test_build_model_from_dataclass_or_dict():
    config = TabPFNv2Config(emsize=16, nlayers=2, nhead=2)
    assert isinstance(registry.build_model("tabpfn_v2", config), TabPFNv2)
    assert registry.build_model("tabpfn_v2", config.to_dict()).config == config
    with pytest.raises(ValueError, match="model_type"):
        registry.build_model("tabpfn_v9", config)
    with pytest.raises(ValueError, match="model_type"):
        TrainConfig(model_type="tabpfn_v9")


@pytest.mark.parametrize("arch", ["v2", "v2.5"])
def test_v2_model_config_is_unchanged(arch):
    # registry를 거쳐도 v2·2.5 설정은 예전처럼 학습 설정의 필드에서 옮긴다
    config = recipe.with_architecture(recipe.paper("regression"), arch)
    assert config.model_type == "tabpfn_v2"
    assert config.model_config() == TabPFNv2Config(
        emsize=config.emsize,
        nlayers=config.nlayers,
        nhead=config.nhead,
        features_per_group=config.features_per_group,
        n_out=config.n_out,
        n_thinking_rows=config.n_thinking_rows,
        encoder=config.encoder,
    )
    assert isinstance(config.build_model(), TabPFNv2)


def test_checkpoint_records_the_model_type_and_old_files_read_as_v2(tmp_path):
    model = TrainConfig.smoke().build_model()
    save_checkpoint(tmp_path / "new.pt", model)
    payload = torch.load(tmp_path / "new.pt", weights_only=True)
    assert payload["model_type"] == "tabpfn_v2"
    del payload["model_type"]  # model_type을 남기기 전에 저장한 파일
    torch.save(payload, tmp_path / "old.pt")
    for name in ("new.pt", "old.pt"):
        loaded, _ = registry.load_checkpoint(tmp_path / name)
        assert isinstance(loaded, TabPFNv2)
        _same_weights(model, loaded)


def test_hub_config_records_the_model_type_and_old_repos_read_as_v2(tmp_path):
    model = TrainConfig.smoke().build_model()
    out = hub.save_pretrained(model, None, tmp_path / "clf")
    config = json.loads((out / hub.CONFIG_NAME).read_text())
    # 예전 from_pretrained는 config["model"]만 읽으므로 model_type은 그 밖에 둔다
    assert config["model_type"] == "tabpfn_v2" and "model_type" not in config["model"]
    del config["model_type"]  # 이미 Hub에 올린 v2 repo
    (out / hub.CONFIG_NAME).write_text(json.dumps(config))
    loaded, _ = hub.from_pretrained(out)
    assert isinstance(loaded, TabPFNv2)
    _same_weights(model, loaded)


def test_other_model_types_train_save_and_load(tmp_path, tiny):
    config = replace(TrainConfig.smoke("classification"), model_type="tiny", steps=4)
    assert config.model_config() == TinyConfig(emsize=config.emsize, nhead=config.nhead, n_out=10)
    result = train(config, tmp_path, mlflow_log=False)
    model, borders = registry.load_checkpoint(result["final_path"])
    # 출력층 가중치·QK-clip attention이 없어도 학습 지표 계산을 건너뛰고 끝까지 간다
    assert isinstance(model, TinyModel) and borders is None
    assert result["history"]["step"][-1] == 4

    loaded, _ = hub.from_pretrained(hub.save_pretrained(model, None, tmp_path / "pretrained"))
    assert isinstance(loaded, TinyModel)
    _same_weights(model, loaded)


def test_qk_clip_needs_attention_weights(tmp_path, tiny):
    config = replace(TrainConfig.smoke(), model_type="tiny", qk_clip=10.0)
    with pytest.raises(ValueError, match="QK-clip"):
        train(config, tmp_path, mlflow_log=False)


def _fake_run(monkeypatch, path, params):
    run = types.SimpleNamespace(data=types.SimpleNamespace(params=params))
    monkeypatch.setattr(
        mlflow.artifacts, "download_artifacts", lambda artifact_uri, dst_path: str(path)
    )
    monkeypatch.setattr(
        mlflow, "MlflowClient", lambda: types.SimpleNamespace(get_run=lambda run_id: run)
    )


def test_training_checkpoints_carry_the_model_settings(tmp_path, monkeypatch, tiny):
    # run param 없이 checkpoint만으로 모델을 만든다 (TinyConfig에만 있는 필드도 그대로)
    config = replace(TrainConfig.smoke(), model_type="tiny", steps=2)
    train(config, tmp_path, mlflow_log=False)
    state = torch.load(tmp_path / "latest.pt", weights_only=True)
    assert state["model_type"] == "tiny" and state["config"] == config.model_config().to_dict()
    _fake_run(monkeypatch, tmp_path / "latest.pt", params={})  # param을 읽으면 KeyError
    loaded, _ = hub.load_model("runs:/abc/checkpoints/latest.pt", "classification", tmp_path)
    assert isinstance(loaded, TinyModel)
    assert isinstance(registry.load_checkpoint(tmp_path / "latest.pt")[0], TinyModel)


def test_old_training_checkpoints_need_the_size_params(tmp_path, monkeypatch):
    # 모델 설정을 남기기 전의 v2 학습 checkpoint는 run param으로 만든다. 크기 param이 없으면 멈춘다
    config = TrainConfig.smoke()
    path = tmp_path / "step_0000100.pt"
    torch.save({"model": config.build_model().state_dict(), "borders": None}, path)
    uri = "runs:/abc/checkpoints/step_0000100.pt"
    params = {"task": "classification", "emsize": "32", "nlayers": "2", "nhead": "2"}
    _fake_run(monkeypatch, path, params)
    loaded, _ = hub.load_model(uri, "classification", tmp_path)
    assert loaded.config == config.model_config()
    _fake_run(monkeypatch, path, {k: v for k, v in params.items() if k != "nhead"})
    with pytest.raises(SystemExit, match="nhead"):
        hub.load_model(uri, "classification", tmp_path)
    _fake_run(monkeypatch, path, {"task": "classification", "model_type": "tiny"})
    with pytest.raises(SystemExit, match="모델 설정이 없다"):
        hub.load_model(uri, "classification", tmp_path)


def test_with_architecture_does_not_switch_other_model_types(tiny):
    config = replace(TrainConfig.smoke(), model_type="tiny")
    with pytest.raises(ValueError, match="tabpfn_v2 모델"):
        recipe.with_architecture(config, "v2")
    assert recipe.with_architecture(TrainConfig.smoke(), "v2.5").model_type == "tabpfn_v2"


def test_init_and_resume_need_the_same_task_and_model(tmp_path):
    config = replace(TrainConfig.smoke(), steps=2, collapse_threshold=None)
    train(config, tmp_path / "a", mlflow_log=False)
    latest = tmp_path / "a" / "latest.pt"
    # head 수만 다르면 가중치 모양이 같아 load_state_dict는 그대로 읽어 버린다
    other_heads = replace(config, nhead=4, steps=4)
    for kw in ({"init_from": latest}, {"resume_from": latest}):
        with pytest.raises(ValueError, match="nhead"):
            train(other_heads, tmp_path / "b", mlflow_log=False, **kw)
    with pytest.raises(ValueError, match="classification checkpoint"):
        regression = replace(TrainConfig.smoke("regression"), steps=2)
        train(regression, tmp_path / "c", init_from=latest, mlflow_log=False)
    # 같은 설정이면 그대로 읽는다 (save_checkpoint 파일도)
    final = tmp_path / "a" / "tabpfn_v2_final.pt"
    train(replace(config, steps=1), tmp_path / "d", init_from=final, mlflow_log=False)


def test_run_params_parse_bools():
    assert hub._parse_param(False, "True") is True
    assert hub._parse_param(True, "False") is False
    assert hub._parse_param(3, "24") == 24 and hub._parse_param("linear", "mlp") == "mlp"
    with pytest.raises(ValueError):
        hub._parse_param(False, "1")
