"""학습한 모델(`minipfn.models` registry의 모델: TabPFN v2·2.5의 cell_transformer 등)을 Hugging
Face Hub에 올리고 받는다. TabPFN v1은 형식이 달라 `minipfn.hub.tabpfn_v1`에 있다.

Hub에는 pickle 없이 읽을 수 있도록 가중치는 `model.safetensors`, 설정은 `config.json`으로 둔다.
회귀 모델은 bar distribution 구간 경계를 `model.safetensors` 안에 `borders`라는 이름으로 함께 둔다.

    from minipfn.hub import from_pretrained
    model, borders = from_pretrained("aiden-jeon/minipfn-tabpfn-v2-classifier")  # borders: None

`load_model`은 체크포인트를 어디에 두었든 같은 방식으로 읽는다(평가·업로드 스크립트가 쓴다).
Hub에서 받을 때는 `hf://owner/name`처럼 적는다(오타 난 로컬 경로를 Hub로 보내지 않도록).
중간 checkpoint는 같은 repo의 branch(`step-100k` 등)에 있고 `hf://owner/name@step-100k`로 받는다.

mlflow와 학습 설정은 쓰는 함수 안에서 불러온다. Hub에서 받아 추론만 할 때(v1 포함) 학습 스택을
불러오지 않도록.
"""

from __future__ import annotations

import json
from dataclasses import fields, replace
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file, save_file
from torch import nn

from minipfn.models import (
    DEFAULT_MODEL_TYPE,
    build_model,
    canonical_model_type,
    config_dict,
    config_fields,
    info_fields,
    model_from_state,
    state_config,
)

WEIGHTS_NAME = "model.safetensors"
CONFIG_NAME = "config.json"
BORDERS_KEY = "borders"
HUB_PREFIX = "hf://"
# 모델 설정을 checkpoint에 남기기 전의 학습 run에서 꼭 있어야 하는 param
_REQUIRED_RUN_PARAMS = ("emsize", "nlayers", "nhead")


def _model_params(model_type: str) -> tuple[str, ...]:
    # 학습 run의 param으로 남는 모델 구조 설정 (그 모델 설정과 TrainConfig에 모두 있는 것). 가중치
    # 모양과 상관없는 필드(`info_fields`)는 구조가 아니라 뺀다
    from minipfn.training.config import TrainConfig

    train_fields = {f.name for f in fields(TrainConfig)} - set(info_fields(model_type))
    return tuple(name for name in config_fields(model_type) if name in train_fields)


def _parse_param(default, text: str):
    # MLflow param은 문자열이다. bool("False")는 True라서 bool만 따로 읽는다
    if isinstance(default, bool):
        if text not in ("True", "False"):
            raise ValueError(f"bool param이 아니다: {text}")
        return text == "True"
    return type(default)(text)


def _is_local_path(repo_id_or_path: str | Path) -> bool:
    """Hub repo id(`owner/name`)가 아니라 로컬 경로를 뜻하는지."""
    if isinstance(repo_id_or_path, Path) or Path(repo_id_or_path).exists():
        return True
    s = str(repo_id_or_path)
    return s.count("/") != 1 or s.startswith((".", "/", "~")) or s.endswith((".pt", ".json"))


def save_pretrained(
    model: nn.Module, borders: torch.Tensor | None, out_dir: str | Path, **extra
) -> Path:
    """`out_dir`에 `model.safetensors`와 `config.json`을 쓴다. `extra`는 config.json에 더 적는다."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tensors = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    if borders is not None:
        tensors[BORDERS_KEY] = borders.detach().cpu().contiguous()
    save_file(tensors, out_dir / WEIGHTS_NAME)
    task = "classification" if borders is None else "regression"
    config = {
        "task": task,
        "model_type": model.model_type,
        "model": config_dict(model.config),
        **extra,
    }
    (out_dir / CONFIG_NAME).write_text(json.dumps(config, indent=2) + "\n")
    return out_dir


def from_pretrained(
    repo_id_or_path: str | Path, revision: str | None = None
) -> tuple[nn.Module, torch.Tensor | None]:
    """Hub repo id 또는 `save_pretrained`로 만든 로컬 폴더에서 (모델, 회귀 구간 경계)를 읽는다."""
    if _is_local_path(repo_id_or_path):
        local = Path(repo_id_or_path).expanduser()
        weights_path, config_path = local / WEIGHTS_NAME, local / CONFIG_NAME
    else:
        repo_id = str(repo_id_or_path)
        weights_path = hf_hub_download(repo_id, WEIGHTS_NAME, revision=revision)
        config_path = hf_hub_download(repo_id, CONFIG_NAME, revision=revision)
    config = json.loads(Path(config_path).read_text())
    tensors = load_file(weights_path)
    borders = tensors.pop(BORDERS_KEY, None)
    # model_type이 없는 repo는 v2로 올린 것이다
    model = build_model(config.get("model_type", DEFAULT_MODEL_TYPE), config["model"])
    model.load_state_dict(tensors)
    return model.eval(), borders


def load_model(spec: str, task: str, dst: str | Path) -> tuple[nn.Module, torch.Tensor | None]:
    """체크포인트를 CPU로 읽는다. `spec`은 다음 가운데 하나다.

    - `runs:/<run_id>/<경로>`: 학습 run이 MLflow에 남긴 파일 또는 폴더 (`dst`로 받는다)
    - `hf://owner/name`: Hub repo (`save_pretrained`로 올린 것). `@<revision>`을 붙이면 그
      branch·tag·commit에서 받는다(예: `hf://owner/name@step-100k`)
    - `save_pretrained`로 만든 로컬 폴더, `minipfn.models.save_checkpoint`로 저장한 파일

    학습 중에 남긴 checkpoint(`checkpoints/step_XXXXXXX.pt`, `latest.pt`)는 그 안에 남긴 모델
    종류와 설정으로 모델을 만든다. 그것을 남기기 전의 v2·2.5 학습 checkpoint만 그 run에 기록된
    param(task·emsize·nlayers·nhead와 2.5 구조 설정)으로 만든다.
    """
    import mlflow  # MLflow를 쓰는 경로에서만 불러온다 (Hub에서 받기만 할 때는 필요 없다)

    from minipfn.training.config import TrainConfig

    if spec.startswith(HUB_PREFIX):
        repo_id, _, revision = spec.removeprefix(HUB_PREFIX).partition("@")
        return _check_task(from_pretrained(repo_id, revision=revision or None), task, spec)
    run_id, path = None, Path(spec)
    if spec.startswith("runs:/"):
        run_id = spec.removeprefix("runs:/").split("/", 1)[0]
        path = Path(mlflow.artifacts.download_artifacts(artifact_uri=spec, dst_path=str(dst)))
    elif not path.exists():
        raise SystemExit(f"{spec}가 없다 (Hub repo면 {HUB_PREFIX}owner/name으로 준다).")
    if path.is_dir():
        return _check_task(from_pretrained(path), task, spec)
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state_config(state) is not None:
        return _check_task(model_from_state(state), task, spec)
    if run_id is None:
        raise SystemExit(
            f"{spec}는 학습 중 checkpoint라 모델 설정이 없다. runs:/<run_id>/...로 준다."
        )
    params = mlflow.MlflowClient().get_run(run_id).data.params
    if params["task"] != task:
        raise SystemExit(f"{spec}는 {params['task']} 모델이다 ({task} 모델이 필요하다).")
    # 모델 설정을 checkpoint에 남기기 전의 run이라 v2·2.5다. 크기 param은 처음부터 기록했으니
    # 없으면 기록이 덜 된 run이다. 2.5 구조 param은 2.5 이전 run에 없으므로 v2 기본값을 쓴다
    if canonical_model_type(params.get("model_type", DEFAULT_MODEL_TYPE)) != DEFAULT_MODEL_TYPE:
        raise SystemExit(f"{spec}에 모델 설정이 없다 (model_type {params['model_type']}).")
    missing = [name for name in _REQUIRED_RUN_PARAMS if name not in params]
    if missing:
        raise SystemExit(f"{spec}의 run에 모델 param {missing}이 없다.")
    config = replace(TrainConfig(task=params["task"]), **run_architecture(params))
    # 학습 때 본 feature 수 등 구조가 아닌 모델 설정도 run param에 있으면 그 값을 쓴다
    info = {
        name: _parse_param(getattr(config, name), params[name])
        for name in info_fields(config.model_type)
        if name in params
    }
    model = replace(config, **info).build_model()
    model.load_state_dict(state["model"])
    return model.eval(), state["borders"]


def run_architecture(params: dict[str, str]) -> dict:
    """학습 run의 MLflow param에서 모델 구조: `model_type`과 그 모델의 구조 설정(TrainConfig 필드).

    run에 없는 값(model_type 이전 run은 v2, TabPFN-2.5 이전 run의 구조 설정)은 설정 기본값(v2)을
    쓴다. 예전 model_type 이름(`tabpfn_v2`)은 지금 이름으로 바꾼다. 반환값을
    `replace(config, **...)`로 학습 설정에 넣는다.
    """
    from minipfn.training.config import TrainConfig

    model_type = canonical_model_type(params.get("model_type", DEFAULT_MODEL_TYPE))
    defaults = TrainConfig(task=params["task"], model_type=model_type)
    names = _model_params(model_type)
    return {
        "model_type": model_type,
        **{
            name: _parse_param(
                getattr(defaults, name), params.get(name, str(getattr(defaults, name)))
            )
            for name in names
        },
    }


def _check_task(loaded: tuple[nn.Module, torch.Tensor | None], task: str, spec: str):
    got = "classification" if loaded[1] is None else "regression"
    if got != task:
        raise SystemExit(f"{spec}는 {got} 모델이다 ({task} 모델이 필요하다).")
    return loaded
