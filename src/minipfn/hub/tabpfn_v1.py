"""학습한 TabPFN을 Hugging Face Hub에 올리고 받는다.

Hub에는 pickle 없이 읽을 수 있도록 가중치는 `model.safetensors`, 설정은 `config.json`으로 둔다.
가중치를 만든 체크포인트는 `source.json`에 따로 적는다. `config.json`에 넣지 않는 까닭은 예전에
설치한 minipfn의 `from_pretrained`가 `config.json`을 `TabPFNConfig(**config)`로 읽어서, 모르는
키가 있으면 깨지기 때문이다.

    from minipfn.hub.tabpfn_v1 import from_pretrained
    model, config = from_pretrained()  # aiden-jeon/minipfn-tabpfn-v1
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file, save_file

from minipfn.hub import _is_local_path
from minipfn.models.row_transformer import RowTransformer
from minipfn.recipes.tabpfn_v1 import HUB_REPO, TabPFNConfig, tabpfn_from_state

HUB_REPO_ID = HUB_REPO
WEIGHTS_NAME = "model.safetensors"
CONFIG_NAME = "config.json"
SOURCE_NAME = "source.json"


def save_pretrained(
    model: torch.nn.Module, config: TabPFNConfig, out_dir: str | Path, source: str | None = None
) -> Path:
    """`out_dir`에 `model.safetensors`와 `config.json`을, `source`를 주면 `source.json`도 쓴다."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    state_dict = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    save_file(state_dict, out_dir / WEIGHTS_NAME)
    (out_dir / CONFIG_NAME).write_text(json.dumps(asdict(config), indent=2) + "\n")
    if source is not None:
        (out_dir / SOURCE_NAME).write_text(json.dumps({"source": source}, indent=2) + "\n")
    return out_dir


def from_pretrained(
    repo_id_or_path: str | Path = HUB_REPO_ID,
    revision: str | None = None,
    map_location: str | torch.device = "cpu",
) -> tuple[RowTransformer, TabPFNConfig]:
    """Hub repo id 또는 `save_pretrained`로 만든 로컬 폴더에서 모델을 읽는다.

    같은 이름의 로컬 폴더가 있으면 Hub보다 먼저 읽는다. `revision`(branch, tag, commit)을 주면
    Hub에서 그 버전을 받는다."""
    if _is_local_path(repo_id_or_path):
        local = Path(repo_id_or_path).expanduser()
        if not local.is_dir():
            raise FileNotFoundError(
                f"{repo_id_or_path}: save_pretrained로 만든 폴더가 아닙니다"
                f" ({WEIGHTS_NAME}와 {CONFIG_NAME}이 든 폴더 또는 Hub repo id 'owner/name')"
            )
        weights_path, config_path = local / WEIGHTS_NAME, local / CONFIG_NAME
    else:
        repo_id = str(repo_id_or_path)
        weights_path = hf_hub_download(repo_id, WEIGHTS_NAME, revision=revision)
        config_path = hf_hub_download(repo_id, CONFIG_NAME, revision=revision)
    config = json.loads(Path(config_path).read_text())
    return tabpfn_from_state(config, load_file(weights_path), map_location)
