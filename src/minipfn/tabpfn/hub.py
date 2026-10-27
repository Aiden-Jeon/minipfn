"""학습한 TabPFN을 Hugging Face Hub에 올리고 받는다.

Hub에는 pickle 없이 읽을 수 있도록 가중치는 `model.safetensors`, 설정은 `config.json`으로 둔다.

    from minipfn.tabpfn.hub import from_pretrained
    model, config = from_pretrained()  # aiden-jeon/minipfn-tabpfn-v1
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file, save_file

from minipfn.tabpfn.model import TabPFNTransformer
from minipfn.tabpfn.train import TabPFNConfig, tabpfn_from_state

HUB_REPO_ID = "aiden-jeon/minipfn-tabpfn-v1"
WEIGHTS_NAME = "model.safetensors"
CONFIG_NAME = "config.json"


def save_pretrained(model: torch.nn.Module, config: TabPFNConfig, out_dir: str | Path) -> Path:
    """`out_dir`에 `model.safetensors`와 `config.json`을 쓴다."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    state_dict = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    save_file(state_dict, out_dir / WEIGHTS_NAME)
    (out_dir / CONFIG_NAME).write_text(json.dumps(asdict(config), indent=2) + "\n")
    return out_dir


def _is_local_path(repo_id_or_path: str | Path) -> bool:
    """Hub repo id(`owner/name`)가 아니라 로컬 경로를 뜻하는지."""
    if isinstance(repo_id_or_path, Path) or Path(repo_id_or_path).exists():
        return True
    s = str(repo_id_or_path)
    return s.count("/") != 1 or s.startswith((".", "/", "~")) or s.endswith((".pt", ".json"))


def from_pretrained(
    repo_id_or_path: str | Path = HUB_REPO_ID,
    revision: str | None = None,
    map_location: str | torch.device = "cpu",
) -> tuple[TabPFNTransformer, TabPFNConfig]:
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
