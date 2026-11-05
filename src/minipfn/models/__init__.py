"""모델 (모델 축). 학습·checkpoint·Hub 코드는 모델 클래스를 직접 부르지 않고 여기서 만든다.

TabPFN-3(arXiv 2605.13986)은 v2·2.5와 뼈대가 달라(셀 토큰의 2-way attention 대신 열 → 행 → 행
ICL 3단계) 다른 클래스로 둔다. 학습 루프·prior·추론 전처리·benchmark는 같이 쓰므로, 모델 클래스가
지킬 것만 여기서 정한다.

- 클래스 속성 `model_type`(아래 `MODEL_TYPES`의 이름)과 `config_class`(dataclass. `n_out`
  필드와 `to_dict()`가 있다)
- `__init__(config)`, `.config`
- `forward(x, y_train)`: x (B, R, F)는 학습 행 다음 테스트 행, y_train (B, n_train) →
  테스트 행 logit (B, R - n_train, n_out)
- `num_parameters()`
- `qk_clip_weights()`: QK-clip이 줄일 attention의 (종류, qkv weight) 목록. 모두 같은 E와
  `config.nhead`를 쓰고, 입력이 affine 없는 정규화 뒤여야 한다
  (`training.stability.qk_logit_bounds`). 종류가 "sample_attention"인 것이 학습 지표
  `qk_bound_sample_max`에 들어간다. 없으면 빈 목록
- `decoder_weight()`: 출력층 가중치 (학습 지표 `decoder_norm`). 없으면 None
"""

from __future__ import annotations

import importlib
from dataclasses import MISSING, fields, is_dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

# 이름 → "모듈:클래스" (쓸 때 import한다) 또는 클래스
MODEL_TYPES: dict[str, str | type[nn.Module]] = {
    "tabpfn_v2": "minipfn.models.tabpfn_v2:TabPFNv2",
}
# model_type을 남기지 않은 이전 checkpoint·학습 run·Hub repo는 모두 v2(2.5 포함)다
DEFAULT_MODEL_TYPE = "tabpfn_v2"


def model_class(model_type: str) -> type[nn.Module]:
    if model_type not in MODEL_TYPES:
        raise ValueError(f"model_type은 {tuple(MODEL_TYPES)} 가운데 하나여야 한다: {model_type}")
    target = MODEL_TYPES[model_type]
    if isinstance(target, str):
        module, name = target.split(":")
        target = getattr(importlib.import_module(module), name)
    return target


def config_fields(model_type: str) -> tuple[str, ...]:
    """그 모델 설정의 필드 이름."""
    return tuple(f.name for f in fields(model_class(model_type).config_class))


def config_dict(config: Any) -> dict:
    """모델 설정을 checkpoint·Hub config.json에 적을 dict로. 설정 클래스의 `ADDED_FIELDS`(나중에
    더한 필드)는 기본값이면 뺀다. 그 필드를 모르는 예전 코드도 기본 구조의 모델은 읽고, 읽을 때
    빠진 필드는 기본값으로 채워진다(`build_model`)."""
    out = config.to_dict()
    added = set(getattr(config, "ADDED_FIELDS", ()))
    for f in fields(config):
        if f.name in added and f.default is not MISSING and out[f.name] == f.default:
            del out[f.name]
    return out


def build_model(model_type: str, config: Any) -> nn.Module:
    """`config`는 그 모델의 설정 dataclass 또는 같은 내용의 dict다."""
    cls = model_class(model_type)
    if not is_dataclass(config):
        config = cls.config_class(**config)
    return cls(config)


def load_checkpoint(
    path: str | Path, map_location: str | torch.device = "cpu"
) -> tuple[nn.Module, torch.Tensor | None]:
    """`save_checkpoint`로 저장한 파일을 읽는다. 모델 종류는 파일에 남긴 `model_type`을 따른다
    (이전 파일에는 없어서 v2로 읽는다)."""
    checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    model_type = checkpoint.get("model_type", DEFAULT_MODEL_TYPE)
    model = build_model(model_type, checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval(), checkpoint["borders"]


def save_checkpoint(
    path: str | Path, model: nn.Module, borders: torch.Tensor | None = None
) -> None:
    """가중치, 설정, (회귀면) bar distribution 경계를 저장한다."""
    payload = {
        "model_type": model.model_type,
        "config": config_dict(model.config),
        "state_dict": model.state_dict(),
        "borders": borders,
    }
    torch.save(payload, path)
