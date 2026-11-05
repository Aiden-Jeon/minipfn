"""모델 (모델 축). 파일은 토큰 하나가 무엇인지로 이름을 붙이고, 논문별 크기·구조는
`minipfn.recipes`에 둔다.

- `point_transformer`: 데이터 점 (x, y) 하나 = 토큰 (PFN recipe)
- `row_transformer`: 표의 행 하나 = 토큰 (TabPFN v1 recipe)
- `cell_transformer`: 셀 묶음 하나 = 토큰, 행·열 두 방향 attention (TabPFN v2·2.5 recipe)
- `bar_distribution`: 회귀 출력을 구간 분류로 다루는 예측분포 (모델에 붙는 부품)

공용 학습 루프·checkpoint·Hub 코드는 모델 클래스를 직접 부르지 않고 아래 registry로 만든다.
TabPFN-3(arXiv 2605.13986)은 v2·2.5와 뼈대가 달라(셀 토큰의 2-way attention 대신 열 → 행 → 행
ICL 3단계) 다른 클래스로 둔다. 학습 루프·prior·추론 전처리·benchmark는 같이 쓰므로, registry
모델 클래스가 지킬 것만 여기서 정한다.

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
- (선택) 설정 클래스의 `ADDED_FIELDS`: 첫 공개 뒤에 더한 필드. 기본값이면 저장하지 않는다
  (`config_dict`)
- (선택) 설정 클래스의 `INFO_FIELDS`: 가중치 모양과 상관없는 필드(학습 때 본 feature 수 등).
  이어 학습할 때 비교하지 않고 학습 run의 모델 구조에도 넣지 않는다 (`info_fields`)
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
    "cell_transformer": "minipfn.models.cell_transformer:CellTransformer",
}
# 예전 이름 → 지금 이름. 예전 checkpoint·학습 run param·Hub config.json에 남아 있어 읽을 때 받는다
MODEL_TYPE_ALIASES: dict[str, str] = {
    "tabpfn_v2": "cell_transformer",
}
# model_type을 남기지 않은 이전 checkpoint·학습 run·Hub repo는 모두 cell_transformer(v2·2.5)다
DEFAULT_MODEL_TYPE = "cell_transformer"


def canonical_model_type(model_type: str) -> str:
    """예전 이름이면 지금 이름으로. 없는 이름은 그대로 둔다(`model_class`가 ValueError를 낸다)."""
    return MODEL_TYPE_ALIASES.get(model_type, model_type)


def model_class(model_type: str) -> type[nn.Module]:
    name = canonical_model_type(model_type)
    if name not in MODEL_TYPES:
        raise ValueError(f"model_type은 {tuple(MODEL_TYPES)} 가운데 하나여야 한다: {model_type}")
    target = MODEL_TYPES[name]
    if isinstance(target, str):
        module, name = target.split(":")
        target = getattr(importlib.import_module(module), name)
    return target


def config_fields(model_type: str) -> tuple[str, ...]:
    """그 모델 설정의 필드 이름."""
    return tuple(f.name for f in fields(model_class(model_type).config_class))


def info_fields(model_type: str) -> tuple[str, ...]:
    """그 모델 설정에서 가중치 모양과 상관없는 필드 (설정 클래스의 `INFO_FIELDS`). checkpoint를
    이어 학습할 때 비교하지 않고 학습 run의 모델 구조에도 넣지 않는다."""
    return tuple(getattr(model_class(model_type).config_class, "INFO_FIELDS", ()))


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


def config_from_dict(model_type: str, values: dict) -> Any:
    """저장한 dict에서 그 모델의 설정. 빠진 필드는 기본값이다. 이 코드가 모르는 필드가 있으면(더
    새 코드나 다른 브랜치에서 저장한 것) 그 이름을 담은 ValueError를 낸다."""
    cls = model_class(model_type).config_class
    unknown = sorted(set(values) - {f.name for f in fields(cls)})
    if unknown:
        raise ValueError(
            f"{model_type} 모델 설정에 이 코드가 모르는 필드 {unknown}가 있다"
            " (더 새 코드나 다른 브랜치에서 저장한 checkpoint일 수 있다)"
        )
    return cls(**values)


def build_model(model_type: str, config: Any) -> nn.Module:
    """`config`는 그 모델의 설정 dataclass 또는 같은 내용의 dict다."""
    if not is_dataclass(config):
        config = config_from_dict(model_type, config)
    return model_class(model_type)(config)


def state_config(state: dict) -> dict | None:
    """checkpoint에 남긴 모델 설정 (없으면 None). `save_checkpoint` 파일은 `config`, 학습 중
    checkpoint는 `model_config`에 둔다. 예전 코드는 `config`가 있으면 `save_checkpoint` 파일로 보고
    `state_dict`를 찾으므로, 학습 중 checkpoint는 다른 키를 써서 예전 코드도 run param으로 읽게
    한다."""
    return state.get("model_config", state.get("config"))


def state_weights(state: dict) -> dict:
    """checkpoint의 가중치: `save_checkpoint` 파일은 `state_dict`, 학습 중 checkpoint는 `model`."""
    return state["state_dict"] if "state_dict" in state else state["model"]


def model_from_state(state: dict) -> tuple[nn.Module, torch.Tensor | None]:
    """모델 설정(`state_config`)이 든 checkpoint에서 (모델, 회귀 구간 경계). `save_checkpoint`
    파일과 학습 중 checkpoint 모두 읽는다. 모델 종류는 파일에 남긴 `model_type`을 따르고, 없는
    이전 파일은 cell_transformer(v2)다."""
    config = state_config(state)
    if config is None:
        raise ValueError("checkpoint에 모델 설정이 없다 (설정을 남기기 전의 학습 중 checkpoint)")
    model = build_model(state.get("model_type", DEFAULT_MODEL_TYPE), config)
    model.load_state_dict(state_weights(state))
    return model.eval(), state["borders"]


def load_checkpoint(
    path: str | Path, map_location: str | torch.device = "cpu"
) -> tuple[nn.Module, torch.Tensor | None]:
    """`save_checkpoint`로 저장한 파일, 또는 모델 설정을 남긴 학습 중 checkpoint를 읽는다."""
    return model_from_state(torch.load(path, map_location=map_location, weights_only=True))


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
