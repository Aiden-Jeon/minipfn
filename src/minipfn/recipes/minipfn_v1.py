"""minipfn-v1 recipe: TabPFN 재현이 아니라 우리 자체 모델의 첫 구조 (issue #110, 2026-11-23 이름).

TabPFN-3.5 구조(`recipes.tabpfn_v3`의 `v3.5`)에서 시작해 Kumo-Tabular 부품(`recipes.kumo_tabular`)을
얹은 것이다. 행 ICL(24층·16 head·폭 1024)과 그 뒤는 v3.5와 같고, 다른 것은:

- 열·행 단계를 6쌍 번갈아 쌓는다(`kumo_tabular.STAGES`: 셀 폭 256, head 4, inducing 256, CLS 4개)
- attention 온도를 Kumo 식으로(`kumo_tabular.ATTENTION_TEMPERATURE`), soft-cap 없음
- 회귀 출력은 분위수 999개(`kumo_tabular.QUANTILE_HEAD`)
- 분류·회귀를 한 모델로(`task="multitask"`)만 학습한다

    config = curriculum(1)  # stage 1. curriculum(2)·(3)은 --init-run-id로 앞 단계에서
    clf = EnsembleClassifier(model, **default_inference(model, "classification"))

모델 클래스는 v3·v3.5와 같은 `column_row_transformer`다. 그래서 추론 설정·권장 크기는 모델 종류가
아니라 구조 판정(`model_architecture`)으로 이 recipe를 고른다(`minipfn.recipes.recipe_for_model`).
이름이 붙기 전(PR #113·#114)에 학습한 run과 checkpoint는 이름 필드가 없지만 구조 필드로 이 구조로
판정된다. 그 run의 MLflow 이름은 `tabpfn-v3-5-…`로 남아 있다.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import TYPE_CHECKING

from torch import nn

from minipfn.models import DEFAULT_MODEL_TYPE, canonical_model_type
from minipfn.recipes import kumo_tabular, tabpfn_v3

if TYPE_CHECKING:
    from minipfn.training.config import TrainConfig

NAME = "minipfn-v1"
MODEL_TYPE = tabpfn_v3.MODEL_TYPE
_V3_5 = tabpfn_v3.ARCHITECTURES["v3.5"]["multitask"]
# 구조를 정하는 값 (`with_architecture`가 넣는다): v3.5 + 열·행 단계 + 온도 + 분위수 출력 +
# soft-cap 없음
STRUCTURE = {
    **_V3_5,
    **kumo_tabular.STAGES,
    **kumo_tabular.ATTENTION_TEMPERATURE,
    **kumo_tabular.QUANTILE_HEAD,
    "logit_softcap": None,
}
# 학습은 multitask만이다. 분류·회귀 키는 multitask 모델을 task별로 평가할 때 판정용이다
# (`model_architecture(model, "classification")`, v3.5와 같다)
ARCHITECTURES = {NAME: dict.fromkeys(("classification", "regression", "multitask"), STRUCTURE)}
TRAIN_TASKS = ("multitask",)
# 권장 크기는 v3.5와 같다 (행 ICL 이후가 같아서. TabArena·BeyondArena에서 이보다 큰 split은 돌리지
# 않는다, `minipfn.recipes.supported_size`)
SUPPORTED_SIZE = {NAME: dict(tabpfn_v3.SUPPORTED_SIZE["v3.5"])}

# 구조 판정. 열·행 단계를 번갈아 쌓았고(`stage_layout`), 행 ICL과 그 뒤가 v3.5와 같으면
# minipfn-v1이다. 열·행 단계의 폭(`emsize`·`n_cls`·`n_inducing`)은 견주지 않고 대신 행 ICL 폭
# (n_cls × emsize)이 v3.5와 같아야 한다. 이름이 붙기 전 `tabpfn_v3.architecture_name`이 이 모델을
# v3.5로 본 규칙 그대로다(그래서 그때 학습한 checkpoint의 추론 설정이 바뀌지 않는다). 블록 수·head
# 수·온도·출력 종류는 v3.5에서 온도를 견주지 않았듯 견주지 않는다
STAGE_WIDTH_FIELDS = ("emsize", "n_cls", "n_inducing")
ICL_FIELDS = {k: v for k, v in _V3_5.items() if k not in STAGE_WIDTH_FIELDS}
ICL_WIDTH = _V3_5["n_cls"] * _V3_5["emsize"]
# smoke(`keep_size`): 층 수·폭·head는 smoke 설정 그대로, inducing point는 8개, 열·행 블록은 2쌍.
# 판정에서는 층 수·head·행 ICL 폭을 보지 않고 inducing point 수로 smoke를 알아본다
SMOKE_MAX_INDUCING = tabpfn_v3.SMOKE_MAX_INDUCING
SMOKE_STAGE_BLOCKS = 2
SMOKE_FREE_FIELDS = ("nlayers", "nhead")


def _check(task: str, arch: str) -> None:
    if arch != NAME:
        raise ValueError(f"arch는 {NAME}여야 한다: {arch}")
    if task not in TRAIN_TASKS:
        raise ValueError(f"{NAME}는 {'·'.join(TRAIN_TASKS)}만 학습한다 (task {task})")


def with_architecture(
    config: TrainConfig, arch: str = NAME, keep_size: bool = False
) -> TrainConfig:
    """모델 구조를 `STRUCTURE`로 (모델 클래스도 `column_row_transformer`로). 다른 모델 클래스로 정한
    설정은 바꾸지 않고 에러를 낸다. `keep_size`(smoke)면 폭·층 수·head 수는 그대로 두고 inducing
    point와 열·행 블록 수만 줄인다. QK-clip은 끈다(`tabpfn_v3.with_architecture`와 같은 까닭)."""
    _check(config.task, arch)
    if canonical_model_type(config.model_type) not in (DEFAULT_MODEL_TYPE, MODEL_TYPE):
        raise ValueError(
            f"arch {arch}는 {MODEL_TYPE} 모델이다 (설정의 model_type: {config.model_type})"
        )
    changes = dict(STRUCTURE)
    if keep_size:
        for name in ("emsize", "nlayers", "nhead"):
            changes[name] = getattr(config, name)
        changes["n_inducing"] = SMOKE_MAX_INDUCING
        changes["column_blocks"] = changes["row_blocks"] = SMOKE_STAGE_BLOCKS
    return replace(config, **changes, qk_clip=None)


def architecture_name(config) -> str | None:
    """`config`(학습 설정, 또는 `task`·`model_type`과 구조 필드가 있는 것)가 minipfn-v1이면 그 이름
    (아니면 None). 판정 규칙은 `ICL_FIELDS` 위 주석. smoke처럼 크기를 줄인 것도 이 구조로 본다."""
    if canonical_model_type(config.model_type) != MODEL_TYPE:
        return None
    if getattr(config, "stage_layout", "sequential") != "interleaved":
        return None
    if config.task not in ARCHITECTURES[NAME]:
        return None
    if all(getattr(config, k) == v for k, v in ICL_FIELDS.items()):
        if config.n_cls * config.emsize == ICL_WIDTH:
            return NAME
    if config.n_inducing == SMOKE_MAX_INDUCING and all(
        getattr(config, k) == v for k, v in ICL_FIELDS.items() if k not in SMOKE_FREE_FIELDS
    ):
        return NAME
    return None


def model_architecture(model: nn.Module, task: str) -> str | None:
    """만든 모델(checkpoint·Hub에서 읽은 것)의 구조 이름 (없으면 None)."""
    config = getattr(model, "config", None)
    if config is None or getattr(model, "model_type", None) != MODEL_TYPE:
        return None
    return architecture_name(
        SimpleNamespace(**{**config.to_dict(), "task": task}, model_type=MODEL_TYPE)
    )


# ---------------------------------------------------------------------------
# 학습: TabICLv2 3단계 curriculum (`tabpfn_v3.curriculum` stage 1~3) 위에 이 구조
# ---------------------------------------------------------------------------
# 시험 학습(8×H100 batch 64, stage 1 처음부터 50k)에서 고른 부품만 넣었다:
# - attention 온도를 Kumo 식으로, soft-cap 없음, stage 1 lr을 TabICLv2 스크립트 값 8e-4로 (issue
#   #102: v3.5 구조는 soft-cap 없이 lr 8e-4면 80k부터 ICL 0층이 폭주했고, Kumo 식은 100k까지 온도가
#   줄었다. TabArena-Lite d −0.013)
# - Kumo식 열·행 단계 (`kumo_tabular.STAGES`)와 회귀 분위수 출력 (`kumo_tabular.QUANTILE_HEAD`,
#   issue #110: 둘 다 BeyondArena 작은 표 d −0.006으로 노이즈 밖, TabArena-Lite는 노이즈 안)
# - decoder q·k norm(효과 없음)·범주를 아는 셀 encoder(나빠짐)는 넣지 않았다 (issue #110)
# stage 2·3 lr은 스크립트 값 1e-4 / 2e-5 (stage 1과 같은 비율, 여기서 정함. 이 두 단계는 시험하지
# 않았다). stage 1 micro batch는 4: 열·행 단계가 넓어 GPU당 표 8개(stage 1 1,024행 × 100열)는
# H100 80 GB를 넘었다(issue #110). stage 2·3은 원래 1이다
# stage 2·3은 2,048행 넘는 표에서 activation checkpointing을 켠다(`tabpfn_v3.curriculum`은 stage 2
# 끔, stage 3 10,240행). 열·행 단계가 넓어 stage 2 가장 큰 표(10,240행) 하나가 GPU 한 장(H100
# 80 GB)을 넘었다(8×H100 smoke, 첫 forward에서 OOM). stage 1은 GPU당 1,024행 표 4개(4,096행)에
# 60 GB여서 기준을 그 절반으로 잡았다(여기서 정함)
# 표 크기·optimizer·step 수·warmup은 `tabpfn_v3.curriculum` 그대로다
ACTIVATION_CHECKPOINTING_ROWS = 2_048
CURRICULUM = {
    1: {"lr": 8e-4, "micro_batch": 4},
    2: {"lr": 1e-4, "activation_checkpointing_rows": ACTIVATION_CHECKPOINTING_ROWS},
    3: {"lr": 2e-5, "activation_checkpointing_rows": ACTIVATION_CHECKPOINTING_ROWS},
}


def curriculum(
    stage: int, task: str = "multitask", arch: str = NAME, steps: int | None = None
) -> TrainConfig:
    """`tabpfn_v3.curriculum`의 stage 1~3을 이 구조로 (`STRUCTURE`) 바꾸고 단계별 lr·micro batch·
    activation checkpointing 기준(`CURRICULUM`)을 넣은 설정. stage 2·3은 앞 단계 checkpoint에서
    가중치만 가져와 시작한다(학습 스크립트가 막는다). `steps`를 주면 그 step 수로 학습하고
    warmup(1%)·cosine도 그것에 맞춘다. multitask만."""
    _check(task, arch)
    if stage not in CURRICULUM:
        raise ValueError(f"{NAME} curriculum stage는 {tuple(CURRICULUM)} 가운데 하나다: {stage}")
    base = tabpfn_v3.curriculum(stage, task, "v3.5", steps)
    return replace(with_architecture(base, arch), **CURRICULUM[stage])


# ---------------------------------------------------------------------------
# 추론: v3.5와 같다 (standard 멤버 8개 + 멤버당 768열 balanced, `tabpfn_v3.INFERENCE`)
# ---------------------------------------------------------------------------
INFERENCE = {NAME: tabpfn_v3.INFERENCE["v3.5"]}
# 이름으로 고르는 멤버 종류 묶음 (`minipfn.recipes.resolve_member_kinds`가 다른 recipe와 합쳐
# 읽는다)
MEMBER_KIND_PRESETS = {name: v["member_kinds"] for name, v in INFERENCE.items()}


def default_inference(model: nn.Module, task: str) -> dict:
    """그 모델의 추론 설정 (`tabpfn_v3.default_inference`와 같은 꼴). weight와 함께 온 설정
    (`model.inference_config`)이 먼저이고, 없는 키는 `INFERENCE`의 것."""
    return tabpfn_v3.inference_settings(model, task, INFERENCE[NAME])
