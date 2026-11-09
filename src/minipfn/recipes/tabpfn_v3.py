"""TabPFN-3(arXiv 2605.13986)·TabPFN-3.5(arXiv 2609.17895) recipe: 학습 설정과 추론 설정 (논문 = 각
축 부품의 조합).

- prior: `graph_scm` (v2·2.5와 같다. 3은 graph SCM에 결합 함수·공간·시간·OOD prior를 더했다고
  적었지만 세부가 없어 지금 prior를 그대로 쓴다)
- 모델: `column_row_transformer`. 구조는 `ARCHITECTURES`의 v3 또는 v3.5
- 학습: `minipfn.training.loop.train`. 학습 step·lr·batch는 두 보고서 모두 공개하지 않아서 v2
  paper 설정(`recipes.tabpfn_v2.paper`)에서 시작한다. 3은 task마다 모델이 따로이고, 3.5는
  분류·회귀를 한 모델로 같이 학습한다(`task="multitask"`: 학습 루프가 step마다 task를 번갈아 쓴다)
- 추론: `minipfn.inference.ensemble`의 `EnsembleClassifier`·`EnsembleRegressor`에
  `default_inference` (3은 2.5와 같은 전처리, 3.5는 standard 멤버 8개)

    config = with_architecture(paper("classification"), "v3")
    clf = EnsembleClassifier(model, **default_inference(model, "classification"))

공개 가중치가 생기면 `TabPFNv3Classifier`·`TabPFNv3Regressor`(3·3.5 공용, 기본 repo만 정하는 얇은
클래스)를 여기에 더한다. repo가 생기기 전에는 만들지 않는다(인자 없이 부르면 404라서).

학습 설정(`TrainConfig`)은 쓰는 함수 안에서 불러온다. 추론만 할 때 학습 스택을 불러오지 않도록.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import TYPE_CHECKING

from torch import nn

from minipfn.models import DEFAULT_MODEL_TYPE, canonical_model_type
from minipfn.recipes import resolve_member_kinds

if TYPE_CHECKING:
    from minipfn.training.config import TrainConfig

MODEL_TYPE = "column_row_transformer"
# 모델 구조. v3는 TabPFN-3 Appendix C 표 2~7: E 128, 셀 3개 묶기, inducing 128, CLS 4개 → 행 폭 512,
# 행 ICL 24층·head 8. 열·행 단계 블록 3개·head 8, ff_factor 2, decoder 6 × 64는 모델 설정
# 기본값이다. v3.5는 TabPFN-3.5가 적은 변경: CLS 8개(행 폭 1024), head 16(head dim 64), q·k RMSNorm,
# Fourier + ECDF 셀 encoder. 3.5의 층 수는 공개되지 않아 3의 24층 그대로 둔다(그러면 multitask가
# 약 219M으로 보고서의 220M과 비슷하다)
_V3 = {
    "model_type": MODEL_TYPE,
    "emsize": 128,
    "nlayers": 24,
    "nhead": 8,
    "features_per_group": 3,
    "n_inducing": 128,
    "n_cls": 4,
    "qk_norm": False,
    "encoder": "linear",
    "target_norm": False,
    "head_mlp": False,
}
# 3.5의 정답 LayerNorm·출력층 앞 MLP는 공개 추론 코드에서 확인했다 (모델 docstring)
_V3_5 = {
    **_V3,
    "n_cls": 8,
    "nhead": 16,
    "qk_norm": True,
    "encoder": "fourier",
    "target_norm": True,
    "head_mlp": True,
}
# task마다 구조가 같다(출력층만 다르다). 분류·회귀를 한 모델로(multitask) 학습하는 것은 3.5만
ARCHITECTURES = {
    "v3": {"classification": _V3, "regression": _V3},
    "v3.5": {"classification": _V3_5, "regression": _V3_5, "multitask": _V3_5},
}
# 논문 권장 크기 (TabArena·BeyondArena에서 이보다 큰 split은 돌리지 않는다,
# `minipfn.recipes.supported_size`).
# 3은 검증 범위를 "1M행·200 feature, 100k행·2,000 feature, 1k행·20,000 feature"로 적어서, 상한
# 하나씩으로 둘 수 있는 가운데 점을 쓴다. 3.5는 보고서의 권장 한계
SUPPORTED_SIZE = {
    "v3": {"max_train_rows": 100_000, "max_features": 2_000},
    "v3.5": {"max_train_rows": 1_000_000, "max_features": 6_000},
}
# smoke(`keep_size`)에서 줄이는 값: 층 수·폭·head는 smoke 설정 그대로, inducing point는 8개까지
SMOKE_MAX_INDUCING = 8


def _model_changes(config: TrainConfig, arch: str, keep_size: bool) -> dict:
    if arch not in ARCHITECTURES:
        raise ValueError(f"arch는 {tuple(ARCHITECTURES)} 가운데 하나여야 한다: {arch}")
    if config.task not in ARCHITECTURES[arch]:
        raise ValueError(
            f"arch {arch}에는 {config.task} 구조가 없다 ({tuple(ARCHITECTURES[arch])})"
        )
    changes = dict(ARCHITECTURES[arch][config.task])
    if canonical_model_type(config.model_type) not in (DEFAULT_MODEL_TYPE, MODEL_TYPE):
        raise ValueError(
            f"arch {arch}는 {MODEL_TYPE} 모델이다 (설정의 model_type: {config.model_type})"
        )
    if keep_size:
        for name in ("emsize", "nlayers", "nhead"):
            changes[name] = getattr(config, name)
        changes["n_inducing"] = min(changes["n_inducing"], SMOKE_MAX_INDUCING)
    return changes


def with_architecture(config: TrainConfig, arch: str, keep_size: bool = False) -> TrainConfig:
    """모델 구조를 `ARCHITECTURES`의 것으로 (모델 클래스도 `column_row_transformer`로). 다른 모델
    클래스로 정한 설정은 바꾸지 않고 에러를 낸다. `keep_size`(smoke)면 폭·층 수·head 수는 그대로
    두고 inducing point만 줄인다(구조가 바뀌는 코드 경로만 확인한다).

    QK-clip은 끈다. 이 모델의 attention logit은 가중치만으로 묶이지 않아서 QK-clip이 볼 attention이
    없다(`ColumnRowTransformer.qk_clip_weights`)."""
    return replace(config, **_model_changes(config, arch, keep_size), qk_clip=None)


def architecture_name(config) -> str | None:
    """`config`(학습 설정, 또는 `task`·`model_type`과 구조 필드가 있는 것)의 구조가
    `ARCHITECTURES`의 어느 것인지 (없으면 None). smoke처럼 크기를 줄인 것도 그 구조로 본다."""
    if canonical_model_type(config.model_type) != MODEL_TYPE:
        return None
    for name in ARCHITECTURES:
        if config.task not in ARCHITECTURES[name]:
            continue
        for keep_size in (False, True):
            changes = _model_changes(config, name, keep_size)
            if all(getattr(config, k) == v for k, v in changes.items()):
                return name
    return None


def model_architecture(model: nn.Module, task: str) -> str | None:
    """만든 모델(checkpoint·Hub에서 읽은 것)의 구조 이름 (없으면 None)."""
    config = getattr(model, "config", None)
    if config is None or getattr(model, "model_type", None) != MODEL_TYPE:
        return None
    return architecture_name(
        SimpleNamespace(**{**config.to_dict(), "task": task}, model_type=MODEL_TYPE)
    )


def paper(task: str = "classification", arch: str = "v3") -> TrainConfig:
    """논문 규모 학습 설정. 보고서에 학습 설정이 없어서 prior·표 크기·batch 64·step·lr·붕괴 규칙은
    v2 paper 그대로이고 구조만 `arch`로 바꾼다. v2 회귀의 QK-clip 250은 이 모델에 쓸 수 없어 끈다
    (`with_architecture`). multitask(3.5)는 붕괴 규칙을 v2 회귀의 것(되돌릴 때 lr 절반)으로 둔다
    — 회귀가 더 자주 무너졌다 (여기서 정함)."""
    from minipfn.recipes import tabpfn_v2

    if task == "multitask":
        return with_architecture(replace(tabpfn_v2.paper("regression"), task=task), arch)
    return with_architecture(tabpfn_v2.paper(task), arch)


# 논문 규모로 실제 학습한 TabPFN-3.5 경로 (issue #29). `paper`에서 결과를 바꾸는 설정만 더한다:
# batch 256(GPU 32장 × 8), Muon(issue #28), 125k step부터 기준 lr 2e-4 → 1e-4와 Muon weight decay
# 0 → 0.1. 125k 구간은 chunk02(lr 2e-4, weight decay 0)가 148~152k부터 gradient norm이 3.7에서
# 15까지 뛰며 불안정해져서 125k checkpoint로 되돌려 바꾼 것이다(여기서 정함). 계산 결과를
# 바꾸지 않는 가속 옵션은 넣지 않는다. 실제 run은 compile·fused Adam·TF32·LAWA 10×100·
# 분산 Muon all을 켰다 (`--compile --fused-adam --tf32 --weight-average 10
# --weight-average-every 100 --muon-distributed all`)
PAPER_RUN = {
    "batch_size": 256,
    "optimizer": "muon",
    "phases": ((125_000, {"lr": 1e-4, "muon_weight_decay": 0.1}),),
}


def paper_run(task: str = "multitask", arch: str = "v3.5") -> TrainConfig:
    """실제로 학습한 TabPFN-3.5 논문 규모 설정 (`paper` + `PAPER_RUN`). 한 번에 처음부터 학습하면
    구간이 바뀌는 step에서 lr·weight decay를 바꾼다(`TrainConfig.phases`). v3.5 multitask만."""
    if (task, arch) != ("multitask", "v3.5"):
        raise ValueError(f"paper_run은 v3.5 multitask만 있다 (task {task}, arch {arch})")
    return replace(paper(task, arch), **PAPER_RUN)


# paper_run으로 학습한 가중치를 lr을 0까지 내리며 짧게 마무리 학습한다 (issue #29, 여기서 정함).
# paper_run의 lr은 2M step cosine이라 200k에서도 거의 줄지 않았고, 100k → 200k에 benchmark가
# 그대로였다(분류 AUC 0.932 → 0.931). 2.5의 long_tables처럼 가중치만 가져와(`init_from`) 새
# optimizer로 warmup 1k + cosine 20k step, lr 1e-4(paper_run 125k 뒤의 lr)에서 0까지. 표 크기는
# paper 그대로다
COOLDOWN = {
    "steps": 20_000,
    "warmup_steps": 1_000,
    "lr": 1e-4,
    "muon_weight_decay": 0.1,
    "keep_every": 5_000,
    "phases": (),
}


def cooldown(task: str = "multitask", arch: str = "v3.5") -> TrainConfig:
    """`paper_run` 가중치의 마무리 학습 설정 (`paper_run` + `COOLDOWN`). v3.5 multitask만."""
    return replace(paper_run(task, arch), **COOLDOWN)


# TabICLv2의 3단계 학습(arXiv 2602.11139 §4.1·Appendix B.1)을 공개 학습 스크립트
# (soda-inria/tabicl `scripts/train_v2_clf_stage{1,2,3}.sh`) 값 그대로 옮긴 것. 단계마다 가중치만
# 넘겨(`init_from`) 새 optimizer로, warmup 1% 뒤 cosine으로 내린다. TabPFN-3·3.5 보고서에 학습
# 설정이 없어서, 구조가 같은 계열(열 → 행 → 행 ICL, QASSMax, graph SCM prior)의 공개 recipe를 쓴다.
# 스크립트와 같은 것:
# - Muon을 모든 파라미터에(`muon_scope="all"`), momentum 0.9, weight decay 0.01 (스크립트 주석:
#   논문은 cautious라고 적었지만 공개 checkpoint run은 일반 weight decay였다)
# - grad clip 10 / 10 / 1, lr 8e-4 / 1e-4 / 2e-5
# - 표 크기: 행 1,024개 고정·학습 30~90% → 400~10,240행 log 균등·학습 79~81% → 400~60,000행
#   (행이 많으면 feature 상한을 줄인다). feature 수 U(1, 100)
# - 분류 클래스 수의 절반은 2 (`binary_prob` 0.5), 예측할 수 없는 그래프·데이터셋 다시 뽑기
# 여기서 바꾼 것 (여기서 정함):
# - batch 256 (GPU 32장). TabICLv2는 분류·회귀 모델이 따로이고 batch 64다. multitask는 step마다
#   task를 번갈아 쓰므로 step 수를 task당 데이터셋 수가 같게 정했다: 500K × 64 = 250K × 256 / 2.
#   stage 2·3도 같은 규칙으로 20K·5K. lr은 스크립트 값 그대로(batch를 키운 만큼 바꾸지 않는다)
# - 모델은 TabPFN-3.5 구조(219M), 분류·회귀 한 모델. 결측값(`missing_dataset_prob`)·회귀 bar
#   distribution·붕괴 되돌리기 규칙은 `paper`의 것 그대로
# - micro batch는 스크립트처럼 4 / 1 / 1 (stage 1을 8로 하면 셀이 1,024 × 100이라 메모리가 빠듯하다)
CURRICULUM_COMMON = {
    "batch_size": 256,
    "optimizer": "muon",
    "muon_scope": "all",
    "muon_momentum": 0.9,
    "muon_weight_decay": 0.01,
    "size_rule": "train_fraction",
    "max_features": 100,
    "binary_prob": 0.5,
    "filter_unpredictable": True,
    "phases": (),
}
CURRICULUM = {
    1: {
        "steps": 250_000,
        "lr": 8e-4,
        "grad_clip": 10.0,
        "min_rows": 1024,
        "max_rows": 1024,
        "min_train_fraction": 0.3,
        "max_train_fraction": 0.9,
        "log_uniform_rows": False,
        "micro_batch": 4,
    },
    2: {
        "steps": 20_000,
        "lr": 1e-4,
        "grad_clip": 10.0,
        "min_rows": 400,
        "max_rows": 10_240,
        "min_train_fraction": 0.79,
        "max_train_fraction": 0.81,
        "log_uniform_rows": True,
        "micro_batch": 1,
        "checkpoint_every": 1_000,
        "keep_every": 5_000,
    },
    3: {
        "steps": 5_000,
        "lr": 2e-5,
        "grad_clip": 1.0,
        "min_rows": 400,
        "max_rows": 60_000,
        "min_train_fraction": 0.79,
        "max_train_fraction": 0.81,
        "log_uniform_rows": True,
        "micro_batch": 1,
        "val_every": 1_000,
        "checkpoint_every": 1_000,
        "keep_every": 1_000,
    },
}
# warmup은 단계 길이의 1% (스크립트 `--warmup_proportion 0.01`)
CURRICULUM_WARMUP = 0.01


def curriculum(stage: int, task: str = "multitask", arch: str = "v3.5") -> TrainConfig:
    """TabICLv2 3단계 학습의 `stage`(1·2·3)번째 단계 설정 (`CURRICULUM`). stage 2·3은 앞 단계
    checkpoint에서 가중치만 가져와(`init_from`) 시작한다(학습 스크립트가 막는다). v3.5
    multitask만."""
    if (task, arch) != ("multitask", "v3.5"):
        raise ValueError(f"curriculum은 v3.5 multitask만 있다 (task {task}, arch {arch})")
    if stage not in CURRICULUM:
        raise ValueError(f"curriculum stage는 {tuple(CURRICULUM)} 가운데 하나다: {stage}")
    changes = {**CURRICULUM_COMMON, **CURRICULUM[stage]}
    changes["max_cells"] = changes["max_rows"] * changes["max_features"]
    changes["warmup_steps"] = int(CURRICULUM_WARMUP * changes["steps"])
    return replace(paper(task, arch), **changes)


# ---------------------------------------------------------------------------
# 추론
# ---------------------------------------------------------------------------
# 3: "추론 전처리(robust scaling + soft clipping, quantile, standard, SVD)는 2.5와 같다" → 2.5의
# 멤버 종류와 v2 논문 멤버 수. 3.5: "quantile·robust·SVD를 빼고 standard만, estimator 8개"
INFERENCE = {
    "v3": {
        "n_estimators": {"classification": 4, "regression": 8},
        "member_kinds": ("quantile", "robust"),
    },
    "v3.5": {"n_estimators": {"classification": 8, "regression": 8}, "member_kinds": ("standard",)},
}
# 이름으로 고르는 멤버 종류 묶음 (`minipfn.recipes.resolve_member_kinds`가 v2·v2.5와 합쳐 읽는다)
MEMBER_KIND_PRESETS = {name: v["member_kinds"] for name, v in INFERENCE.items()}


def default_inference(model: nn.Module, task: str) -> dict:
    """그 모델의 추론 설정 `{"n_estimators": int, "member_kinds": tuple}`. weight와 함께 온 설정
    (`model.inference_config`, Hub config.json의 `inference`)이 먼저이고, 없으면 모델 구조(3.5면
    v3.5, 그 밖은 v3)의 것."""
    saved = getattr(model, "inference_config", None) or {}
    preset = INFERENCE[model_architecture(model, task) or "v3"]
    return {
        "n_estimators": int(saved.get("n_estimators", preset["n_estimators"][task])),
        "member_kinds": resolve_member_kinds(saved.get("member_kinds") or preset["member_kinds"]),
    }
