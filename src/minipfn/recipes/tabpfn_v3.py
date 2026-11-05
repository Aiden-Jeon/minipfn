"""TabPFN-3(arXiv 2605.13986)·TabPFN-3.5(arXiv 2609.17895) recipe: 학습 설정과 추론 설정 (논문 = 각
축 부품의 조합).

- prior: `graph_scm` (v2·2.5와 같다. 3은 graph SCM에 결합 함수·공간·시간·OOD prior를 더했다고
  적었지만 세부가 없어 지금 prior를 그대로 쓴다)
- 모델: `column_row_transformer`. 구조는 `ARCHITECTURES`의 v3 또는 v3.5
- 학습: `minipfn.training.loop.train`. 학습 step·lr·batch는 두 보고서 모두 공개하지 않아서 v2
  paper 설정(`recipes.tabpfn_v2.paper`)에서 시작한다
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
# Fourier + ECDF 셀 encoder. 3.5의 층 수는 공개되지 않아 3의 24층 그대로 둔다(그러면 약 215M으로
# 보고서의 220M과 비슷하다)
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
}
_V3_5 = {**_V3, "n_cls": 8, "nhead": 16, "qk_norm": True, "encoder": "fourier"}
# 3·3.5 모두 분류·회귀가 같은 구조다 (출력층만 다르다. 3.5의 한 checkpoint 학습은 아직 없다)
ARCHITECTURES = {
    "v3": {"classification": _V3, "regression": _V3},
    "v3.5": {"classification": _V3_5, "regression": _V3_5},
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
    (`with_architecture`)."""
    from minipfn.recipes import tabpfn_v2

    return with_architecture(tabpfn_v2.paper(task), arch)


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
