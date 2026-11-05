"""TabPFN v2(Nature 2025)·TabPFN-2.5(arXiv 2511.08667) recipe: 학습 설정과 추론 설정 (논문 = 각 축
부품의 조합).

- prior: `graph_scm` (TabICLv2 graph SCM + 표 크기 규칙. 크기 기본값이 v2 논문의 것이다)
- 모델: `cell_transformer`. 구조는 `ARCHITECTURES`의 v2 또는 v2.5
- 학습: `minipfn.training.loop.train`, 논문 규모 설정은 `paper`, 긴 표 이어 학습은 `long_tables`
- 추론: `TabPFNv2Classifier`·`TabPFNv2Regressor` (`minipfn.inference.ensemble`에 논문 멤버 수와
  멤버 종류 묶음 `MEMBER_KIND_PRESETS`의 v2 또는 v2.5)

    config = with_architecture(paper("regression"), "v2.5")
    clf = TabPFNv2Classifier(model).fit(x_train, y_train)  # 추론 설정은 모델이 정한다

추론 설정(멤버 수와 멤버 종류)은 weight와 함께 온다(`default_inference`). Hub config.json에 적힌
값이 있으면 그것, 없으면 모델 구조가 2.5면 v2.5, 아니면 v2다. 인자로 주면 그 값을 쓴다(실험용).

학습 설정(`TrainConfig`)은 쓰는 함수 안에서 불러온다. Hub에서 받아 추론만 할 때 학습 스택을
불러오지 않도록.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from minipfn.inference import ensemble
from minipfn.inference.ensemble import EnsembleClassifier, EnsembleRegressor
from minipfn.models import DEFAULT_MODEL_TYPE, canonical_model_type

if TYPE_CHECKING:
    from minipfn.training.config import TrainConfig

# 모델 구조. v2는 Nature 논문 기본 모델, v2.5는 TabPFN-2.5(arXiv 2511.08667 "Architecture")가
# 적은 변경: 층 수(회귀 18, 분류 24), 셀 3개 묶기, thinking 행 64개, 회귀 encoder를 2층 MLP로.
# emsize·head 수는 2.5 논문에 없어서 v2 그대로 둔다. 둘 다 모델 클래스는 CellTransformer다
_V2 = {
    "model_type": "cell_transformer",
    "nlayers": 12,
    "features_per_group": 2,
    "n_thinking_rows": 0,
    "encoder": "linear",
}
_V2_5 = {**_V2, "features_per_group": 3, "n_thinking_rows": 64}  # 두 task 공통
ARCHITECTURES = {
    "v2": {"classification": _V2, "regression": _V2},
    "v2.5": {
        "classification": {**_V2_5, "nlayers": 24},
        "regression": {**_V2_5, "nlayers": 18, "encoder": "mlp"},
    },
}
# long_tables의 GPU당 micro batch. v2(12층)는 2에서 GPU 메모리 52.6GB였다. 2.5는 층이 두 배라 1
LONG_TABLES_MICRO_BATCH = {"v2": 2, "v2.5": 1}


def architecture_name(config: TrainConfig) -> str | None:
    """`config`의 모델 구조가 `ARCHITECTURES`의 어느 것인지 (없으면 None). smoke처럼 층 수를
    줄인 설정(`with_architecture(..., keep_size=True)`로 만든 것)도 그 구조로 본다. `task`,
    `model_type`과 구조 필드가 있으면 학습 설정이 아니어도 된다(`model_architecture`)."""
    for name in ARCHITECTURES:
        for keep_size in (False, True):
            try:
                changes = _model_changes(config, name, keep_size)
            except ValueError:  # 다른 모델 클래스
                return None
            if all(getattr(config, k) == v for k, v in changes.items()):
                return name
    return None


def model_architecture(model: nn.Module, task: str) -> str | None:
    """만든 모델(checkpoint·Hub에서 읽은 것)의 구조가 `ARCHITECTURES`의 어느 것인지
    (없으면 None)."""
    config = getattr(model, "config", None)
    if config is None or not hasattr(model, "model_type"):
        return None
    return architecture_name(
        SimpleNamespace(**config.to_dict(), task=task, model_type=model.model_type)
    )


def paper(task: str = "classification") -> TrainConfig:
    from minipfn.training.config import TrainConfig

    # 분류는 2e-4에서 붕괴가 드물어(300k까지 rewind 1번) lr을 그대로 둔다.
    # 회귀의 QK-clip 250: 50k scratch 비교에서 250·500 모두 상한을 지켰고 val NLL도 비슷했다
    # (−1.08·−1.07, 끈 것 −1.04). 그중 logit을 더 낮게 묶는 쪽. 100은 10k부터 잘라 학습이 늦었다
    if task == "regression":
        return TrainConfig(task=task, rewind_lr_factor=0.5, qk_clip=250.0)
    return TrainConfig(task=task)


def long_tables(task: str = "classification", arch: str = "v2") -> TrainConfig:
    """paper로 학습한 모델을 더 긴·넓은 표로 짧게 이어 학습한다 (논문에 없음, 값은 여기서 정함).

    TabPFN-2.5는 prior를 "더 많은 행과 feature로 키웠다"고만 적었다(추론 권장 크기는 행 50,000,
    feature 2,000). 그 크기로 처음부터 학습하는 대신 TabICL처럼 짧은 표로 오래 학습한 뒤 긴 표로
    잠깐 더 학습한다: 학습 행 512~8,192개 log 균등, feature 수 분포를 500까지 펴고, 셀은
    400,000개까지. 가중치만 가져와(`init_from`) 새 optimizer로 lr 5e-5, warmup 1k + cosine
    20k step. 검증 batch는 데이터셋 4개다. GPU당 micro batch는 구조에 맞춰
    `LONG_TABLES_MICRO_BATCH`로 정한다.
    """
    if arch not in LONG_TABLES_MICRO_BATCH:
        raise ValueError(f"long_tables의 arch는 {tuple(LONG_TABLES_MICRO_BATCH)} 가운데서: {arch}")
    sizes = replace(
        paper(task),
        min_train_rows=512,
        max_train_rows=8192,
        log_uniform_rows=True,
        max_features=500,
        feature_scale=500,
        max_cells=400_000,
        val_batch=4,
        steps=20_000,
        warmup_steps=1_000,
        lr=5e-5,
        keep_every=5_000,
    )
    return replace(with_architecture(sizes, arch), micro_batch=LONG_TABLES_MICRO_BATCH[arch])


def _model_changes(config: TrainConfig, arch: str, keep_size: bool) -> dict:
    # `arch` 구조로 바꿀 모델 설정 (with_architecture와 architecture_name이 같이 쓴다)
    if arch not in ARCHITECTURES:
        raise ValueError(f"arch는 {tuple(ARCHITECTURES)} 가운데 하나여야 한다: {arch}")
    changes = dict(ARCHITECTURES[arch][config.task])
    if canonical_model_type(config.model_type) not in (DEFAULT_MODEL_TYPE, changes["model_type"]):
        raise ValueError(
            f"arch {arch}는 {changes['model_type']} 모델이다"
            f" (설정의 model_type: {config.model_type})"
        )
    if keep_size:
        changes["nlayers"] = config.nlayers
        if "n_thinking_rows" in changes:
            changes["n_thinking_rows"] = min(changes["n_thinking_rows"], 4)
    return changes


def with_architecture(config: TrainConfig, arch: str, keep_size: bool = False) -> TrainConfig:
    """모델 구조를 `ARCHITECTURES`의 것으로. 구조가 모델 클래스(`model_type`)도 정한다. 이미
    기본값이 아닌 다른 모델 클래스로 정한 설정은 바꾸지 않고 에러를 낸다(다른 모델을 v2로 조용히
    바꿔 학습하지 않도록). `keep_size`(smoke)면 층 수는 그대로 두고 thinking 행은 4개까지만 써서
    작은 크기를 유지한다(구조가 바뀌는 코드 경로만 확인한다).

    micro batch는 건드리지 않는다. 긴 표의 micro batch는 `long_tables`가 구조에 맞춰 정하고
    (`LONG_TABLES_MICRO_BATCH`), 구조만 바꾸면 그 값이 그대로 남는다.
    """
    return replace(config, **_model_changes(config, arch, keep_size))


# ---------------------------------------------------------------------------
# 추론 (Methods "Inference details", "Regression details")
# ---------------------------------------------------------------------------
N_ESTIMATORS = {"classification": 4, "regression": 8}  # 논문 기본 멤버 수
MEMBER_KINDS_V2 = ("quantile", "power")  # 분위수 + SVD, Yeo-Johnson
# TabPFN-2.5: "robust scaling과 soft clipping을 분위수·표준 변환과 함께"
# (`minipfn.inference.ensemble` docstring)
MEMBER_KINDS_V2_5 = ("quantile", "robust")
# 이름으로 고르는 멤버 종류 묶음 (평가 스크립트의 `--member-kinds v2.5` 등)
MEMBER_KIND_PRESETS = {"v2": MEMBER_KINDS_V2, "v2.5": MEMBER_KINDS_V2_5}


def resolve_member_kinds(value: str | tuple[str, ...] | list[str]) -> tuple[str, ...]:
    """멤버 전처리 종류. `MEMBER_KIND_PRESETS`의 이름, 쉼표로 구분한 문자열, 종류 목록을 받는다."""
    return ensemble.resolve_member_kinds(value, MEMBER_KIND_PRESETS)


def default_inference(model: nn.Module, task: str) -> dict:
    """그 모델의 추론 설정 `{"n_estimators": int, "member_kinds": tuple}`. weight와 함께 온 설정
    (`model.inference_config`: Hub config.json의 `inference`. `minipfn.hub.from_pretrained`가
    붙인다)이 있으면 그 값을 쓴다. 없으면(그 키가 없는 repo, 학습 checkpoint) 멤버 수는 논문
    기본값이고 멤버 종류는 모델 구조로 정한다: 2.5 구조면 v2.5, 그 밖은 v2."""
    saved = getattr(model, "inference_config", None) or {}
    kinds = saved.get("member_kinds") or model_architecture(model, task) or "v2"
    return {
        "n_estimators": int(saved.get("n_estimators", N_ESTIMATORS[task])),
        "member_kinds": resolve_member_kinds(kinds),
    }


def _inference_kwargs(model: nn.Module, task: str, n_estimators, member_kinds) -> dict:
    # 인자로 준 값이 먼저, 없으면 모델의 추론 설정 (`default_inference`)
    defaults = default_inference(model, task)
    return {
        "n_estimators": defaults["n_estimators"] if n_estimators is None else n_estimators,
        "member_kinds": defaults["member_kinds"]
        if member_kinds is None
        else resolve_member_kinds(member_kinds),
    }


# 공개 가중치 (`TabPFNv2Classifier()`·`TabPFNv2Regressor()`의 기본값)
HUB_REPOS = {
    "classification": "aiden-jeon/minipfn-tabpfn-v2-classifier",
    "regression": "aiden-jeon/minipfn-tabpfn-v2-regressor",
}


def _load(model, task: str) -> tuple[nn.Module, torch.Tensor | None]:
    # 모델이 아니면 Hub repo id·로컬 폴더(None이면 공개 가중치)에서 받는다
    if isinstance(model, nn.Module):
        return model, None
    from minipfn.hub import model_or_pretrained

    loaded, borders = model_or_pretrained(model, HUB_REPOS[task], task)
    if not hasattr(loaded, "model_type"):
        raise ValueError(f"{model or HUB_REPOS[task]}는 TabPFN v2·2.5 모델이 아니다")
    return loaded, borders


class TabPFNv2Classifier(EnsembleClassifier):
    """v2·2.5 분류 `EnsembleClassifier`. `model`을 주지 않으면 공개 가중치
    (`HUB_REPOS["classification"]`)를 받는다. 다른 가중치는 Hub repo id(`owner/name`,
    `owner/name@branch`), 로컬 폴더, 또는 모델로 준다. 멤버 수와 멤버 종류의 기본값은 모델이
    정한다(`default_inference`). 인자로 주면 그 값을 쓴다.

        clf = TabPFNv2Classifier().fit(x_train, y_train)
    """

    def __init__(
        self,
        model: nn.Module | str | Path | None = None,
        *,
        n_estimators: int | None = None,
        member_kinds: str | tuple[str, ...] | None = None,
        **kwargs: Any,
    ) -> None:
        model, _ = _load(model, "classification")
        inference = _inference_kwargs(model, "classification", n_estimators, member_kinds)
        super().__init__(model, **inference, **kwargs)


class TabPFNv2Regressor(EnsembleRegressor):
    """v2·2.5 회귀 `EnsembleRegressor`. `model`을 주지 않으면 공개 가중치
    (`HUB_REPOS["regression"]`)와 그 구간 경계를 받는다. 다른 가중치는 Hub repo id, 로컬 폴더(경계도
    거기서), 또는 모델과 `borders`로 준다. 멤버 수와 멤버 종류의 기본값은 모델이 정한다
    (`default_inference`). 인자로 주면 그 값을 쓴다.

        reg = TabPFNv2Regressor().fit(x_train, y_train)
    """

    def __init__(
        self,
        model: nn.Module | str | Path | None = None,
        borders: torch.Tensor | None = None,
        *,
        n_estimators: int | None = None,
        member_kinds: str | tuple[str, ...] | None = None,
        **kwargs: Any,
    ) -> None:
        if isinstance(model, nn.Module):
            if borders is None:
                raise ValueError("모델을 직접 줄 때는 회귀 구간 경계 borders도 준다")
        else:
            if borders is not None:
                raise ValueError("borders는 모델을 직접 줄 때만 준다 (받는 가중치에 경계가 있다)")
            model, borders = _load(model, "regression")
        inference = _inference_kwargs(model, "regression", n_estimators, member_kinds)
        super().__init__(model, borders, **inference, **kwargs)
