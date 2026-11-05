"""TabPFN v2(Nature 2025)·TabPFN-2.5(arXiv 2511.08667) 학습 설정 (논문 = 4축 설정 조합).

- prior: `tabicl` (TabICLv2 graph prior + v2 논문 크기 규칙)
- 모델: `tabpfn_v2` 클래스. 구조는 `ARCHITECTURES`의 v2 또는 v2.5
- 학습: `minipfn.training.loop.train`, 논문 규모 설정은 `paper`, 긴 표 이어 학습은 `long_tables`
- 추론: `minipfn.inference.tabpfn_v2` (멤버 전처리 v2 또는 2.5)

    config = with_architecture(paper("regression"), "v2.5")
"""

from __future__ import annotations

from dataclasses import replace

from minipfn.training.config import TrainConfig

# 모델 구조. v2는 Nature 논문 기본 모델, v2.5는 TabPFN-2.5(arXiv 2511.08667 "Architecture")가
# 적은 변경: 층 수(회귀 18, 분류 24), 셀 3개 묶기, thinking 행 64개, 회귀 encoder를 2층 MLP로.
# emsize·head 수는 2.5 논문에 없어서 v2 그대로 둔다. 둘 다 모델 클래스는 TabPFNv2다
_V2 = {
    "model_type": "tabpfn_v2",
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


def architecture_name(config: TrainConfig) -> str | None:
    """`config`의 모델 구조가 `ARCHITECTURES`의 어느 것과 같은지 (없으면 None)."""
    for name, by_task in ARCHITECTURES.items():
        if all(getattr(config, k) == v for k, v in by_task[config.task].items()):
            return name
    return None


def paper(task: str = "classification") -> TrainConfig:
    # 분류는 2e-4에서 붕괴가 드물어(300k까지 rewind 1번) lr을 그대로 둔다.
    # 회귀의 QK-clip 250: 50k scratch 비교에서 250·500 모두 상한을 지켰고 val NLL도 비슷했다
    # (−1.08·−1.07, 끈 것 −1.04). 그중 logit을 더 낮게 묶는 쪽. 100은 10k부터 잘라 학습이 늦었다
    if task == "regression":
        return TrainConfig(task=task, rewind_lr_factor=0.5, qk_clip=250.0)
    return TrainConfig(task=task)


def long_tables(task: str = "classification") -> TrainConfig:
    """paper로 학습한 모델을 더 긴·넓은 표로 짧게 이어 학습한다 (논문에 없음, 값은 여기서 정함).

    TabPFN-2.5는 prior를 "더 많은 행과 feature로 키웠다"고만 적었다(추론 권장 크기는 행 50,000,
    feature 2,000). 그 크기로 처음부터 학습하는 대신 TabICL처럼 짧은 표로 오래 학습한 뒤 긴 표로
    잠깐 더 학습한다: 학습 행 512~8,192개 log 균등, feature 수 분포를 500까지 펴고, 셀은
    400,000개까지. 가중치만 가져와(`init_from`) 새 optimizer로 lr 5e-5, warmup 1k + cosine
    20k step. v2(12층)는 GPU당 micro batch 2에서 GPU 메모리 52.6GB였는데 2.5는 층이 두 배라
    micro batch 1로 둔다. 검증 batch는 데이터셋 4개다.
    """
    return replace(
        paper(task),
        min_train_rows=512,
        max_train_rows=8192,
        log_uniform_rows=True,
        max_features=500,
        feature_scale=500,
        max_cells=400_000,
        micro_batch=1,
        val_batch=4,
        steps=20_000,
        warmup_steps=1_000,
        lr=5e-5,
        keep_every=5_000,
    )


def with_architecture(config: TrainConfig, arch: str, keep_size: bool = False) -> TrainConfig:
    """모델 구조를 `ARCHITECTURES`의 것으로. `keep_size`(smoke)면 층 수는 그대로 두고 thinking
    행은 4개까지만 써서 작은 크기를 유지한다(구조가 바뀌는 코드 경로만 확인한다)."""
    if arch not in ARCHITECTURES:
        raise ValueError(f"arch는 {tuple(ARCHITECTURES)} 가운데 하나여야 한다: {arch}")
    changes = dict(ARCHITECTURES[arch][config.task])
    if keep_size:
        changes["nlayers"] = config.nlayers
        if "n_thinking_rows" in changes:
            changes["n_thinking_rows"] = min(changes["n_thinking_rows"], 4)
    return replace(config, **changes)
