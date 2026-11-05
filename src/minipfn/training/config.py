"""학습 설정 (학습 축). 모델·prior·최적화·실행 설정을 한 dataclass에 평평하게 둔다.

평평하게 두는 까닭: MLflow run param이 이 필드 이름 그대로 남고, 평가·Hub 업로드가 그 param으로
모델을 다시 만든다(`minipfn.hub.load_model`). 어떤 모델·prior를 쓸지는 `model_type`·`prior_type`이
정하고, 각 모델·prior는 이 설정에서 자기에게 있는 필드만 읽는다.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Literal

from minipfn.models import (
    DEFAULT_MODEL_TYPE,
    build_model,
    canonical_model_type,
    config_fields,
    model_class,
)
from minipfn.priors import DEFAULT_PRIOR_TYPE, canonical_prior_type, prior_module
from minipfn.priors.table_size import MAX_CELLS, MAX_FEATURES, MAX_TRAIN_ROWS, N_EVAL_ROWS

N_BUCKETS = 5000


@dataclass
class TrainConfig:
    """모델·prior·최적화 설정. 기본값은 TabPFN v2 논문 규모다. 일반 크기 smoke(로컬 CPU)·
    medium(짧은 실측)은 여기에, 논문별 설정(paper, long_tables, 구조)은 `minipfn.recipes`에 있다."""

    task: Literal["classification", "regression"] = "classification"
    # 학습 데이터 종류 (`minipfn.priors.PRIOR_TYPES`). 예전 이름(`PRIOR_TYPE_ALIASES`)은 지금
    # 이름으로 바꾼다
    prior_type: str = DEFAULT_PRIOR_TYPE
    # 모델 클래스 (`minipfn.models.MODEL_TYPES`). 아래 모델 설정 가운데 그 클래스의 설정에 있는 것만
    # 쓴다. 예전 이름(`MODEL_TYPE_ALIASES`)은 지금 이름으로 바꾼다
    model_type: str = DEFAULT_MODEL_TYPE
    # 모델 (논문 기본 모델 크기). 아래 셋은 TabPFN-2.5가 바꾼 것
    # (`recipes.tabpfn_v2.with_architecture(config, "v2.5")`)
    emsize: int = 192
    nlayers: int = 12
    nhead: int = 6
    features_per_group: int = 2
    n_thinking_rows: int = 0
    encoder: str = "linear"
    # prior / 표 크기 (논문 Methods, `minipfn.priors.table_size`). max_features는 모델 설정에도
    # 남아 추론이 읽는다(학습 때 본 feature 수)
    max_train_rows: int = MAX_TRAIN_ROWS
    n_eval_rows: int = N_EVAL_ROWS
    max_features: int = MAX_FEATURES
    # 아래 넷은 논문에 없음 (long_tables에서 쓴다). 기본값이면 논문 크기 규칙 그대로다
    min_train_rows: int = 1
    log_uniform_rows: bool = False  # 학습 행 수를 [min, max]에서 log 균등으로
    max_cells: int = MAX_CELLS  # 표 하나의 셀 수 상한 (검증 행 포함)
    feature_scale: int = MAX_FEATURES  # feature 수 분포를 펼 최대값 (sample_table_size)
    missing_dataset_prob: float = 0.3
    border_datasets: int = 20_000  # 회귀 구간 경계를 정할 때 뽑는 데이터셋 수 (여기서 정함)
    # 최적화 (논문: 약 200만 step × batch 64. warmup 길이와 lr은 여기서 정함)
    batch_size: int = 64
    micro_batch: int = 8
    steps: int = 2_000_000  # 학습률 스케줄의 전체 길이
    stop_step: int | None = None  # 이번 실행을 멈출 step (None이면 steps까지). 나눠 학습할 때 쓴다
    deadline: float | None = None  # 이 시각(Unix time)을 넘기면 checkpoint를 남기고 멈춘다
    warmup_steps: int = 10_000
    # 회귀는 200k 언저리부터 이 lr에서 붕괴가 되풀이됐다. 낮추는 건 rewind_lr_factor 규칙이 맡는다
    lr: float = 2e-4
    grad_clip: float = 1.0
    # 회귀에서 반복되는 target 값의 정답을 이웃 구간 몇 개에 흩을지 (0이면 끔, dequantize_targets).
    # 25k step 비교에서 100이어야 grad norm 증가가 멈췄다(10은 거의 효과 없음)
    dequantize_buckets: float = 100.0
    # 실행
    seed: int = 0
    num_workers: int = 8
    log_every: int = 100
    val_every: int = 5_000
    val_batch: int = 8  # 검증 batch의 데이터셋 수. micro_batch와 따로 둬서 검증 세트가 늘 같다
    checkpoint_every: int = 5_000
    # latest.pt는 덮어써지므로 이 step마다 `step_XXXXXXX.pt`로 따로 남긴다(되돌아갈 지점)
    keep_every: int | None = 25_000
    # 붕괴 감지 (CollapseGuard): log_every마다 낸 train loss 평균이 최근 건강한 평균보다
    # collapse_threshold 넘게 높은 채로 collapse_patience번 이어지면, 메모리에 둔 건강한 상태로
    # 되돌리고 데이터 시드를 바꿔 다시 간다. None이면 끈다. 값은 여기서 정함(논문 규모 run에서
    # 붕괴는 분류 0.8 → 2.0, 회귀 −1.0 → +1.1로 뛰었고, 평소 흔들림은 0.1 안팎이었다)
    collapse_threshold: float | None = 0.5
    collapse_patience: int = 3
    snapshot_every: int = 2_000  # 이 step마다 건강하면 상태를 메모리에 남긴다 (최근 2개)
    max_rewinds: int = 10  # 한 번 실행에서 이보다 많이 되돌려야 하면 멈춘다
    # 붕괴로 되돌릴 때마다 기준 lr에 곱하는 값 (1이면 lr 그대로). 되돌려도 같은 lr이면 붕괴가
    # 되풀이됐고(회귀 run에서 rewind 8~10번), 같은 checkpoint에서 lr만 낮추면 멈췄다(275k에서
    # 35k step 비교: 1e-4는 weight decay를 더해도 3번씩, 5e-5는 0번). 사람이 중간에 lr을 고치는 대신
    # 미리 정한 규칙으로 낮춘다. warmup·cosine 모양은 그대로이고 다음 chunk에도 이어진다
    rewind_lr_factor: float = 1.0
    # 규칙으로는 기준 lr을 이보다 낮추지 않는다(그 뒤로는 되돌리기만 한다). 2e-4에서 세 번 절반,
    # 5e-5가 안정했던 것의 절반까지. 이어서 학습할 때 TABPFN_V2_LR로 준 lr은 이 값과 상관없다
    rewind_min_lr: float = 2.5e-5
    # QK-clip (논문에 없음, 여기서 정함. Kimi K2의 MuonClip과 같은 생각): optimizer step마다
    # head별 attention logit의 상한(`qk_logit_bounds`)이 이 값을 넘으면 그 head의 Wq·Wk를 줄여
    # 상한을 이 값에 맞춘다. 학습 절차만 바뀌고 모델 구조·추론은 그대로다. None이면 끈다.
    # 논문 규모 회귀에서 sample attention의 상한이 수만~수십만으로 커지며(정상 수백 이하) 행 하나에
    # 몰리는 attention sink가 됐고, 그 뒤 학습이 불안정해져 benchmark가 크게 나빠졌다
    qk_clip: float | None = None
    # 상한 계산(head마다 작은 QR·SVD)은 GPU와 CPU를 맞추느라 step마다 하면 step 시간이 1.7배가 된다.
    # 그래서 이 step마다만 한다. 폭주할 때도 상한은 수천 step에 걸쳐 커져서 50 step 사이에 넘는 양은
    # 몇 % 안팎이다
    qk_clip_every: int = 50

    def __post_init__(self) -> None:
        self.model_type = canonical_model_type(self.model_type)
        self.prior_type = canonical_prior_type(self.prior_type)
        model_class(self.model_type)  # 없는 이름이면 ValueError
        prior_module(self.prior_type).check_config(self)  # 없는 이름이거나 크기 설정이 틀리면
        if not 0.0 < self.rewind_lr_factor <= 1.0:
            raise ValueError(f"rewind_lr_factor는 0 초과 1 이하여야 한다: {self.rewind_lr_factor}")
        if self.qk_clip is not None and self.qk_clip <= 0:
            raise ValueError(f"qk_clip은 0보다 커야 한다: {self.qk_clip}")
        if self.qk_clip_every < 1:
            raise ValueError(f"qk_clip_every는 1 이상이어야 한다: {self.qk_clip_every}")

    @property
    def n_out(self) -> int:
        # 분류는 prior가 만드는 클래스 수 상한, 회귀는 bar distribution 구간 수
        if self.task == "classification":
            return prior_module(self.prior_type).MAX_CLASSES
        return N_BUCKETS

    @classmethod
    def medium(cls, task: str = "classification") -> TrainConfig:
        """논문 모델·prior·batch 그대로, step만 1/100. step 시간과 학습 곡선을 재는 용도."""
        return cls(
            task=task,
            steps=20_000,
            warmup_steps=1_000,
            val_every=1_000,
            checkpoint_every=1_000,
            keep_every=5_000,
            border_datasets=5_000,
        )

    @classmethod
    def smoke(cls, task: str = "classification") -> TrainConfig:
        """로컬 CPU에서 1분 안에 끝나는 크기. 코드 경로 확인용."""
        return cls(
            task=task,
            emsize=32,
            nlayers=2,
            nhead=2,
            max_train_rows=64,
            n_eval_rows=16,
            max_features=8,
            border_datasets=40,
            batch_size=8,
            micro_batch=4,
            steps=20,
            warmup_steps=2,
            lr=1e-3,
            num_workers=0,
            log_every=5,
            val_every=10,
            checkpoint_every=10,
            keep_every=10,
        )

    def model_config(self):
        """`model_type` 모델의 설정. 그 설정에 있는 필드 가운데 여기에도 있는 것을 옮기고, 나머지는
        그 설정의 기본값이다."""
        mine = {f.name for f in fields(self)}
        values = {n: getattr(self, n) for n in config_fields(self.model_type) if n in mine}
        return model_class(self.model_type).config_class(n_out=self.n_out, **values)

    def build_model(self):
        return build_model(self.model_type, self.model_config())

    def for_world_size(self, world_size: int) -> TrainConfig:
        """GPU가 많아 micro_batch × world_size가 batch를 넘으면 GPU당 micro batch를 줄인다."""
        if self.micro_batch * world_size <= self.batch_size:
            return self
        return replace(self, micro_batch=self.batch_size // world_size)

    def grad_accumulation(self, world_size: int) -> int:
        per_forward = self.micro_batch * world_size
        if self.batch_size % per_forward:
            raise ValueError(
                f"batch_size {self.batch_size}가 micro_batch × world_size = {per_forward}로"
                " 나눠떨어지지 않는다"
            )
        return self.batch_size // per_forward

    def make_prior(self, seed: int):
        """`prior_type` prior를 이 설정의 데이터셋 크기로 만든다."""
        return prior_module(self.prior_type).make_prior(self, seed)
