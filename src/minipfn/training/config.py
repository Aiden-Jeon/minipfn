"""학습 설정 (학습 축). 모델·prior·최적화·실행 설정을 한 dataclass에 평평하게 둔다.

평평하게 두는 까닭: MLflow run param이 이 필드 이름 그대로 남고, 평가·Hub 업로드가 그 param으로
모델을 다시 만든다(`minipfn.hub.load_model`). 어떤 모델·prior를 쓸지는 `model_type`·`prior_type`이
정하고, 각 모델·prior는 이 설정에서 자기에게 있는 필드만 읽는다.
"""

from __future__ import annotations

import ast
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
# 학습 가속 옵션 (issue #46)과 optimizer 종류 (issue #28). 이어 학습하면 원래 run의 값을 따른다
# (학습 스크립트, `check_resume_step`)
SPEED_OPTIONS = (
    "compile",
    "fused_adam",
    "tf32",
    "schedule_free",
    "weight_average",
    "weight_average_every",
    "optimizer",
    "muon_weight_decay",
    "muon_distributed",
    "phases",
    "shared_table_size",
)
OPTIMIZERS = ("adam", "muon")
# step 구간별로 바꿀 수 있는 설정 (`TrainConfig.phases`)
PHASE_KEYS = ("lr", "muon_weight_decay")
# 분산 Muon의 범위 (issue #56, `muon_distributed`)
MUON_DISTRIBUTED = ("off", "node", "all")
# 한 task 학습과, 분류·회귀를 한 모델로 같이 학습하는 multitask (`task_configs`)
TASKS = ("classification", "regression", "multitask")


@dataclass
class TrainConfig:
    """모델·prior·최적화 설정. 기본값은 TabPFN v2 논문 규모다. 일반 크기 smoke(로컬 CPU)·
    medium(짧은 실측)은 여기에, 논문별 설정(paper, long_tables, 구조)은 `minipfn.recipes`에 있다."""

    # multitask: 분류·회귀를 한 모델로 (그런 모델 클래스만. 학습 루프가 step마다 task를 번갈아 쓴다)
    task: Literal["classification", "regression", "multitask"] = "classification"
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
    # column_row_transformer(TabPFN-3·3.5, `recipes.tabpfn_v3`)에만 있는 구조 설정. emsize·
    # nlayers·nhead·features_per_group·encoder는 위 필드를 같이 쓴다(그 모델에서 nlayers·nhead는
    # 행 ICL의 것)
    n_inducing: int = 128
    n_cls: int = 4
    qk_norm: bool = False
    target_norm: bool = False
    head_mlp: bool = False
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
    # 학습 가속 (nanoTabPFN speedrun, arXiv 2606.03681에서 가져온 것. issue #46). 기본값은 모두
    # 꺼져 있어 예전 학습과 비트 단위로 같다
    # - compile: 학습 forward를 torch.compile(dynamic=True)로. 표 크기가 batch마다 달라 shape는
    #   동적으로 둔다
    # - fused_adam: CUDA fused Adam (한 kernel로 모든 파라미터를 갱신)
    # - tf32: float32 matmul을 TF32로 (bf16 autocast 밖의 matmul. `set_float32_matmul_precision`)
    compile: bool = False
    fused_adam: bool = False
    tf32: bool = False
    # schedule-free AdamW (Defazio et al. 2024, speedrun의 Adam 쪽): cosine 대신 warmup 뒤 일정한
    # lr로 가고, 가중치의 평균을 optimizer가 들고 있다가 검증·저장할 때 그 평균을 쓴다. 학습 길이를
    # 미리 정하지 않아도 되어 chunk로 나눠 학습하기 쉽다. β1은 0.9, weight decay는 0
    schedule_free: bool = False
    # LAWA(latest weight averaging, speedrun record 7): 이 step마다 남긴 가중치 최근 K개를 평균해
    # 검증(`val_*_avg`)하고 마지막에 `final_avg.pt`로 저장한다. 학습 자체는 바꾸지 않는다.
    # 0이면 끈다
    weight_average: int = 0
    weight_average_every: int = 500
    # optimizer (issue #28): adam(기본, 논문) 또는 muon(블록 안 가중치 행렬은 Muon, 나머지는 Adam,
    # `training.muon`). muon은 Adam과 같은 lr·스케줄을 쓴다(갱신 크기를 AdamW RMS에 맞춘다).
    # muon_weight_decay는 Muon 묶음의 decoupled weight decay (Adam 묶음은 지금처럼 없음)
    optimizer: Literal["adam", "muon"] = "adam"
    muon_weight_decay: float = 0.0
    # 분산 Muon (issue #56, `training.muon`): Newton–Schulz 직교화를 GPU마다 똑같이 계산하지 않고
    # GPU끼리 행렬 조각을 나눠 맡은 뒤 all-gather로 모은다. "node"는 노드 안 GPU끼리, "all"은 모든
    # GPU끼리 나눈다. 갱신 결과는 "off"(기본, 모든 GPU가 다 계산)와 비트 단위로 같고 checkpoint
    # 형식도 같아서 이어 학습할 때 켜거나 끌 수 있다
    muon_distributed: Literal["off", "node", "all"] = "off"
    # step 구간별 설정 변경 (issue #29):
    # `((시작 step, {"lr": ..., "muon_weight_decay": ...}), ...)`.
    # 시작 step의 갱신부터 그 값을 쓰고, 첫 구간 전에는 위 `lr`·`muon_weight_decay` 필드 값이다.
    # 구간에 없는 설정은 앞 구간의 값을 잇는다. lr은 기준 lr만 바꾸고 warmup·cosine 모양은 그대로다.
    # 붕괴로 낮춘 lr 배율(`rewind_lr_factor`)은 구간이 바뀌어도 이어진다. 논문 규모 학습 도중 바꾼
    # 설정을 레시피에 남겨, 명령 하나로 같은 경로를 밟게 한다(`recipes.tabpfn_v3.paper_run`).
    # MLflow param의 문자열로도 받는다 (`normalize_phases`)
    phases: tuple = ()
    # 학습 진단 지표 (`training.diagnostics`). 학습 계산은 바꾸지 않는다(켜도 비트 단위로 같다).
    # 묶음별 가중치·gradient norm(issue #65)은 늘 낸다
    # - grad_cosine (issue #64): multitask에서 log_every마다 앞 4 step의 gradient로 두 task
    #   gradient의 묶음별 cosine을 낸다. rank 0에 gradient 3 step치(bf16)를 잠깐 더 둔다
    # - rank_timing (issue #68): GPU마다 forward·backward 시간과 표 크기를 재서 log_every마다
    #   rank 0으로 모은다(지표 `rank_*`, step마다 한 줄인 `rank_timing.jsonl`)
    grad_cosine: bool = False
    rank_timing: bool = False
    # 표 크기를 step마다 GPU끼리 맞춘다 (issue #68, 학습 가속 옵션. 논문에 없음, 여기서 정함).
    # 끄면(기본) GPU마다 표 크기를 따로 뽑아, 가장 큰 표를 받은 GPU를 나머지가 all-reduce에서
    # 기다린다(8 GPU에서 셀 수 최대/평균 약 2.7배, forward 최대/평균 약 1.4배). 켜면 i번째 batch의
    # 학습 행·feature 수가 모든 GPU에서 같고 표 내용만 다르다(`training.data.PriorBatches`).
    # 학습 전체의 크기 분포는 같고, 한 step 안의 크기가 GPU 수만큼에서 하나로 준다
    shared_table_size: bool = False

    def __post_init__(self) -> None:
        self.phases = normalize_phases(self.phases)
        self.model_type = canonical_model_type(self.model_type)
        self.prior_type = canonical_prior_type(self.prior_type)
        model_class(self.model_type)  # 없는 이름이면 ValueError
        if self.task not in TASKS:
            raise ValueError(f"task는 {TASKS} 가운데 하나여야 한다: {self.task}")
        for config in self.task_configs().values():
            prior_module(self.prior_type).check_config(config)  # 없는 이름이거나 크기가 틀리면
        if not 0.0 < self.rewind_lr_factor <= 1.0:
            raise ValueError(f"rewind_lr_factor는 0 초과 1 이하여야 한다: {self.rewind_lr_factor}")
        if self.qk_clip is not None and self.qk_clip <= 0:
            raise ValueError(f"qk_clip은 0보다 커야 한다: {self.qk_clip}")
        if self.qk_clip_every < 1:
            raise ValueError(f"qk_clip_every는 1 이상이어야 한다: {self.qk_clip_every}")
        if self.schedule_free and self.qk_clip is not None:
            # QK-clip은 학습 중 가중치(y)를 줄이는데, schedule-free는 평균 가중치(x)를 optimizer
            # 상태에서 다시 만든다. 둘을 같이 쓰는 방식은 정하지 않았다
            raise ValueError("schedule_free와 qk_clip은 같이 쓰지 않는다")
        if self.schedule_free and self.fused_adam:
            raise ValueError("fused_adam은 Adam에만 있다 (schedule_free와 같이 쓰지 않는다)")
        if self.optimizer not in OPTIMIZERS:
            raise ValueError(f"optimizer는 {OPTIMIZERS} 가운데 하나여야 한다: {self.optimizer}")
        if self.optimizer == "muon" and self.schedule_free:
            raise ValueError("schedule_free는 Adam 쪽 방법이다 (optimizer=muon과 같이 쓰지 않는다)")
        if self.muon_weight_decay < 0 or (self.muon_weight_decay and self.optimizer != "muon"):
            raise ValueError("muon_weight_decay는 0 이상이고 optimizer=muon일 때만 쓴다")
        if self.muon_distributed not in MUON_DISTRIBUTED:
            raise ValueError(
                f"muon_distributed는 {MUON_DISTRIBUTED} 가운데 하나여야 한다:"
                f" {self.muon_distributed}"
            )
        if self.muon_distributed != "off" and self.optimizer != "muon":
            raise ValueError("muon_distributed는 optimizer=muon일 때만 쓴다")
        if self.weight_average < 0 or self.weight_average_every < 1:
            raise ValueError("weight_average는 0 이상, weight_average_every는 1 이상이어야 한다")
        if self.grad_cosine and self.task != "multitask":
            raise ValueError("grad_cosine은 두 task의 gradient를 견준다 (multitask에서만 쓴다)")
        if self.grad_cosine and self.log_every < 4:
            raise ValueError("grad_cosine은 log_every마다 앞 4 step을 쓴다 (log_every 4 이상)")
        self._check_phases()

    def _check_phases(self) -> None:
        starts = [start for start, _ in self.phases]
        if any(b <= a for a, b in zip(starts, starts[1:], strict=False)) or any(
            s < 0 for s in starts
        ):
            raise ValueError(f"phases의 시작 step은 0 이상이고 커지는 순서여야 한다: {starts}")
        if starts and starts[-1] >= self.steps:
            # 닿지 않는 구간을 받아 두면 run에는 구간이 있다고 남는데 실제로는 바뀌지 않는다
            raise ValueError(f"phases의 시작 step은 steps({self.steps})보다 작아야 한다: {starts}")
        for start, changes in self.phases:
            settings = dict(changes)
            if not settings or set(settings) - set(PHASE_KEYS):
                raise ValueError(f"phases는 {PHASE_KEYS}만 바꾼다 (step {start}: {settings})")
            if settings.get("lr", 1.0) <= 0:
                raise ValueError(f"phases의 lr은 0보다 커야 한다 (step {start})")
            if "lr" in settings and self.schedule_free:
                raise ValueError("schedule_free는 phases로 lr을 바꾸지 않는다")
            wd = settings.get("muon_weight_decay", 0.0)
            if wd < 0 or (wd and self.optimizer != "muon"):
                raise ValueError(
                    "phases의 muon_weight_decay는 0 이상이고 optimizer=muon일 때만 쓴다"
                    f" (step {start})"
                )

    def phase_value(self, name: str, step: int) -> float:
        """step번째 갱신(0부터)에 쓰는 `name`(`PHASE_KEYS`)의 값. 시작 step이 `step` 이하인 마지막
        구간 가운데 그 설정이 있는 것, 없으면 필드 값."""
        value = getattr(self, name)
        for start, changes in self.phases:
            if start > step:
                break
            value = dict(changes).get(name, value)
        return value

    @property
    def n_out(self) -> int:
        # 분류(multitask면 그 분류 쪽)는 prior가 만드는 클래스 수 상한, 회귀는 bar distribution
        # 구간 수. multitask 모델의 회귀 구간 수는 모델 설정의 n_buckets(= N_BUCKETS)다
        if self.task in ("classification", "multitask"):
            return prior_module(self.prior_type).MAX_CLASSES
        return N_BUCKETS

    @property
    def tasks(self) -> tuple[str, ...]:
        """학습하는 task. multitask면 (분류, 회귀)."""
        return ("classification", "regression") if self.task == "multitask" else (self.task,)

    def task_configs(self) -> dict[str, TrainConfig]:
        """task마다 그 task 하나의 설정 (prior·회귀 구간 경계·검증 batch를 만든다). 한 task면
        자기 자신."""
        if self.task != "multitask":
            return {self.task: self}
        # task 하나의 설정에는 두 task를 견주는 grad_cosine이 없다
        return {task: replace(self, task=task, grad_cosine=False) for task in self.tasks}

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
        if self.task == "multitask" and "task" not in config_fields(self.model_type):
            # 모델 설정에 task가 없는 모델(cell_transformer 등)은 한 task 출력뿐이다. 설정은 구조를
            # 바꾸기 전(기본 cell_transformer)에도 만들 수 있게 두고, 모델을 만들 때 막는다
            raise ValueError(f"{self.model_type} 모델은 multitask로 학습할 수 없다")
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
        """`prior_type` prior를 이 설정의 데이터셋 크기로 만든다. multitask는 task마다 prior가
        따로라 `task_configs()`의 설정으로 만든다."""
        if self.task == "multitask":
            raise ValueError("multitask 설정의 prior는 task_configs()의 task마다 만든다")
        return prior_module(self.prior_type).make_prior(self, seed)


def normalize_phases(phases) -> tuple:
    """`TrainConfig.phases`를 `((시작 step, (("lr", 값), ...)), ...)` 꼴로 (설정 이름 순서, 값은
    float). `{시작 step: {설정: 값}}` dict나 (시작 step, dict) 목록으로 써도 되고, MLflow
    param·tag에 남은 문자열(`str(phases)`)도 읽는다."""
    if isinstance(phases, str):
        phases = ast.literal_eval(phases) if phases.strip() else ()
    if isinstance(phases, dict):  # {시작 step: {설정: 값}}
        phases = sorted(phases.items())
    out = []
    for start, changes in phases or ():
        items = changes.items() if isinstance(changes, dict) else changes
        out.append((int(start), tuple(sorted((str(k), float(v)) for k, v in items))))
    return tuple(out)
