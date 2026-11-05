"""TabPFN v1(arXiv:2207.01848) recipe: 논문 수치와 출처, 그리고 그 수치로 각 축의 부품을 묶은 것.

- prior: `minipfn.priors.prior_bag.PriorBag` + 여기의 `PRIOR` 설정 → `TabPFNPrior`
- 모델: `minipfn.models.row_transformer.RowTransformer` (논문 크기 `PAPER_MODEL`)
- 학습: 여기의 `train_tabpfn` (v1 `train.py`의 루프, 설정은 `TabPFNConfig`)
- 추론: `minipfn.inference.rotation_ensemble` + 논문 앙상블 설정 → `TabPFNClassifier`
- 평가: `minipfn.benchmarks.cc18` (Table 1)

값은 논문 본문·부록과 공식 v1 코드(github.com/automl/TabPFN, `tabpfn_v1` 브랜치, commit bef0cf2,
Apache 2.0, Copyright 2022 University of Freiburg)에서 가져왔다. 출처 표기:
- "논문 E.1" 등: 논문 절
- "v1 PriorFittingCustomPrior": 공개 체크포인트를 학습한 설정이 적힌 노트북 셀
- "v1 model_configs": `tabpfn/scripts/model_configs.py`의 prior 설정
- "추정": 직접 확인하지 못해 가장 근거 있는 값을 고른 것
논문 표와 v1 코드가 다르면 체크포인트를 만든 v1 코드를 따른다(차이는 주석에 남긴다).

학습 루프 (논문 3절, v1 `train.py`): 매 optimizer step마다 prior에서 데이터셋 `batch_size`개를
GPU들에 나눠 만들고, forward batch마다 학습 행 / 테스트 행으로 나눠 테스트 행의 cross-entropy를
줄인다. GPU가 여러 장이면
`torchrun`으로 띄운 DDP로 gradient를 평균한다. 유효 batch는 v1과 같이

    batch_size = group_size × groups_per_batch × grad_accumulation × world_size

이고, grad_accumulation은 나머지 값으로 자동 계산한다. 학습률은 v1처럼 linear warmup 뒤
cosine으로 줄이되, cosine 길이(schedule_steps)와 실제 학습 step 수(steps)를 따로 둔다
(논문은 400 epoch 스케줄의 약 141 epoch 지점인 18,000 step에서 멈췄다).
"""

from __future__ import annotations

import contextlib
import json
import shutil
import time
from dataclasses import asdict, dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from minipfn.inference import rotation_ensemble
from minipfn.inference.rotation_ensemble import FitMode, RotationEnsembleClassifier
from minipfn.models.row_transformer import RowTransformer, configure_attention_backends
from minipfn.priors.prior_bag import PriorBag, PriorBagConfig
from minipfn.priors.tabular import IGNORE_INDEX
from minipfn.training.data import PriorGroups
from minipfn.training.losses import classification_loss
from minipfn.utils import lr_factor, setup_distributed

# ---------------------------------------------------------------------------
# 모델 (논문 E.1, v1 PriorFittingCustomPrior)
# ---------------------------------------------------------------------------
EMSIZE = 512
NLAYERS = 12
NHEAD = 4  # emsize // 128
NHID = 1024  # emsize * nhid_factor(2)
DROPOUT = 0.0
MAX_FEATURES = 100  # 부족한 feature는 0으로 채우고 100/k로 스케일 (논문 E.2)
MAX_CLASSES = 10  # 출력 logit 수
PAPER_N_PARAMETERS = 25_820_000  # 논문 E.1: 25.82M

# ---------------------------------------------------------------------------
# 학습 (논문 3절·E.3, v1 PriorFittingCustomPrior + train.py)
# ---------------------------------------------------------------------------
SEQ_LEN = 1152  # bptt = 1024 + 128 (v1)
# single_eval_pos(학습 행 수) ~ Uniform{1, ..., 999}.
# v1은 0부터 뽑는다 (prior.TabPFNPriorDataset 참고).
MAX_EVAL_POS = 1000
# 한 forward batch의 데이터셋 8개가 prior 하이퍼파라미터를 공유한다 (v1)
DATASETS_PER_PRIOR_GROUP = 8
GRAD_ACCUMULATION = 8  # aggregate_k_gradients (v1)
BATCH_SIZE = 512  # optimizer step당 데이터셋 수 = 8 × 8 × GPU 8장 (논문 3절)
TOTAL_STEPS = 18_000  # 논문 3절: 18,000 step × batch 512
STEPS_PER_EPOCH = 128  # num_steps = 1024 // aggregate_k_gradients (v1)
WARMUP_STEPS = 20 * STEPS_PER_EPOCH  # warmup_epochs = 20 (v1 model_builder)
SCHEDULE_STEPS = (
    400 * STEPS_PER_EPOCH
)  # cosine은 400 epoch 기준. 논문은 약 141 epoch(18,000 step)에서 멈췄다.
LR = 1.1555e-4  # 추정: v1 노트북 출력의 lr (탐색 범위 log-uniform [1e-4, 1.5e-4])
WEIGHT_DECAY = 0.0  # AdamW (v1 train.py)
GRAD_CLIP = 1.0  # v1 train.py
# v1은 fp16 + GradScaler. H100에서는 bf16 autocast를 쓴다 (차이: 수치 형식만).

# ---------------------------------------------------------------------------
# Prior (v1 model_configs.get_diff_causal 등, PriorFittingCustomPrior에서 덮어쓴 값 반영)
# 논문 E.4 표와 다른 점: 활성화 함수 후보가 v1에서는 {Tanh, Identity, ReLU}다
# (논문 표: Tanh, LeakyReLU, ELU, Identity).
# ---------------------------------------------------------------------------
# prior bag: softmax([1, w])로 GP 또는 MLP prior를 고른다. w ~ U(2, 10)이라 대부분 MLP다.
PRIOR_BAG_MLP_WEIGHT = {"distribution": "uniform", "min": 2.0, "max": 10.0}

# MLP prior: is_causal=True면 SCM prior, False면 BNN prior (논문 4.3·4.4).
# 둘의 비율은 meta_choice 가중치로 정해진다 (평균 50/50).
MLP_HYPERPARAMETERS = {
    "num_layers": {
        "distribution": "meta_gamma",
        "max_alpha": 2,
        "max_scale": 3,
        "round": True,
        "lower_bound": 2,
    },
    "prior_mlp_hidden_dim": {
        "distribution": "meta_gamma",
        "max_alpha": 3,
        "max_scale": 100,
        "round": True,
        "lower_bound": 4,
    },
    "prior_mlp_dropout_prob": {"distribution": "meta_beta", "scale": 0.6, "min": 0.1, "max": 5.0},
    "noise_std": {
        "distribution": "meta_trunc_norm_log_scaled",
        "max_mean": 0.3,
        "min_mean": 1e-4,
        "round": False,
        "lower_bound": 0.0,
    },
    "init_std": {
        "distribution": "meta_trunc_norm_log_scaled",
        "max_mean": 10.0,
        "min_mean": 0.01,
        "round": False,
        "lower_bound": 0.0,
    },
    "num_causes": {
        "distribution": "meta_gamma",
        "max_alpha": 3,
        "max_scale": 7,
        "round": True,
        "lower_bound": 2,
    },
    "is_causal": {"distribution": "meta_choice", "choice_values": [True, False]},
    "pre_sample_weights": {"distribution": "meta_choice", "choice_values": [True, False]},
    "y_is_effect": {"distribution": "meta_choice", "choice_values": [True, False]},
    "prior_mlp_activations": {
        "distribution": "meta_choice_mixed",
        "choice_values": ["tanh", "identity", "relu"],
    },
    "block_wise_dropout": {"distribution": "meta_choice", "choice_values": [True, False]},
    "sort_features": {"distribution": "meta_choice", "choice_values": [True, False]},
    "in_clique": {"distribution": "meta_choice", "choice_values": [True, False]},
}
PRE_SAMPLE_CAUSES = True  # 원인 노드의 평균·표준편차를 데이터셋마다 미리 뽑는다
PRIOR_MLP_SCALE_WEIGHTS_SQRT = True
RANDOM_FEATURE_ROTATION = True

# GP prior (prior bag의 나머지)
GP_HYPERPARAMETERS = {
    "outputscale": {
        "distribution": "meta_trunc_norm_log_scaled",
        "max_mean": 10.0,
        "min_mean": 1e-5,
        "round": False,
        "lower_bound": 0.0,
    },
    "lengthscale": {
        "distribution": "meta_trunc_norm_log_scaled",
        "max_mean": 10.0,
        "min_mean": 1e-5,
        "round": False,
        "lower_bound": 0.0,
    },
    "noise": {"distribution": "meta_choice", "choice_values": [1e-5, 1e-4, 1e-2]},
}

# 표 데이터 처리 (v1 flexible_categorical, PriorFittingCustomPrior)
NUM_FEATURES_USED = (1, MAX_FEATURES)  # round(U(1, 100))
NUM_CLASSES = (2, MAX_CLASSES)  # 상한 round(U(2, 10)), 그 뒤 50% 확률로 2클래스 (class_sampler_f)
CATEGORICAL_FEATURE_P = 0.2  # 그룹의 20%에서 일부 열을 범주형으로 바꾼다
CATEGORICAL_ORDERED_P = 0.3
NAN_PROB = 0.0  # 공개 체크포인트는 결측값 주입을 끄고 학습했다 (nan_prob_* = 0)
OUTPUT_MULTICLASS_ORDERED_P = 0.0  # 클래스 번호는 항상 섞는다
NORMALIZE_BY_USED_FEATURES = True
CHECK_IS_COMPATIBLE = (
    True  # 학습/테스트 행의 클래스 집합이 같아야 한다. 아니면 행을 섞고, 10번 실패하면 무시(-100)
)
# IGNORE_INDEX(-100, 무시할 target)는 `minipfn.priors.tabular`의 것을 쓴다

# ---------------------------------------------------------------------------
# 평가 (논문 5.2절·부록 F, v1 RunFullDatasetAnalyses, tabular_evaluation,
# transformer_prediction_interface)
# ---------------------------------------------------------------------------
# OpenML-CC18에서 샘플 ≤2000, feature ≤100, 클래스 ≤10인 30개 (v1 datasets/__init__.py open_cc_dids)
OPENML_CC18_TEST_DIDS = [
    11,
    14,
    15,
    16,
    18,
    22,
    23,
    29,
    31,
    37,
    50,
    54,
    188,
    458,
    469,
    1049,
    1050,
    1063,
    1068,
    1510,
    1494,
    1480,
    1462,
    1464,
    6332,
    23381,
    40966,
    40982,
    40994,
    40975,
]
# 논문 Table 1은 이 중 수치형 feature만 있고 결측값이 없는 18개다.
# 평가할 때 OpenML 메타데이터로 걸러낸다.
TABLE1_N_DATASETS = 18
EVAL_SEQ_LEN = 2000  # 학습 + 테스트 행
EVAL_TRAIN_SIZE = 1000  # eval_position
EVAL_MAX_SAMPLES = 10_000  # 로드 시 seed 13으로 섞은 뒤 자른다
N_SPLITS = 5
N_ENSEMBLE = 32  # 논문 5.2절
SOFTMAX_TEMPERATURE = 0.8  # logits / 0.8 (v1)
ROC_AUC_MULTI_CLASS = "ovo"  # v1 tabular_metrics.auc_metric

# 논문 Table 1 (18개 데이터셋, 평균 ROC AUC)
PAPER_TABLE1_ROC_AUC = {
    "TabPFN": 0.934,
    "TabPFN (no ensemble)": 0.932,
    "AutoGluon": 0.930,
    "Auto-sklearn 2.0": 0.929,
    "XGBoost": 0.924,
    "CatBoost": 0.924,
    "LightGBM": 0.920,
}


# 위 수치를 prior bag 설정으로 (`minipfn.priors.prior_bag`)
PRIOR = PriorBagConfig(
    mlp_hyperparameters=MLP_HYPERPARAMETERS,
    gp_hyperparameters=GP_HYPERPARAMETERS,
    mlp_weight=PRIOR_BAG_MLP_WEIGHT,
    min_classes=NUM_CLASSES[0],
    categorical_feature_p=CATEGORICAL_FEATURE_P,
    categorical_ordered_p=CATEGORICAL_ORDERED_P,
    output_multiclass_ordered_p=OUTPUT_MULTICLASS_ORDERED_P,
    normalize_by_used_features=NORMALIZE_BY_USED_FEATURES,
    check_is_compatible=CHECK_IS_COMPATIBLE,
    scale_weights_sqrt=PRIOR_MLP_SCALE_WEIGHTS_SQRT,
    random_feature_rotation=RANDOM_FEATURE_ROTATION,
)

# 논문 크기 모델 (`RowTransformer(**PAPER_MODEL)`, 25.82M 파라미터)
PAPER_MODEL = {
    "num_features": MAX_FEATURES,
    "n_out": MAX_CLASSES,
    "emsize": EMSIZE,
    "nhead": NHEAD,
    "nhid": NHID,
    "nlayers": NLAYERS,
    "dropout": DROPOUT,
}


class TabPFNPrior(PriorBag):
    """v1 논문 prior(`PRIOR`)와 데이터셋 크기를 기본값으로 둔 prior bag."""

    def __init__(
        self,
        seq_len: int = SEQ_LEN,
        num_features: int = MAX_FEATURES,
        max_classes: int = MAX_CLASSES,
        group_size: int = DATASETS_PER_PRIOR_GROUP,
        device: torch.device | str = "cpu",
        seed: int = 0,
    ) -> None:
        super().__init__(
            PRIOR,
            seq_len=seq_len,
            num_features=num_features,
            max_classes=max_classes,
            group_size=group_size,
            device=device,
            seed=seed,
        )


# ---------------------------------------------------------------------------
# 학습 (논문 3절, v1 train.py)
# ---------------------------------------------------------------------------


@dataclass
class TabPFNConfig:
    """모델·prior·최적화 설정. smoke(로컬 CPU), medium(GPU 1장, 예산 1/10), paper(논문)."""

    # 모델
    emsize: int = EMSIZE
    nhead: int = NHEAD
    nhid: int = NHID
    nlayers: int = NLAYERS
    dropout: float = DROPOUT
    num_features: int = MAX_FEATURES
    n_out: int = MAX_CLASSES
    # prior / 데이터셋 구성
    seq_len: int = SEQ_LEN
    max_eval_pos: int = MAX_EVAL_POS
    group_size: int = DATASETS_PER_PRIOR_GROUP
    groups_per_batch: int = 1
    # 최적화
    batch_size: int = BATCH_SIZE
    steps: int = TOTAL_STEPS
    warmup_steps: int = WARMUP_STEPS
    schedule_steps: int = SCHEDULE_STEPS
    lr: float = LR
    weight_decay: float = WEIGHT_DECAY
    grad_clip: float = GRAD_CLIP
    # 실행
    seed: int = 0
    num_workers: int = 8
    log_every: int = 25
    val_every: int = 500
    checkpoint_every: int = 1000
    # latest.pt는 덮어써지므로 이 step마다 `step_XXXXXXX.pt`로 따로 남긴다(평가·재개할 지점)
    keep_every: int | None = 3000
    profile_steps: int = 0  # > 0이면 rank 0에서 torch.profiler로 이만큼의 step을 재서 출력한다

    @classmethod
    def paper(cls) -> TabPFNConfig:
        return cls()

    @classmethod
    def medium(cls) -> TabPFNConfig:
        """논문 모델·prior·batch 그대로, step과 스케줄만 1/10."""
        return cls(
            steps=TOTAL_STEPS // 10,
            warmup_steps=WARMUP_STEPS // 10,
            schedule_steps=SCHEDULE_STEPS // 10,
            val_every=200,
            checkpoint_every=300,
            keep_every=600,
        )

    @classmethod
    def smoke(cls) -> TabPFNConfig:
        """로컬 CPU에서 몇 분 안에 끝나는 크기. 코드 경로 확인용."""
        return cls(
            emsize=32,
            nhead=2,
            nhid=64,
            nlayers=2,
            seq_len=128,
            max_eval_pos=100,
            group_size=4,
            batch_size=8,
            steps=20,
            warmup_steps=2,
            schedule_steps=20,
            lr=1e-3,
            num_workers=0,
            log_every=5,
            val_every=10,
            checkpoint_every=10,
            keep_every=10,
        )

    def grad_accumulation(self, world_size: int) -> int:
        per_forward = self.group_size * self.groups_per_batch * world_size
        if self.batch_size % per_forward:
            raise ValueError(
                f"batch_size {self.batch_size}가 group_size × groups_per_batch × world_size"
                f" = {per_forward}로 나눠떨어지지 않습니다"
            )
        return self.batch_size // per_forward


def build_tabpfn_model(config: TabPFNConfig) -> RowTransformer:
    return RowTransformer(
        num_features=config.num_features,
        n_out=config.n_out,
        emsize=config.emsize,
        nhead=config.nhead,
        nhid=config.nhid,
        nlayers=config.nlayers,
        dropout=config.dropout,
    )


def save_tabpfn(model: torch.nn.Module, config: TabPFNConfig, path: str | Path) -> None:
    """가중치와 config만 저장한다 (`weights_only=True`로 다시 읽을 수 있는 형식)."""
    torch.save({"state_dict": model.state_dict(), "config": asdict(config)}, path)


def tabpfn_from_state(
    config: dict, state_dict: dict[str, torch.Tensor], map_location: str | torch.device = "cpu"
) -> tuple[RowTransformer, TabPFNConfig]:
    """저장된 config(dict)와 가중치로 추론용 모델을 만든다 (`.pt`와 Hub 형식이 함께 쓴다)."""
    config = TabPFNConfig(**config)
    model = build_tabpfn_model(config)
    model.load_state_dict(state_dict)
    return model.to(map_location).eval(), config


def load_tabpfn(
    path: str | Path, map_location: str | torch.device = "cpu"
) -> tuple[RowTransformer, TabPFNConfig]:
    ckpt = torch.load(path, map_location=map_location, weights_only=True)
    return tabpfn_from_state(ckpt["config"], ckpt["state_dict"], map_location)


# 검증용 prior group 수. 검증 데이터셋은 VAL_GROUPS × group_size개다 (plot_card.py도 쓴다).
VAL_GROUPS = 8


def _make_val_batch(config: TabPFNConfig, n_groups: int = VAL_GROUPS):
    """학습과 겹치지 않는 seed로 검증용 prior 데이터셋을 고정해 둔다.

    학습 행 수는 max_eval_pos의 절반으로 둔다.
    """
    prior = TabPFNPrior(
        seq_len=config.seq_len,
        num_features=config.num_features,
        max_classes=config.n_out,
        group_size=config.group_size,
        seed=10_000_000 + config.seed,
    )
    sep = config.max_eval_pos // 2
    groups = [prior.sample_group(sep)[:2] for _ in range(n_groups)]
    return [(x, y, sep) for x, y in groups]


@torch.no_grad()
def _validate(model, val_batch, device, autocast) -> dict[str, float]:
    model.eval()
    total_loss, correct, count = 0.0, 0, 0
    for x, y, sep in val_batch:
        x, y = x.to(device), y.to(device)
        with autocast():
            logits = model(x[:, :sep], y[:, :sep], x[:, sep:])
        y_test = y[:, sep:]
        total_loss += classification_loss(logits, y_test).item()
        valid = y_test != IGNORE_INDEX
        correct += (logits.argmax(-1)[valid] == y_test[valid]).sum().item()
        count += valid.sum().item()
    model.train()
    return {"val_ce": total_loss / len(val_batch), "val_acc": correct / max(1, count)}


def train_tabpfn(
    config: TabPFNConfig,
    out_dir: str | Path,
    resume_from: str | Path | None = None,
    mlflow_log: bool = True,
) -> dict:
    """prior-fitting을 실행하고 마지막 가중치 경로와 loss 기록을 반환한다.

    rank 0만 MLflow에 기록한다. 이미 활성 run이 있으면 거기에 이어서 쓴다(AI Runtime은
    `MLFLOW_RUN_ID`로 run을 넘겨준다). `checkpoint_every` step마다 학습 상태 전체를
    `out_dir/latest.pt`에 저장하고 MLflow 아티팩트(`checkpoints/`)로 올린다. `keep_every`
    step마다는 덮어쓰지 않는 `step_XXXXXXX.pt`로도 남긴다. `resume_from`에 그 파일을 주면
    이어서 학습한다.
    """
    d, device = setup_distributed()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    accumulation = config.grad_accumulation(d.world_size)
    use_cuda = device.type == "cuda"
    configure_attention_backends()

    torch.manual_seed(config.seed + d.rank)
    model = build_tabpfn_model(config).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: lr_factor(s, config.warmup_steps, config.schedule_steps)
    )
    start_step, skipped_steps = 0, 0
    history: dict[str, list] = {
        "step": [],
        "train_ce": [],
        "val_step": [],
        "val_ce": [],
        "val_acc": [],
    }
    if resume_from is not None:
        state = torch.load(resume_from, map_location=device, weights_only=True)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["scheduler"])
        start_step = int(state["step"])
        # 이전 구간의 loss 기록도 이어 붙여야 history.json·loss 곡선이 처음부터 남는다.
        # (history가 없는 예전 체크포인트는 resume 이후만 기록한다)
        history = state.get("history", history)
        skipped_steps = int(state.get("skipped_steps", 0))
    ddp_model = (
        DistributedDataParallel(model, device_ids=[d.local_rank]) if d.world_size > 1 else model
    )

    def autocast():
        if use_cuda:
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    # 이어서 학습할 때 같은 데이터를 다시 보지 않도록 seed에 시작 step을 섞는다.
    make_prior = partial(
        TabPFNPrior,
        seq_len=config.seq_len,
        num_features=config.num_features,
        max_classes=config.n_out,
        group_size=config.group_size,
    )
    dataset = PriorGroups(
        make_prior,
        max_eval_pos=config.max_eval_pos,
        groups_per_batch=config.groups_per_batch,
        seed=(config.seed, start_step, d.rank),
    )
    loader_kwargs = {"num_workers": config.num_workers, "pin_memory": use_cuda}
    if config.num_workers > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=4)
    batches = iter(DataLoader(dataset, batch_size=None, **loader_kwargs))

    mlf, val_batch = None, None
    if d.is_main:
        val_batch = _make_val_batch(config)
        if mlflow_log:
            import mlflow as mlf

            if mlf.active_run() is None:
                mlf.start_run()
            # 같은 run에서 이어서 학습하면 이미 기록돼 있다. 새 run에서 이어 가면 다시 기록한다.
            run_params = mlf.get_run(mlf.active_run().info.run_id).data.params
            if "steps" not in run_params:
                mlf.log_params(asdict(config))
                mlf.log_params(
                    {
                        "world_size": d.world_size,
                        "grad_accumulation": accumulation,
                        "n_parameters": model.num_parameters(),
                        "device": str(device),
                    }
                )

    ddp_model.train()
    window_loss, window_wait, window_start, window_finite = 0.0, 0.0, time.time(), 0
    last_log_step, nan_reports = start_step, 0
    prof = None
    if config.profile_steps > 0 and d.is_main:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if use_cuda:
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        schedule = torch.profiler.schedule(wait=1, warmup=1, active=config.profile_steps)
        prof = torch.profiler.profile(activities=activities, schedule=schedule)
        prof.start()
    for step in range(start_step, config.steps):
        step_loss = torch.zeros((), device=device)
        for micro in range(accumulation):
            t_wait = time.time()
            x, y, sep = next(batches)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            window_wait += time.time() - t_wait
            sync = micro == accumulation - 1 or d.world_size == 1
            ctx = contextlib.nullcontext() if sync else ddp_model.no_sync()
            with ctx:
                with autocast():
                    logits = ddp_model(x[:, :sep], y[:, :sep], x[:, sep:])
                loss = classification_loss(logits, y[:, sep:]) / accumulation
                loss.backward()
            if nan_reports < 5 and not torch.isfinite(loss):
                nan_reports += 1
                print(
                    f"[rank {d.rank}] step {step} micro {micro}: loss {loss.item()} sep {sep} "
                    f"x finite {bool(torch.isfinite(x).all())} |x|max {x.abs().max().item():.1f} "
                    f"logits finite {bool(torch.isfinite(logits).all())}",
                    flush=True,
                )
            step_loss += loss.detach()
        # v1의 fp16 GradScaler처럼 gradient가 inf/nan이면 이 step을 건너뛴다.
        # (DDP가 gradient를 평균한 뒤라 모든 GPU가 같은 판단을 한다.)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        if torch.isfinite(grad_norm):
            opt.step()
        else:
            skipped_steps += 1
        opt.zero_grad(set_to_none=True)
        sched.step()

        if d.world_size > 1:
            dist.all_reduce(step_loss, op=dist.ReduceOp.AVG)
        if torch.isfinite(step_loss):
            window_loss += step_loss.item()
            window_finite += 1

        done = step + 1
        if prof is not None:
            prof.step()
            if done - start_step == config.profile_steps + 2:
                prof.stop()
                sort_key = "cuda_time_total" if use_cuda else "cpu_time_total"
                print(prof.key_averages().table(sort_by=sort_key, row_limit=25), flush=True)
                print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=15), flush=True)
                prof = None
        if d.is_main and (done % config.log_every == 0 or done == config.steps):
            n = done - last_log_step
            elapsed = time.time() - window_start
            metrics = {
                "train_ce": window_loss / max(1, window_finite),
                "lr": sched.get_last_lr()[0],
                "grad_norm": float(grad_norm),
                "skipped_steps": skipped_steps,
                "sec_per_step": elapsed / n,
                "data_wait_frac": window_wait / max(elapsed, 1e-9),
                "datasets_per_sec": config.batch_size * n / max(elapsed, 1e-9),
            }
            history["step"].append(done)
            history["train_ce"].append(metrics["train_ce"])
            print(
                f"step {done}/{config.steps} ce {metrics['train_ce']:.4f} "
                f"lr {metrics['lr']:.2e} {metrics['sec_per_step']:.2f}s/step "
                f"wait {metrics['data_wait_frac']:.0%} skipped {skipped_steps}",
                flush=True,
            )
            if mlf is not None:
                mlf.log_metrics(metrics, step=done)
            window_loss, window_wait, window_start, window_finite = 0.0, 0.0, time.time(), 0
            last_log_step = done

        if d.is_main and (done % config.val_every == 0 or done == config.steps):
            val = _validate(model, val_batch, device, autocast)
            history["val_step"].append(done)
            history["val_ce"].append(val["val_ce"])
            history["val_acc"].append(val["val_acc"])
            print(f"  val ce {val['val_ce']:.4f} acc {val['val_acc']:.3f}", flush=True)
            if mlf is not None:
                mlf.log_metrics(val, step=done)

        keep = bool(config.keep_every) and done % config.keep_every == 0
        if d.is_main and (done % config.checkpoint_every == 0 or keep or done == config.steps):
            state = {
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "scheduler": sched.state_dict(),
                "step": done,
                "history": history,
                "skipped_steps": skipped_steps,
            }
            torch.save(state, out_dir / "latest.pt")
            kept = None
            if keep:
                kept = out_dir / f"step_{done:07d}.pt"
                shutil.copyfile(out_dir / "latest.pt", kept)
            if mlf is not None:
                mlf.log_artifact(str(out_dir / "latest.pt"), "checkpoints")
                if kept is not None:
                    mlf.log_artifact(str(kept), "checkpoints")
                mlf.log_metric("checkpoint_step", done, step=done)

    final_path = out_dir / "tabpfn_final.pt"
    if d.is_main:
        save_tabpfn(model, replace(config), final_path)
        (out_dir / "history.json").write_text(json.dumps(history))
        if mlf is not None:
            mlf.log_artifact(str(final_path), "model")
            mlf.log_artifact(str(out_dir / "history.json"))
    if d.world_size > 1:
        dist.barrier()
        dist.destroy_process_group()
    return {"final_path": str(final_path), "history": history, "is_main": d.is_main}


# ---------------------------------------------------------------------------
# 추론 (논문 5.2절, v1 transformer_predict)
# ---------------------------------------------------------------------------
class TabPFNClassifier(RotationEnsembleClassifier):
    """v1 논문 앙상블(멤버 `N_ENSEMBLE`개, 온도 `SOFTMAX_TEMPERATURE`)을 기본값으로 둔
    `RotationEnsembleClassifier`.

        clf = TabPFNClassifier(model).fit(x_train, y_train)
        proba = clf.predict_proba(x_test)
    """

    def __init__(
        self,
        model: RowTransformer,
        *,
        n_ensemble: int = N_ENSEMBLE,
        softmax_temperature: float = SOFTMAX_TEMPERATURE,
        fit_mode: FitMode = "fit_preprocessors",
        seed: int = 0,
        batch_size: int = 16,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__(
            model,
            n_ensemble=n_ensemble,
            softmax_temperature=softmax_temperature,
            fit_mode=fit_mode,
            seed=seed,
            batch_size=batch_size,
            device=device,
        )


def tabpfn_predict_proba(
    model: RowTransformer,
    x_train: Any,
    y_train: Any,
    x_test: Any,
    n_ensemble: int = N_ENSEMBLE,
    seed: int = 0,
    batch_size: int = 16,
    device: torch.device | str | None = None,
) -> np.ndarray:
    """테스트 행의 클래스 확률 (m, C). context(학습 행)가 호출마다 바뀔 때 쓰는 stateless 버전
    (`TabPFNClassifier.predict_proba`와 같은 앙상블). 열 순서는 정렬한 레이블 순서다."""
    return rotation_ensemble.predict_proba(
        model,
        x_train,
        y_train,
        x_test,
        n_ensemble=n_ensemble,
        softmax_temperature=SOFTMAX_TEMPERATURE,
        seed=seed,
        batch_size=batch_size,
        device=device,
    )
