"""TabPFN v2 prior-fitting 학습 루프 (Methods "Training details").

매 optimizer step: GPU마다 prior에서 데이터셋 `micro_batch`개짜리 batch를 받아, 학습 행을 보고
정답을 가린 검증 행 128개의 정답을 맞히는 손실을 줄인다. 분류는 cross-entropy, 회귀는 구간
5,000개 bar distribution의 NLL이다. 유효 batch는

    batch_size = micro_batch × grad_accumulation × world_size

이고 grad_accumulation은 나머지 값으로 자동 계산한다. GPU가 많아 micro_batch × world_size가
batch_size를 넘으면 GPU당 micro batch를 batch_size / world_size로 줄인다. 학습률은 linear
warmup 뒤 cosine으로 줄인다(논문: Adam, warmup + cosine, lr ∈ [1e-4, 5e-4]).

v1 재현에서 GPU로 학습하며 고친 세 가지를 그대로 둔다.
- bf16에는 GradScaler가 없으므로 gradient가 inf/nan이면 그 step을 건너뛴다
- 학습 행 수가 forward마다 바뀌므로 cuDNN attention을 끈다(shape마다 그래프를 새로 만들어 느리다)
- 학습 행은 1개 이상이다(0개면 그 GPU만 attention 가중치를 쓰지 않아 DDP 동기화가 어긋난다)
"""

from __future__ import annotations

import contextlib
import copy
import json
import math
import os
import shutil
import time
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Literal

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from minipfn.bar_distribution import FullSupportBarDistribution
from minipfn.tabpfn_v2.model import save_checkpoint
from minipfn.tabpfn_v2.prior import (
    MAX_CELLS,
    MAX_CLASSES,
    MAX_FEATURES,
    MAX_TRAIN_ROWS,
    MIN_CLASSIFICATION_TRAIN_ROWS,
    N_EVAL_ROWS,
    TabPFNv2Prior,
    check_table_size,
    regression_borders,
)
from minipfn.tabpfn_v2.registry import (
    DEFAULT_MODEL_TYPE,
    build_model,
    config_fields,
    model_class,
)
from minipfn.utils import DistContext, lr_factor, setup_distributed

N_BUCKETS = 5000

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
# long_tables의 GPU당 micro batch. v2(12층)는 2에서 GPU 메모리 52.6GB였다. 2.5는 층이 두 배라 1
LONG_TABLES_MICRO_BATCH = {"v2": 2, "v2.5": 1}
ARCHITECTURES = {
    "v2": {"classification": _V2, "regression": _V2},
    "v2.5": {
        "classification": {**_V2_5, "nlayers": 24},
        "regression": {**_V2_5, "nlayers": 18, "encoder": "mlp"},
    },
}


@dataclass
class TabPFNv2TrainConfig:
    """모델·prior·최적화 설정. smoke(로컬 CPU), medium(짧은 실측), paper(논문 규모),
    long_tables(paper로 학습한 모델을 긴·넓은 표로 이어 학습)."""

    task: Literal["classification", "regression"] = "classification"
    # 모델 클래스 (`registry.MODEL_TYPES`). 아래 모델 설정 가운데 그 클래스의 설정에 있는 것만 쓴다
    model_type: str = DEFAULT_MODEL_TYPE
    # 모델 (논문 기본 모델 크기). 아래 셋은 TabPFN-2.5가 바꾼 것(`with_architecture("v2.5")`)
    emsize: int = 192
    nlayers: int = 12
    nhead: int = 6
    features_per_group: int = 2
    n_thinking_rows: int = 0
    encoder: str = "linear"
    # prior / 데이터셋 크기 (논문 Methods)
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
        model_class(self.model_type)  # 없는 이름이면 ValueError
        min_rows = self.min_train_rows
        if self.task == "classification":
            min_rows = max(min_rows, MIN_CLASSIFICATION_TRAIN_ROWS)
        check_table_size(min_rows, self.max_train_rows, self.n_eval_rows, self.max_cells)
        if not 0.0 < self.rewind_lr_factor <= 1.0:
            raise ValueError(f"rewind_lr_factor는 0 초과 1 이하여야 한다: {self.rewind_lr_factor}")
        if self.qk_clip is not None and self.qk_clip <= 0:
            raise ValueError(f"qk_clip은 0보다 커야 한다: {self.qk_clip}")
        if self.qk_clip_every < 1:
            raise ValueError(f"qk_clip_every는 1 이상이어야 한다: {self.qk_clip_every}")

    @property
    def n_out(self) -> int:
        return MAX_CLASSES if self.task == "classification" else N_BUCKETS

    @classmethod
    def paper(cls, task: str = "classification") -> TabPFNv2TrainConfig:
        # 분류는 2e-4에서 붕괴가 드물어(300k까지 rewind 1번) lr을 그대로 둔다.
        # 회귀의 QK-clip 250: 50k scratch 비교에서 250·500 모두 상한을 지켰고 val NLL도 비슷했다
        # (−1.08·−1.07, 끈 것 −1.04). 그중 logit을 더 낮게 묶는 쪽. 100은 10k부터 잘라 학습이 늦었다
        if task == "regression":
            return cls(task=task, rewind_lr_factor=0.5, qk_clip=250.0)
        return cls(task=task)

    @classmethod
    def long_tables(cls, task: str = "classification", arch: str = "v2") -> TabPFNv2TrainConfig:
        """paper로 학습한 모델을 더 긴·넓은 표로 짧게 이어 학습한다 (논문에 없음, 값은 여기서 정함).

        TabPFN-2.5는 prior를 "더 많은 행과 feature로 키웠다"고만 적었다(추론 권장 크기는 행 50,000,
        feature 2,000). 그 크기로 처음부터 학습하는 대신 TabICL처럼 짧은 표로 오래 학습한 뒤 긴 표로
        잠깐 더 학습한다: 학습 행 512~8,192개 log 균등, feature 수 분포를 500까지 펴고, 셀은
        400,000개까지. 가중치만 가져와(`init_from`) 새 optimizer로 lr 5e-5, warmup 1k + cosine
        20k step. GPU당 micro batch는 구조마다 다르므로(`LONG_TABLES_MICRO_BATCH`) 구조를 여기서
        `arch`로 받는다(나중에 `with_architecture`로 바꾸면 micro batch는 따라 바뀌지 않는다).
        검증 batch는 데이터셋 4개다.
        """
        if arch not in LONG_TABLES_MICRO_BATCH:
            raise ValueError(
                f"long_tables의 arch는 {tuple(LONG_TABLES_MICRO_BATCH)} 가운데서: {arch}"
            )
        return replace(
            cls.paper(task).with_architecture(arch),
            min_train_rows=512,
            max_train_rows=8192,
            log_uniform_rows=True,
            max_features=500,
            feature_scale=500,
            max_cells=400_000,
            micro_batch=LONG_TABLES_MICRO_BATCH[arch],
            val_batch=4,
            steps=20_000,
            warmup_steps=1_000,
            lr=5e-5,
            keep_every=5_000,
        )

    @classmethod
    def medium(cls, task: str = "classification") -> TabPFNv2TrainConfig:
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
    def smoke(cls, task: str = "classification") -> TabPFNv2TrainConfig:
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

    def with_architecture(self, arch: str, keep_size: bool = False) -> TabPFNv2TrainConfig:
        """모델 구조를 `ARCHITECTURES`의 것으로. 구조가 모델 클래스(`model_type`)도 정한다. 이미
        기본값이 아닌 다른 모델 클래스로 정한 설정은 바꾸지 않고 에러를 낸다(다른 모델을 v2로
        조용히 바꿔 학습하지 않도록). `keep_size`(smoke)면 층 수는 그대로 두고 thinking 행은
        4개까지만 써서 작은 크기를 유지한다(구조가 바뀌는 코드 경로만 확인한다)."""
        if arch not in ARCHITECTURES:
            raise ValueError(f"arch는 {tuple(ARCHITECTURES)} 가운데 하나여야 한다: {arch}")
        changes = dict(ARCHITECTURES[arch][self.task])
        if self.model_type not in (DEFAULT_MODEL_TYPE, changes["model_type"]):
            raise ValueError(
                f"arch {arch}는 {changes['model_type']} 모델이다 (설정의 model_type:"
                f" {self.model_type})"
            )
        if keep_size:
            changes["nlayers"] = self.nlayers
            if "n_thinking_rows" in changes:
                changes["n_thinking_rows"] = min(changes["n_thinking_rows"], 4)
        return replace(self, **changes)

    def for_world_size(self, world_size: int) -> TabPFNv2TrainConfig:
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

    def table_sizes(self) -> dict:
        """prior의 표 크기 설정 (학습 prior와 회귀 구간 경계의 prior가 같이 쓴다)."""
        return {
            "min_train_rows": self.min_train_rows,
            "max_train_rows": self.max_train_rows,
            "n_eval_rows": self.n_eval_rows,
            "max_features": self.max_features,
            "log_uniform_rows": self.log_uniform_rows,
            "max_cells": self.max_cells,
            "feature_scale": self.feature_scale,
        }

    def make_prior(self, seed: int) -> TabPFNv2Prior:
        return TabPFNv2Prior(
            self.task,
            seed=seed,
            missing_dataset_prob=self.missing_dataset_prob,
            **self.table_sizes(),
        )


class PriorBatches(IterableDataset):
    """DataLoader worker마다 다른 시드로 prior batch를 끝없이 만든다."""

    def __init__(self, config: TabPFNv2TrainConfig, seed: tuple[int, ...]) -> None:
        self.config = config
        self.seed = seed

    def __iter__(self):
        info = get_worker_info()
        worker = 0 if info is None else info.id
        torch.set_num_threads(1)
        seed = hash((*self.seed, worker)) % (2**31)
        prior = self.config.make_prior(seed)
        while True:
            batch = prior.sample_batch(self.config.micro_batch)
            yield batch.x, batch.y, batch.n_train


def prior_loss(
    logits: torch.Tensor,
    y_eval: torch.Tensor,
    task: str,
    bar: FullSupportBarDistribution | None,
) -> torch.Tensor:
    """검증 행의 평균 손실. 분류: cross-entropy, 회귀: bar distribution NLL."""
    logits = logits.float()
    if task == "classification":
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y_eval.reshape(-1).long())
    return bar(logits, y_eval.float()).mean()


def dequantize_targets(
    y: torch.Tensor, n_train: int, bar: FullSupportBarDistribution, n_buckets: float
) -> torch.Tensor:
    """데이터셋 안에서 반복되는 target 값의 검증 행 정답을 이웃 구간 `n_buckets`개에 흩는다.

    같은 값이 반복되는 이산 target은 늘 같은 구간에 떨어지므로, 그 구간에 확률을 몰수록 NLL이
    끝없이 낮아진다(최적 logit이 무한대). 그러면 출력층이 계속 커지고 gradient가 연속 target의
    수백 배가 되어 학습이 무너진다(논문 규모 회귀 run에서 겪음). 구간 번호 좌표에서 균일 noise를
    더해 정답을 이웃 구간에 퍼뜨리면 최적이 유한해진다. 연속 target, 양 끝 꼬리 구간의 값, 결측은
    그대로 둔다(여기서 정함). 반환은 검증 행 정답 `(batch, n_eval)`이다.
    """
    y_eval = y[:, n_train:]
    # 데이터셋마다 정렬해 이웃과 같은 값이면 반복 (GPU 동기화 없이 batch 전체를 한 번에)
    y_sorted, order = y.sort(dim=1)
    same = y_sorted[:, 1:] == y_sorted[:, :-1]
    dup = torch.zeros_like(y_sorted, dtype=torch.bool)
    dup[:, 1:] |= same
    dup[:, :-1] |= same
    repeated = torch.zeros_like(dup).scatter_(1, order, dup)[:, n_train:]
    borders, widths, last = bar.borders, bar.widths, bar.num_buckets - 1
    nan = torch.isnan(y_eval)
    y_safe = torch.where(nan, borders[1], y_eval)  # 결측은 바꾸지 않지만 index 계산은 유한하게
    idx = bar._bucket_index(y_safe.contiguous())
    interior = (idx >= 1) & (idx <= last - 1) & ~nan
    coord = idx + (y_safe - borders[idx]) / widths[idx]  # 구간 번호 좌표 (k ~ k+1이 구간 k)
    moved = coord + (torch.rand_like(y_eval) - 0.5) * n_buckets
    # 꼬리 구간으로는 보내지 않는다. 잘라내면 경계에 점 질량이 다시 생기므로 경계에서 반사한다
    lo, hi = 1.0, float(last)
    moved = torch.where(moved < lo, 2 * lo - moved, moved)
    moved = torch.where(moved > hi, 2 * hi - moved, moved).clamp(lo, hi)
    j = moved.floor().long().clamp(1, last - 1)
    shifted = borders[j] + (moved - j) * widths[j]
    # float 반올림으로 오른쪽 꼬리 경계(borders[-2])에 닿지 않게 그 바로 아래 값까지만
    shifted = torch.minimum(shifted, torch.nextafter(borders[-2], borders[-3]))
    return torch.where(repeated & interior, shifted, y_eval)


def _borders(
    config: TabPFNv2TrainConfig, d: DistContext, device: torch.device
) -> torch.Tensor | None:
    """회귀면 rank 0에서 구간 경계를 계산해 모든 GPU에 나눠 준다."""
    if config.task != "regression":
        return None
    borders = torch.empty(N_BUCKETS + 1, device=device)
    if d.is_main:
        start = time.time()
        borders = regression_borders(
            N_BUCKETS,
            config.border_datasets,
            seed=20_000_000 + config.seed,
            n_jobs=max(1, min(32, (os.cpu_count() or 1) // 2)),
            **config.table_sizes(),
        ).to(device)
        print(f"regression borders: {config.border_datasets} datasets, {time.time() - start:.0f}s")
    if d.world_size > 1:
        dist.broadcast(borders, src=0)
    return borders


def _make_val_batches(config: TabPFNv2TrainConfig, n_batches: int = 8):
    """학습과 겹치지 않는 시드로 검증용 prior batch를 고정해 둔다 (val_batch × n_batches개)."""
    prior = config.make_prior(10_000_000 + config.seed)
    return [prior.sample_batch(config.val_batch) for _ in range(n_batches)]


@torch.no_grad()
def _validate(model, val_batches, device, autocast, task, bar) -> dict[str, float]:
    model.eval()
    total, correct, count = 0.0, 0, 0
    for batch in val_batches:
        x, y = batch.x.to(device), batch.y.to(device)
        with autocast():
            logits = model(x, y[:, : batch.n_train])
        y_eval = y[:, batch.n_train :]
        total += prior_loss(logits, y_eval, task, bar).item()
        if task == "classification":
            correct += (logits.argmax(-1) == y_eval.long()).sum().item()
            count += y_eval.numel()
    model.train()
    out = {"val_loss": total / len(val_batches)}
    if task == "classification":
        out["val_acc"] = correct / max(1, count)
    return out


def _set_base_lr(opt, sched, lr: float, step: int, verbose: bool) -> None:
    """기준 lr(base_lrs, initial_lr)을 바꾸고 지금 lr을 그 기준의 `step` 값으로 맞춘다.

    optimizer와 scheduler 상태를 불러오면 lr도 그 상태의 값으로 돌아가므로, 이어서 학습할 때 lr을
    바꾸거나 붕괴로 되돌린 뒤 lr을 낮추려면 기준 lr과 지금 lr을 함께 고친다. warmup·cosine 모양은
    그대로다. 같은 lr이면 아무것도 바뀌지 않는다.
    """
    old = sched.base_lrs[0]
    factor = sched.lr_lambdas[0](step)  # scheduler가 쓰는 스케줄 그대로
    sched.base_lrs = [lr] * len(opt.param_groups)
    for group in opt.param_groups:
        group["initial_lr"] = lr
        group["lr"] = lr * factor
    sched._last_lr = [group["lr"] for group in opt.param_groups]
    if verbose and not math.isclose(old, lr, rel_tol=1e-12):
        print(f"lr: 기준 {old:.2e} → {lr:.2e} (step {step}부터 {lr * factor:.3e})", flush=True)


@torch.no_grad()
def qk_logit_bounds(model) -> torch.Tensor:
    """attention마다 head별 logit 상한 σ_max(Wq_hᵀ Wk_h) · E / √d. 모양 (attention 수, head 수).

    attention 순서는 `model.qk_clip_weights()`와 같다. 층 입력은 affine 없는 LayerNorm 뒤라 토큰
    크기가 √E로 같으므로 |q·k| / √d ≤ σ_max · E / √d이다. 논문 규모 회귀 checkpoint에서 이 값은
    실제 최대 logit과 거의 같았다(678k vs 676k). Wq_hᵀ = Q_q R_q, Wk_hᵀ = Q_k R_k로 QR 분해하면
    σ_max(Wq_hᵀ Wk_h) = σ_max(R_q R_kᵀ)라 E×E 대신 d×d에서 계산한다.
    """
    weights = model.qk_clip_weights()
    if not weights:
        return torch.zeros(0, model.config.nhead)
    nhead = model.config.nhead
    e = weights[0][1].shape[1]
    d = e // nhead
    w = torch.stack([w for _, w in weights]).double()  # (A, 3E, E)
    q = w[:, :e].reshape(len(weights), nhead, d, e)
    k = w[:, e : 2 * e].reshape(len(weights), nhead, d, e)
    r_q = torch.linalg.qr(q.transpose(-1, -2)).R  # (A, H, d, d)
    r_k = torch.linalg.qr(k.transpose(-1, -2)).R
    sigma = torch.linalg.matrix_norm(r_q @ r_k.transpose(-1, -2), ord=2)
    return (sigma * e / math.sqrt(d)).float()


def _sample_rows(model) -> list[int]:
    # qk_logit_bounds의 행 가운데 sample attention인 것
    return [i for i, (kind, _) in enumerate(model.qk_clip_weights()) if kind == "sample_attention"]


@torch.no_grad()
def apply_qk_clip(model, tau: float, world_size: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    """logit 상한이 tau를 넘는 head의 Wq_h·Wk_h를 각각 √(tau / 상한)배로 줄인다.

    반환: (줄인 head 수, 줄이기 전 가장 큰 상한). 매 step 부르므로 GPU에 둔 채 돌려준다(값을 읽는
    것은 기록할 때만). GPU마다 계산이 아주 조금 다를 수 있어 배율은 모든 GPU의 최솟값으로
    맞춘다(가중치가 GPU마다 같게 남도록).
    """
    bounds = qk_logit_bounds(model)
    scale = (tau / bounds).clamp(max=1.0).sqrt()
    if world_size > 1:
        dist.all_reduce(scale, op=dist.ReduceOp.MIN)
    nhead = model.config.nhead
    for (_, w), s in zip(model.qk_clip_weights(), scale, strict=True):
        e = w.shape[1]
        head = s.to(w.dtype).repeat_interleave(e // nhead)[:, None]  # (E, 1): 행마다 그 head의 배율
        w[:e].mul_(head)
        w[e : 2 * e].mul_(head)
    return (scale < 1).sum(), bounds.max()


def check_checkpoint(state: dict, config: TabPFNv2TrainConfig, name: str) -> None:
    """`state`(학습 중 checkpoint 또는 `save_checkpoint` 파일)가 `config`와 같은 task·모델인지.

    모양이 같은 다른 설정(head 수 등)은 `load_state_dict`가 그대로 읽어 버리므로 미리 맞춰 본다.
    모델 설정을 남기기 전의 학습 중 checkpoint는 task만 본다.
    """
    regression = state.get("borders") is not None
    if regression != (config.task == "regression"):
        got = "regression" if regression else "classification"
        raise ValueError(f"{name}은 {got} checkpoint다 (설정의 task: {config.task})")
    if "config" not in state:
        return
    saved_type = state.get("model_type", DEFAULT_MODEL_TYPE)
    if saved_type != config.model_type:
        raise ValueError(f"{name}은 {saved_type} 모델이다 (설정의 model_type: {config.model_type})")
    # 저장한 뒤에 생긴 설정 필드는 그 기본값으로 채워 비교한다
    saved = model_class(saved_type).config_class(**state["config"]).to_dict()
    expected = config.model_config().to_dict()
    diff = {k: (saved.get(k), v) for k, v in expected.items() if saved.get(k) != v}
    if diff:
        raise ValueError(f"{name}의 모델 설정이 이번 설정과 다르다 (저장된 값, 설정): {diff}")


def checkpoint_lr(path: str | Path) -> float:
    """checkpoint가 쓰던 기준 lr. 이어서 학습할 때 lr을 따로 주지 않으면 이 값을 이어 간다."""
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    return float(state["scheduler"]["base_lrs"][0])


def _to_cpu(obj):
    """state_dict 안의 텐서를 CPU로 복사한다 (메모리에 남겨 둘 사본)."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().to("cpu", copy=True)
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_cpu(v) for v in obj]
    return obj


class CollapseGuard:
    """train loss가 갑자기 뛰어 돌아오지 않는 붕괴를 알아챈다.

    논문 규모 run에서 분류·회귀 모두 grad spike 없이도 수백 step 안에 상수 출력으로 무너진 적이
    있다. 같은 시드로 다시 돌리면 무너지지 않아서(GPU 비결정성), 막기보다 알아채고 되돌린다.
    log_every마다 window 평균 loss를 받아, 최근 건강한 window들의 중앙값보다 `threshold` 넘게
    높은 window가 `patience`번 이어지면 붕괴로 본다. 모든 GPU가 같은 평균(all-reduce 뒤)을 보므로
    판단도 같다.
    """

    def __init__(self, threshold: float, patience: int, n_windows: int = 20, min_windows: int = 5):
        self.threshold, self.patience, self.min_windows = threshold, patience, min_windows
        self.healthy: list[float] = []  # 최근 건강한 window 평균
        self.n_windows = n_windows
        self.bad = 0  # 이어진 나쁜 window 수

    def update(self, window_mean: float) -> bool:
        """window 평균을 넣고, 붕괴로 판단하면 True.

        유한한 loss가 하나도 없던 window(inf)는 나쁜 window로 센다.
        """
        bad = not math.isfinite(window_mean)
        if not bad and len(self.healthy) >= self.min_windows:
            ref = sorted(self.healthy)[len(self.healthy) // 2]
            bad = window_mean > ref + self.threshold
        if bad:
            self.bad += 1
            return self.bad >= self.patience
        self.bad = 0
        self.healthy = (self.healthy + [window_mean])[-self.n_windows :]
        return False

    @property
    def suspect(self) -> bool:
        return self.bad > 0


def train_tabpfn_v2(
    config: TabPFNv2TrainConfig,
    out_dir: str | Path,
    resume_from: str | Path | None = None,
    mlflow_log: bool = True,
    init_from: str | Path | None = None,
) -> dict:
    """prior-fitting을 실행하고 마지막 체크포인트 경로와 loss 기록을 반환한다.

    `resume_from`은 학습 상태 전체(optimizer·스케줄·step)를 이어 가고, `init_from`은 가중치(와
    회귀 구간 경계)만 가져와 step 0부터 새 optimizer로 시작한다(long_tables).

    rank 0만 MLflow에 기록한다. 활성 run이 있으면 거기에 이어서 쓴다(AI Runtime은
    `MLFLOW_RUN_ID`로 run을 넘겨준다). `checkpoint_every` step마다 학습 상태 전체를
    `out_dir/latest.pt`에 저장하고 MLflow 아티팩트(`checkpoints/`)로 올린다. `keep_every`
    step마다는 덮어쓰지 않는 `step_XXXXXXX.pt`로도 남긴다. `CollapseGuard`가 붕괴를 알아채면
    `snapshot_every`마다 메모리에 남긴 건강한 상태로 되돌리고 데이터 시드를 바꿔 다시 간다
    (MLflow `rewinds`). 붕괴 중일 수 있는 상태는 checkpoint로 남기지 않는다.
    """
    d, device = setup_distributed()
    config = config.for_world_size(d.world_size)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    accumulation = config.grad_accumulation(d.world_size)
    use_cuda = device.type == "cuda"
    if use_cuda:
        torch.backends.cuda.enable_cudnn_sdp(False)

    torch.manual_seed(config.seed)  # 모든 GPU가 같은 초기 가중치에서 시작한다
    model = config.build_model().to(device)
    if config.qk_clip is not None and not model.qk_clip_weights():
        raise ValueError(f"{config.model_type} 모델에는 QK-clip이 볼 attention이 없다")
    opt = torch.optim.Adam(model.parameters(), lr=config.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: lr_factor(s, config.warmup_steps, config.steps)
    )
    start_step, skipped_steps, borders, rewinds = 0, 0, None, 0
    history: dict[str, list] = {"step": [], "train_loss": [], "val_step": [], "val_loss": []}
    if resume_from is not None and init_from is not None:
        raise ValueError("resume_from과 init_from은 함께 줄 수 없다")
    if init_from is not None:
        state = torch.load(init_from, map_location=device, weights_only=True)
        check_checkpoint(state, config, "init_from")
        # 학습 중 checkpoint는 "model", `save_checkpoint`로 저장한 파일은 "state_dict"
        model.load_state_dict(state["model"] if "model" in state else state["state_dict"])
        borders = state["borders"]
    if resume_from is not None:
        state = torch.load(resume_from, map_location=device, weights_only=True)
        check_checkpoint(state, config, "resume_from")
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["scheduler"])
        start_step = int(state["step"])
        history = state["history"]
        skipped_steps = int(state["skipped_steps"])
        # 이어서 학습해도 붕괴 뒤의 데이터를 다시 보지 않도록 되돌린 횟수를 이어받는다
        rewinds = int(state.get("rewinds", 0))
        _set_base_lr(opt, sched, config.lr, start_step, d.is_main)
        borders = state["borders"]
    if borders is None:
        borders = _borders(config, d, device)
    bar = FullSupportBarDistribution(borders).to(device) if borders is not None else None
    ddp_model = (
        DistributedDataParallel(model, device_ids=[d.local_rank]) if d.world_size > 1 else model
    )

    def autocast():
        if use_cuda:
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    loader_kwargs = {"num_workers": config.num_workers, "pin_memory": use_cuda}
    if config.num_workers > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=4)

    def make_batches(from_step: int, rewinds: int):
        # 이어서 학습할 때 같은 데이터를 다시 보지 않도록 시드에 시작 step을 섞는다.
        # 붕괴로 되돌아가면 되돌린 횟수도 섞어 다른 데이터로 다시 간다
        seed = (config.seed, from_step, d.rank) + ((rewinds,) if rewinds else ())
        return iter(DataLoader(PriorBatches(config, seed=seed), batch_size=None, **loader_kwargs))

    batches = make_batches(start_step, rewinds)

    mlf, val_batches = None, None
    if d.is_main:
        val_batches = _make_val_batches(config)
        if mlflow_log:
            import mlflow as mlf

            if mlf.active_run() is None:
                mlf.start_run()
            run_params = mlf.get_run(mlf.active_run().info.run_id).data.params
            if "steps" not in run_params:
                mlf.log_params(asdict(config))
                mlf.log_params(
                    {
                        "world_size": d.world_size,
                        "grad_accumulation": accumulation,
                        "n_parameters": model.num_parameters(),
                        "device": str(device),
                        "cpu_count": os.cpu_count(),
                    }
                )
            else:
                # 이어서 쓰는 run은 param을 바꿀 수 없으니, 그 뒤에 생긴 설정만 더한다
                new = {k: v for k, v in asdict(config).items() if k not in run_params}
                if new:
                    mlf.log_params(new)

    guard = None
    if config.collapse_threshold is not None:
        guard = CollapseGuard(config.collapse_threshold, config.collapse_patience)
    rewinds_here = 0  # 이번 실행에서 되돌린 횟수 (rewinds는 이전 chunk까지 포함한 누계)
    snapshots: list[dict] = []  # 메모리에 둔 건강한 상태 (최근 2개)

    def snapshot(at_step: int) -> None:
        snapshots.append(
            {
                "model": _to_cpu(model.state_dict()),
                "optimizer": _to_cpu(opt.state_dict()),
                "scheduler": copy.deepcopy(sched.state_dict()),
                "step": at_step,
                "history": copy.deepcopy(history),
                "skipped_steps": skipped_steps,
                "healthy": list(guard.healthy),
            }
        )
        del snapshots[:-2]

    ddp_model.train()
    window_loss, window_wait, window_start, window_finite = 0.0, 0.0, time.time(), 0
    window_clipped, window_bound = 0, 0.0  # QK-clip: 줄인 head 수, 줄이기 전 최대 상한
    last_log_step, skip_reports = start_step, 0
    stop_step = min(config.stop_step or config.steps, config.steps)
    if config.qk_clip is not None:
        # 이어서 학습하는 checkpoint가 clip 없이 학습됐을 수 있어, 첫 snapshot 전에 한 번 맞춘다
        apply_qk_clip(model, config.qk_clip, d.world_size)
        if mlf is not None:  # 이어 쓰는 run은 param을 못 바꾸므로 chunk마다 쓴 값을 지표로 남긴다
            mlf.log_metric("qk_clip", config.qk_clip, step=start_step)
    if guard is not None:
        snapshot(start_step)
    step = start_step - 1
    while step + 1 < stop_step:
        step += 1
        step_loss = torch.zeros((), device=device)
        micro_info = []
        for micro in range(accumulation):
            t_wait = time.time()
            x, y, n_train = next(batches)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            window_wait += time.time() - t_wait
            sync = micro == accumulation - 1 or d.world_size == 1
            ctx = contextlib.nullcontext() if sync else ddp_model.no_sync()
            with ctx:
                with autocast():
                    logits = ddp_model(x, y[:, :n_train])
                y_eval = y[:, n_train:]
                if bar is not None and config.dequantize_buckets:
                    y_eval = dequantize_targets(y, n_train, bar, config.dequantize_buckets)
                loss = prior_loss(logits, y_eval, config.task, bar) / accumulation
                loss.backward()
            step_loss += loss.detach()
            micro_info.append((loss.detach(), x, y, n_train))
        # gradient가 inf/nan이면 이 step을 건너뛴다 (DDP 평균 뒤라 모든 GPU가 같은 판단을 한다)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        if torch.isfinite(grad_norm):
            opt.step()
            if config.qk_clip is not None and (step + 1) % config.qk_clip_every == 0:
                clipped, bound = apply_qk_clip(model, config.qk_clip, d.world_size)
                window_clipped = window_clipped + clipped  # GPU에 둔 채 더한다 (읽는 건 기록할 때)
                window_bound = torch.maximum(
                    torch.as_tensor(window_bound, device=bound.device), bound
                )
        else:
            skipped_steps += 1
            if skip_reports < 5:  # 원인을 찾도록 처음 몇 번은 rank마다 batch 정보를 남긴다
                skip_reports += 1
                for loss_, x_, y_, n_ in micro_info:
                    finite_x = x_[torch.isfinite(x_)]
                    print(
                        f"[rank {d.rank}] step {step + 1} non-finite grad: loss {loss_.item():.4g} "
                        f"x {tuple(x_.shape)} n_train {n_} "
                        f"|x|max {finite_x.abs().max().item() if finite_x.numel() else 0:.3g} "
                        f"y [{y_.min().item():.3g}, {y_.max().item():.3g}]",
                        flush=True,
                    )
        opt.zero_grad(set_to_none=True)
        sched.step()

        if d.world_size > 1:
            dist.all_reduce(step_loss, op=dist.ReduceOp.AVG)
        if torch.isfinite(step_loss):
            window_loss += step_loss.item()
            window_finite += 1

        done = step + 1
        # 시각을 넘겼는지는 log_every마다 보고, 모든 GPU가 같은 step에서 멈추도록 판단을 맞춘다
        if config.deadline is not None and done % config.log_every == 0:
            late = torch.tensor(float(time.time() > config.deadline), device=device)
            if d.world_size > 1:
                dist.all_reduce(late, op=dist.ReduceOp.MAX)
            if late.item() > 0:
                stop_step = done
        last = done == stop_step

        rewound_at_end = False
        if guard is not None:
            collapsed = False
            if done % config.log_every == 0:
                # window_loss는 all-reduce 뒤 값이라 모든 GPU에서 같다
                collapsed = guard.update(window_loss / window_finite if window_finite else math.inf)
            # 무너졌거나, 의심스러운 채로 끝나면 마지막 건강한 상태로 되돌린다
            if collapsed or (last and guard.suspect):
                rewinds, rewinds_here = rewinds + 1, rewinds_here + 1
                if rewinds_here > config.max_rewinds:
                    raise RuntimeError(f"step {done}: 붕괴로 {config.max_rewinds}번 넘게 되돌렸다")
                # 붕괴면 무너지기 시작한 지점보다 앞서도록 오래된 쪽으로, 의심스러운 채로 끝나는
                # 것뿐이면(한 번 튄 것일 수 있다) 가장 최근 건강한 상태로 되돌린다
                snap = snapshots[0] if collapsed else snapshots[-1]
                # snapshot이 이전 lr 인하보다 앞설 수 있으니 되돌리기 전의 lr을 기준으로 낮춘다
                lr_now = sched.base_lrs[0]
                model.load_state_dict(snap["model"])
                opt.load_state_dict(snap["optimizer"])
                sched.load_state_dict(snap["scheduler"])
                # 한 번 튄 채로 끝난 것뿐이면 붕괴가 아니므로 lr은 그대로 둔다
                lr_next = lr_now
                if collapsed:  # 하한까지만 낮춘다 (이미 하한보다 낮으면 그대로)
                    lr_next = min(
                        lr_now, max(lr_now * config.rewind_lr_factor, config.rewind_min_lr)
                    )
                _set_base_lr(opt, sched, lr_next, snap["step"], d.is_main)
                history = copy.deepcopy(snap["history"])
                skipped_steps = snap["skipped_steps"]
                guard.healthy, guard.bad = list(snap["healthy"]), 0
                snapshots[:] = [snap]
                if d.is_main:
                    print(
                        f"step {done}: train loss 붕괴 → step {snap['step']}로 되돌림", flush=True
                    )
                    if mlf is not None:
                        mlf.log_metric("rewinds", rewinds, step=done)
                del batches
                batches = make_batches(snap["step"], rewinds)
                window_loss, window_wait, window_start, window_finite = 0.0, 0.0, time.time(), 0
                window_clipped, window_bound = 0, 0.0  # QK-clip: 줄인 head 수, 줄이기 전 최대 상한
                last_log_step = snap["step"]
                if not last:
                    step = snap["step"] - 1
                    continue
                done, rewound_at_end = snap["step"], True  # 되돌린 상태를 남기고 멈춘다
        if d.is_main and (done % config.log_every == 0 or last) and not rewound_at_end:
            n = done - last_log_step
            elapsed = time.time() - window_start
            metrics = {
                "train_loss": window_loss / max(1, window_finite),
                "lr": sched.get_last_lr()[0],
                # 스케줄의 기준 lr. 이어 갈 때 바꾸거나 붕괴 규칙으로 낮추면 config.lr과 달라진다
                "base_lr": sched.base_lrs[0],
                "grad_norm": float(grad_norm),
                "skipped_steps": skipped_steps,
                "sec_per_step": elapsed / n,
                "data_wait_frac": window_wait / max(elapsed, 1e-9),
                "datasets_per_sec": config.batch_size * n / max(elapsed, 1e-9),
            }
            # 출력층 가중치 크기. 계속 커지면 출력이 날카로워지며 학습이 불안정해진다
            if (decoder := model.decoder_weight()) is not None:
                metrics["decoder_norm"] = decoder.norm().item()
            if use_cuda:
                metrics["gpu_mem_gb"] = torch.cuda.max_memory_allocated(device) / 1e9
            # attention logit 상한 중 가장 큰 값 (sample attention이 폭주하면 수만 이상으로 커진다)
            bounds = qk_logit_bounds(model)
            if bounds.numel():
                metrics["qk_bound_max"] = bounds.max().item()
            if sample := _sample_rows(model):
                metrics["qk_bound_sample_max"] = bounds[sample].max().item()
            if config.qk_clip is not None:
                # 이 window에서 clip할 때마다 줄인 head 수의 합
                metrics["qk_clipped_heads"] = float(window_clipped)
                metrics["qk_bound_before_clip"] = float(window_bound)
            history["step"].append(done)
            history["train_loss"].append(metrics["train_loss"])
            print(
                f"step {done}/{config.steps} loss {metrics['train_loss']:.4f} "
                f"lr {metrics['lr']:.2e} {metrics['sec_per_step']:.3f}s/step "
                f"wait {metrics['data_wait_frac']:.0%} skipped {skipped_steps}",
                flush=True,
            )
            if mlf is not None:
                mlf.log_metrics(metrics, step=done)
        if done % config.log_every == 0 or last:
            window_loss, window_wait, window_start, window_finite = 0.0, 0.0, time.time(), 0
            window_clipped, window_bound = 0, 0.0  # QK-clip: 줄인 head 수, 줄이기 전 최대 상한
            last_log_step = done

        if d.is_main and (done % config.val_every == 0 or last) and not rewound_at_end:
            val = _validate(model, val_batches, device, autocast, config.task, bar)
            history["val_step"].append(done)
            history["val_loss"].append(val["val_loss"])
            print("  val " + " ".join(f"{k} {v:.4f}" for k, v in val.items()), flush=True)
            if mlf is not None:
                mlf.log_metrics(val, step=done)

        keep = bool(config.keep_every) and done % config.keep_every == 0
        suspect = guard is not None and guard.suspect  # 무너지는 중일 수 있는 상태는 남기지 않는다
        if d.is_main and (done % config.checkpoint_every == 0 or keep or last) and not suspect:
            state = {
                # 모델 종류와 설정 (학습 run param 없이도 이 파일만으로 모델을 만든다)
                "model_type": config.model_type,
                "config": model.config.to_dict(),
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "scheduler": sched.state_dict(),
                "step": done,
                "history": history,
                "skipped_steps": skipped_steps,
                "borders": borders,
                "rewinds": rewinds,
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
        if last:
            break
        if guard is not None and done % config.snapshot_every == 0 and not guard.suspect:
            snapshot(done)

    final_path = out_dir / "tabpfn_v2_final.pt"
    if d.is_main:
        save_checkpoint(final_path, model, None if borders is None else borders.cpu())
        (out_dir / "history.json").write_text(json.dumps(history))
        (out_dir / "train_config.json").write_text(json.dumps(asdict(config)))
        if mlf is not None:
            mlf.log_artifact(str(final_path), "model")
            mlf.log_artifact(str(out_dir / "history.json"))
    if d.world_size > 1:
        dist.barrier()
        dist.destroy_process_group()
    return {"final_path": str(final_path), "history": history, "is_main": d.is_main}
