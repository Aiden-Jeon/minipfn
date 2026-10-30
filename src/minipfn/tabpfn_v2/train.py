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
import json
import os
import shutil
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from minipfn.bar_distribution import FullSupportBarDistribution
from minipfn.tabpfn_v2.model import TabPFNv2, TabPFNv2Config, save_checkpoint
from minipfn.tabpfn_v2.prior import (
    MAX_CLASSES,
    MAX_FEATURES,
    MAX_TRAIN_ROWS,
    N_EVAL_ROWS,
    TabPFNv2Prior,
    regression_borders,
)
from minipfn.utils import DistContext, lr_factor, setup_distributed

N_BUCKETS = 5000


@dataclass
class TabPFNv2TrainConfig:
    """모델·prior·최적화 설정. smoke(로컬 CPU), medium(짧은 실측), paper(논문 규모)."""

    task: Literal["classification", "regression"] = "classification"
    # 모델 (논문 기본 모델 크기)
    emsize: int = 192
    nlayers: int = 12
    nhead: int = 6
    # prior / 데이터셋 크기 (논문 Methods)
    max_train_rows: int = MAX_TRAIN_ROWS
    n_eval_rows: int = N_EVAL_ROWS
    max_features: int = MAX_FEATURES
    missing_dataset_prob: float = 0.3
    border_datasets: int = 20_000  # 회귀 구간 경계를 정할 때 뽑는 데이터셋 수 (여기서 정함)
    # 최적화 (논문: 약 200만 step × batch 64. warmup 길이와 lr은 여기서 정함)
    batch_size: int = 64
    micro_batch: int = 8
    steps: int = 2_000_000  # 학습률 스케줄의 전체 길이
    stop_step: int | None = None  # 이번 실행을 멈출 step (None이면 steps까지). 나눠 학습할 때 쓴다
    deadline: float | None = None  # 이 시각(Unix time)을 넘기면 checkpoint를 남기고 멈춘다
    warmup_steps: int = 10_000
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

    @property
    def n_out(self) -> int:
        return MAX_CLASSES if self.task == "classification" else N_BUCKETS

    @classmethod
    def paper(cls, task: str = "classification") -> TabPFNv2TrainConfig:
        return cls(task=task)

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

    def model_config(self) -> TabPFNv2Config:
        return TabPFNv2Config(
            emsize=self.emsize, nlayers=self.nlayers, nhead=self.nhead, n_out=self.n_out
        )

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

    def make_prior(self, seed: int) -> TabPFNv2Prior:
        return TabPFNv2Prior(
            self.task,
            seed=seed,
            missing_dataset_prob=self.missing_dataset_prob,
            max_train_rows=self.max_train_rows,
            n_eval_rows=self.n_eval_rows,
            max_features=self.max_features,
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
            max_train_rows=config.max_train_rows,
            n_eval_rows=config.n_eval_rows,
            max_features=config.max_features,
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


def train_tabpfn_v2(
    config: TabPFNv2TrainConfig,
    out_dir: str | Path,
    resume_from: str | Path | None = None,
    mlflow_log: bool = True,
) -> dict:
    """prior-fitting을 실행하고 마지막 체크포인트 경로와 loss 기록을 반환한다.

    rank 0만 MLflow에 기록한다. 활성 run이 있으면 거기에 이어서 쓴다(AI Runtime은
    `MLFLOW_RUN_ID`로 run을 넘겨준다). `checkpoint_every` step마다 학습 상태 전체를
    `out_dir/latest.pt`에 저장하고 MLflow 아티팩트(`checkpoints/`)로 올린다. `keep_every`
    step마다는 덮어쓰지 않는 `step_XXXXXXX.pt`로도 남긴다.
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
    model = TabPFNv2(config.model_config()).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=config.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: lr_factor(s, config.warmup_steps, config.steps)
    )
    start_step, skipped_steps, borders = 0, 0, None
    history: dict[str, list] = {"step": [], "train_loss": [], "val_step": [], "val_loss": []}
    if resume_from is not None:
        state = torch.load(resume_from, map_location=device, weights_only=True)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["scheduler"])
        start_step = int(state["step"])
        history = state["history"]
        skipped_steps = int(state["skipped_steps"])
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

    # 이어서 학습할 때 같은 데이터를 다시 보지 않도록 시드에 시작 step을 섞는다
    dataset = PriorBatches(config, seed=(config.seed, start_step, d.rank))
    loader_kwargs = {"num_workers": config.num_workers, "pin_memory": use_cuda}
    if config.num_workers > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=4)
    batches = iter(DataLoader(dataset, batch_size=None, **loader_kwargs))

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

    ddp_model.train()
    window_loss, window_wait, window_start, window_finite = 0.0, 0.0, time.time(), 0
    last_log_step, skip_reports = start_step, 0
    stop_step = min(config.stop_step or config.steps, config.steps)
    for step in range(start_step, stop_step):
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
        if d.is_main and (done % config.log_every == 0 or last):
            n = done - last_log_step
            elapsed = time.time() - window_start
            metrics = {
                "train_loss": window_loss / max(1, window_finite),
                "lr": sched.get_last_lr()[0],
                "grad_norm": float(grad_norm),
                # 출력층 가중치 크기. 계속 커지면 출력이 날카로워지며 학습이 불안정해진다
                "decoder_norm": model.decoder[-1].weight.norm().item(),
                "skipped_steps": skipped_steps,
                "sec_per_step": elapsed / n,
                "data_wait_frac": window_wait / max(elapsed, 1e-9),
                "datasets_per_sec": config.batch_size * n / max(elapsed, 1e-9),
            }
            if use_cuda:
                metrics["gpu_mem_gb"] = torch.cuda.max_memory_allocated(device) / 1e9
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
            window_loss, window_wait, window_start, window_finite = 0.0, 0.0, time.time(), 0
            last_log_step = done

        if d.is_main and (done % config.val_every == 0 or last):
            val = _validate(model, val_batches, device, autocast, config.task, bar)
            history["val_step"].append(done)
            history["val_loss"].append(val["val_loss"])
            print("  val " + " ".join(f"{k} {v:.4f}" for k, v in val.items()), flush=True)
            if mlf is not None:
                mlf.log_metrics(val, step=done)

        keep = bool(config.keep_every) and done % config.keep_every == 0
        if d.is_main and (done % config.checkpoint_every == 0 or keep or last):
            state = {
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "scheduler": sched.state_dict(),
                "step": done,
                "history": history,
                "skipped_steps": skipped_steps,
                "borders": borders,
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
