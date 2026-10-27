"""TabPFN prior-fitting 학습 루프 (논문 3절, v1 `train.py`).

매 optimizer step: prior에서 데이터셋 `batch_size`개를 GPU들에 나눠 만들고, forward batch마다
학습 행 / 테스트 행으로 나눠 테스트 행의 cross-entropy를 줄인다. GPU가 여러 장이면
`torchrun`으로 띄운 DDP로 gradient를 평균한다. 유효 batch는 v1과 같이

    batch_size = group_size × groups_per_batch × grad_accumulation × world_size

이고, grad_accumulation은 나머지 값으로 자동 계산한다. 학습률은 v1처럼 linear warmup 뒤
cosine으로 줄이되, cosine 길이(schedule_steps)와 실제 학습 step 수(steps)를 따로 둔다
(논문은 400 epoch 스케줄의 약 141 epoch 지점인 18,000 step에서 멈췄다).
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from minipfn.tabpfn import spec
from minipfn.tabpfn.model import (
    TabPFNTransformer,
    classification_loss,
    configure_attention_backends,
)
from minipfn.tabpfn.prior import TabPFNPrior, TabPFNPriorDataset


@dataclass
class TabPFNConfig:
    """모델·prior·최적화 설정. smoke(로컬 CPU), medium(GPU 1장, 예산 1/10), paper(논문)."""

    # 모델
    emsize: int = spec.EMSIZE
    nhead: int = spec.NHEAD
    nhid: int = spec.NHID
    nlayers: int = spec.NLAYERS
    dropout: float = spec.DROPOUT
    num_features: int = spec.MAX_FEATURES
    n_out: int = spec.MAX_CLASSES
    # prior / 데이터셋 구성
    seq_len: int = spec.SEQ_LEN
    max_eval_pos: int = spec.MAX_EVAL_POS
    group_size: int = spec.DATASETS_PER_PRIOR_GROUP
    groups_per_batch: int = 1
    # 최적화
    batch_size: int = spec.BATCH_SIZE
    steps: int = spec.TOTAL_STEPS
    warmup_steps: int = spec.WARMUP_STEPS
    schedule_steps: int = spec.SCHEDULE_STEPS
    lr: float = spec.LR
    weight_decay: float = spec.WEIGHT_DECAY
    grad_clip: float = spec.GRAD_CLIP
    # 실행
    seed: int = 0
    num_workers: int = 8
    log_every: int = 25
    val_every: int = 500
    checkpoint_every: int = 1000
    profile_steps: int = 0  # > 0이면 rank 0에서 torch.profiler로 이만큼의 step을 재서 출력한다

    @classmethod
    def paper(cls) -> TabPFNConfig:
        return cls()

    @classmethod
    def medium(cls) -> TabPFNConfig:
        """논문 모델·prior·batch 그대로, step과 스케줄만 1/10."""
        return cls(
            steps=spec.TOTAL_STEPS // 10,
            warmup_steps=spec.WARMUP_STEPS // 10,
            schedule_steps=spec.SCHEDULE_STEPS // 10,
            val_every=200,
            checkpoint_every=300,
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
        )

    def grad_accumulation(self, world_size: int) -> int:
        per_forward = self.group_size * self.groups_per_batch * world_size
        if self.batch_size % per_forward:
            raise ValueError(
                f"batch_size {self.batch_size}가 group_size × groups_per_batch × world_size"
                f" = {per_forward}로 나눠떨어지지 않습니다"
            )
        return self.batch_size // per_forward


def build_tabpfn_model(config: TabPFNConfig) -> TabPFNTransformer:
    return TabPFNTransformer(
        num_features=config.num_features,
        n_out=config.n_out,
        emsize=config.emsize,
        nhead=config.nhead,
        nhid=config.nhid,
        nlayers=config.nlayers,
        dropout=config.dropout,
    )


def lr_factor(step: int, warmup_steps: int, schedule_steps: int) -> float:
    """v1 get_cosine_schedule_with_warmup: 0에서 선형으로 올린 뒤 cosine으로 0까지."""
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, schedule_steps - warmup_steps)
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))


def save_tabpfn(model: torch.nn.Module, config: TabPFNConfig, path: str | Path) -> None:
    """가중치와 config만 저장한다 (`weights_only=True`로 다시 읽을 수 있는 형식)."""
    torch.save({"state_dict": model.state_dict(), "config": asdict(config)}, path)


def tabpfn_from_state(
    config: dict, state_dict: dict[str, torch.Tensor], map_location: str | torch.device = "cpu"
) -> tuple[TabPFNTransformer, TabPFNConfig]:
    """저장된 config(dict)와 가중치로 추론용 모델을 만든다 (`.pt`와 Hub 형식이 함께 쓴다)."""
    config = TabPFNConfig(**config)
    model = build_tabpfn_model(config)
    model.load_state_dict(state_dict)
    return model.to(map_location).eval(), config


def load_tabpfn(
    path: str | Path, map_location: str | torch.device = "cpu"
) -> tuple[TabPFNTransformer, TabPFNConfig]:
    ckpt = torch.load(path, map_location=map_location, weights_only=True)
    return tabpfn_from_state(ckpt["config"], ckpt["state_dict"], map_location)


@dataclass
class _Dist:
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def _setup_distributed() -> tuple[_Dist, torch.device]:
    """torchrun 환경변수가 있으면 NCCL 프로세스 그룹을 연다."""
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        d = _Dist(
            int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), int(os.environ["LOCAL_RANK"])
        )
        torch.cuda.set_device(d.local_rank)
        dist.init_process_group("nccl")
        return d, torch.device("cuda", d.local_rank)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _Dist(), device


def _make_val_batch(config: TabPFNConfig, n_groups: int = 8):
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
        valid = y_test != spec.IGNORE_INDEX
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
    `out_dir/latest.pt`에 저장하고 MLflow 아티팩트(`checkpoints/`)로 올린다. `resume_from`에
    그 파일을 주면 이어서 학습한다.
    """
    d, device = _setup_distributed()
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
    dataset = TabPFNPriorDataset(
        seq_len=config.seq_len,
        max_eval_pos=config.max_eval_pos,
        num_features=config.num_features,
        max_classes=config.n_out,
        group_size=config.group_size,
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

        if d.is_main and (done % config.checkpoint_every == 0 or done == config.steps):
            state = {
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "scheduler": sched.state_dict(),
                "step": done,
                "history": history,
                "skipped_steps": skipped_steps,
            }
            torch.save(state, out_dir / "latest.pt")
            if mlf is not None:
                mlf.log_artifact(str(out_dir / "latest.pt"), "checkpoints")

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
