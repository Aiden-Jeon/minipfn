"""공용 prior-fitting 학습 루프 (TabPFN v2 논문 Methods "Training details"의 절차).

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

import collections
import contextlib
import copy
import json
import math
import os
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from minipfn.models import (
    DEFAULT_MODEL_TYPE,
    canonical_model_type,
    config_dict,
    config_from_dict,
    info_fields,
    save_checkpoint,
    state_config,
    state_weights,
)
from minipfn.models.bar_distribution import FullSupportBarDistribution
from minipfn.priors import prior_module
from minipfn.training.config import N_BUCKETS, TrainConfig
from minipfn.training.data import PriorBatches
from minipfn.training.losses import dequantize_targets, prior_loss
from minipfn.training.muon import MuonAdam, shard_group
from minipfn.training.schedule_free import ScheduleFreeAdamW, warmup_factor
from minipfn.training.stability import (
    CollapseGuard,
    apply_qk_clip,
    qk_logit_bounds,
    sample_attention_rows,
)
from minipfn.utils import DistContext, lr_factor, setup_distributed


def _borders(config: TrainConfig, d: DistContext, device: torch.device) -> torch.Tensor | None:
    """회귀(multitask의 회귀 쪽 포함)면 rank 0에서 구간 경계를 계산해 모든 GPU에 나눠 준다."""
    if "regression" not in config.tasks:
        return None
    config = config.task_configs()["regression"]
    borders = torch.empty(N_BUCKETS + 1, device=device)
    if d.is_main:
        start = time.time()
        borders = (
            prior_module(config.prior_type)
            .make_borders(
                config,
                N_BUCKETS,
                seed=20_000_000 + config.seed,
                n_jobs=max(1, min(32, (os.cpu_count() or 1) // 2)),
            )
            .to(device)
        )
        print(f"regression borders: {config.border_datasets} datasets, {time.time() - start:.0f}s")
    if d.world_size > 1:
        dist.broadcast(borders, src=0)
    return borders


def _make_val_batches(config: TrainConfig, n_batches: int = 8):
    """학습과 겹치지 않는 시드로 검증용 prior batch를 고정해 둔다 (val_batch × n_batches개)."""
    prior = config.make_prior(10_000_000 + config.seed)
    return [prior.sample_batch(config.val_batch) for _ in range(n_batches)]


def _forward_kwargs(config: TrainConfig, task: str) -> dict:
    """forward에 줄 task. multitask 모델만 받는다(한 task 모델의 호출은 예전 그대로)."""
    return {"task": task} if config.task == "multitask" else {}


@torch.no_grad()
def _validate(
    model, val_batches, device, autocast, task, bar, forward: dict | None = None
) -> dict[str, float]:
    forward = forward or {}  # multitask 모델에 줄 task (`_forward_kwargs`)
    model.eval()
    total, correct, count = 0.0, 0, 0
    for batch in val_batches:
        x, y = batch.x.to(device), batch.y.to(device)
        with autocast():
            logits = model(x, y[:, : batch.n_train], **forward)
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


def _validate_tasks(model, val_sets, device, autocast, config, bar) -> dict[str, float]:
    """task마다 검증. 한 task면 예전 지표 그대로, multitask면 지표 이름 뒤에 task를 붙이고
    `val_loss`는 두 task 손실의 평균이다."""
    if config.task != "multitask":
        return _validate(model, val_sets[config.task], device, autocast, config.task, bar)
    out = {}
    for task, batches in val_sets.items():
        val = _validate(model, batches, device, autocast, task, bar, _forward_kwargs(config, task))
        out.update({f"{k}_{task}": v for k, v in val.items()})
    out["val_loss"] = sum(out[f"val_loss_{t}"] for t in val_sets) / len(val_sets)
    return out


def _average_states(states) -> dict:
    """state_dict 여러 개의 평균 (LAWA). 실수 텐서는 float32로 더해 원래 dtype으로 돌려준다."""
    out = {}
    for key, first in states[0].items():
        if first.is_floating_point():
            out[key] = (sum(s[key].float() for s in states) / len(states)).to(first.dtype)
        else:
            out[key] = first
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


def check_checkpoint(state: dict, config: TrainConfig, name: str) -> None:
    """`state`(학습 중 checkpoint 또는 `save_checkpoint` 파일)가 `config`와 같은 task·모델인지.

    모양이 같은 다른 설정(head 수 등)은 `load_state_dict`가 그대로 읽어 버리므로 미리 맞춰 본다.
    모델 설정을 남기기 전의 학습 중 checkpoint는 task만 본다.
    """
    # 회귀 구간 경계는 회귀를 내는 모델(회귀, multitask)에만 있다
    has_borders = state.get("borders") is not None
    if has_borders != ("regression" in config.tasks):
        got = "regression" if has_borders else "classification"
        if "multitask" in (config.task, (state_config(state) or {}).get("task")):
            got = "회귀 구간 경계가 있는" if has_borders else "회귀 구간 경계가 없는"
        raise ValueError(f"{name}은 {got} checkpoint다 (설정의 task: {config.task})")
    saved_config = state_config(state)
    if saved_config is None:
        return
    saved_type = canonical_model_type(state.get("model_type", DEFAULT_MODEL_TYPE))
    if saved_type != canonical_model_type(config.model_type):
        raise ValueError(f"{name}은 {saved_type} 모델이다 (설정의 model_type: {config.model_type})")
    # 적지 않은 필드(기본값이라 빠진 것, 저장한 뒤에 생긴 것)는 그 기본값으로 채워 비교한다.
    # 이 코드가 모르는 필드가 있으면 config_from_dict가 그 이름으로 멈춘다
    try:
        saved = config_from_dict(saved_type, saved_config).to_dict()
    except ValueError as e:
        raise ValueError(f"{name}: {e}") from None
    expected = config.model_config().to_dict()
    # 가중치 모양과 상관없는 필드(학습 때 본 feature 수 등)는 이어 학습하며 바뀔 수 있다
    skip = set(info_fields(saved_type))
    diff = {
        k: (saved.get(k), v) for k, v in expected.items() if k not in skip and saved.get(k) != v
    }
    if diff:
        raise ValueError(f"{name}의 모델 설정이 이번 설정과 다르다 (저장된 값, 설정): {diff}")


def check_resume_step(state: dict, config: TrainConfig) -> None:
    """이어 갈 checkpoint의 step이 이번에 멈출 step보다 앞인지. 아니면 학습할 것이 없다.

    다른 설정의 run(예: paper run을 long_tables로)을 잘못 이으면 학습 없이 끝나고 결과를 그 run에
    덮어쓰게 된다.
    """
    saved_schedule_free = bool(state.get("schedule_free", False))
    if saved_schedule_free != config.schedule_free:
        # optimizer 상태가 다르다 (Adam: exp_avg·step, schedule-free: z·k). checkpoint의 가중치도
        # schedule-free면 평균 x라 Adam으로는 그대로 이을 수 없다
        raise ValueError(
            f"이어 갈 checkpoint는 schedule_free={saved_schedule_free}로 학습했다"
            f" (설정은 {config.schedule_free}). 원래 run과 같은 optimizer로 이어 간다"
        )
    saved_optimizer = state.get("optimizer_type", "adam")  # 이 키가 생기기 전의 run은 Adam
    if saved_optimizer != config.optimizer:
        raise ValueError(
            f"이어 갈 checkpoint는 optimizer={saved_optimizer}로 학습했다"
            f" (설정은 {config.optimizer}). 원래 run과 같은 optimizer로 이어 간다"
        )
    step, stop = int(state["step"]), min(config.stop_step or config.steps, config.steps)
    if step >= stop:
        raise ValueError(
            f"이어 갈 checkpoint가 이미 step {step}이라 멈출 step {stop}까지 학습할 것이 없다"
            " (같은 설정의 run을 이어 가는지, steps·stop_step이 맞는지 확인)"
        )


def check_source_checkpoint(path: str | Path, config: TrainConfig, *, resume: bool) -> float:
    """이어 가거나(`resume`) 가중치를 가져올 checkpoint를 이 설정으로 쓸 수 있는지 미리 본다
    (task·모델·이어 갈 step). `train`도 같은 것을 보지만, 학습 스크립트는 MLflow run을 열기 전에
    불러서 원래 run의 tag·코드를 덮어쓰거나 빈 run을 남기지 않는다. 틀리면 ValueError.

    반환값은 checkpoint가 쓰던 기준 lr이다. 이어서 학습할 때 lr을 따로 주지 않으면 이 값을 잇는다.
    checkpoint는 이 함수에서 한 번만 읽는다(mmap)."""
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    check_checkpoint(state, config, "resume_from" if resume else "init_from")
    if resume:
        check_resume_step(state, config)
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


def train(
    config: TrainConfig,
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
    # 인자만 보고 알 수 있는 잘못은 분산 프로세스 그룹을 열기 전에 막는다
    if resume_from is not None and init_from is not None:
        raise ValueError("resume_from과 init_from은 함께 줄 수 없다")
    d, device = setup_distributed()
    # 가속 옵션 tf32는 프로세스 전체 설정이라 끝나면 되돌린다
    # (같은 프로세스의 다음 학습·평가에 남지 않게)
    matmul_precision = torch.get_float32_matmul_precision()
    try:
        return _train(
            config,
            out_dir,
            d,
            device,
            resume_from=resume_from,
            mlflow_log=mlflow_log,
            init_from=init_from,
        )
    except BaseException:
        # 모델을 만든 뒤의 검사(QK-clip, checkpoint 모양)에서 멈추면 다른 GPU가 기다리므로
        # 프로세스 그룹을 닫고 올린다 (정상 경로는 `_train` 끝에서 닫는다)
        if d.world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()
        raise
    finally:
        torch.set_float32_matmul_precision(matmul_precision)


def _train(
    config: TrainConfig,
    out_dir: str | Path,
    d: DistContext,
    device: torch.device,
    *,
    resume_from: str | Path | None,
    mlflow_log: bool,
    init_from: str | Path | None,
) -> dict:
    """`train`의 본체. 프로세스 그룹은 `train`이 연다.
    정상으로 끝나면 여기서, 실패하면 `train`이 닫는다."""
    config = config.for_world_size(d.world_size)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    accumulation = config.grad_accumulation(d.world_size)
    use_cuda = device.type == "cuda"
    if use_cuda:
        torch.backends.cuda.enable_cudnn_sdp(False)
    if config.tf32:  # 가속 옵션 (TrainConfig 주석). 끄면 PyTorch 기본값(highest) 그대로
        torch.set_float32_matmul_precision("high")

    torch.manual_seed(config.seed)  # 모든 GPU가 같은 초기 가중치에서 시작한다
    model = config.build_model().to(device)
    if config.qk_clip is not None and not model.qk_clip_weights():
        raise ValueError(f"{config.model_type} 모델에는 QK-clip이 볼 attention이 없다")
    adam_kwargs = {"fused": True} if config.fused_adam and use_cuda else {}
    if config.schedule_free:
        # warmup 뒤 일정한 lr. 평균은 optimizer가 든다 (`training.schedule_free`)
        opt = ScheduleFreeAdamW(model.parameters(), lr=config.lr)
        schedule = lambda s: warmup_factor(s, config.warmup_steps)  # noqa: E731
    elif config.optimizer == "muon":
        # 블록 안 가중치 행렬은 Muon, 나머지는 Adam (`training.muon`). 스케줄은 Adam과 같다.
        # 분산 Muon이면 직교화를 GPU끼리 나눈다 (모든 GPU가 같이 group을 만든다)
        group = None if config.muon_distributed == "off" else shard_group(config.muon_distributed)
        opt = MuonAdam(
            model,
            lr=config.lr,
            weight_decay=config.muon_weight_decay,
            adam_kwargs=adam_kwargs,
            process_group=group,
        )
        schedule = lambda s: lr_factor(s, config.warmup_steps, config.steps)  # noqa: E731
    else:
        opt = torch.optim.Adam(model.parameters(), lr=config.lr, **adam_kwargs)
        schedule = lambda s: lr_factor(s, config.warmup_steps, config.steps)  # noqa: E731
    sched = torch.optim.lr_scheduler.LambdaLR(opt, schedule)

    eval_cache: dict = {}  # 이번 step의 평균 x (검증·LAWA·checkpoint가 같은 step에 한 번만 계산)

    def eval_state() -> dict:
        # 검증·저장할 가중치. schedule-free면 평균 x (`ScheduleFreeAdamW.average`), 아니면 그대로
        if not config.schedule_free:
            return model.state_dict()
        if "state" not in eval_cache:
            state = model.state_dict()
            for name, p in model.named_parameters():
                state[name] = opt.average(p)
            eval_cache["state"] = state
        return eval_cache["state"]

    start_step, skipped_steps, borders, rewinds = 0, 0, None, 0
    resumed_train_seconds = 0.0
    history: dict[str, list] = {"step": [], "train_loss": [], "val_step": [], "val_loss": []}
    if init_from is not None:
        state = torch.load(init_from, map_location=device, weights_only=True)
        check_checkpoint(state, config, "init_from")
        model.load_state_dict(state_weights(state))
        borders = state["borders"]
    if resume_from is not None:
        state = torch.load(resume_from, map_location=device, weights_only=True)
        check_checkpoint(state, config, "resume_from")
        check_resume_step(state, config)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        if config.schedule_free:
            opt.average_to_train_point()  # checkpoint에는 평균 x를 저장했다. 학습 점 y로
        sched.load_state_dict(state["scheduler"])
        start_step = int(state["step"])
        history = state["history"]
        skipped_steps = int(state["skipped_steps"])
        # 이어서 학습해도 붕괴 뒤의 데이터를 다시 보지 않도록 되돌린 횟수를 이어받는다
        rewinds = int(state.get("rewinds", 0))
        resumed_train_seconds = float(state.get("train_seconds", 0.0))
        _set_base_lr(opt, sched, config.lr, start_step, d.is_main)
        borders = state["borders"]
    if borders is None:
        borders = _borders(config, d, device)
    bar = FullSupportBarDistribution(borders).to(device) if borders is not None else None
    # multitask는 step마다 한 task의 정답 encoder·출력층만 쓰므로 쓰지 않은 파라미터를 허용한다
    # (모든 GPU가 같은 step에 같은 task를 쓴다)
    multitask = config.task == "multitask"
    ddp_model = (
        DistributedDataParallel(model, device_ids=[d.local_rank], find_unused_parameters=multitask)
        if d.world_size > 1
        else model
    )
    # 학습 forward. 가속 옵션 compile이면 반복 블록마다 compile한다(모델 전체를 compile하면 표
    # 크기에 따라 갈리는 Python 분기 때문에 batch마다 다시 compile된다). 블록 목록이 없는 모델은
    # 전체를
    train_forward = ddp_model
    if config.compile:
        # 폭이 다른 블록(TabPFN-3의 행 단계·ICL)이 같은 forward 코드를 써서 shape마다 한 벌씩
        # compile된다. 기본 cache 한도(8)를 넘으면 조용히 eager로 돌아가므로 넉넉히 둔다
        torch._dynamo.config.cache_size_limit = max(64, torch._dynamo.config.cache_size_limit)
        blocks = model.repeated_blocks() if hasattr(model, "repeated_blocks") else []
        for block in blocks:
            block.compile(dynamic=True)
        if not blocks:
            train_forward = torch.compile(ddp_model, dynamic=True)
    # rank 0의 검증용 사본: schedule-free의 평균 x나 LAWA 평균을 담는다
    # (학습 가중치는 건드리지 않는다)
    eval_copy_model = None
    averages = collections.deque(maxlen=config.weight_average) if config.weight_average else None
    if averages is not None and start_step and d.is_main:
        # 평균에 쓸 가중치는 메모리에만 있다(모델 K개를 checkpoint에 넣기에는 크다)
        print(
            f"LAWA: 이어 학습이라 평균 창이 비어 있다. step {start_step}부터"
            f" {config.weight_average * config.weight_average_every} step 뒤에 다시 찬다",
            flush=True,
        )
    # 순수 학습 시간 (검증·checkpoint·기록 빼고). 목표 도달 시간을 잰다. checkpoint에 남겨 이어
    # 학습해도 누적된다
    train_seconds = resumed_train_seconds

    def eval_copy(state: dict):
        nonlocal eval_copy_model
        if eval_copy_model is None:
            eval_copy_model = copy.deepcopy(model)
        eval_copy_model.load_state_dict(state)
        return eval_copy_model

    def autocast():
        if use_cuda:
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    task_configs = config.task_configs()
    # prior 생성 worker는 task끼리 나눠 쓴다 (multitask여도 GPU당 CPU 사용량은 같게)
    workers = config.num_workers // len(task_configs) if config.num_workers else 0
    if config.num_workers:
        workers = max(1, workers)
    loader_kwargs = {"num_workers": workers, "pin_memory": use_cuda}
    if workers > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=4)

    def make_batches(from_step: int, rewinds: int) -> dict:
        # 이어서 학습할 때 같은 데이터를 다시 보지 않도록 시드에 시작 step을 섞는다.
        # 붕괴로 되돌아가면 되돌린 횟수도 섞어 다른 데이터로 다시 간다. multitask는 task 번호도
        seed = (config.seed, from_step, d.rank) + ((rewinds,) if rewinds else ())
        return {
            task: iter(
                DataLoader(
                    PriorBatches(task_config, seed=seed + ((i,) if multitask else ())),
                    batch_size=None,
                    **loader_kwargs,
                )
            )
            for i, (task, task_config) in enumerate(task_configs.items())
        }

    batches = make_batches(start_step, rewinds)

    mlf, val_sets = None, None
    if d.is_main:
        val_sets = {task: _make_val_batches(c) for task, c in task_configs.items()}
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
    window_tasks = {t: [0.0, 0] for t in config.tasks}
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
        t_step = time.time()
        eval_cache.clear()  # 가중치가 바뀌므로 지난 step의 평균은 버린다
        # multitask는 step마다 task를 번갈아 쓴다 (모든 GPU가 같은 task. 비율은 1:1, 여기서 정함)
        task = config.tasks[step % len(config.tasks)]
        step_loss = torch.zeros((), device=device)
        micro_info = []
        for micro in range(accumulation):
            t_wait = time.time()
            x, y, n_train = next(batches[task])
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            window_wait += time.time() - t_wait
            sync = micro == accumulation - 1 or d.world_size == 1
            ctx = contextlib.nullcontext() if sync else ddp_model.no_sync()
            with ctx:
                with autocast():
                    logits = train_forward(x, y[:, :n_train], **_forward_kwargs(config, task))
                y_eval = y[:, n_train:]
                if task == "regression" and config.dequantize_buckets:
                    y_eval = dequantize_targets(y, n_train, bar, config.dequantize_buckets)
                loss = prior_loss(logits, y_eval, task, bar) / accumulation
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
            if multitask:
                window_tasks[task][0] += step_loss.item()
                window_tasks[task][1] += 1
        train_seconds += time.time() - t_step  # 위 .item()에서 GPU를 기다렸다

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
                if averages is not None:  # 붕괴 중이었을 수 있는 가중치는 평균에서 뺀다
                    averages.clear()
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
                window_tasks = {t: [0.0, 0] for t in config.tasks}
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
            if multitask:  # train_loss는 두 task를 번갈아 쓴 평균이다
                for t, (total, count) in window_tasks.items():
                    if count:
                        metrics[f"train_loss_{t}"] = total / count
            # 출력층 가중치 크기. 계속 커지면 출력이 날카로워지며 학습이 불안정해진다
            if (decoder := model.decoder_weight()) is not None:
                metrics["decoder_norm"] = decoder.norm().item()
            if use_cuda:
                metrics["gpu_mem_gb"] = torch.cuda.max_memory_allocated(device) / 1e9
            # attention logit 상한 중 가장 큰 값 (sample attention이 폭주하면 수만 이상으로 커진다)
            bounds = qk_logit_bounds(model)
            if bounds.numel():
                metrics["qk_bound_max"] = bounds.max().item()
            if sample := sample_attention_rows(model):
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
            window_tasks = {t: [0.0, 0] for t in config.tasks}
            window_clipped, window_bound = 0, 0.0  # QK-clip: 줄인 head 수, 줄이기 전 최대 상한
            last_log_step = done

        if (
            averages is not None
            and d.is_main
            and done % config.weight_average_every == 0
            and not rewound_at_end
        ):
            averages.append(_to_cpu(eval_state()))
        if d.is_main and (done % config.val_every == 0 or last) and not rewound_at_end:
            val_model = eval_copy(eval_state()) if config.schedule_free else model
            val = _validate_tasks(val_model, val_sets, device, autocast, config, bar)
            # LAWA는 창(K개)이 다 찼을 때만 낸다. 이어 학습하면 창이 비어서 다시 찬다
            if averages is not None and len(averages) == config.weight_average:
                averaged = eval_copy(_average_states(list(averages)))
                avg = _validate_tasks(averaged, val_sets, device, autocast, config, bar)
                val.update({f"{k}_avg": v for k, v in avg.items()})
            val["train_seconds"] = train_seconds
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
                # `config`가 아니라 `model_config`: 예전 코드가 save_checkpoint 파일로 오인하지
                # 않도록 (`models.state_config`)
                "model_config": config_dict(model.config),
                # schedule-free면 평균 x (평가가 그대로 읽는다. 이어 학습은 x와 z로 y를 만든다)
                "model": eval_state(),
                "optimizer": opt.state_dict(),
                "scheduler": sched.state_dict(),
                "step": done,
                "history": history,
                "skipped_steps": skipped_steps,
                "borders": borders,
                "rewinds": rewinds,
                "train_seconds": train_seconds,
                # optimizer 종류 (이어 학습할 때 같은지 본다, `check_resume_step`)
                "schedule_free": config.schedule_free,
                "optimizer_type": config.optimizer,
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

    final_path = out_dir / "final.pt"
    if d.is_main:
        final_borders = None if borders is None else borders.cpu()
        final_model = eval_copy(eval_state()) if config.schedule_free else model
        save_checkpoint(final_path, final_model, final_borders)
        if averages is not None and len(averages) == config.weight_average:  # LAWA 평균
            save_checkpoint(
                out_dir / "final_avg.pt", eval_copy(_average_states(list(averages))), final_borders
            )
            if mlf is not None:
                mlf.log_artifact(str(out_dir / "final_avg.pt"), "model")
        (out_dir / "history.json").write_text(json.dumps(history))
        (out_dir / "train_config.json").write_text(json.dumps(asdict(config)))
        if mlf is not None:
            mlf.log_artifact(str(final_path), "model")
            mlf.log_artifact(str(out_dir / "history.json"))
    if d.world_size > 1:
        dist.barrier()
        dist.destroy_process_group()
    return {"final_path": str(final_path), "history": history, "is_main": d.is_main}
