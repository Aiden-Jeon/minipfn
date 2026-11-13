"""activation checkpointing을 켠 학습 step과 끈 학습 step을 견준다 (issue #79).

GPU 한 장에서 세 가지를 본다.

1. 같은지 (`check`): float32(TF32 끔)에서 같은 batch의 손실·gradient. TabPFN-3.5 multitask 모델
   (curriculum 설정, soft-cap 50) 두 task, 가속 옵션 compile을 끈 것과 켠 것. 분류 decoder가 score
   행렬을 만드는 것(`fused` 끔, 그러면 비트 단위로 같아야 한다)과 만들지 않는 것(기본)
2. 학습 step (`bench`): 학습 루프와 같은 한 step — bf16 autocast forward, 학습 손실(회귀는 bar
   distribution), backward, gradient clip, Muon(모든 파라미터, curriculum 설정) 갱신. 블록 compile·
   TF32는 실제 run처럼 켠다. stage 2의 가장 큰 표와 stage 3의 feature 상한마다 가장 큰 표에서 step
   시간과 최대 메모리(optimizer 상태 포함). checkpointing 끔·켬
3. 검증 (`val`): 학습 루프의 검증처럼 gradient 없이 데이터셋 8개 batch. 학습 forward 경로와 추론
   경로(`inference=True`, checkpointing을 켜면 큰 표의 검증이 이것을 쓴다)

  databricks air run -f scripts/tabpfn_v3/activation_checkpointing.yaml --watch \\
    --override mlflow_experiment_directory=$MLFLOW_EXPERIMENT_DIR
"""

from __future__ import annotations

import argparse
import json
import time

import torch

from minipfn.models import column_row_transformer as M
from minipfn.models.bar_distribution import FullSupportBarDistribution
from minipfn.recipes import tabpfn_v3
from minipfn.training.losses import prior_loss
from minipfn.training.muon import MuonAdam

# (micro batch, 행 수, feature 수, 학습 행 비율). stage 2의 가장 큰 표와 stage 3의 feature 상한
# (`priors.table_size.LONG_TABLE_FEATURE_CAPS`)마다 가장 큰 표. 학습 행은 81%까지
TABLES = {
    "2": (1, 10_240, 100, 0.81),
    "3_20k": (1, 20_000, 80, 0.81),
    "3_30k": (1, 30_000, 60, 0.81),
    "3_40k": (1, 40_000, 40, 0.81),
    "3_50k": (1, 50_000, 30, 0.81),
    "3_60k": (1, 60_000, 20, 0.81),
}
TASKS = ("classification", "regression")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tables", default=",".join(TABLES))
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--modes", default="on,off", help="checkpointing on·off 가운데 잴 것")
    parser.add_argument("--val-batch", type=int, default=8)
    parser.add_argument("--skip-check", action="store_true")
    parser.add_argument("--skip-val", action="store_true")
    parser.add_argument("--out", default="activation_checkpointing.json")
    return parser.parse_args(argv)


def stage_config():
    return tabpfn_v3.curriculum(3)


def build(device, compile_blocks: bool):
    torch.manual_seed(0)
    model = stage_config().build_model().to(device).train()
    if compile_blocks:  # 학습 루프(`training.loop`)와 같게 블록마다, 모델의 inductor 설정으로
        torch._dynamo.config.cache_size_limit = max(64, torch._dynamo.config.cache_size_limit)
        options = model.compile_options()
        for block in model.repeated_blocks():
            block.compile(dynamic=True, options=options)
    return model


def make_bar(device) -> FullSupportBarDistribution:
    # 구간 경계는 값만 다르고 계산은 같다 (학습은 prior에서 정한 경계 5,000개)
    n = stage_config().model_config().n_buckets
    return FullSupportBarDistribution(torch.linspace(-4, 4, n + 1)).to(device)


def batch(task: str, b: int, r: int, c: int, frac: float, device, seed: int = 0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    n_train = int(r * frac)
    x = torch.randn(b, r, c, generator=g)
    if task == "classification":
        y = torch.randint(0, 10, (b, r), generator=g).float()
    else:
        y = torch.randn(b, r, generator=g)
    return x.to(device), y.to(device), n_train


def set_mode(model, on: bool) -> None:
    model.activation_checkpointing_rows = 0 if on else None


def loss_of(model, bar, x, y, n_train, task, autocast: bool) -> torch.Tensor:
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
        logits = model(x, y[:, :n_train], task)
    return prior_loss(logits, y[:, n_train:], task, bar)


def grads(model, bar, x, y, n_train, task):
    model.zero_grad(set_to_none=True)
    loss = loss_of(model, bar, x, y, n_train, task, autocast=False)
    loss.backward()
    g = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    return loss.item(), g


def compare(a, b) -> dict:
    """손실 차이, 모든 gradient를 이은 벡터의 상대 차이, 파라미터마다 gradient 상대 차이의 중앙값,
    비트 단위로 같은지."""
    (la, ga), (lb, gb) = a, b
    names = list(ga)
    flat_a = torch.cat([ga[n].flatten() for n in names])
    flat_b = torch.cat([gb[n].flatten() for n in names])
    rel = torch.tensor(
        [((ga[n] - gb[n]).norm() / ga[n].norm().clamp_min(1e-30)).item() for n in names]
    )
    return {
        "loss": la,
        "loss_diff": abs(la - lb),
        "grad_rel_all": ((flat_a - flat_b).norm() / flat_a.norm()).item(),
        "grad_rel_median": rel.median().item(),
        "same_params": sorted(ga) == sorted(gb),
        "bitwise": la == lb and all(torch.equal(ga[n], gb[n]) for n in names),
    }


def check(device) -> dict:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    bar = make_bar(device)
    result = {}
    for compile_blocks in (False, True):
        model = build(device, compile_blocks)
        for task in TASKS:
            x, y, n_train = batch(task, 2, 2048, 24, 0.8, device, seed=1)
            set_mode(model, False)
            reference = grads(model, bar, x, y, n_train, task)
            for fused in (False, True):
                M.CHECKPOINT_FUSED_DECODER = fused
                set_mode(model, True)
                run = grads(model, bar, x, y, n_train, task)
                name = f"compile_{'on' if compile_blocks else 'off'}_{task}_fused_{fused}"
                result[name] = compare(reference, run)
                print(name, json.dumps(result[name]), flush=True)
        M.CHECKPOINT_FUSED_DECODER = True
        del model
        torch._dynamo.reset()
        torch.cuda.empty_cache()
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    return result


def bench(device, tables, modes, steps, warmup) -> list[dict]:
    """학습 루프의 step 하나 (`training.loop._train`): forward·손실·backward·clip·Muon 갱신."""
    config = stage_config()
    torch.backends.cuda.enable_cudnn_sdp(False)  # 학습 루프와 같다
    torch.set_float32_matmul_precision("high")  # 가속 옵션 tf32
    model = build(device, compile_blocks=True)
    opt = MuonAdam(
        model,
        lr=config.lr,
        weight_decay=config.muon_weight_decay,
        scope=config.muon_scope,
        momentum=config.muon_momentum,
    )
    bar = make_bar(device)
    rows = []
    for name in tables:
        b, r, c, frac = TABLES[name]
        for mode in modes:
            set_mode(model, mode == "on")
            for task in TASKS:
                x, y, n_train = batch(task, b, r, c, frac, device)
                row = dict(table=name, rows=r, features=c, checkpointing=mode, task=task)
                opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                try:
                    times = []
                    for i in range(warmup + steps):
                        torch.cuda.synchronize()
                        t0 = time.perf_counter()
                        loss = loss_of(model, bar, x, y, n_train, task, autocast=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                        opt.step()
                        opt.zero_grad(set_to_none=True)
                        torch.cuda.synchronize()
                        if i >= warmup:
                            times.append(time.perf_counter() - t0)
                    row["seconds"] = sum(times) / len(times)
                    row["peak_gb"] = torch.cuda.max_memory_allocated() / 2**30
                    row["reserved_gb"] = torch.cuda.max_memory_reserved() / 2**30
                    row["loss"] = loss.item()
                except torch.OutOfMemoryError:
                    row["oom"] = True
                except Exception as e:  # 한 설정이 실패해도 나머지는 잰다
                    row["error"] = f"{type(e).__name__}: {str(e)[:500]}"
                loss = None
                opt.zero_grad(set_to_none=True)
                print(json.dumps(row), flush=True)
                rows.append(row)
    return rows


@torch.no_grad()
def val(device, tables, val_batch) -> list[dict]:
    """학습 루프의 검증 (`training.loop._validate`): eval, gradient 없음, 데이터셋 `val_batch`개."""
    model = build(device, compile_blocks=True).eval()
    bar = make_bar(device)
    rows = []
    for name in tables:
        _, r, c, frac = TABLES[name]
        for inference in (True, False):
            for task in TASKS:
                x, y, n_train = batch(task, val_batch, r, c, frac, device)
                row = dict(table=name, rows=r, features=c, inference=inference, task=task)
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                try:
                    times = []
                    for _ in range(2):
                        torch.cuda.synchronize()
                        t0 = time.perf_counter()
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            logits = model(x, y[:, :n_train], task, inference=inference)
                        loss = prior_loss(logits, y[:, n_train:], task, bar).item()
                        torch.cuda.synchronize()
                        times.append(time.perf_counter() - t0)
                    row["seconds"] = times[-1]
                    row["peak_gb"] = torch.cuda.max_memory_allocated() / 2**30
                    row["loss"] = loss
                except torch.OutOfMemoryError:
                    row["oom"] = True
                except Exception as e:
                    row["error"] = f"{type(e).__name__}: {str(e)[:500]}"
                logits = None
                print(json.dumps(row), flush=True)
                rows.append(row)
    return rows


def main(argv=None) -> None:
    args = parse_args(argv)
    device = torch.device("cuda")
    tables = args.tables.split(",")
    result = {}
    if not args.skip_check:
        result["check"] = check(device)
    result["bench"] = bench(device, tables, args.modes.split(","), args.steps, args.warmup)
    torch._dynamo.reset()
    torch.cuda.empty_cache()
    if not args.skip_val:
        result["val"] = val(device, tables, args.val_batch)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    try:
        import mlflow

        if mlflow.active_run() is None:
            mlflow.start_run()
        mlflow.log_artifact(args.out)
    except Exception as e:  # MLflow가 없는 로컬 실행
        print(f"MLflow 기록 건너뜀: {e}", flush=True)


if __name__ == "__main__":
    main()
