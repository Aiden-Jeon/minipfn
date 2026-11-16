"""TabPFN-2.5 긴 표 단계(tuned_stage2·3)의 학습 step 메모리·시간 (issue #81).

GPU 한 장에서 tuned_stage2·3 설정(`recipes.tabpfn_v2.tuned_long`, 2.5 구조)의 가장 큰 표마다:

1. 학습 step (`bench`): 학습 루프와 같은 한 step — 층마다 compile, TF32, bf16 autocast forward,
   학습 손실(회귀는 bar distribution), backward, gradient clip, optimizer 갱신(분류 fused Adam,
   회귀 Muon). activation checkpointing 끔·켬. 최대 메모리(optimizer 상태 포함)와 step 시간
2. 검증 (`val`): 학습 루프의 검증처럼 gradient 없이 데이터셋 `val_batch`개

표는 stage 2의 셀 상한(10,240 × 100)과 stage 3의 셀 상한(50,000 × 30)으로 정한 가장 큰 표다
(`TABLES`). 학습 행은 81%.

  databricks air run -f scripts/tabpfn_v2_5/long_tables_memory.yaml --watch \\
    --override mlflow_experiment_directory=$MLFLOW_EXPERIMENT_DIR
"""

from __future__ import annotations

import argparse
import json
import time

import torch

from minipfn.models.bar_distribution import FullSupportBarDistribution
from minipfn.recipes import tabpfn_v2 as recipe
from minipfn.training.config import N_BUCKETS
from minipfn.training.losses import prior_loss
from minipfn.training.muon import MuonAdam

# 이름: (stage, 행 수, feature 수). stage 2: 셀 1,024,000개(10,240행 × 100)까지, 짧고 넓은 표 포함.
# stage 3: 셀 1,500,000개(50,000행 × 30)까지
TABLES = {
    "2_10k": (2, 10_240, 100),
    # activation checkpointing을 켜지 않는 가장 큰 표 (`TUNED_LONG_COMMON`의 4,096행)
    "2_4k": (2, 4_096, 100),
    "2_5k": (2, 5_000, 204),
    "2_wide": (2, 2_048, 500),
    "3_20k": (3, 20_000, 75),
    "3_30k": (3, 30_000, 50),
    "3_40k": (3, 40_000, 37),
    "3_50k": (3, 50_000, 30),
}
TASKS = ("classification", "regression")
TRAIN_FRACTION = 0.81


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tables", default=",".join(TABLES))
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--modes", default="off,on", help="activation checkpointing 끔·켬")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--val-batch", type=int, default=4)
    parser.add_argument("--skip-val", action="store_true")
    parser.add_argument("--out", default="long_tables_memory.json")
    return parser.parse_args(argv)


def stage_config(stage: int, task: str):
    return recipe.with_architecture(recipe.tuned_long(stage, task), "v2.5")


def build(task: str, device):
    torch.manual_seed(0)
    config = stage_config(3, task)
    model = config.build_model().to(device).train()
    torch._dynamo.config.cache_size_limit = max(64, torch._dynamo.config.cache_size_limit)
    for block in model.repeated_blocks():  # 학습 루프(`training.loop`)와 같게 층마다
        block.compile(dynamic=True)
    if config.optimizer == "muon":
        opt = MuonAdam(
            model,
            lr=config.lr,
            weight_decay=config.muon_weight_decay,
            adam_kwargs={"fused": True},
            scope=config.muon_scope,
            momentum=config.muon_momentum,
        )
    else:
        opt = torch.optim.Adam(model.parameters(), lr=config.lr, fused=True)
    return config, model, opt


def make_bar(device) -> FullSupportBarDistribution:
    # 구간 경계는 값만 다르고 계산은 같다 (학습은 prior에서 정한 경계)
    return FullSupportBarDistribution(torch.linspace(-4, 4, N_BUCKETS + 1)).to(device)


def batch(task: str, b: int, r: int, c: int, device, seed: int = 0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    n_train = int(r * TRAIN_FRACTION)
    x = torch.randn(b, r, c, generator=g)
    if task == "classification":
        y = torch.randint(0, 10, (b, r), generator=g).float()
    else:
        y = torch.randn(b, r, generator=g)
    return x.to(device), y.to(device), n_train


def bench(device, tables, tasks, modes, steps, warmup) -> list[dict]:
    torch.backends.cuda.enable_cudnn_sdp(False)  # 학습 루프와 같다
    torch.set_float32_matmul_precision("high")  # 가속 옵션 tf32
    bar = make_bar(device)
    rows = []
    for task in tasks:
        config, model, opt = build(task, device)
        for name in tables:
            stage, r, c = TABLES[name]
            for mode in modes:
                model.activation_checkpointing_rows = 0 if mode == "on" else None
                x, y, n_train = batch(task, 1, r, c, device)
                row = dict(table=name, stage=stage, rows=r, features=c, cells=r * c, task=task)
                row["checkpointing"] = mode
                opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                try:
                    times = []
                    for i in range(warmup + steps):
                        torch.cuda.synchronize()
                        t0 = time.perf_counter()
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            logits = model(x, y[:, :n_train])
                        loss = prior_loss(logits, y[:, n_train:], task, bar)
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
                logits = loss = None
                opt.zero_grad(set_to_none=True)
                print(json.dumps(row), flush=True)
                rows.append(row)
        del model, opt
        torch._dynamo.reset()
        torch.cuda.empty_cache()
    return rows


@torch.no_grad()
def val(device, tables, tasks, val_batch) -> list[dict]:
    """학습 루프의 검증 (`training.loop._validate`): eval, gradient 없음, 데이터셋 `val_batch`개."""
    bar = make_bar(device)
    rows = []
    for task in tasks:
        _, model, _ = build(task, device)
        model.eval()
        for name in tables:
            stage, r, c = TABLES[name]
            x, y, n_train = batch(task, val_batch, r, c, device)
            row = dict(table=name, stage=stage, rows=r, features=c, task=task, val_batch=val_batch)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            try:
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(x, y[:, :n_train])
                row["loss"] = prior_loss(logits, y[:, n_train:], task, bar).item()
                torch.cuda.synchronize()
                row["seconds"] = time.perf_counter() - t0
                row["peak_gb"] = torch.cuda.max_memory_allocated() / 2**30
            except torch.OutOfMemoryError:
                row["oom"] = True
            except Exception as e:
                row["error"] = f"{type(e).__name__}: {str(e)[:500]}"
            logits = None
            print(json.dumps(row), flush=True)
            rows.append(row)
        del model
        torch._dynamo.reset()
        torch.cuda.empty_cache()
    return rows


def main(argv=None) -> None:
    args = parse_args(argv)
    device = torch.device("cuda")
    tables, tasks = args.tables.split(","), args.tasks.split(",")
    result = {"bench": bench(device, tables, tasks, args.modes.split(","), args.steps, args.warmup)}
    if not args.skip_val:
        result["val"] = val(device, tables, tasks, args.val_batch)
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
