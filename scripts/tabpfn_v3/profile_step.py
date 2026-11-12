"""TabPFN-3.5 multitask(Muon) 학습 step을 단계별로 나눠 잰다 (issue #56).

학습 루프(`minipfn.training.loop`)와 같은 모델·DDP·prior batch로 step마다 아래를 따로 잰다. 단계마다
GPU를 기다린(`torch.cuda.synchronize`) 뒤 시간을 재므로 학습 루프처럼 겹쳐 돌지는 않는다.

- data: 다음 prior batch를 기다린 시간
- fwd_bwd_local: forward·backward (`no_sync`, gradient all-reduce 없음)
- fwd_bwd_ddp: 같은 batch의 forward·backward + DDP gradient all-reduce. 둘의 차이가 all-reduce
  가운데 backward와 겹치지 않은 몫이다
- allreduce: 모든 gradient(float32)를 한 덩어리로 all-reduce한 시간 (겹침 없는 통신 시간)
- clip: gradient norm clip
- ns_only: Muon이 맡는 조각 전부의 Newton–Schulz만 (GPU 한 장이 지금 하는 직교화)
- muon_off·muon_node·muon_all: Muon step 전체 (momentum·직교화·갱신). off는 지금처럼 모든 GPU가 다
  계산, node는 노드 안 GPU끼리, all은 모든 GPU끼리 나눈다(`muon_distributed`). 같은 시작 상태에서
  돌리고, 나눈 결과가 off와 비트 단위로 같은지 본다(`*_equal`)
- adam: Adam 묶음 step

rank마다 잰 평균 가운데 가장 큰 값(가장 느린 GPU가 step을 정한다)을 MLflow 지표와 `profile.json`으로
남긴다.

  databricks air run -f scripts/tabpfn_v3/profile_step.yaml --watch \\
    --override mlflow_experiment_directory=$MLFLOW_EXPERIMENT_DIR
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import time
from dataclasses import replace

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from minipfn.models.bar_distribution import FullSupportBarDistribution
from minipfn.recipes import tabpfn_v3
from minipfn.training.config import N_BUCKETS, TrainConfig
from minipfn.training.data import PriorBatches
from minipfn.training.losses import dequantize_targets, prior_loss
from minipfn.training.muon import MuonAdam, _newton_schulz, shard_group
from minipfn.utils import setup_distributed


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--batch-size", type=int, default=256)  # chunk01 (32 GPU, GPU당 8)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--compile", action="store_true")  # chunk01은 켰다. 켜면 준비가 오래 걸린다
    parser.add_argument("--out", default="profile.json")
    parser.add_argument("--smoke", action="store_true", help="작은 v3.5로 (로컬 CPU 점검)")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    d, device = setup_distributed()
    base = tabpfn_v3.paper("multitask", "v3.5")
    if args.smoke:
        base = tabpfn_v3.with_architecture(TrainConfig.smoke("multitask"), "v3.5", keep_size=True)
    use_cuda = device.type == "cuda"
    config = replace(
        base,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        optimizer="muon",
        fused_adam=use_cuda,
        tf32=use_cuda,
    ).for_world_size(d.world_size)
    if use_cuda:
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.set_float32_matmul_precision("high")
    torch.manual_seed(config.seed)
    model = config.build_model().to(device)
    groups = {"off": None}
    if d.world_size > 1:
        groups["node"] = shard_group("node")
        groups["all"] = shard_group("all")
    opt = MuonAdam(model, lr=config.lr, adam_kwargs={"fused": True} if use_cuda else {})
    ddp = model
    if d.world_size > 1:
        ddp = DistributedDataParallel(model, device_ids=[d.local_rank], find_unused_parameters=True)
    forward = ddp
    if args.compile:
        torch._dynamo.config.cache_size_limit = max(64, torch._dynamo.config.cache_size_limit)
        options = model.compile_options()  # 학습 루프와 같은 inductor 설정
        for block in model.repeated_blocks():
            block.compile(dynamic=True, options=options)
    # 시간만 재므로 회귀 구간 경계는 고르게 둔다 (학습은 prior에서 계산한다)
    bar = FullSupportBarDistribution(torch.linspace(-4, 4, N_BUCKETS + 1)).to(device)
    workers = max(1, config.num_workers // len(config.tasks))
    batches = {
        task: iter(
            DataLoader(
                PriorBatches(c, seed=(config.seed, 0, d.rank, i)),
                batch_size=None,
                num_workers=workers,
                pin_memory=use_cuda,
                persistent_workers=True,
                prefetch_factor=4,
            )
        )
        for i, (task, c) in enumerate(config.task_configs().items())
    }
    accumulation = config.grad_accumulation(d.world_size)
    muon_params = [p for g in opt.muon.param_groups for p in g["params"]]

    def clock() -> float:
        if use_cuda:
            torch.cuda.synchronize(device)
        return time.perf_counter()

    def fwd_bwd(x, y, n_train, task, sync: bool) -> None:
        ctx = contextlib.nullcontext() if sync or d.world_size == 1 else ddp.no_sync()
        autocast = (
            torch.autocast("cuda", dtype=torch.bfloat16) if use_cuda else contextlib.nullcontext()
        )
        with ctx:
            with autocast:
                logits = forward(x, y[:, :n_train], task=task)
            y_eval = y[:, n_train:]
            if task == "regression" and config.dequantize_buckets:
                y_eval = dequantize_targets(y, n_train, bar, config.dequantize_buckets)
            (prior_loss(logits, y_eval, task, bar) / accumulation).backward()

    totals: dict[str, float] = {}
    equal = {name: True for name in groups if name != "off"}
    for step in range(args.warmup + args.steps):
        task = config.tasks[step % len(config.tasks)]
        t: dict[str, float] = {}
        start = clock()
        micro = [next(batches[task]) for _ in range(accumulation)]
        micro = [(x.to(device), y.to(device), n) for x, y, n in micro]
        t["data"] = clock() - start

        start = clock()
        for x, y, n in micro:
            fwd_bwd(x, y, n, task, sync=False)
        t["fwd_bwd_local"] = clock() - start
        opt.zero_grad(set_to_none=True)
        start = clock()
        for i, (x, y, n) in enumerate(micro):
            fwd_bwd(x, y, n, task, sync=i == accumulation - 1)
        t["fwd_bwd_ddp"] = clock() - start

        grads = [p.grad for p in model.parameters() if p.grad is not None]
        flat = torch.cat([g.reshape(-1) for g in grads])
        if d.world_size > 1:
            dist.barrier()
            start = clock()
            dist.all_reduce(flat)
            t["allreduce"] = clock() - start
        del flat

        start = clock()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        t["clip"] = clock() - start

        start = clock()
        for g in opt.muon.param_groups:
            for p in g["params"]:
                if p.grad is None:
                    continue
                rows = p.shape[0] // g["split"]
                for piece in p.grad.split(rows):
                    _newton_schulz(piece)
        t["ns_only"] = clock() - start

        # 같은 시작 상태(가중치·momentum)에서 Muon step을 범위마다 돌린다
        weights = [p.detach().clone() for p in muon_params]
        buffers = [
            opt.muon.state[p]["momentum_buffer"].clone() if p in opt.muon.state else None
            for p in muon_params
        ]
        results = {}
        for name, group in groups.items():
            for p, w, b in zip(muon_params, weights, buffers, strict=True):
                p.data.copy_(w)
                if b is None:
                    opt.muon.state.pop(p, None)
                else:
                    opt.muon.state[p]["momentum_buffer"].copy_(b)
            opt.muon.process_group = group
            if d.world_size > 1:
                dist.barrier()
            start = clock()
            opt.muon.step()
            t[f"muon_{name}"] = clock() - start
            results[name] = [p.detach().clone() for p in muon_params]
        for name in equal:
            same = all(
                torch.equal(a, b) for a, b in zip(results[name], results["off"], strict=True)
            )
            equal[name] = equal[name] and same
        del weights, buffers, results

        start = clock()
        opt.adam.step()
        t["adam"] = clock() - start
        opt.zero_grad(set_to_none=True)

        if step >= args.warmup:
            for k, v in t.items():
                totals[k] = totals.get(k, 0.0) + v
        if d.is_main:
            times = " ".join(f"{k} {v:.4f}" for k, v in t.items())
            print(f"step {step} {task} {times}", flush=True)

    names = sorted(totals)
    means = torch.tensor([totals[k] / args.steps for k in names], device=device)
    flags = torch.tensor([float(v) for v in equal.values()], device=device)
    if d.world_size > 1:
        dist.all_reduce(means, op=dist.ReduceOp.MAX)
        dist.all_reduce(flags, op=dist.ReduceOp.MIN)
    if d.is_main:
        result = {f"sec_{k}": float(v) for k, v in zip(names, means.tolist(), strict=True)}
        result.update({f"{k}_equal": float(v) for k, v in zip(equal, flags.tolist(), strict=True)})
        result.update(
            world_size=d.world_size,
            local_world_size=int(os.environ.get("LOCAL_WORLD_SIZE", d.world_size)),
            micro_batch=config.micro_batch,
            grad_accumulation=accumulation,
            muon_pieces=sum(g["split"] * len(g["params"]) for g in opt.muon.param_groups),
            compile=args.compile,
        )
        print(json.dumps(result, indent=2), flush=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        try:
            import mlflow

            if mlflow.active_run() is None:
                mlflow.start_run()
            mlflow.log_metrics({k: v for k, v in result.items() if isinstance(v, float)})
            mlflow.log_params({k: v for k, v in result.items() if not isinstance(v, float)})
            mlflow.log_artifact(args.out)
        except Exception as e:  # MLflow가 없는 로컬 실행
            print(f"MLflow 기록 건너뜀: {e}", flush=True)
    if d.world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
