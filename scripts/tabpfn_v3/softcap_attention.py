"""soft-cap attention의 flex attention kernel을 logit을 직접 만드는 구현과 견준다 (issue #77).

GPU 한 장에서 세 가지를 본다.

1. 같은지 (`check`): float32(TF32 끔)에서 같은 입력의 출력·gradient. attention 하나(cap이 실제로
   걸리도록 logit을 크게)와 TabPFN-3.5 모델 전체(열 단계 cross-attention·QASSMax, 행 단계 RoPE·CLS
   query, 행 ICL multi-query 테스트 행을 다 지난다). bf16 autocast에서의 차이도 적는다
2. attention 하나 (`micro`): stage 1·2 표의 attention 종류마다 forward·backward 시간과 메모리.
   soft-cap 없음(SDPA), manual(compile), flex(기본 block, 작은 block, autotune)
3. 모델 전체 (`bench`): TabPFN-3.5 multitask 모델의 forward·backward 한 번(bf16 autocast)을
   curriculum stage 1·2·3의 가장 큰 표로(stage 3은 feature 상한마다). 구현은 soft-cap 없음
   (SDPA), 직접 만드는 logit(manual), flex. 가속 옵션 compile을 켠 것(실제 run)과 끈 것

  databricks air run -f scripts/tabpfn_v3/softcap_attention.yaml --watch \\
    --override mlflow_experiment_directory=$MLFLOW_EXPERIMENT_DIR
"""

from __future__ import annotations

import argparse
import json
import time

import torch

from minipfn.models import column_row_transformer as M
from minipfn.recipes import tabpfn_v3

CAP = 50.0
VARIANTS = ("none", "manual", "flex")
# curriculum의 가장 큰 표: (micro batch, 행 수, feature 수, 학습 행 비율). 학습 행은 stage 1이
# 90%까지, stage 2·3은 81%까지. stage 3은 행이 많으면 feature 상한이 줄어서
# (`priors.table_size.LONG_TABLE_FEATURE_CAPS`) 상한마다 가장 큰 표 셋을 잰다
STAGES = {
    "1": (8, 1024, 100, 0.9),
    "2": (1, 10_240, 100, 0.81),
    "3a": (1, 20_000, 80, 0.81),
    "3b": (1, 40_000, 40, 0.81),
    "3c": (1, 60_000, 20, 0.81),
}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stages", default=",".join(STAGES))
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--compile", default="off,on", help="off·on 가운데 잴 것")
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--skip-micro", action="store_true")
    parser.add_argument("--skip-check", action="store_true")
    parser.add_argument("--out", default="softcap_attention.json")
    return parser.parse_args(argv)


def set_variant(model: torch.nn.Module, variant: str) -> None:
    """none: soft-cap 없음(SDPA), 나머지는 `M.SOFTCAP_KERNEL`: manual(logit을 다 만든다),
    flex(flex attention kernel)."""
    M.SOFTCAP_KERNEL = "manual" if variant == "none" else variant
    for m in model.modules():
        if isinstance(m, M.Attention):
            m.logit_softcap = None if variant == "none" else CAP


def model_config():
    return tabpfn_v3.curriculum(1).model_config()  # logit_softcap 50


def batch(task: str, b: int, r: int, c: int, frac: float, device, seed: int = 0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    n_train = int(r * frac)
    x = torch.randn(b, r, c, generator=g)
    if task == "classification":
        y = torch.randint(0, 10, (b, n_train), generator=g).float()
    else:
        y = torch.randn(b, n_train, generator=g)
    return x.to(device), y.to(device)


def step(model, x, y, task, autocast: bool = True) -> torch.Tensor:
    model.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
        out = model(x, y, task)
    out.float().square().mean().backward()
    return out


def grads(model, x, y, task, autocast: bool):
    out = step(model, x, y, task, autocast)
    grad = {n: p.grad.float().clone() for n, p in model.named_parameters() if p.grad is not None}
    return out.detach().float(), grad


def compare(a_out, a_grad, b_out, b_grad) -> dict:
    """출력의 최대 상대 차이(최대 절댓값 대비)와 파라미터마다 gradient 상대 차이(norm 비)의
    최댓값·중앙값, 가장 큰 파라미터와 그 gradient norm, 모든 gradient를 이은 벡터의 상대 차이."""
    names = list(a_grad)
    rel = torch.tensor(
        [((a_grad[n] - b_grad[n]).norm() / a_grad[n].norm().clamp_min(1e-30)).item() for n in names]
    )
    worst = names[int(rel.argmax())]
    flat_a = torch.cat([a_grad[n].flatten() for n in names])
    flat_b = torch.cat([b_grad[n].flatten() for n in names])
    return {
        "out_rel": ((a_out - b_out).abs().max() / a_out.abs().max()).item(),
        "grad_rel_max": rel.max().item(),
        "grad_rel_median": rel.median().item(),
        "grad_rel_worst": worst,
        "grad_norm_worst": a_grad[worst].norm().item(),
        "grad_rel_all": ((flat_a - flat_b).norm() / flat_a.norm()).item(),
    }


def attempt(fn, *args):
    """실패하면 오류를 적고 넘어간다 (한 구현이 실패해도 나머지를 본다)."""
    try:
        return fn(*args)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {str(e)[:500]}"}


def check(device) -> dict:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    result = {}
    # attention 하나: QK-norm gain을 키워 logit이 cap(50)을 넘게 한다
    torch.manual_seed(0)
    attn = M.Attention(1024, 16, qk_norm=True, qassmax=True, zero_out=False, logit_softcap=CAP)
    attn = attn.to(device)
    with torch.no_grad():
        attn.q_norm.weight.fill_(6.0)
        attn.k_norm.weight.fill_(6.0)
    x = torch.randn(3, 700, 1024, device=device)
    q = attn._heads(torch.nn.functional.linear(x, attn.qkv.weight[:1024]))
    k = attn._heads(torch.nn.functional.linear(x, attn.qkv.weight[1024:2048]))
    logits = attn.q_norm(q) @ attn.k_norm(k).transpose(-1, -2) / 8
    result["attention_logit_absmax"] = logits.abs().max().item()

    def attention_run(variant):
        M.SOFTCAP_KERNEL = variant
        x_ = x.clone().requires_grad_()
        attn.zero_grad(set_to_none=True)
        out = attn(x_, n_context=550)
        out.square().mean().backward()
        g = {n: p.grad.clone() for n, p in attn.named_parameters() if p.grad is not None}
        g["input"] = x_.grad.clone()
        return out.detach(), g

    reference = attention_run("manual")
    run = attempt(attention_run, "flex")
    result["attention_fp32"] = run if isinstance(run, dict) else compare(*reference, *run)
    # 모델 전체 (두 task)
    torch.manual_seed(0)
    model = M.ColumnRowTransformer(model_config()).to(device)
    for task in ("classification", "regression"):
        x, y = batch(task, 2, 512, 20, 0.8, device, seed=1)
        for autocast in (False, True):
            set_variant(model, "manual")
            reference = grads(model, x, y, task, autocast)
            set_variant(model, "flex")
            run = attempt(grads, model, x, y, task, autocast)
            if not isinstance(run, dict):
                run = compare(*reference, *run)
            result[f"model_{task}_{'bf16' if autocast else 'fp32'}"] = run
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    return result


def micro_shapes(stage: str) -> dict[str, tuple[tuple[int, ...], tuple[int, ...]]]:
    """stage 표 하나에서 attention 종류마다 (q 모양, k·v 모양). TabPFN-3.5 크기: 열·행 단계 head
    8 × 16, 행 ICL head 16 × 64, inducing point 128개, CLS 8개."""
    b, r, c, frac = STAGES[stage]
    n_train = int(r * frac)
    tokens = c + 8
    return {
        "column_summarize": ((b * c, 8, 128, 16), (b * c, 8, n_train, 16)),
        "column_broadcast": ((b * c, 8, r, 16), (b * c, 8, 128, 16)),
        "row": ((b * r, 8, tokens, 16), (b * r, 8, tokens, 16)),
        "row_cls": ((b * r, 8, 8, 16), (b * r, 8, tokens, 16)),
        "icl_train": ((b, 16, n_train, 64), (b, 16, n_train, 64)),
        "icl_test": ((b, 1, 16 * (r - n_train), 64), (b, 1, n_train, 64)),
    }


def micro(device, stages, steps, warmup) -> list[dict]:
    """attention 하나의 forward·backward (bf16): soft-cap 없음(SDPA), manual(compile), flex(기본
    kernel 설정, 작은 block, autotune)."""
    from torch.nn.attention.flex_attention import flex_attention

    torch._dynamo.config.cache_size_limit = max(256, torch._dynamo.config.cache_size_limit)
    score_mod = M._softcap_score_mod(CAP)
    small = dict(M.FLEX_SMALL_BLOCKS)  # 모델이 head 폭 16(열·행 단계)에 쓰는 block

    def manual(q, k, v):
        logits = (q @ k.transpose(-1, -2)) * q.shape[-1] ** -0.5
        logits = CAP * torch.tanh(logits / CAP)
        return logits.softmax(-1, dtype=torch.float32).to(v.dtype) @ v

    def flex(q, k, v):
        return flex_attention(q, k, v, score_mod=score_mod)

    def flex_small(q, k, v):
        return flex_attention(q, k, v, score_mod=score_mod, kernel_options=small)

    impls = {
        "sdpa": torch.nn.functional.scaled_dot_product_attention,
        "manual": torch.compile(manual),
        "flex": torch.compile(flex),
        "flex_small": torch.compile(flex_small),
        "flex_autotune": torch.compile(flex, mode="max-autotune-no-cudagraphs"),
    }
    rows = []
    for stage in stages:
        for kind, (q_shape, k_shape) in micro_shapes(stage).items():
            for name, fn in impls.items():
                row = dict(stage=stage, kind=kind, impl=name, q=list(q_shape), k=list(k_shape))
                torch.cuda.empty_cache()
                try:
                    g = torch.Generator(device=device).manual_seed(0)
                    q, k, v = (
                        torch.randn(
                            shape, device=device, generator=g, dtype=torch.bfloat16
                        ).requires_grad_()
                        for shape in (q_shape, k_shape, k_shape)
                    )
                    grad = torch.randn(q_shape, device=device, dtype=torch.bfloat16)
                    base = torch.cuda.memory_allocated()
                    torch.cuda.reset_peak_memory_stats()
                    times = []
                    for i in range(warmup + steps):
                        torch.cuda.synchronize()
                        t0 = time.perf_counter()
                        fn(q, k, v).backward(grad)
                        torch.cuda.synchronize()
                        if i >= warmup:
                            times.append(time.perf_counter() - t0)
                        q.grad = k.grad = v.grad = None
                    row["ms"] = 1000 * sum(times) / len(times)
                    row["extra_gb"] = (torch.cuda.max_memory_allocated() - base) / 2**30
                except torch.OutOfMemoryError:
                    row["oom"] = True
                except Exception as e:
                    row["error"] = f"{type(e).__name__}: {str(e)[:300]}"
                print(json.dumps(row), flush=True)
                rows.append(row)
    return rows


def bench(device, stages, modes, variants, steps, warmup) -> list[dict]:
    rows = []
    torch.set_float32_matmul_precision("high")
    for mode in modes:
        torch.manual_seed(0)
        model = M.ColumnRowTransformer(model_config()).to(device)
        if mode == "on":  # 학습 루프(`training.loop`)와 같게 블록마다, 모델의 inductor 설정으로
            torch._dynamo.config.cache_size_limit = max(64, torch._dynamo.config.cache_size_limit)
            options = model.compile_options()
            for block in model.repeated_blocks():
                block.compile(dynamic=True, options=options)
        for stage in stages:
            b, r, c, frac = STAGES[stage]
            for variant in variants:
                set_variant(model, variant)
                for task in ("classification", "regression"):
                    x, y = batch(task, b, r, c, frac, device)
                    row = dict(compile=mode, stage=stage, variant=variant, task=task, rows=r)
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    try:
                        times = []
                        for i in range(warmup + steps):
                            torch.cuda.synchronize()
                            t0 = time.perf_counter()
                            step(model, x, y, task)
                            torch.cuda.synchronize()
                            if i >= warmup:
                                times.append(time.perf_counter() - t0)
                        row["seconds"] = sum(times) / len(times)
                        row["peak_gb"] = torch.cuda.max_memory_allocated() / 2**30
                    except torch.OutOfMemoryError:
                        row["oom"] = True
                    except Exception as e:  # 한 구현이 실패해도 나머지는 잰다
                        row["error"] = f"{type(e).__name__}: {str(e)[:300]}"
                    model.zero_grad(set_to_none=True)
                    print(json.dumps(row), flush=True)
                    rows.append(row)
        del model
        torch._dynamo.reset()
        torch.cuda.empty_cache()
    return rows


def main(argv=None) -> None:
    args = parse_args(argv)
    device = torch.device("cuda")
    result = {}
    if not args.skip_check:
        result["check"] = check(device)
        print(json.dumps(result["check"], indent=2), flush=True)
    stages = args.stages.split(",")
    modes = args.compile.split(",")
    if not args.skip_micro:
        result["micro"] = micro(device, ["1", "2"], 10, args.warmup)
    variants = args.variants.split(",")
    result["bench"] = bench(device, stages, modes, variants, args.steps, args.warmup)
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
