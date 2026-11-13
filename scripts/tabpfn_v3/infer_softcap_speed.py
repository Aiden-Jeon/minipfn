"""soft-cap attention 모델의 추론 시간: SDPA(cap 끔)·manual(logit을 직접 만든다)·flex(#78)를 견준다.

soft-cap(`logit_softcap`)을 켠 checkpoint 하나로, attention 계산만 바꿔 실제 평가와 같은 앙상블
(`default_inference`: 멤버 수·전처리)의 fit + predict 시간을 표 크기마다 잰다(분류·회귀, 1×H100).

- none: soft-cap을 끄고 fused SDPA (soft-cap이 없는 모델의 속도 기준. 출력은 cap만큼 다르다)
- manual: logit 행렬을 다 만들어 c·tanh(logit / c)와 float32 softmax를 거친다. 추론에서 행렬이
  `SOFTCAP_CHUNK_LOGITS`보다 크면 query를 나눈다. #78 전의 GPU 경로이자 지금의 CPU·MPS 경로
- flex: flex attention kernel의 `score_mod` (#78 뒤의 GPU 경로, 기본값)

시간은 두 번째 호출이다(첫 호출은 flex compile이 들어간다, `first_s`). 끝에 10,000행 분류 표에서
manual과 none의 CUDA op별 시간(profiler)을 찍는다.

  databricks air run -f scripts/tabpfn_v3/infer_softcap_speed.yaml --watch \\
    --override mlflow_experiment_directory=$MLFLOW_EXPERIMENT_DIR \\
    --override "env_variables.SPEED_ARGS=--checkpoint runs:/<run_id>/checkpoints/step_0250000.pt"
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from minipfn.hub import load_model
from minipfn.inference.ensemble import EnsembleClassifier, EnsembleRegressor
from minipfn.models import column_row_transformer as M
from minipfn.recipes import default_inference

VARIANTS = ("none", "manual", "flex")
# 학습 행 × 테스트 행 × feature
SIZES = "1000x200x20,3000x1000x50,10000x2000x100,30000x5000x50"
PROFILE_SIZE = (10_000, 2_000, 100)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--checkpoint", required=True, help="soft-cap checkpoint (runs:/, hf://, 파일)"
    )
    parser.add_argument("--sizes", default=SIZES)
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--tasks", default="classification,regression")
    parser.add_argument("--skip-profile", action="store_true")
    parser.add_argument("--out", default="infer_softcap_speed.json")
    return parser.parse_args(argv)


def set_variant(model: torch.nn.Module, variant: str, cap: float) -> None:
    """none: soft-cap을 끈다(SDPA). 나머지는 `M.SOFTCAP_KERNEL`(manual·flex)과 원래 cap."""
    M.SOFTCAP_KERNEL = "manual" if variant == "none" else variant
    for m in model.modules():
        if isinstance(m, M.Attention):
            m.logit_softcap = None if variant == "none" else cap


def model_cap(model: torch.nn.Module) -> float:
    caps = {m.logit_softcap for m in model.modules() if isinstance(m, M.Attention)}
    if len(caps) != 1 or None in caps:
        raise ValueError(f"soft-cap checkpoint가 아니다 (attention의 cap: {caps})")
    return caps.pop()


def table(task: str, n_train: int, n_test: int, n_features: int, seed: int = 0):
    """선형 신호 표 (분류는 3클래스). 시간만 재므로 값은 상관없다."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n_train + n_test, n_features)).astype(np.float32)
    z = x @ rng.standard_normal(n_features)
    if task == "classification":
        y = (z > np.median(z)).astype(int) + (z > np.quantile(z, 0.8)).astype(int)
    else:
        y = z + 0.1 * rng.standard_normal(len(z))
    return x[:n_train], y[:n_train], x[n_train:]


def estimator(model, borders, task: str, inference: dict):
    kw = {
        "n_estimators": inference["n_estimators"],
        "member_kinds": inference["member_kinds"],
        "device": "cuda",
        "random_state": 0,
    }
    if task == "classification":
        return EnsembleClassifier(model, **kw)
    return EnsembleRegressor(model, borders, **kw)


def fit_predict(est, task: str, x, y, x_test) -> None:
    est.fit(x, y)
    est.predict_proba(x_test) if task == "classification" else est.predict(x_test)


def timed(est, task: str, x, y, x_test) -> float:
    torch.cuda.synchronize()
    start = time.perf_counter()
    fit_predict(est, task, x, y, x_test)
    torch.cuda.synchronize()
    return time.perf_counter() - start


def main(argv=None) -> None:
    args = parse_args(argv)
    results = []
    for task in args.tasks.split(","):
        model, borders = load_model(args.checkpoint, task, f"/tmp/infer_softcap_speed_{task}")
        model = model.cuda().eval()
        cap = model_cap(model)
        inference = default_inference(model, task)
        print(task, f"cap {cap}", inference, flush=True)
        for size in args.sizes.split(","):
            n_train, n_test, n_features = map(int, size.split("x"))
            x, y, x_test = table(task, n_train, n_test, n_features)
            for variant in args.variants.split(","):
                set_variant(model, variant, cap)
                est = estimator(model, borders, task, inference)
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                row = {"task": task, "size": size, "variant": variant}
                try:
                    first = timed(est, task, x, y, x_test)
                    second = timed(est, task, x, y, x_test)
                    peak = torch.cuda.max_memory_allocated() / 2**30
                    row.update(first_s=round(first, 2), s=round(second, 2), peak_gb=round(peak, 1))
                except torch.OutOfMemoryError:
                    row["error"] = "OOM"
                print(json.dumps(row), flush=True)
                results.append(row)
        if task == "classification" and not args.skip_profile:
            x, y, x_test = table(task, *PROFILE_SIZE)
            for variant in ("manual", "none"):
                set_variant(model, variant, cap)
                est = estimator(model, borders, task, inference)
                fit_predict(est, task, x, y, x_test)  # 준비 (한 번)
                cuda = torch.profiler.ProfilerActivity.CUDA
                with torch.profiler.profile(activities=[cuda]) as prof:
                    fit_predict(est, task, x, y, x_test)
                    torch.cuda.synchronize()
                print(f"== profile {variant} {'x'.join(map(str, PROFILE_SIZE))}", flush=True)
                ops = prof.key_averages().table(sort_by="cuda_time_total", row_limit=15)
                print(ops, flush=True)
        del model
        torch.cuda.empty_cache()
    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
