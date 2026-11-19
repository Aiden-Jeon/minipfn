"""TabPFN-3.5 checkpoint의 attention 진단: 층마다 logit 크기와 온도 배율 (PR #74, issue #102).

checkpoint마다 같은 검증 batch(curriculum stage 1 prior, 학습 검증과 같은 시드. 분류·회귀 각
`--n-batches` batch × 8 데이터셋)로 forward 하며 다음을 잰다.

- attention마다(열 단계 요약·펼침, 행 단계, 행 ICL, 분류 retrieval decoder): soft-cap을 걸기 전
  logit |max|, 정규화 entropy(1이면 균등), 가장 큰 확률의 평균
- 온도 배율: QASSMax·LogScale이 q에 곱한 배율 |q 뒤| / |q 앞|의 최대·평균(원소마다, query에 따른
  gate 포함), QK-norm gain(q·k 곱)의 최대, LogScale의 a_h
- 행 ICL Wq·Wk의 spectral norm, 검증 손실

  databricks air run -f scripts/tabpfn_v3/attention_probe.yaml --watch \\
    --override mlflow_experiment_directory=$MLFLOW_EXPERIMENT_DIR \\
    --override env_variables.PROBE_CKPTS="<runs:/RUN/checkpoints/step_0005000.pt ...>" \\
    --override env_variables.PROBE_LABELS="<RUN=A,...>"

`--local-test`는 작은 smoke 모델 하나로 CPU에서 경로만 본다.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import tempfile
import time
from collections import defaultdict
from dataclasses import replace

import torch

from minipfn.models import column_row_transformer as M
from minipfn.models import model_from_state
from minipfn.models.bar_distribution import FullSupportBarDistribution
from minipfn.recipes import kumo_tabular, tabpfn_v3
from minipfn.training.config import TrainConfig
from minipfn.training.losses import prior_loss

# attention batch(열 단계는 데이터셋 × 열, 행 단계는 데이터셋 × 행) 앞 몇 개만 logit을 만든다
# (메모리)
MAX_QUERY_BATCH = 16
STAGE_ORDER = ("column_blocks", "row_blocks", "icl_layers", "decoder")


class Recorder:
    """forward 하는 동안 attention 호출의 통계를 모은다 (`install`)."""

    def __init__(self) -> None:
        self.current: str | None = None
        self.logits: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
        self.scales: dict[str, list[tuple[float, float]]] = defaultdict(list)

    def logit_stats(self, q: torch.Tensor, k: torch.Tensor) -> None:
        if self.current is None:
            return
        with torch.no_grad():
            qq, kk = q[:MAX_QUERY_BATCH].float(), k[:MAX_QUERY_BATCH].float()
            logits = qq @ kk.transpose(-1, -2) / math.sqrt(q.shape[-1])
            p = logits.softmax(-1)
            entropy = -(p * p.clamp_min(1e-30).log()).sum(-1) / math.log(max(k.shape[-2], 2))
            self.logits[self.current].append(
                (logits.abs().amax().item(), entropy.mean().item(), p.amax(-1).mean().item())
            )

    def scale_stats(self, name: str, before: torch.Tensor, after: torch.Tensor) -> None:
        with torch.no_grad():
            b, a = before.float(), after.float()
            ratio = a.abs() / b.abs().clamp_min(1e-6)
            ratio = ratio[b.abs() > 1e-3]  # q가 0 근처인 원소는 배율이 뜻이 없다
            if ratio.numel():
                self.scales[name].append((ratio.amax().item(), ratio.mean().item()))

    def install(self, model: torch.nn.Module) -> list:
        handles = []
        for name, module in model.named_modules():
            if isinstance(module, (M.Attention, M.RetrievalDecoder)):

                def pre(_m, _inp, name=name):
                    self.current = name

                def post(_m, _inp, _out):
                    self.current = None

                handles += [
                    module.register_forward_pre_hook(pre),
                    module.register_forward_hook(post),
                ]
            if isinstance(module, (M.QASSMax, M.LogScale)):
                owner = name.rsplit(".", 1)[0]

                def scale_hook(_m, inp, out, owner=owner):
                    self.scale_stats(owner, inp[0], out)

                handles.append(module.register_forward_hook(scale_hook))
        return handles


# 지금 forward의 기록 (`probe`가 바꾼다). attention 함수는 process에서 한 번만 감싼다
RECORDER: list[Recorder | None] = [None]


def patch_attention() -> None:
    """`Attention._sdpa`(soft-cap 전 q·k가 지나는 곳)와 retrieval decoder의 score를 감싼다. 두 번
    부르면 아무것도 하지 않는다."""
    if getattr(M.Attention._sdpa, "probe_patched", False):
        return
    original, original_retrieve = M.Attention._sdpa, M._retrieve

    def sdpa(self, q, k, v):
        if RECORDER[0] is not None:
            RECORDER[0].logit_stats(q, k)
        return original(self, q, k, v)

    def retrieve(q, k, onehot):
        if RECORDER[0] is not None:
            RECORDER[0].logit_stats(q, k)
        return original_retrieve(q, k, onehot)

    sdpa.probe_patched = True
    M.Attention._sdpa, M._retrieve = sdpa, retrieve


def stage_key(name: str) -> str:
    m = re.match(r"(column_blocks|row_blocks|icl_layers)\.(\d+)\.?(.*)", name)
    if m:
        part = {"summarize": "summary", "broadcast": "spread"}.get(m.group(3), "")
        return f"{m.group(1)}.{int(m.group(2)):02d}" + (f".{part}" if part else "")
    if name.startswith("heads.classification"):
        return "decoder"
    return name


def param_stats(model: torch.nn.Module) -> dict[str, float]:
    out = {}
    for name, module in model.named_modules():
        if not isinstance(module, (M.Attention, M.RetrievalDecoder)):
            continue
        key = stage_key(name)
        if getattr(module, "qk_norm", False) and module.q_norm.weight is not None:
            gq, gk = module.q_norm.weight.detach().float(), module.k_norm.weight.detach().float()
            out[f"{key}/qk_gain_prod_max"] = (gq.abs() * gk.abs()).max().item()
        if getattr(module, "log_scale", None) is not None:
            a = module.log_scale.head_scale.detach().float()
            out[f"{key}/head_scale_max"] = a.abs().max().item()
            out[f"{key}/head_scale_mean"] = a.mean().item()
        if key.startswith("icl_layers"):
            w = module.qkv.weight.detach().float()
            d = w.shape[1]
            out[f"{key}/Wq_spec"] = torch.linalg.matrix_norm(w[:d], ord=2).item()
            out[f"{key}/Wk_spec"] = torch.linalg.matrix_norm(w[d : 2 * d], ord=2).item()
    return out


def val_batches(config: TrainConfig, n_batches: int) -> dict:
    sets = {}
    for task, c in config.task_configs().items():
        prior = c.make_prior(10_000_000 + c.seed)  # 학습 루프의 검증과 같은 시드
        sets[task] = [prior.sample_batch(c.val_batch) for _ in range(n_batches)]
    return sets


def probe(model, borders, sets, device) -> dict:
    model.to(device).eval()
    # 추론 forward(`inference=True`)로 부른다: retrieval decoder가 fused 경로라 q·k가 `_retrieve`를
    # 지나 통계를 잰다(score는 float32. 학습 검증의 bf16 score와 손실이 조금 다를 수 있다)
    recorder = Recorder()
    patch_attention()
    RECORDER[0] = recorder
    handles = recorder.install(model)
    bar = FullSupportBarDistribution(borders.to(device)) if borders is not None else None
    losses = {}
    cuda = device.type == "cuda"
    try:
        with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=cuda):
            for task, batches in sets.items():
                total = 0.0
                for b in batches:
                    x, y = b.x.to(device), b.y.to(device)
                    logits = model(x, y[:, : b.n_train], task=task, inference=True)
                    total += prior_loss(logits.float(), y[:, b.n_train :], task, bar).item()
                losses[task] = total / len(batches)
    finally:
        RECORDER[0] = None
        for h in handles:
            h.remove()
    attention = {}
    for name, rows in recorder.logits.items():
        t = torch.tensor(rows)
        attention[stage_key(name)] = {
            "logit_absmax": t[:, 0].max().item(),
            "entropy": t[:, 1].mean().item(),
            "pmax": t[:, 2].mean().item(),
        }
    for name, rows in recorder.scales.items():
        t = torch.tensor(rows)
        entry = attention.setdefault(stage_key(name), {})
        entry["scale_max"], entry["scale_mean"] = t[:, 0].max().item(), t[:, 1].mean().item()
    return {"val_loss": losses, "attention": attention, "params": param_stats(model)}


def summary(label: str, res: dict) -> str:
    loss = res["val_loss"]
    lines = [f"== {label}  val cls {loss['classification']:.4f}  reg {loss['regression']:.4f}"]
    att = res["attention"]
    for stage in STAGE_ORDER:
        items = {k: v for k, v in att.items() if k.startswith(stage) and "logit_absmax" in v}
        if not items:
            continue
        worst = max(items, key=lambda k: items[k]["logit_absmax"])
        scales = {k: v["scale_max"] for k, v in items.items() if "scale_max" in v}
        scale = f"  scale max {max(scales.values()):8.2f}" if scales else ""
        low = min(items, key=lambda k: items[k]["entropy"])
        lines.append(
            f"  {stage:13s} logit|max| {items[worst]['logit_absmax']:10.1f} ({worst}){scale}"
            f"  min entropy {items[low]['entropy']:.3f} ({low})"
        )
    icl = [att[k]["logit_absmax"] for k in sorted(att) if k.startswith("icl_layers")]
    if icl:
        lines.append("  ICL logit|max| by layer: " + " ".join(f"{v:.0f}" for v in icl))
    p = res["params"]
    for kind in ("qk_gain_prod_max", "head_scale_max", "Wq_spec", "Wk_spec"):
        values = {k: v for k, v in p.items() if k.endswith(kind)}
        if values:
            worst = max(values, key=values.get)
            lines.append(f"  {kind:17s} max {values[worst]:8.3f} ({worst.rsplit('/', 1)[0]})")
    return "\n".join(lines)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ckpt", nargs="*", default=os.environ.get("PROBE_CKPTS", "").split())
    parser.add_argument("--labels", default=os.environ.get("PROBE_LABELS", ""))
    parser.add_argument("--n-batches", type=int, default=4)
    parser.add_argument("--out", default="attention_probe.json")
    parser.add_argument("--local-test", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}
    if args.local_test:
        config = tabpfn_v3.with_architecture(TrainConfig.smoke("multitask"), "v3.5", keep_size=True)
        common = {k: v for k, v in tabpfn_v3.CURRICULUM_COMMON.items() if k != "batch_size"}
        common |= {"min_rows": 48, "max_rows": 48, "max_features": 8, "max_cells": 48 * 8}
        for label, temps in (("default", {}), ("kumo", kumo_tabular.ATTENTION_TEMPERATURE)):
            c = replace(config, **common, **temps, val_batch=4)
            res = probe(c.build_model(), torch.linspace(-3, 3, 5001), val_batches(c, 1), device)
            results[label] = res
            print(summary(label, res), flush=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1)
        return
    import mlflow

    sets = val_batches(tabpfn_v3.curriculum(1), args.n_batches)
    print("검증 batch", {t: [tuple(b.x.shape) for b in bs] for t, bs in sets.items()}, flush=True)
    names = dict(x.split("=", 1) for x in args.labels.split(",") if "=" in x)
    for spec in args.ckpt:
        start = time.time()
        with tempfile.TemporaryDirectory() as d:
            path = mlflow.artifacts.download_artifacts(artifact_uri=spec, dst_path=d)
            state = torch.load(path, map_location="cpu", weights_only=True)
        model, borders = model_from_state(state)
        step = state.get("step")
        if step is None:  # 파일 이름에서 (latest.pt처럼 없으면 -1)
            found = re.search(r"step_(\d+)", spec)
            step = int(found.group(1)) if found else -1
        del state
        res = probe(model, borders, sets, device)
        run = spec.removeprefix("runs:/").split("/", 1)[0]
        label = f"{names.get(run, run[:8])}@{step}"
        results[label] = {"spec": spec, "step": step, **res}
        print(summary(label, res), f"  ({time.time() - start:.0f}s)", flush=True)
        with open(args.out, "w") as f:  # 뒤 checkpoint에서 멈춰도 앞 결과가 남도록
            json.dump(results, f, indent=1)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)
    if mlflow.active_run() or os.environ.get("MLFLOW_RUN_ID"):
        mlflow.log_artifact(args.out)
    print("JSON", json.dumps(results))


if __name__ == "__main__":
    main()
