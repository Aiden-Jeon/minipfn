"""GP prior PFN을 학습하고 analytic GP posterior와 대조해 MLflow에 기록한다.

AI Runtime(`databricks air run -f runs/train_pfn.yaml`)과 로컬에서 같은 스크립트를
쓴다.
- AI Runtime: 플랫폼이 만든 MLflow run(`MLFLOW_RUN_ID`)에 이어서 기록한다.
- 로컬: 새 run을 연다.
  예) `uv run python scripts/train_pfn.py --config-name small --steps 50`

규모는 인자 또는 환경변수(`PFN_CONFIG_NAME`, `PFN_STEPS`)로 정한다. air에서는
`--override env_variables.PFN_STEPS=...`로 yaml을 고치지 않고 바꿀 수 있다.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import replace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from minipfn import (  # noqa: E402
    PFNConfig,
    build_criterion,
    build_model,
    build_prior_sampler,
    evaluate_against_gp,
    get_device,
    log_run_source,
    set_seed,
    train_pfn,
)

CONFIG_NAMES = ("small", "large", "paper")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config-name",
        choices=CONFIG_NAMES,
        default=os.environ.get("PFN_CONFIG_NAME", "large"),
        help="PFNConfig preset (기본: $PFN_CONFIG_NAME 또는 large)",
    )
    steps_env = os.environ.get("PFN_STEPS")
    parser.add_argument(
        "--steps",
        type=int,
        default=int(steps_env) if steps_env else None,
        help="config의 step 수 덮어쓰기 (기본: $PFN_STEPS 또는 config 기본값)",
    )
    return parser.parse_args(argv)


def plot_losses(losses: np.ndarray, window: int = 50) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    ax.plot(losses, color="C0", alpha=0.4, lw=0.8)
    if len(losses) >= window:
        smooth = np.convolve(losses, np.ones(window) / window, mode="valid")
        ax.plot(np.arange(len(smooth)) + window // 2, smooth, color="C0", lw=2)
    ax.set_xlabel("step")
    ax.set_ylabel("query NLL")
    ax.set_title("Prior-fitting loss")
    return fig


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = getattr(PFNConfig, args.config_name)()
    if args.steps is not None:
        config = replace(config, steps=args.steps)

    device = get_device()
    if device.type == "mps":  # 로컬 MPS는 일부 연산 미지원 → CPU
        device = torch.device("cpu")
    print("device:", device)
    if device.type == "cuda":
        print("gpu   :", torch.cuda.get_device_name(0))
    print("config_name:", args.config_name)
    print(config)

    set_seed(config.seed)
    model = build_model(config).to(device)
    criterion = build_criterion(config).to(device)
    prior_sampler = build_prior_sampler(config)
    print("parameters:", f"{model.num_parameters():,}")

    # air가 넘겨준 run이 있으면 거기에, 없으면(로컬) 새 run에 기록한다
    run_id = os.environ.get("MLFLOW_RUN_ID")
    run_name = None if run_id else f"{args.config_name}-{config.steps}steps"
    with mlflow.start_run(run_id=run_id, run_name=run_name) as run:
        print("MLflow run_id:", run.info.run_id)
        log_run_source(__file__, env_prefix="PFN_")
        history = train_pfn(
            model,
            prior_sampler,
            criterion,
            config,
            device=device,
            progress=False,
            mlflow_log=True,
        )
        losses = np.array(history["losses"])
        mlflow.log_figure(plot_losses(losses), "loss_curve.png")

        title = f"PFN ({args.config_name}, {config.steps} steps) vs analytic GP posterior"
        metrics, fig = evaluate_against_gp(model, criterion, title=title)
        mlflow.log_metrics(metrics)
        mlflow.log_figure(fig, "pfn_overlay.png")

    print("final train NLL ~", round(float(losses[-50:].mean()), 3))
    print(f"best val NLL={history['best_val_nll']:.3f} (step {history['best_step']})")
    print(f"RMSE(mean)={metrics['rmse_mean']:.4f}  mean|Δσ|={metrics['mean_abs_dsigma']:.4f}")
    print(
        f"PFN NLL={metrics['pfn_nll']:.3f}  GP NLL(하한)={metrics['gp_nll']:.3f}  "
        f"gap={metrics['nll_gap']:+.3f}"
    )


if __name__ == "__main__":
    main()
