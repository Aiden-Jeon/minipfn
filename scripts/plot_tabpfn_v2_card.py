"""학습 체크포인트와 benchmark 평가 run으로 TabPFN v2 Hub 모델 카드의 그림을 만든다.

예) 저장소 루트에서 (.env.local을 불러온 셸)
  MLFLOW_TRACKING_URI=databricks uv run python scripts/plot_tabpfn_v2_card.py \
    --checkpoint runs:/<학습 run_id>/checkpoints/step_0500000.pt \
    --eval-runs 100000=<평가 run_id>,200000=<평가 run_id>,...

체크포인트의 `history`(이어서 학습한 chunk까지 step 0부터 이어진 기록)와 `eval_tabpfn_v2.py`가 남긴
`eval/eval_per_dataset.csv`를 읽어 `--out-dir`(기본 `hub/minipfn-tabpfn-v2-classifier/images/`)에
세 그림을 쓴다. 데이터셋별 그림은 step이 가장 큰 평가 run으로 그린다.

- `training_curves.png`: train CE(1,000 step 평균)와 val CE(고정한 prior 데이터셋 64개)
- `benchmark_by_step.png`: step별 평균 ROC AUC와 논문 값
- `benchmark_roc_auc.png`: 데이터셋별 ROC AUC, 이 모델과 논문

카드 본문에 적을 수치도 함께 출력한다. 그림 안의 글자는 영어로 쓴다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
# v1 카드와 같은 색. 논문 값은 범주가 아니라 기준선이라 회색으로 둔다
BLUE, GRAY = "#2a78d6", "#8a8984"
INK, MUTED, GRID, SURFACE = "#1f1f1e", "#5f5e58", "#e6e5e0", "#fcfcfb"
STYLE = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "text.color": INK,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
    "legend.frameon": False,
}
SMOOTH = 10  # train loss는 100 step마다 남으므로 10개(1,000 step)씩 평균한다


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True, help="학습 checkpoint의 runs:/ URI")
    parser.add_argument("--eval-runs", required=True, help="step=평가 run_id를 쉼표로 구분")
    parser.add_argument(
        "--out-dir", type=Path, default=REPO_ROOT / "hub/minipfn-tabpfn-v2-classifier/images"
    )
    parser.add_argument("--cache-dir", default="tabpfn_v2_hub/card")
    return parser.parse_args(argv)


def _k(x, _=None) -> str:
    return f"{x / 1000:.0f}k" if x else "0"


def plot_training_curves(history: dict, path: Path) -> tuple[float, int]:
    step = np.asarray(history["step"])
    train = np.asarray(history["train_loss"])
    n = len(step) // SMOOTH * SMOOTH
    step_s = step[:n].reshape(-1, SMOOTH)[:, -1]
    train_s = train[:n].reshape(-1, SMOOTH).mean(axis=1)
    # chunk 경계에서 같은 step의 검증이 두 번 남은 적이 있어 마지막 값을 쓴다
    val = pd.Series(history["val_loss"], index=history["val_step"]).groupby(level=0).last()

    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.plot(step_s, train_s, color=BLUE, lw=1.2, alpha=0.55, label="train CE (1k-step mean)")
    ax.plot(val.index, val.values, color=BLUE, lw=2, label="val CE (64 fixed prior datasets)")
    ax.set_ylim(0.6, 1.2)
    ax.set(xlabel="optimizer step", ylabel="cross-entropy", title="Prior-fitting loss")
    ax.xaxis.set_major_locator(plt.MultipleLocator(100_000))
    ax.xaxis.set_major_formatter(_k)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return float(val.iloc[-1]), int(val.index[-1])


def plot_by_step(means: pd.Series, paper: float, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.axhline(paper, color=GRAY, lw=1.5, ls="--")
    ax.annotate(
        f"paper {paper:.4f}",
        (means.index[0], paper),
        xytext=(0, 5),
        textcoords="offset points",
        color=MUTED,
    )
    ax.plot(means.index, means.values, "o-", color=BLUE, lw=2, ms=8, mec=SURFACE, mew=2)
    for x, y in means.items():
        ax.annotate(
            f"{y:.4f}", (x, y), xytext=(0, -16), textcoords="offset points", ha="center", color=INK
        )
    ax.set_ylim(0.91, 0.935)
    ax.set_xlim(means.index[0] - 40_000, means.index[-1] + 40_000)
    ax.xaxis.set_major_locator(plt.MultipleLocator(100_000))
    ax.xaxis.set_major_formatter(_k)
    ax.set(xlabel="optimizer step", ylabel="mean ROC AUC (29 datasets)")
    ax.set_title("Benchmark ROC AUC by training step", loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_per_dataset(per_dataset: pd.DataFrame, step: int, path: Path) -> None:
    df = per_dataset.assign(diff=per_dataset["score"] - per_dataset["paper"])
    df = df.sort_values("diff", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 8))
    y = np.arange(len(df))
    lo = df[["score", "paper"]].min(axis=1)
    hi = df[["score", "paper"]].max(axis=1)
    ax.hlines(y, lo, hi, color=GRID, lw=2, zorder=1)
    ax.scatter(df["paper"], y, s=45, marker="s", color=GRAY, zorder=2, label="paper (TabPFN v2)")
    ax.scatter(
        df["score"],
        y,
        s=55,
        color=BLUE,
        edgecolors=SURFACE,
        linewidths=1.5,
        zorder=3,
        label=f"this model ({_k(step)} steps)",
    )
    ax.set_yticks(y, df["dataset"], fontsize=8.5)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("ROC AUC (mean over 10 folds)")
    ax.set_title(
        f"AMLB classification datasets, sorted by gap to the paper\n"
        f"mean ROC AUC {df['score'].mean():.4f} (paper {df['paper'].mean():.4f})",
        loc="left",
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache_dir)
    path = mlflow.artifacts.download_artifacts(
        artifact_uri=args.checkpoint, dst_path=str(cache / "checkpoint")
    )
    history = torch.load(path, map_location="cpu", weights_only=True)["history"]
    evals = {}
    for item in args.eval_runs.split(","):
        step, run_id = item.split("=")
        csv = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path="eval/eval_per_dataset.csv", dst_path=str(cache / run_id)
        )
        evals[int(step)] = pd.read_csv(csv)
    evals = dict(sorted(evals.items()))
    means = pd.Series({step: df["score"].mean() for step, df in evals.items()})
    last_step, last = list(evals.items())[-1]

    plt.rcParams.update(STYLE)
    val_ce, val_step = plot_training_curves(history, args.out_dir / "training_curves.png")
    plot_by_step(means, last["paper"].mean(), args.out_dir / "benchmark_by_step.png")
    plot_per_dataset(last, last_step, args.out_dir / "benchmark_roc_auc.png")
    print("saved:", sorted(p.name for p in args.out_dir.glob("*.png")))
    print(f"checkpoint {args.checkpoint}")
    print(f"val CE at {val_step}: {val_ce:.3f}")
    for step, mean in means.items():
        print(f"step {step}: mean ROC AUC {mean:.4f}")
    print(f"paper {last['paper'].mean():.4f}")


if __name__ == "__main__":
    main()
