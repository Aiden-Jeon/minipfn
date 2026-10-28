"""학습 run의 기록으로 Hub 모델 카드의 그림을 만든다.

예) 저장소 루트에서 (.env.local을 불러온 셸)
  MLFLOW_TRACKING_URI=databricks \
    uv run python scripts/plot_tabpfn_card.py --run-id <학습 run_id>

run의 `history.json`과 `eval_table1/eval_per_dataset.csv`(`eval_tabpfn.py --run-id`가 남긴다)를
내려받아 `--out-dir`(기본 `hub/minipfn-tabpfn-v1/images/`)에 두 그림을 쓴다.

- `training_curves.png`: train CE(25 step 평균)와 val CE, val accuracy
- `table1_roc_auc.png`: Table 1 데이터셋별 ROC AUC, TabPFN과 sklearn 기본값 중 가장 좋은 것

카드 본문에 적을 수치(마지막 val CE·accuracy, 평균 ROC AUC)도 함께 출력한다.
그림 안의 글자는 영어로 쓴다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd

from minipfn.tabpfn import spec

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINES = [
    "LogisticRegression (default)",
    "HistGradientBoosting (default)",
    "RandomForest (default)",
]
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#8a8984"
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-id", required=True, help="학습 run (history.json, eval_table1/)")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "hub/minipfn-tabpfn-v1/images")
    return parser.parse_args(argv)


def plot_training_curves(h: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    ax = axes[0]
    ax.plot(
        h["step"], h["train_ce"], color=BLUE, lw=1.5, alpha=0.8, label="train CE (25-step mean)"
    )
    ax.plot(
        h["val_step"],
        h["val_ce"],
        "o-",
        color=ORANGE,
        lw=2,
        ms=4,
        label="val CE (64 prior datasets)",
    )
    # 처음 loss(약 2.3)와 끝 loss(약 0.46)를 한 그림에 담으려고 log 축을 쓴다.
    ax.set_yscale("log")
    ax.set_yticks([0.5, 0.75, 1.0, 1.5, 2.0], ["0.5", "0.75", "1.0", "1.5", "2.0"])
    ax.minorticks_off()
    ax.set(xlabel="optimizer step", ylabel="cross-entropy (log scale)", title="Prior-fitting loss")
    ax.legend(loc="upper right")
    ax = axes[1]
    ax.plot(h["val_step"], h["val_acc"], "o-", color=ORANGE, lw=2, ms=4)
    ax.annotate(
        f"{h['val_acc'][-1]:.3f}",
        (h["val_step"][-1], h["val_acc"][-1]),
        xytext=(-4, -14),
        textcoords="offset points",
        ha="right",
        color=INK,
    )
    ax.set(
        xlabel="optimizer step",
        ylabel="accuracy",
        title="Validation accuracy (held-out prior datasets)",
    )
    for a in axes:
        a.xaxis.set_major_locator(plt.MultipleLocator(3000))
        a.xaxis.set_major_formatter(lambda x, _: f"{x / 1000:.0f}k" if x else "0")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_table1(per_dataset: pd.DataFrame, path: Path) -> float:
    wide = per_dataset.pivot(index="dataset", columns="method", values="roc_auc")
    wide["baseline"] = wide[BASELINES].max(axis=1)
    wide = wide.sort_values("TabPFN")
    mean = wide["TabPFN"].mean()

    fig, ax = plt.subplots(figsize=(8, 6.5))
    y = np.arange(len(wide))
    lo = wide[["TabPFN", "baseline"]].min(axis=1)
    hi = wide[["TabPFN", "baseline"]].max(axis=1)
    ax.hlines(y, lo, hi, color=GRID, lw=2, zorder=1)
    ax.scatter(
        wide["baseline"],
        y,
        s=45,
        marker="s",
        color=GRAY,
        zorder=2,
        label="best sklearn default (LR / HGB / RF)",
    )
    ax.scatter(
        wide["TabPFN"],
        y,
        s=55,
        color=BLUE,
        edgecolors=SURFACE,
        linewidths=1.5,
        zorder=3,
        label="minipfn TabPFN (ensemble 32)",
    )
    ax.set_yticks(y, wide.index, fontsize=9)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("ROC AUC (mean over 5 splits)")
    paper = spec.PAPER_TABLE1_ROC_AUC["TabPFN"]
    ax.set_title(
        f"Table 1 datasets (OpenML-CC18)\nmean ROC AUC {mean:.3f} (paper {paper:.3f})", loc="left"
    )
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return mean


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    history = json.loads(
        Path(
            mlflow.artifacts.download_artifacts(run_id=args.run_id, artifact_path="history.json")
        ).read_text()
    )
    per_dataset = pd.read_csv(
        mlflow.artifacts.download_artifacts(
            run_id=args.run_id, artifact_path="eval_table1/eval_per_dataset.csv"
        )
    )

    plt.rcParams.update(STYLE)
    plot_training_curves(history, args.out_dir / "training_curves.png")
    mean = plot_table1(per_dataset, args.out_dir / "table1_roc_auc.png")
    print("saved:", sorted(p.name for p in args.out_dir.glob("*.png")))
    print(f"run {args.run_id}")
    print(f"final val CE {history['val_ce'][-1]:.3f}, val accuracy {history['val_acc'][-1]:.3f}")
    print(f"Table 1 mean ROC AUC (ensemble 32) {mean:.3f}")


if __name__ == "__main__":
    main()
