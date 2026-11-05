"""학습 run의 기록으로 Hub 모델 카드의 그림을 만든다.

예) 저장소 루트에서 (.env.local을 불러온 셸)
  MLFLOW_TRACKING_URI=databricks \
    uv run python scripts/tabpfn/plot_card.py --run-id <학습 run_id> --eval-run-id <평가 run_id>

학습 run의 `history.json`과 평가 run의 `eval/eval_per_dataset.csv`(`eval.py`가 남긴다)를
내려받아 `--out-dir`(기본 `hub/minipfn-tabpfn-v1/images/`)에 두 그림을 쓴다. `--eval-run-id`를
주지 않으면 학습 run의 `eval_table1/eval_per_dataset.csv`를 읽는다(지금 카드의 run
`977614dce2104e44915f81b02705a178`처럼 평가를 학습 run에 남기던 때의 위치).

- `training_curves.png`: train CE(25 step 평균)와 val CE, val accuracy
- `table1_roc_auc.png`: Table 1 데이터셋별 ROC AUC, TabPFN과 sklearn 기본값 중 가장 좋은 것

그림의 설명(train CE 평균 구간, val 데이터셋 수, ensemble 수, split 수)은 run param에서 읽는다.
예전 run처럼 param이 없으면 기본값(`TabPFNConfig`, `spec`)을 쓴다.

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

from minipfn.recipes import tabpfn_v1 as spec
from minipfn.recipes.tabpfn_v1 import VAL_GROUPS, TabPFNConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
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
    parser.add_argument("--run-id", required=True, help="학습 run (history.json)")
    parser.add_argument(
        "--eval-run-id", help="Table 1 평가 run (eval/; 없으면 학습 run의 eval_table1/)"
    )
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "hub/minipfn-tabpfn-v1/images")
    return parser.parse_args(argv)


def run_params(run_id: str) -> dict[str, str]:
    return mlflow.MlflowClient().get_run(run_id).data.params


def step_label(x: float, _pos=None) -> str:
    return f"{x / 1000:g}k" if x >= 1000 else f"{x:g}"


def step_tick(last_step: int, max_ticks: int = 6) -> int:
    """눈금이 `max_ticks`칸을 넘지 않는 가장 작은 1·2·3·5×10^k 간격 (18,000 step이면 3,000)."""
    scale = 1
    while True:
        for base in (1, 2, 3, 5):
            if base * scale * max_ticks >= last_step:
                return base * scale
        scale *= 10


def ce_label(t: float) -> str:
    return f"{t:.1f}" if round(t, 1) == t else f"{t:g}"  # 0.5, 0.75, 1.0, 2.0


def plot_training_curves(h: dict, path: Path, log_every: int, n_val: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    ax = axes[0]
    ax.plot(
        h["step"],
        h["train_ce"],
        color=BLUE,
        lw=1.5,
        alpha=0.8,
        label=f"train CE ({log_every}-step mean)",
    )
    ax.plot(
        h["val_step"],
        h["val_ce"],
        "o-",
        color=ORANGE,
        lw=2,
        ms=4,
        label=f"val CE ({n_val} prior datasets)",
    )
    # 처음 loss(약 2.3)와 끝 loss(약 0.46)를 한 그림에 담으려고 log 축을 쓴다.
    # 눈금은 읽기 쉬운 값 가운데 실제 loss 범위에 드는 것만 쓴다.
    ax.set_yscale("log")
    ces = [*h["train_ce"], *h["val_ce"]]
    lo, hi = min(ces), max(ces)
    ticks = [t for t in (0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0) if lo <= t <= hi] or [lo, hi]
    ax.set_yticks(ticks, [ce_label(t) for t in ticks])
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
        # 학습 길이에 따라 눈금 간격을 정한다 (paper 18,000 step, medium 1,800 step 등)
        a.xaxis.set_major_locator(plt.MultipleLocator(step_tick(h["step"][-1])))
        a.xaxis.set_major_formatter(step_label)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_table1(per_dataset: pd.DataFrame, path: Path, n_ensemble: int, n_splits: int) -> float:
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
        label=f"minipfn TabPFN (ensemble {n_ensemble})",
    )
    ax.set_yticks(y, wide.index, fontsize=9)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(f"ROC AUC (mean over {n_splits} splits)")
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
    if args.eval_run_id:
        eval_run, eval_path = args.eval_run_id, "eval/eval_per_dataset.csv"
    else:
        eval_run, eval_path = args.run_id, "eval_table1/eval_per_dataset.csv"
    per_dataset = pd.read_csv(
        mlflow.artifacts.download_artifacts(run_id=eval_run, artifact_path=eval_path)
    )
    missing = sorted({"TabPFN", *BASELINES} - set(per_dataset["method"]))
    if missing:
        # --baselines-only나 --no-baselines로 평가한 run은 이 그림을 그릴 수 없다
        raise SystemExit(f"{eval_run}의 {eval_path}에 {', '.join(missing)} 결과가 없습니다.")

    train = run_params(args.run_id)
    default = TabPFNConfig()
    log_every = int(train.get("log_every", default.log_every))
    n_val = VAL_GROUPS * int(train.get("group_size", default.group_size))
    # 학습 run에 평가를 남기던 예전 run은 평가 설정을 param으로 남기지 않았다 (기본값으로 돌렸다)
    evaluated = run_params(args.eval_run_id) if args.eval_run_id else {}
    n_ensemble = int(evaluated.get("n_ensemble", spec.N_ENSEMBLE))
    n_splits = int(evaluated.get("splits", spec.N_SPLITS))

    plt.rcParams.update(STYLE)
    plot_training_curves(history, args.out_dir / "training_curves.png", log_every, n_val)
    mean = plot_table1(per_dataset, args.out_dir / "table1_roc_auc.png", n_ensemble, n_splits)
    print("saved:", sorted(p.name for p in args.out_dir.glob("*.png")))
    print(f"run {args.run_id}" + (f", eval run {args.eval_run_id}" if args.eval_run_id else ""))
    print(f"final val CE {history['val_ce'][-1]:.3f}, val accuracy {history['val_acc'][-1]:.3f}")
    print(f"Table 1 mean ROC AUC (ensemble {n_ensemble}) {mean:.3f}")


if __name__ == "__main__":
    main()
