"""학습 체크포인트와 benchmark 평가 run으로 TabPFN-2.5 Hub 모델 카드의 그림을 만든다.

2.5는 두 단계로 학습했다: paper 설정으로 500k step(prior-fitting) 뒤, 그 가중치에서 긴·넓은 표로
20k step 더(`long_tables`, 새 optimizer). 그래서 v2 카드 그림(`scripts/tabpfn_v2/plot_card.py`)과
달리 두 단계를 나눠 그리고, 같은 benchmark의 v2 점수를 함께 그린다.

예) 저장소 루트에서 (.env.local을 불러온 셸)
  MLFLOW_TRACKING_URI=databricks uv run python scripts/tabpfn_v2_5/plot_card.py \
    --checkpoint runs:/<paper run_id>/checkpoints/step_0500000.pt \
    --long-tables-checkpoint runs:/<long_tables run_id>/checkpoints/step_0020000.pt \
    --eval-runs 100000=<평가 run_id>,...,500000=<평가 run_id> \
    --long-tables-eval-runs 10000=<평가 run_id>,20000=<평가 run_id> \
    --v2-eval-runs 100000=<v2 평가 run_id>,...,500000=<v2 평가 run_id>
  (회귀는 --task regression)

`--out-dir`(기본 `hub/minipfn-tabpfn-v2-5-<classifier|regressor>/images/`)에 세 그림을 쓴다.
데이터셋별 그림은 long_tables의 step이 가장 큰 평가 run(카드가 설명하는 가중치)으로 그린다.

- `training_curves.png`: 왼쪽 prior-fitting(0–500k), 오른쪽 long_tables(0–20k). 두 단계의 val은
  서로 다른 고정 prior 데이터셋(64개, long_tables는 긴 표 32개)이라 값을 서로 견주지 않는다
- `benchmark_by_step.png`: step별 benchmark 점수(분류 평균 ROC AUC, 회귀 RMSE 비율의 기하평균),
  v2(회색)와 논문 값. long_tables 점수는 500k 오른쪽에 `500k + LT` 로 둔다
- `benchmark_roc_auc.png`(분류)·`benchmark_rmse_ratio.png`(회귀): 데이터셋별로 이 모델, v2 500k,
  논문

카드 본문에 적을 수치도 함께 출력한다. 그림 안의 글자는 영어로 쓴다.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]


def _v2_card():
    # v2 카드 스크립트의 색·글꼴·보조 함수를 그대로 쓴다 (같은 이름 plot_card.py라 경로로 읽는다)
    path = REPO_ROOT / "scripts/tabpfn_v2/plot_card.py"
    spec = importlib.util.spec_from_file_location("tabpfn_v2_plot_card", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v2 = _v2_card()
BLUE, GRAY, INK, MUTED, GRID, SURFACE = v2.BLUE, v2.GRAY, v2.INK, v2.MUTED, v2.GRID, v2.SURFACE
ORANGE = "#d9822b"  # long_tables 단계
SMOOTH = v2.SMOOTH
_k, geomean, rmse_ratio = v2._k, v2.geomean, v2.rmse_ratio
PAPER_STEPS = 500_000  # long_tables 점수를 이 step 오른쪽에 둔다
LT_GAP = 60_000  # 그림에서 500k와 `500k + LT` 점 사이 거리


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True, help="paper 학습 checkpoint의 runs:/ URI")
    parser.add_argument(
        "--long-tables-checkpoint", required=True, help="long_tables checkpoint의 runs:/ URI"
    )
    parser.add_argument("--eval-runs", required=True, help="paper step=평가 run_id, 쉼표로 구분")
    parser.add_argument(
        "--long-tables-eval-runs", required=True, help="long_tables step=평가 run_id"
    )
    parser.add_argument("--v2-eval-runs", required=True, help="v2 step=평가 run_id")
    parser.add_argument(
        "--task", choices=["classification", "regression"], default="classification"
    )
    parser.add_argument(
        "--out-dir", type=Path, help="기본: hub/minipfn-tabpfn-v2-5-<classifier|regressor>/images"
    )
    parser.add_argument("--cache-dir", default="tabpfn_v2_hub/card_v2_5")
    args = parser.parse_args(argv)
    if args.out_dir is None:
        name = "classifier" if args.task == "classification" else "regressor"
        args.out_dir = REPO_ROOT / f"hub/minipfn-tabpfn-v2-5-{name}/images"
    return args


def load_history(uri: str, cache: Path) -> dict:
    path = mlflow.artifacts.download_artifacts(artifact_uri=uri, dst_path=str(cache))
    return torch.load(path, map_location="cpu", weights_only=True)["history"]


def load_evals(spec: str, task: str, cache: Path) -> dict[int, pd.DataFrame]:
    evals = {}
    for item in spec.split(","):
        step, run_id = item.split("=")
        csv = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path="eval/eval_per_dataset.csv", dst_path=str(cache / run_id)
        )
        df = pd.read_csv(csv)
        if "suite" in df.columns:
            df = df[df["suite"] == task]
        if df.empty:
            raise SystemExit(f"평가 run {run_id}에 {task} 결과가 없다 (--task를 확인).")
        evals[int(step)] = df.set_index("task_id")
    return dict(sorted(evals.items()))


def _smoothed(history: dict) -> tuple[np.ndarray, np.ndarray, pd.Series]:
    step = np.asarray(history["step"])
    train = np.asarray(history["train_loss"])
    n = len(step) // SMOOTH * SMOOTH
    step_s = step[:n].reshape(-1, SMOOTH)[:, -1]
    train_s = train[:n].reshape(-1, SMOOTH).mean(axis=1)
    # chunk 경계에서 같은 step의 검증이 두 번 남은 적이 있어 마지막 값을 쓴다
    val = pd.Series(history["val_loss"], index=history["val_step"]).groupby(level=0).last()
    return step_s, train_s, val


def plot_training_curves(paper: dict, lt: dict, path: Path, loss: str, ylabel: str) -> dict:
    fig, (ax, bx) = plt.subplots(
        1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [3, 1.1]}, sharey=False
    )
    step_s, train_s, val = _smoothed(paper)
    ax.plot(step_s, train_s, color=BLUE, lw=1.2, alpha=0.55, label=f"train {loss} (1k-step mean)")
    ax.plot(val.index, val.values, color=BLUE, lw=2, label=f"val {loss} (64 fixed prior tables)")
    after = np.concatenate([train_s[step_s >= 25_000], val[val.index >= 25_000].values])
    lo, hi = np.nanpercentile(after, 0.5), np.nanpercentile(after, 99.5)
    pad = (hi - lo) * 0.15
    ax.set_ylim(lo - pad, hi + pad)
    ax.set(xlabel="optimizer step", ylabel=ylabel, title="Prior-fitting (paper sizes)")
    ax.xaxis.set_major_locator(plt.MultipleLocator(100_000))
    ax.xaxis.set_major_formatter(_k)
    ax.legend(loc="upper right")

    lt_step, lt_train, lt_val = _smoothed(lt)
    bx.plot(lt_step, lt_train, color=ORANGE, lw=1.2, alpha=0.6, label="train (1k-step mean)")
    bx.plot(lt_val.index, lt_val.values, "o-", color=ORANGE, lw=2, ms=5, label="val (long tables)")
    bx.set(xlabel="long-tables step", title="Long tables (from 500k)")
    bx.xaxis.set_major_locator(plt.MultipleLocator(10_000))
    bx.xaxis.set_major_formatter(_k)
    bx.legend(loc="center right", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return {
        "paper_val": (int(val.index[-1]), float(val.iloc[-1])),
        "lt_val": (int(lt_val.index[-1]), float(lt_val.iloc[-1])),
    }


def _score(df: pd.DataFrame, task: str) -> float:
    return float(df["score"].mean()) if task == "classification" else geomean(rmse_ratio(df))


def plot_by_step(evals: dict, lt_evals: dict, v2_evals: dict, task: str, path: Path) -> None:
    ours = pd.Series({s: _score(df, task) for s, df in evals.items()})
    base = pd.Series({s: _score(df, task) for s, df in v2_evals.items()})
    lt_step, lt_df = list(lt_evals.items())[-1]
    lt_x, lt_y = PAPER_STEPS + LT_GAP, _score(lt_df, task)
    if task == "classification":
        paper, fmt, ylabel = float(lt_df["paper"].mean()), "{:.4f}", "mean ROC AUC (29 datasets)"
        title = "Benchmark ROC AUC by training step"
    else:
        paper, fmt, ylabel = 1.0, "{:.3f}", "RMSE / paper RMSE (geo. mean)"
        title = "Benchmark RMSE ratio by training step (lower is better)"

    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.axhline(paper, color=GRAY, lw=1.5, ls="--")
    label = f"paper {paper:.4f}" if task == "classification" else "paper (ratio 1)"
    ax.annotate(
        label, (ours.index[0], paper), xytext=(0, 5), textcoords="offset points", color=MUTED
    )
    ax.plot(
        base.index,
        base.values,
        "o-",
        color=GRAY,
        lw=1.5,
        ms=6,
        mec=SURFACE,
        mew=1.5,
        label="minipfn TabPFN v2",
    )
    ax.plot(
        ours.index,
        ours.values,
        "o-",
        color=BLUE,
        lw=2,
        ms=8,
        mec=SURFACE,
        mew=2,
        label="minipfn TabPFN-2.5",
    )
    ax.plot([ours.index[-1], lt_x], [ours.iloc[-1], lt_y], ":", color=ORANGE, lw=1.5)
    ax.plot(
        [lt_x],
        [lt_y],
        "D",
        color=ORANGE,
        ms=8,
        mec=SURFACE,
        mew=2,
        label=f"+ long tables ({_k(lt_step)} steps)",
    )
    off = -16 if task == "classification" else 9
    for x, y in [*ours.items(), (lt_x, lt_y)]:
        ax.annotate(
            fmt.format(y),
            (x, y),
            xytext=(0, off),
            textcoords="offset points",
            ha="center",
            color=INK,
            fontsize=9,
        )
    ticks = list(ours.index) + [lt_x]
    ax.set_xticks(ticks, [_k(t) for t in ours.index] + ["500k\n+ LT"])
    ax.set_xlim(ours.index[0] - 40_000, lt_x + 40_000)
    values = np.concatenate([ours.values, base.values, [lt_y, paper]])
    pad = (values.max() - values.min()) * 0.15
    ax.set_ylim(values.min() - pad, values.max() + pad)
    ax.set(xlabel="optimizer step", ylabel=ylabel)
    ax.set_title(title, loc="left")
    ax.legend(loc="lower right" if task == "classification" else "upper right", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_per_dataset(final: pd.DataFrame, v2_final: pd.DataFrame, task: str, path: Path) -> None:
    df = final[["dataset", "score", "paper"]].join(v2_final["score"].rename("v2"))
    if task == "classification":
        df = df.assign(x=df["score"], v2x=df["v2"], px=df["paper"])
        df = df.assign(gap=df["x"] - df["px"]).sort_values("gap", ascending=False)
        xlabel = "ROC AUC (mean over 10 folds)"
        title = (
            f"AMLB classification datasets, sorted by gap to the paper\n"
            f"mean ROC AUC {df['x'].mean():.4f} (v2 500k {df['v2x'].mean():.4f}, "
            f"paper {df['px'].mean():.4f})"
        )
    else:
        df = df.assign(x=df["score"] / df["paper"], v2x=df["v2"] / df["paper"], px=1.0)
        df = df.sort_values("x", ascending=False)
        xlabel = "RMSE / paper RMSE (mean over 10 folds, log scale; lower is better)"
        title = (
            f"AMLB regression datasets, sorted by RMSE ratio to the paper\n"
            f"geometric mean ratio {geomean(df['x']):.3f} (v2 500k {geomean(df['v2x']):.3f}),"
            f" median {df['x'].median():.3f}"
        )
    fig, ax = plt.subplots(figsize=(8, 8.5))
    y = np.arange(len(df))
    lo = df[["x", "px"]].min(axis=1)
    hi = df[["x", "px"]].max(axis=1)
    ax.hlines(y, lo, hi, color=GRID, lw=2, zorder=1)
    if task == "classification":
        ax.scatter(df["px"], y, s=45, marker="s", color=GRAY, zorder=2, label="paper (TabPFN v2)")
    else:
        ax.axvline(1.0, color=GRAY, lw=1.5, ls="--", zorder=1, label="paper (TabPFN v2)")
    ax.scatter(
        df["v2x"],
        y,
        s=40,
        facecolors="none",
        edgecolors=GRAY,
        linewidths=1.5,
        zorder=2,
        label="minipfn v2 500k",
    )
    ax.scatter(
        df["x"],
        y,
        s=55,
        color=BLUE,
        edgecolors=SURFACE,
        linewidths=1.5,
        zorder=3,
        label="this model (500k + LT 20k)",
    )
    if task == "regression":
        ax.set_xscale("log")
        lo_t, hi_t = min(df["x"].min(), df["v2x"].min(), 1.0), max(df["x"].max(), df["v2x"].max())
        ticks = [
            t for t in (0.8, 0.9, 1, 1.25, 1.5, 2, 3, 4, 6, 8) if lo_t / 1.05 <= t <= hi_t * 1.05
        ]
        ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
        ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_yticks(y, df["dataset"], fontsize=8.5)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache_dir)
    paper = load_history(args.checkpoint, cache / "paper")
    lt = load_history(args.long_tables_checkpoint, cache / "long_tables")
    evals = load_evals(args.eval_runs, args.task, cache)
    lt_evals = load_evals(args.long_tables_eval_runs, args.task, cache)
    v2_evals = load_evals(args.v2_eval_runs, args.task, cache)
    plt.rcParams.update(v2.STYLE)

    if args.task == "classification":
        loss, ylabel, per_dataset = "CE", "cross-entropy", "benchmark_roc_auc.png"
    else:
        loss, ylabel, per_dataset = "NLL", "negative log-likelihood", "benchmark_rmse_ratio.png"
    vals = plot_training_curves(paper, lt, args.out_dir / "training_curves.png", loss, ylabel)
    plot_by_step(evals, lt_evals, v2_evals, args.task, args.out_dir / "benchmark_by_step.png")
    final = list(lt_evals.values())[-1]
    v2_final = list(v2_evals.values())[-1]
    plot_per_dataset(final, v2_final, args.task, args.out_dir / per_dataset)

    print("saved:", sorted(p.name for p in args.out_dir.glob("*.png")))
    print(f"checkpoint {args.long_tables_checkpoint} (from {args.checkpoint})")
    print(
        f"val {loss} prior-fitting at {vals['paper_val'][0]}: {vals['paper_val'][1]:.3f}; "
        f"long tables at {vals['lt_val'][0]}: {vals['lt_val'][1]:.3f}"
    )
    rows = [(f"{_k(s)}", df) for s, df in evals.items()]
    rows += [(f"500k + LT {_k(s)}", df) for s, df in lt_evals.items()]
    for name, df in rows:
        if args.task == "classification":
            print(f"{name}: mean ROC AUC {df['score'].mean():.4f}")
        else:
            r = rmse_ratio(df)
            print(
                f"{name}: RMSE ratio geo. mean {geomean(r):.3f}, median {r.median():.3f}, "
                f"within 1.1x {(r <= 1.1).sum()}/{len(r)}, mean neg RMSE {df['score'].mean():.1f}"
            )
    for s, df in v2_evals.items():
        print(f"v2 {_k(s)}: {_score(df, args.task):.4f}")


if __name__ == "__main__":
    main()
