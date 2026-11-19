"""학습·평가 run으로 TabPFN-3.5 Hub 모델 카드(`hub/minipfn-tabpfn-v3-5/`)의 그림을 만든다.

3.5는 분류·회귀를 한 모델로(multitask), TabICLv2 3단계 curriculum
(`recipes.tabpfn_v3.CURRICULUM`)으로 학습했다. 단계마다 가중치만 넘겨 새 optimizer로 시작했고, 한
단계 안에서도 이어 학습하면 MLflow run이 나뉜다. 그래서 학습 곡선은 checkpoint의 `history`가 아니라
단계마다 run들의 MLflow 지표(task별)를 이어 붙여 그린다. 같은 step이 두 run에 있으면(이어 학습한
지점) 뒤 run의 값을 쓴다.

예) 저장소 루트에서 (.env.local을 불러온 셸)
  MLFLOW_TRACKING_URI=databricks uv run python scripts/tabpfn_v3/plot_card.py \
    --train-runs 1=<run_id>,1=<run_id>,1=<run_id>,2=<run_id>,3=<run_id>,3=<run_id>,3=<run_id> \
    --benchmark-runs 1=<평가 run_id>,2=<평가 run_id>,3=<평가 run_id> \
    --tabarena-runs 1=<평가 run_id>,2=...,3=... --beyondarena-runs 1=...,2=...,3=... \
    --v2-5-benchmark-runs classification=<평가 run_id>,regression=<평가 run_id> \
    --v2-5-tabarena-run <평가 run_id> --v2-5-beyondarena-run <평가 run_id>

`--out-dir`(기본 `hub/minipfn-tabpfn-v3-5/images/`)에 그림 다섯 개를 쓴다. 비교 대상 2.5는 같은
평가 코드로 잰 minipfn TabPFN-2.5(Hub의 v2.5 분류·회귀 모델)다.

- `training_curves.png`: 위 분류 CE, 아래 회귀 NLL, 열마다 curriculum 단계 하나. val은 단계마다 그
  단계 표 크기로 고정한 prior 표 64개라 단계끼리 값을 견주지 않는다
- `benchmark_by_stage.png`: 단계별 v2 benchmark 점수(분류 평균 ROC AUC, 회귀 논문 대비 RMSE 비율의
  기하평균)와 2.5·논문 값
- `arena_elo.png`: 단계별 TabArena-Lite·BeyondArena Elo(95% CI)와 같은 leaderboard의 공개 방법
- `benchmark_roc_auc.png`·`benchmark_rmse_ratio.png`: 마지막 단계의 데이터셋별 점수, 2.5와 논문

카드 본문에 적을 수치도 함께 출력한다. 그림 안의 글자는 영어로 쓴다.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd

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
ORANGE = "#d9822b"  # minipfn TabPFN-2.5
_k, geomean, rmse_ratio = v2._k, v2.geomean, v2.rmse_ratio
TASKS = ("classification", "regression")
STAGES = (1, 2, 3)
STAGE_NAMES = {1: "Stage 1", 2: "Stage 2", 3: "Stage 3"}
# 단계마다 표 크기 (그림 제목. `recipes.tabpfn_v3.CURRICULUM`)
STAGE_ROWS = {1: "1,024 rows", 2: "400-10k rows", 3: "400-60k rows"}
# train loss는 100 step마다 남는다. 단계 길이가 달라 평균하는 step 수를 단계마다 정한다
SMOOTH_STEPS = {1: 2_000, 2: 500, 3: 500}
# 같은 leaderboard에 있는 공개 방법 (그림에 기준선으로 둔다). 이름은 suite마다 조금 달라서
# 대소문자와 앞의 "TA-"를 무시하고 이름 전체가 같은 것을 찾는다
REFERENCES = {
    "TabPFN-3.5 (official)": "TABPFN-3.5 (DEFAULT)",
    "TabICLv2": "TABICLV2 (DEFAULT)",
    "RealTabPFN-2.5": "REALTABPFN-V2.5 (DEFAULT)",
}
ARENA = {"tabarena": "tabarena-lite", "beyondarena": "beyondarena"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--train-runs",
        required=True,
        help="stage=학습 run_id, 쉼표로 구분 (단계 안에서는 순서대로)",
    )
    parser.add_argument("--benchmark-runs", required=True, help="stage=benchmark 평가 run_id")
    parser.add_argument(
        "--tabarena-runs",
        required=True,
        help="stage=TabArena-Lite 평가 run_id (또는 leaderboard CSV 경로)",
    )
    parser.add_argument(
        "--beyondarena-runs",
        required=True,
        help="stage=BeyondArena 평가 run_id (또는 leaderboard CSV 경로)",
    )
    parser.add_argument(
        "--v2-5-benchmark-runs",
        required=True,
        help="minipfn TabPFN-2.5 benchmark 평가: classification=<run_id>,regression=<run_id>",
    )
    parser.add_argument("--v2-5-tabarena-run", required=True)
    parser.add_argument("--v2-5-beyondarena-run", required=True)
    parser.add_argument(
        "--out-dir", type=Path, default=REPO_ROOT / "hub/minipfn-tabpfn-v3-5/images"
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("tabpfn_v3_hub/card"))
    return parser.parse_args(argv)


def parse_pairs(spec: str, key=int) -> list[tuple]:
    """`a=b,c=d`를 [(key(a), b), ...]로 (적은 순서 그대로)."""
    pairs = []
    for item in spec.split(","):
        k, _, v = item.partition("=")
        if not v:
            raise SystemExit(f"`키=run_id` 형식이 아니다: {item}")
        pairs.append((key(k), v))
    return pairs


def metric_history(client, run_id: str, key: str, cache: Path) -> pd.Series:
    """run의 지표 기록 (step → 값). 받은 것은 `cache`에 둔다 (그림만 고쳐 다시 그릴 때)."""
    path = cache / "metrics" / f"{run_id}_{key}.json"
    if not path.is_file():
        points = [(m.step, m.value) for m in client.get_metric_history(run_id, key)]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(points))
    points = json.loads(path.read_text())
    return pd.Series(dict(points), dtype=float).sort_index()


def stage_metric(client, run_ids: list[str], key: str, cache: Path) -> pd.Series:
    """한 단계의 run들(이어 학습한 순서)을 이어 붙인 지표. 같은 step은 뒤 run의 값."""
    out = pd.Series(dtype=float)
    for run_id in run_ids:
        part = metric_history(client, run_id, key, cache)
        out = pd.concat([out[out.index < part.index.min()], part]) if len(part) else out
    return out


def _block_mean(s: pd.Series, every: int) -> pd.Series:
    # `every` step 묶음마다 평균 (x는 묶음의 마지막 step)
    groups = (s.index - 1) // every
    return s.groupby(groups).mean().set_axis(s.groupby(groups).apply(lambda g: g.index.max()))


def plot_training_curves(curves: dict, path: Path) -> dict:
    """curves[stage][task] = (train, val). 단계마다 열 하나, task마다 행 하나."""
    widths = [3.0, 1.2, 1.2]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.2), gridspec_kw={"width_ratios": widths})
    finals = {}
    for col, stage in enumerate(STAGES):
        for row, task in enumerate(TASKS):
            ax = axes[row, col]
            train, val = curves[stage][task]
            smooth = _block_mean(train, SMOOTH_STEPS[stage])
            ax.plot(
                smooth.index,
                smooth.values,
                color=BLUE,
                lw=1.1,
                alpha=0.55,
                label=f"train ({_k(SMOOTH_STEPS[stage])}-step mean)",
            )
            ax.plot(
                val.index, val.values, "o-", color=BLUE, lw=2, ms=4, label="val (64 prior tables)"
            )
            # stage 1 앞부분은 손실이 커서 나머지가 납작해진다. 5% 뒤부터로 범위를 잡는다
            start = val.index.max() * 0.05 if stage == 1 else 0
            shown = np.concatenate([smooth[smooth.index >= start].values, val.values])
            lo, hi = np.nanpercentile(shown, 0.5), np.nanpercentile(shown, 99.5)
            pad = (hi - lo) * 0.15 or 0.01
            ax.set_ylim(lo - pad, hi + pad)
            ax.xaxis.set_major_formatter(_k)
            if stage == 1:
                ax.xaxis.set_major_locator(plt.MultipleLocator(50_000))
            if row == 0:
                ax.set_title(f"{STAGE_NAMES[stage]}: {STAGE_ROWS[stage]}", loc="left")
            if row == 1:
                ax.set_xlabel("optimizer step (in stage)")
            if col == 0:
                ax.set_ylabel("classification CE" if task == "classification" else "regression NLL")
                ax.legend(loc="upper right", fontsize=8.5)
            finals[(stage, task)] = (int(val.index[-1]), float(val.iloc[-1]))
    fig.suptitle(
        "Training curves (one multitask model; val tables differ by stage, so compare within"
        " a stage)",
        x=0.01,
        ha="left",
        color=MUTED,
        fontsize=9.5,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return finals


def load_per_dataset(run_id: str, cache: Path, task: str) -> pd.DataFrame:
    """benchmark 평가 run의 데이터셋별 점수 (그 task만, task_id가 index)."""
    csv = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path="eval/eval_per_dataset.csv", dst_path=str(cache / run_id)
    )
    df = pd.read_csv(csv)
    df = df[df["suite"] == task]
    if df.empty:
        raise SystemExit(f"평가 run {run_id}에 {task} 결과가 없다.")
    return df.set_index("task_id")


def score(df: pd.DataFrame, task: str) -> float:
    """분류는 평균 ROC AUC, 회귀는 논문 대비 RMSE 비율의 기하평균."""
    return float(df["score"].mean()) if task == "classification" else geomean(rmse_ratio(df))


def plot_by_stage(bench: dict, v25: dict, path: Path) -> None:
    """bench[stage][task], v25[task]: 데이터셋별 점수."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    x = np.arange(len(STAGES))
    for ax, task in zip(axes, TASKS, strict=True):
        ours = [score(bench[s][task], task) for s in STAGES]
        ref = score(v25[task], task)
        n = len(bench[STAGES[-1]][task])
        if task == "classification":
            paper = float(bench[STAGES[-1]][task]["paper"].mean())
            fmt, ylabel = "{:.4f}", f"mean ROC AUC ({n} datasets)"
            title, paper_label = "Classification (higher is better)", f"TabPFN v2 paper {paper:.4f}"
        else:
            paper, fmt, ylabel = 1.0, "{:.3f}", f"RMSE / paper RMSE (geo. mean, {n} datasets)"
            title, paper_label = "Regression (lower is better)", "TabPFN v2 paper (ratio 1)"
        ax.axhline(paper, color=GRAY, lw=1.5, ls="--", label=paper_label)
        ax.axhline(ref, color=ORANGE, lw=1.5, ls=":", label=f"minipfn TabPFN-2.5 {fmt.format(ref)}")
        ax.plot(x, ours, "o-", color=BLUE, lw=2, ms=8, mec=SURFACE, mew=2, label="this model")
        off = 9 if task == "classification" else -15
        for xi, y in zip(x, ours, strict=True):
            ax.annotate(
                fmt.format(y),
                (xi, y),
                xytext=(0, off),
                textcoords="offset points",
                ha="center",
                color=INK,
                fontsize=9,
            )
        ax.set_xticks(x, ["stage 1\n250k", "stage 2\n+20k", "stage 3\n+5k"])
        ax.set_xlim(-0.4, len(STAGES) - 0.6)
        values = np.array([*ours, ref, paper])
        pad = (values.max() - values.min()) * 0.25
        ax.set_ylim(values.min() - pad, values.max() + pad)
        ax.set(ylabel=ylabel)
        ax.set_title(title, loc="left")
        ax.legend(loc="lower right" if task == "classification" else "upper right", fontsize=8)
    fig.suptitle("TabPFN v2 benchmark (AMLB, 10 folds) by curriculum stage", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def load_leaderboard(run_id: str, suite: str, cache: Path) -> pd.DataFrame:
    """평가 run의 `eval/<suite>/leaderboard.csv`. run_id 대신 leaderboard CSV 경로를 주면 그
    파일을 읽는다(평가 run 없이 split 일부만 바꿔 다시 집계한 leaderboard, 예: issue #109)."""
    if run_id.endswith(".csv"):
        return pd.read_csv(run_id)
    csv = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path=f"eval/{suite}/leaderboard.csv", dst_path=str(cache / run_id)
    )
    return pd.read_csv(csv)


def own_row(board: pd.DataFrame) -> pd.Series:
    """leaderboard에서 minipfn 방법의 행 (평가 run마다 하나)."""
    rows = board[board["method"].str.startswith("minipfn")]
    if len(rows) != 1:
        raise SystemExit(f"leaderboard에 minipfn 방법이 {len(rows)}개다: {list(rows['method'])}")
    return rows.iloc[0]


def reference_rows(board: pd.DataFrame) -> dict[str, pd.Series]:
    names = board["method"].str.upper().str.removeprefix("TA-")
    out = {}
    for label, name in REFERENCES.items():
        hit = board[names == name]
        if len(hit) != 1:
            # 그림에서 기준선이 조용히 빠지면 카드의 수치와 어긋난다
            raise SystemExit(f"leaderboard에서 {name}을 하나로 찾지 못했다 ({len(hit)}개)")
        out[label] = hit.iloc[0]
    return out


def plot_arena(boards: dict, v25: dict, path: Path) -> dict:
    """boards[suite][stage], v25[suite]: leaderboard. 공개 방법은 마지막 단계의 leaderboard에서."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9))
    x = np.arange(len(STAGES))
    out = {}
    for ax, (suite, title) in zip(
        axes,
        (("tabarena", "TabArena-Lite (51 datasets)"), ("beyondarena", "BeyondArena core2k")),
        strict=True,
    ):
        rows = [own_row(boards[suite][s]) for s in STAGES]
        elo = np.array([r["elo"] for r in rows])
        err = np.array([[r["elo-"] for r in rows], [r["elo+"] for r in rows]])
        refs = reference_rows(boards[suite][STAGES[-1]])
        mini25 = float(own_row(v25[suite])["elo"])
        lines = [("minipfn TabPFN-2.5", mini25, ORANGE, ":")]
        lines += [(label, float(r["elo"]), GRAY, "--") for label, r in refs.items()]
        for label, value, color, ls in lines:
            ax.axhline(value, color=color, lw=1.3, ls=ls)
            ax.annotate(
                f"{label} {value:.0f}",
                (-0.75, value),
                xytext=(0, 3),
                textcoords="offset points",
                ha="left",
                color=MUTED if color == GRAY else color,
                fontsize=8,
            )
        ax.errorbar(x, elo, yerr=err, fmt="o-", color=BLUE, lw=2, ms=8, mec=SURFACE, capsize=4)
        for xi, y in zip(x, elo, strict=True):
            ax.annotate(
                f"{y:.0f}",
                (xi, y),
                xytext=(10, -4),
                textcoords="offset points",
                color=INK,
                fontsize=9,
            )
        ax.set_xticks(x, ["stage 1\n250k", "stage 2\n+20k", "stage 3\n+5k"])
        ax.set_xlim(-0.8, len(STAGES) - 0.5)
        ax.set(ylabel="Elo (95% CI)")
        ax.set_title(title, loc="left")
        out[suite] = {
            "stages": list(elo),
            "v2.5": mini25,
            **{k: float(r["elo"]) for k, r in refs.items()},
        }
    fig.suptitle("Arena Elo by curriculum stage (public leaderboard pool)", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return out


def plot_per_dataset(final: pd.DataFrame, v25: pd.DataFrame, task: str, path: Path) -> None:
    df = final[["dataset", "score", "paper"]].join(v25["score"].rename("v25"))
    if task == "classification":
        df = df.assign(x=df["score"], refx=df["v25"], px=df["paper"])
        df = df.assign(gap=df["x"] - df["px"]).sort_values("gap", ascending=False)
        xlabel = "ROC AUC (mean over 10 folds)"
        title = (
            "AMLB classification datasets, sorted by gap to the paper\n"
            f"mean ROC AUC {df['x'].mean():.4f} (minipfn 2.5 {df['refx'].mean():.4f},"
            f" paper {df['px'].mean():.4f})"
        )
    else:
        df = df.assign(x=df["score"] / df["paper"], refx=df["v25"] / df["paper"], px=1.0)
        df = df.sort_values("x", ascending=False)
        xlabel = "RMSE / paper RMSE (mean over 10 folds, log scale; lower is better)"
        title = (
            "AMLB regression datasets, sorted by RMSE ratio to the paper\n"
            f"geometric mean ratio {geomean(df['x']):.3f} (minipfn 2.5 {geomean(df['refx']):.3f}),"
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
        df["refx"],
        y,
        s=40,
        facecolors="none",
        edgecolors=ORANGE,
        linewidths=1.5,
        zorder=2,
        label="minipfn TabPFN-2.5",
    )
    ax.scatter(
        df["x"],
        y,
        s=55,
        color=BLUE,
        edgecolors=SURFACE,
        linewidths=1.5,
        zorder=3,
        label="this model (stage 3)",
    )
    if task == "regression":
        ax.set_xscale("log")
        lo_t = min(df["x"].min(), df["refx"].min(), 1.0)
        hi_t = max(df["x"].max(), df["refx"].max())
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


def by_stage(spec: str) -> dict[int, list[str]]:
    out: dict[int, list[str]] = {}
    for stage, run_id in parse_pairs(spec):
        out.setdefault(stage, []).append(run_id)
    if sorted(out) != list(STAGES):
        raise SystemExit(f"단계 {STAGES}가 모두 있어야 한다: {sorted(out)}")
    return out


def one_per_stage(spec: str) -> dict[int, str]:
    runs = by_stage(spec)
    if any(len(v) != 1 for v in runs.values()):
        raise SystemExit(f"평가 run은 단계마다 하나다: {spec}")
    return {stage: v[0] for stage, v in runs.items()}


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache = args.cache_dir
    client = mlflow.MlflowClient()
    plt.rcParams.update(v2.STYLE)

    train_runs = by_stage(args.train_runs)
    curves = {
        stage: {
            task: (
                stage_metric(client, runs, f"train_loss_{task}", cache),
                stage_metric(client, runs, f"val_loss_{task}", cache),
            )
            for task in TASKS
        }
        for stage, runs in train_runs.items()
    }
    finals = plot_training_curves(curves, args.out_dir / "training_curves.png")

    bench_runs = one_per_stage(args.benchmark_runs)
    bench = {s: {t: load_per_dataset(r, cache, t) for t in TASKS} for s, r in bench_runs.items()}
    v25_runs = dict(parse_pairs(args.v2_5_benchmark_runs, key=str))
    if set(v25_runs) != set(TASKS):
        raise SystemExit(f"--v2-5-benchmark-runs에는 {TASKS}가 모두 있어야 한다")
    v25 = {t: load_per_dataset(v25_runs[t], cache, t) for t in TASKS}
    plot_by_stage(bench, v25, args.out_dir / "benchmark_by_stage.png")
    final = bench[STAGES[-1]]
    plot_per_dataset(
        final["classification"],
        v25["classification"],
        "classification",
        args.out_dir / "benchmark_roc_auc.png",
    )
    plot_per_dataset(
        final["regression"],
        v25["regression"],
        "regression",
        args.out_dir / "benchmark_rmse_ratio.png",
    )

    boards = {
        "tabarena": {
            s: load_leaderboard(r, ARENA["tabarena"], cache)
            for s, r in one_per_stage(args.tabarena_runs).items()
        },
        "beyondarena": {
            s: load_leaderboard(r, ARENA["beyondarena"], cache)
            for s, r in one_per_stage(args.beyondarena_runs).items()
        },
    }
    v25_boards = {
        "tabarena": load_leaderboard(args.v2_5_tabarena_run, ARENA["tabarena"], cache),
        "beyondarena": load_leaderboard(args.v2_5_beyondarena_run, ARENA["beyondarena"], cache),
    }
    elo = plot_arena(boards, v25_boards, args.out_dir / "arena_elo.png")

    print("saved:", sorted(p.name for p in args.out_dir.glob("*.png")))
    for (stage, task), (step, value) in finals.items():
        print(f"stage {stage} {task}: val at {step} {value:.4f}")
    for stage in STAGES:
        cls, reg = bench[stage]["classification"], bench[stage]["regression"]
        r = rmse_ratio(reg)
        print(
            f"stage {stage}: mean ROC AUC {cls['score'].mean():.4f} (>= paper"
            f" {(cls['score'] >= cls['paper']).sum()}/{len(cls)}), RMSE ratio geo. mean"
            f" {geomean(r):.3f}, median {r.median():.3f}, within 1.1x {(r <= 1.1).sum()}/{len(r)}"
        )
    print(
        f"v2.5: ROC AUC {score(v25['classification'], 'classification'):.4f}, RMSE ratio"
        f" {score(v25['regression'], 'regression'):.3f}"
    )
    for suite, values in elo.items():
        print(
            suite,
            {
                k: (round(v) if not isinstance(v, list) else [round(x) for x in v])
                for k, v in values.items()
            },
        )
    for suite, stages in boards.items():
        for stage, board in stages.items():
            row = own_row(board)
            ci = f"+{row['elo+']:.0f}/-{row['elo-']:.0f}"
            print(
                f"{suite} stage {stage}: elo {row['elo']:.0f} ({ci})"
                f" rank {board.index[board['method'] == row['method']][0] + 1}/{len(board)}"
                f" imputed {row['imputed']:.1%}"
            )


if __name__ == "__main__":
    main()
