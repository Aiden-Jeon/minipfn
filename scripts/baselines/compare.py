"""방법별 불확실성 평가 결과를 한 표로 모으고 점 예측 순위와 분포 지표 순위를 견준다 (issue #48).

입력은 `이름=출처`이고, 출처는 `eval.py`·`scripts/tabpfn_v2/eval.py --uncertainty`의 결과 폴더나
MLflow run(`runs:/<run_id>`, artifact `eval/`)이다. 출처 끝에 `#regression`·`#classification`을
붙이면 그 suite만 쓴다. 같은 이름을 여러 번 주면 합친다(v2는 분류와 회귀를 다른 run에서
가져온다). `eval_uncertainty_per_dataset.csv`에 점 예측 점수(`score`)가 없으면
`eval_per_dataset.csv`에서 가져온다.

  uv run python scripts/baselines/compare.py \
      v2=runs:/<run>#regression v2=runs:/<run>#classification \
      lightgbm=runs:/<run> tabpfn_v2=baselines_eval/tabpfn_v2 --out-dir baselines_compare

suite마다 모든 방법에 결과가 있는 데이터셋만 쓴다(데이터셋 수를 같이 낸다). 출력:

- 방법 × 지표의 데이터셋 평균 (`compare_means.csv`, 표준 출력은 markdown)
- 지표마다 데이터셋별 순위(1이 가장 좋다)의 평균 (`compare_ranks.csv`). coverage는 목표(90%)와의
  거리, SSCS·폭 5분위별 coverage 최솟값은 클수록 좋다고 본다
- 데이터셋마다 방법들의 점 예측 순위(회귀 RMSE, 분류 ROC AUC)와 분포 지표 순위(회귀 CRPS, 분류 log
  loss)의 Spearman 상관을 낸 뒤 평균한 값 (1이면 두 순위가 같다)
- 데이터셋별 긴 표 (`compare_per_dataset.csv`)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

KEYS = ["suite", "task_id", "dataset"]
TARGET = 0.9  # 가운데 90% 구간·conformal α = 0.1

# 지표: "min" 작을수록, "max" 클수록, 숫자면 그 값에 가까울수록 좋다
METRICS = {
    "regression": {
        "rmse_scaled": "min",
        "crps_scaled": "min",
        "interval_score_90_scaled": "min",
        "coverage_90": TARGET,
        "coverage_min_by_width_90": "max",
        "pit_ks": "min",
    },
    "classification": {
        "score": "max",
        "log_loss": "min",
        "brier": "min",
        "ece": "min",
        "lac_coverage": TARGET,
        "lac_set_size": "min",
        "lac_sscs": "max",
        "aps_set_size": "min",
        "aps_sscs": "max",
    },
}
POINT_METRIC = {"regression": "rmse_scaled", "classification": "score"}
DISTRIBUTION_METRIC = {"regression": "crps_scaled", "classification": "log_loss"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", help="이름=출처 (결과 폴더나 runs:/<run_id>)")
    parser.add_argument("--out-dir", default="baselines_compare")
    return parser.parse_args(argv)


def resolve(source: str) -> Path:
    """결과 폴더. MLflow run이면 artifact `eval/`을 내려받는다."""
    if source.startswith("runs:/"):
        import mlflow

        uri = source.rstrip("/") + ("" if source.rstrip("/").endswith("/eval") else "/eval")
        return Path(mlflow.artifacts.download_artifacts(artifact_uri=uri))
    return Path(source)


def load(source: str) -> pd.DataFrame:
    """출처 하나의 데이터셋별 표. `출처#suite`면 그 suite만 남긴다."""
    source, _, suite = source.partition("#")
    if suite and suite not in METRICS:
        raise SystemExit(f"suite는 {tuple(METRICS)} 가운데 하나: {suite}")
    path = resolve(source)
    table = pd.read_csv(path / "eval_uncertainty_per_dataset.csv")
    if "score" not in table.columns:
        scores = pd.read_csv(path / "eval_per_dataset.csv")[[*KEYS, "score"]]
        table = table.merge(scores, on=KEYS, how="left")
    return table[table["suite"] == suite] if suite else table


def collect(specs: list[str]) -> pd.DataFrame:
    tables = []
    for spec in specs:
        name, sep, source = spec.partition("=")
        if not sep or not name or not source:
            raise SystemExit(f"이름=출처로 준다: {spec}")
        tables.append(load(source).assign(method=name))
    df = pd.concat(tables, ignore_index=True)
    duplicated = df.duplicated(["method", *KEYS])
    if duplicated.any():
        rows = df.loc[duplicated, ["method", "dataset"]].drop_duplicates().values.tolist()
        raise SystemExit(f"같은 방법·데이터셋이 두 번 있다: {rows}")
    return df


def common_datasets(df: pd.DataFrame) -> pd.DataFrame:
    """suite마다 모든 방법에 결과가 있는 데이터셋만."""
    out = []
    for _, group in df.groupby("suite"):
        n_methods = group["method"].nunique()
        counts = group.groupby("task_id")["method"].nunique()
        out.append(group[group["task_id"].isin(counts[counts == n_methods].index)])
    return pd.concat(out, ignore_index=True)


def loss(values: pd.Series, direction) -> pd.Series:
    """작을수록 좋은 값으로 바꾼다."""
    if direction == "min":
        return values
    if direction == "max":
        return -values
    return (values - direction).abs()


def ranks(group: pd.DataFrame, suite: str) -> pd.DataFrame:
    """지표마다 데이터셋 안에서 방법의 순위 (1이 가장 좋다, 같으면 평균 순위)."""
    out = group[["method", *KEYS]].copy()
    for metric, direction in METRICS[suite].items():
        if metric in group.columns:
            by_dataset = loss(group[metric], direction).groupby(group["task_id"])
            out[metric] = by_dataset.rank(method="average")
    return out


def rank_agreement(rank_table: pd.DataFrame, suite: str) -> tuple[float, int]:
    """데이터셋마다 점 예측 순위와 분포 지표 순위의 Spearman 상관의 평균, 쓴 데이터셋 수."""
    point, dist = POINT_METRIC[suite], DISTRIBUTION_METRIC[suite]
    rhos = []
    for _, g in rank_table.groupby("task_id"):
        if g[point].nunique() > 1 and g[dist].nunique() > 1:
            rhos.append(stats.spearmanr(g[point], g[dist]).statistic)
    return (float(np.mean(rhos)) if rhos else float("nan")), len(rhos)


def markdown(table: pd.DataFrame, digits: int = 4) -> str:
    cells = table.reset_index()
    header = "| " + " | ".join(map(str, cells.columns)) + " |"
    rule = "| " + " | ".join("---" for _ in cells.columns) + " |"
    body = [
        "| " + " | ".join(f"{v:.{digits}f}" if isinstance(v, float) else str(v) for v in row) + " |"
        for row in cells.itertuples(index=False)
    ]
    return "\n".join([header, rule, *body])


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = common_datasets(collect(args.runs))
    df.to_csv(out_dir / "compare_per_dataset.csv", index=False)
    order = list(dict.fromkeys(spec.partition("=")[0] for spec in args.runs))
    all_means, all_ranks = [], []
    for suite, group in df.groupby("suite"):
        columns = [m for m in METRICS[suite] if m in group.columns]
        n = group["task_id"].nunique()
        means = (
            group.groupby("method")[columns]
            .mean()
            .reindex([m for m in order if m in set(group["method"])])
        )
        rank_table = ranks(group, suite)
        mean_ranks = rank_table.groupby("method")[columns].mean().reindex(means.index)
        rho, n_rho = rank_agreement(rank_table, suite)
        print(f"\n## {suite} (datasets {n}, methods {len(means)})\n")
        print("데이터셋 평균\n")
        print(markdown(means))
        print("\n평균 순위 (1이 가장 좋다)\n")
        print(markdown(mean_ranks, digits=2))
        print(
            f"\n점 예측({POINT_METRIC[suite]})과 분포({DISTRIBUTION_METRIC[suite]}) 순위의 "
            f"데이터셋별 Spearman 평균: {rho:.3f} (datasets {n_rho})"
        )
        all_means.append(means.assign(suite=suite, datasets=n))
        all_ranks.append(mean_ranks.assign(suite=suite, datasets=n, rank_spearman=rho))
    pd.concat(all_means).to_csv(out_dir / "compare_means.csv")
    pd.concat(all_ranks).to_csv(out_dir / "compare_ranks.csv")


if __name__ == "__main__":
    main()
