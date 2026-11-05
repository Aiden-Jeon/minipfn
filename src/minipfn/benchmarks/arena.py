"""TabArena·BeyondArena 평가의 공통 부분: sklearn 형식 estimator를 공식 split으로 돌리고 공개
leaderboard와 같은 방식으로 점수를 낸다.

데이터, split, 지표, 집계는 TabArena 공식 패키지 `tabarena`(extra `arena`:
`uv sync --extra arena`)를 쓴다. 직접 만드는 것은 estimator를 그 패키지의 system
(`ExternalSystemModel`, 전처리·검증을 스스로 하는 방법)으로 감싸는 부분, 모델이 다룰 수 없는
데이터셋을 고르는 규칙, 결과를 저장하는 형식이다.

- 데이터셋마다 split 지표는 TabArena 정의대로 이진 분류 ROC AUC, 다중 분류 log loss, 회귀 RMSE이고,
  집계는 오차(`metric_error`, 작을수록 좋다)로 한다.
- 집계(`tabarena`의 `compare`): Elo(bootstrap 95% 구간, 기준 방법을 1000에 맞춘다. TabArena는
  RandomForest (default), BeyondArena는 XGBoost (default)), 정규화 점수(데이터셋마다 가장 좋은 방법
  1, 중앙값 방법 0, 음수는 0), improvability(가장 좋은 방법보다 오차가 몇 % 큰지의 평균), 순위.
  baseline 방법 약 100개의 split별 결과는 패키지가 처음 쓸 때 받아 `~/.cache/tabarena`에 둔다.
- 모델 상한(`ModelLimits`: 학습 행·feature·클래스 수, 문제 종류)을 넘는 split은 돌리지 않는다.
  leaderboard처럼 그 결과는 RandomForest (default)의 결과로 채워지고(imputed) 비율을 함께 낸다.
- 공개 leaderboard와 다른 점: TabArena는 모든 방법을 학습 split 안에서 8-fold bagging해
  돌리지만, 여기서는 기본으로 학습 split 전체에 한 번 fit한다. `bagging=k`를 주면 학습 split을
  k-fold로 나눠 k개를 fit하고 예측을 평균한다(그 fold의 k−1개를 학습 행으로 쓴다).
- 텍스트·날짜 열: estimator에 그대로 넘기지 않는다. 날짜 열은 초 단위 수로, 숫자가 아닌 열(텍스트
  포함)은 범주형으로 넘긴다(estimator가 번호를 매긴다).

    from minipfn.benchmarks import arena, tabarena
    result = arena.run(tabarena.SUITE, make_estimator, limits=..., out_dir="out")

`make_estimator(problem_type, categorical_indices)`는 `problem_type`("binary", "multiclass",
"regression")에 맞는 estimator를 돌려준다. 분류는 `fit`·`predict_proba`·`classes_`, 회귀는
`fit`·`predict`가 있으면 된다.
"""

from __future__ import annotations

import json
import math
import sys
import urllib.parse
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROBLEM_TYPES = ("binary", "multiclass", "regression")
PUBLIC_LEADERBOARD = "https://huggingface.co/spaces/TabArena/leaderboard/resolve/main/"
# 승률을 견줄 방법 {leaderboard 표시 이름: `tabarena` 결과의 방법 이름}. 결과의 이름은 suite마다
# 앞에 "TA-"가 붙거나 대소문자가 달라서 `_method_key`로 맞춘다. 결과가 없는 방법은 건너뛴다
WINRATE_BASELINES = {
    "XGBoost (default)": "XGB (default)",
    "XGBoost (tuned + ensembled)": "XGB (tuned + ensemble)",
    "CatBoost (default)": "CAT (default)",
    "TabPFNv2 (default)": "TABPFNV2_GPU (default)",
    "RealTabPFN-2.5 (default)": "REALTABPFN-V2.5 (default)",
    "TabICLv2 (default)": "TABICLV2 (default)",
}


def _method_key(method: str) -> str:
    return str(method).removeprefix("TA-").upper()


@dataclass(frozen=True)
class Suite:
    """benchmark 하나: `tabarena` context, 평가할 split, 공개 leaderboard의 같은 결과.

    `subset`은 context의 subset 식(예: `["lite"]`, `["core2k"]`), `public_csv`는
    `PUBLIC_LEADERBOARD` 아래 경로다. `breakdown`은 subset별 점수 {이름: (subset 식, 공개 CSV)}이다.
    """

    name: str
    context: str  # "TabArenaContext" | "BeyondArenaContext"
    subset: tuple[str, ...]
    public_csv: str
    breakdown: dict[str, tuple[tuple[str, ...], str]] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelLimits:
    """모델이 돌 수 있는 split의 상한. 넘으면 그 split은 imputed가 된다. None은 상한 없음."""

    max_train_rows: int | None = None
    max_features: int | None = None
    max_classes: int | None = None
    problem_types: tuple[str, ...] = PROBLEM_TYPES

    def skip_reason(self, row: pd.Series) -> str | None:
        """task grid 한 행(split 하나)을 돌리지 않는 이유. 돌리면 None."""
        if row["problem_type"] not in self.problem_types:
            return f"problem_type {row['problem_type']}"
        if self.max_train_rows is not None and row["max_train_rows"] > self.max_train_rows:
            return f"train rows {int(row['max_train_rows'])} > {self.max_train_rows}"
        if self.max_features is not None and row["n_features"] > self.max_features:
            return f"features {int(row['n_features'])} > {self.max_features}"
        n_classes = row["n_classes"]
        if self.max_classes is not None and n_classes > self.max_classes:
            return f"classes {int(n_classes)} > {self.max_classes}"
        return None


def make_context(suite: Suite):
    try:
        from tabarena import contexts
    except ImportError as e:  # pragma: no cover - extra가 없을 때만
        raise ImportError(
            "TabArena·BeyondArena 평가에는 tabarena 패키지가 필요하다: uv sync --extra arena"
        ) from e
    return getattr(contexts, suite.context)(backend="native")


Limits = ModelLimits | dict[str, ModelLimits]


def skip_reason(limits: Limits, row: pd.Series) -> str | None:
    """`limits`는 `ModelLimits` 하나 또는 문제 종류("binary" 등)마다 하나.
    없는 종류는 돌리지 않는다."""
    if isinstance(limits, ModelLimits):
        return limits.skip_reason(row)
    if row["problem_type"] not in limits:
        return f"problem_type {row['problem_type']}"
    return limits[row["problem_type"]].skip_reason(row)


def task_grid(context, suite: Suite, limits: Limits) -> pd.DataFrame:
    """suite의 split 목록(데이터셋, fold, repeat, 크기)과 돌리지 않는 이유(`skip_reason`)."""
    grid = context.task_metadata_collection.task_grid()
    grid = grid[_subset_mask(context, grid, suite.subset)].reset_index(drop=True)
    grid["skip_reason"] = [skip_reason(limits, row) for _, row in grid.iterrows()]
    return grid


def _subset_mask(context, grid: pd.DataFrame, subset: tuple[str, ...]) -> pd.Series:
    # subset 식 하나하나(AND)를 context의 predicate로. "!이름"은 NOT
    mask = pd.Series(True, index=grid.index)
    for expr in subset:
        negate = expr.startswith("!")
        m = context.subset_predicates[expr.lstrip("!")](grid)
        mask &= ~m if negate else m
    return mask


def run(
    suite: Suite,
    make_estimator: Callable[[str, list[int]], Any],
    *,
    limits: Limits,
    out_dir: str | Path,
    method: str = "minipfn",
    bagging: int = 0,
    datasets: list[str] | None = None,
    breakdown: bool = True,
) -> dict:
    """suite의 split을 돌리고 점수를 낸다. 결과는 `out_dir`에 쓰고 요약을 돌려준다.

    split 결과는 `out_dir/runs`에 split마다 저장해 두므로, 같은 `out_dir`로 다시 부르면 끝난
    split은 다시 돌리지 않는다. `datasets`를 주면 그 데이터셋만 돌린다(나머지는 imputed).

    쓰는 파일:
    - `tasks.csv`: split마다 돌렸는지(`skip_reason`, 돌렸는데 실패한 split은 "failed")
    - `results_per_split.csv`: 이 방법의 split 결과 (`metric_error` 등, imputed 포함)
    - `leaderboard.csv`: baseline과 함께 낸 leaderboard (`tabarena` 형식, 모든 방법)
    - `leaderboard_<이름>.csv`: subset별 leaderboard (`breakdown`)
    - `comparison.csv`: 공개 leaderboard(같은 split)와 이 실행의 점수를 방법마다 나란히
    - `summary.json`: 이 방법의 점수, imputed 비율, 승률, 설정
    - `compare.log`: `tabarena` 집계의 출력
    """
    from minipfn.benchmarks import _arena_system as system

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    context = make_context(suite)
    grid = task_grid(context, suite, limits)
    if datasets is not None:
        unknown = sorted(set(datasets) - set(grid["dataset"]))
        if unknown:
            raise ValueError(f"{suite.name}에 없는 데이터셋: {unknown}")
        grid.loc[~grid["dataset"].isin(datasets), "skip_reason"] = "not selected"
    grid.to_csv(out_dir / "tasks.csv", index=False)
    todo = grid[grid["skip_reason"].isna()]
    print(
        f"{suite.name}: splits {len(grid)} ({grid['dataset'].nunique()} datasets), "
        f"run {len(todo)}, imputed {len(grid) - len(todo)}",
        flush=True,
    )

    results = []
    if len(todo):
        key = system.register_factory(make_estimator)
        experiments = system.experiments(suite, method, key, bagging)
        jobs = context.build_jobs(
            experiments,
            subset=list(suite.subset),
            dataset_names=sorted(todo["dataset"].unique()),
        )
        wanted = set(zip(todo["dataset"], todo["fold"], todo["repeat"], strict=True))
        jobs = [j for j in jobs if _job_split(j) in wanted]
        # split 하나가 실패해도 멈추지 않는다: 그 split은 결과가 없어 imputed가 된다(`failed`)
        results = context.run_jobs(
            jobs,
            expname=str(out_dir / "runs"),
            register=False,
            debug_mode=True,
            raise_on_failure=False,
        )
    if results:
        context.register(results, scope_to_valid_tasks=False)
    pool = model_pool(context, method)

    log = out_dir / "compare.log"
    leaderboard, per_split = _quiet(
        context.compare,
        log,
        output_dir=None,
        subset=list(suite.subset),
        ta_results=pool,
        return_results=True,
    )
    name = _new_method_name(context, leaderboard, method)
    fillna = context.fillna_method
    leaderboard.to_csv(out_dir / "leaderboard.csv", index=False)
    ours = with_imputed(per_split, name, fillna)
    ours.drop(columns=["method_metadata"], errors="ignore").to_csv(
        out_dir / "results_per_split.csv", index=False
    )
    # 돌렸는데 결과가 없는 split (results의 fold는 split 번호다)
    filled = ours[ours["imputed"]]
    imputed = set(zip(filled["dataset"], filled["fold"], strict=True))
    failed = grid["skip_reason"].isna() & [
        (d, s) in imputed for d, s in zip(grid["dataset"], grid["split"], strict=True)
    ]
    grid.loc[failed, "skip_reason"] = "failed"
    grid.to_csv(out_dir / "tasks.csv", index=False)
    if failed.any():
        print(f"실패한 split {int(failed.sum())}개 (imputed): tasks.csv", file=sys.stderr)
    website = context.leaderboard_to_website_format(leaderboard=leaderboard)
    public = load_public_leaderboard(suite.public_csv)
    comparison = compare_with_public(website, public, name)
    comparison.to_csv(out_dir / "comparison.csv", index=False)

    summary = {
        "suite": suite.name,
        "method": name,
        "bagging": bagging,
        "limits": _limits_dict(limits),
        "splits": len(grid),
        "datasets": int(grid["dataset"].nunique()),
        "run_splits": len(todo),
        "failed_splits": int(failed.sum()),
        "imputed_pct": 100.0 * float(ours["imputed"].mean()),
        "skip_reasons": _reason_counts(grid["skip_reason"]),
        **_scores(leaderboard, name),
        "winrate": winrates(per_split, ours),
        "breakdown": {},
    }
    if breakdown:
        for label, (subset, public_csv) in suite.breakdown.items():
            part = _quiet(_breakdown, log, context, pool, subset, name, public_csv)
            if part is None:
                continue
            board, scores = part
            board.to_csv(out_dir / f"leaderboard_{label}.csv", index=False)
            summary["breakdown"][label] = scores
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default)
    )
    return summary


def _reason_counts(reasons: pd.Series) -> dict:
    """split 수를 이유의 종류로 센다 ("train rows 11640 > 10000" → "train rows > 10000")."""
    kinds = reasons.fillna("run").str.replace(r" \d+ > ", " > ", regex=True)
    return kinds.value_counts().to_dict()


def _limits_dict(limits: Limits) -> dict:
    if isinstance(limits, ModelLimits):
        return asdict(limits)
    return {k: asdict(v) for k, v in limits.items()}


def _quiet(fn, log: Path, *args, **kwargs):
    # tabarena의 집계는 leaderboard 전체를 stdout에 찍는다. 로그 파일로 돌린다
    import contextlib

    with open(log, "a") as f, contextlib.redirect_stdout(f):
        return fn(*args, **kwargs)


def model_pool(context, method: str) -> pd.DataFrame:
    """비교할 방법들의 split 결과: 공개 leaderboard의 기본 표처럼 model만 (AutoGluon 같은 system은
    뺀다) + 이 실행의 방법(`method`로 시작하는 이름)."""
    from tabarena.evaluation.entrants import DEFAULT_ENTRANT_POOL, filter_results_to_pool

    results = context.load_results()
    ours = results["method"].astype(str).str.startswith(method)
    pool = filter_results_to_pool(
        results, DEFAULT_ENTRANT_POOL, context.method_metadata_collection.info()
    )
    return pd.concat([pool[~pool.index.isin(results.index[ours])], results[ours]])


def _job_split(job) -> tuple[str, int, int]:
    task = job.task if hasattr(job, "task") else job
    return (task.dataset, int(task.fold), int(task.repeat))


def _new_method_name(context, leaderboard: pd.DataFrame, method: str) -> str:
    # 등록한 방법의 leaderboard 이름 (system 설정 이름이 붙는다)
    names = [m for m in leaderboard["method"] if str(m).startswith(method)]
    if len(names) != 1:
        raise RuntimeError(f"leaderboard에서 {method}를 찾지 못했다: {names}")
    return names[0]


def _scores(leaderboard: pd.DataFrame, name: str) -> dict:
    """leaderboard에서 이 방법의 점수. 공개 leaderboard와 같은 단위로 바꾼다: score는
    1 − 정규화 오차, improvability와 imputed는 %, harmonic rank는 1 / mrr."""
    row = leaderboard[leaderboard["method"] == name].iloc[0]
    return {
        "elo": float(row["elo"]),
        "elo_ci_plus": float(row["elo+"]),
        "elo_ci_minus": float(row["elo-"]),
        "score": 1.0 - float(row["normalized-error"]),
        "improvability": 100.0 * float(row["improvability"]),
        "rank": float(row["rank"]),
        "harmonic_rank": 1.0 / float(row["mrr"]),
        "imputed_pct_leaderboard": 100.0 * float(row["imputed"]),
        "n_methods": int(len(leaderboard)),
        "position": int((leaderboard["elo"] > row["elo"]).sum()) + 1,
    }


def with_imputed(per_split: pd.DataFrame, name: str, fillna: str) -> pd.DataFrame:
    """이 방법의 split 결과에 돌리지 않은 split을 `fillna` 방법(RandomForest (default))의 결과로
    채운다(`imputed=True`). leaderboard가 내부에서 채우는 것과 같다. `per_split`의 split 전체가
    범위다."""
    ours = per_split[per_split["method"] == name].copy()
    ours["imputed"] = False
    filler = per_split[per_split["method"] == fillna]
    done = set(zip(ours["dataset"], ours["fold"], strict=True))
    missing = filler[[k not in done for k in zip(filler["dataset"], filler["fold"], strict=True)]]
    missing = missing.assign(method=name, imputed=True, impute_method=fillna)
    out = pd.concat([ours, missing], ignore_index=True)
    return out.sort_values(["dataset", "fold"]).reset_index(drop=True)


def winrates(per_split: pd.DataFrame, ours: pd.DataFrame) -> dict:
    """`WINRATE_BASELINES`마다 split별로 오차가 작으면 1, 같으면 0.5로 센 승률. `ours`는
    `with_imputed`의 결과다. `all`은 imputed split까지(leaderboard와 같은 범위), `run`은 직접 돌린
    split만이다."""
    ours = ours.set_index(["dataset", "fold"])
    out = {}
    keys = per_split["method"].map(_method_key)
    for baseline, internal in WINRATE_BASELINES.items():
        other = per_split[keys == _method_key(internal)]
        if not len(other):
            continue
        other = other.set_index(["dataset", "fold"])["metric_error"]
        both = ours.join(other.rename("other"), how="inner")
        win = np.where(
            both["metric_error"] < both["other"],
            1.0,
            np.where(both["metric_error"] == both["other"], 0.5, 0.0),
        )
        imputed = both["imputed"].astype(bool).to_numpy()
        out[baseline] = {
            "all": float(win.mean()) if len(win) else math.nan,
            "run": float(win[~imputed].mean()) if (~imputed).any() else math.nan,
            "splits": int(len(win)),
            "run_splits": int((~imputed).sum()),
        }
    return out


def _breakdown(context, pool, subset: tuple[str, ...], name: str, public_csv: str | None):
    try:
        board, per_split = context.compare(
            output_dir=None, subset=list(subset), ta_results=pool, return_results=True
        )
    except Exception as e:  # subset에 데이터셋이 너무 적으면 집계가 실패할 수 있다
        print(f"subset {subset}: 집계 실패 ({e})", file=sys.stderr, flush=True)
        return None
    if name not in set(board["method"]):
        return None
    ours = with_imputed(per_split, name, context.fillna_method)
    scores = {
        **_scores(board, name),
        "datasets": int(per_split["dataset"].nunique()),
        "splits": int(len(ours)),
        "imputed_pct": 100.0 * float(ours["imputed"].mean()),
        "winrate": winrates(per_split, ours),
    }
    if public_csv:
        website = context.leaderboard_to_website_format(leaderboard=board)
        try:
            public = load_public_leaderboard(public_csv)
        except Exception as e:  # 공개 CSV가 없는 조합
            print(f"공개 leaderboard {public_csv}: {e}", file=sys.stderr, flush=True)
        else:
            board = compare_with_public(website, public, name)
    return board, scores


def load_public_leaderboard(path: str) -> pd.DataFrame:
    """공개 leaderboard CSV (Hugging Face Space `TabArena/leaderboard`의 `path`)."""
    url = PUBLIC_LEADERBOARD + urllib.parse.quote(path)
    df = pd.read_csv(url)
    df["name"] = df["Model"].map(display_name)
    return df


def display_name(model: str) -> str:
    """leaderboard의 Model 칸(`[이름 (default)](링크) [x% IMPUTED]`)에서 이름만."""
    model = str(model)
    if model.startswith("[") and "](" in model:
        model = model[1 : model.index("](")]
    return model.split(" [")[0].strip()


def compare_with_public(website: pd.DataFrame, public: pd.DataFrame, name: str) -> pd.DataFrame:
    """방법마다 공개 leaderboard의 Elo·점수와 이 실행에서 다시 낸 값을 나란히 둔다.
    이 실행의 방법(`name`)은 공개 쪽 값이 비어 있다. Elo 순."""
    cols = {
        "Elo [⬆️]": "elo",
        "Elo 95% CI": "elo_ci",
        "Score [⬆️]": "score",
        "Improvability (%) [⬇️]": "improvability",
        "Imputed (%) [⬇️]": "imputed_pct",
    }
    ours = website.copy()
    ours["name"] = ours["Model"].map(display_name)
    ours.loc[ours["Model"].astype(str).str.contains(name, regex=False), "name"] = name
    ours = ours[["name", *[c for c in cols if c in ours]]].rename(columns=cols)
    theirs = public[["name", *[c for c in cols if c in public]]].rename(columns=cols)
    merged = ours.merge(theirs, on="name", how="outer", suffixes=("", "_public"))
    merged["is_new"] = merged["name"] == name
    return merged.sort_values("elo", ascending=False, na_position="last").reset_index(drop=True)


def _json_default(value: Any):
    if isinstance(value, np.integer | np.floating):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(f"JSON으로 쓸 수 없는 값: {value!r}")
