"""TabArena·BeyondArena 평가의 공통 부분: sklearn 형식 estimator를 공식 split으로 돌리고 공개
leaderboard와 같은 방식으로 점수를 낸다.

데이터, split, 지표, 집계는 TabArena 공식 패키지 `tabarena`(extra `arena`:
`uv sync --no-dev --extra arena --group arena-dev`. pandas 2.3이 필요해 dev group과 따로 설치한다)를
쓴다. 직접 만드는 것은 estimator를 그 패키지의 system
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
import pickle
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


# temporal split: 시간 열을 이만큼의 연속 구간으로 나눠 group으로 쓴다. greedy 선택의 기본 fold 수
# (`ensemble_selection.N_FOLDS`)와 같은 값이다. 빈 시각은 따로 한 group(-1)이 되어, fold를 나눌 때
# 다른 구간과 한 fold에 묶일 수 있다
TIME_INTERVALS = 8


def _time_values(t: pd.Series) -> pd.Series:
    """시간 열을 순서를 지키는 수로 (빈 값·읽을 수 없는 값은 NaN). 순서 있는 범주는 범주 순서,
    날짜·날짜 문자열은 시각, 그 밖은 수로 읽는다."""
    import warnings

    if isinstance(t.dtype, pd.CategoricalDtype) and t.cat.ordered:
        return t.cat.codes.where(t.cat.codes >= 0).astype(float)
    if pd.api.types.is_bool_dtype(t) or pd.api.types.is_numeric_dtype(t):
        return pd.to_numeric(t, errors="coerce").astype(float)
    if not pd.api.types.is_datetime64_any_dtype(t):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # 날짜 형식 추론 경고
            t = pd.to_datetime(t.astype(object), errors="coerce", utc=True)
    if getattr(t.dt, "tz", None) is not None:
        t = t.dt.tz_convert(None)
    return (t - pd.Timestamp(0)) / pd.Timedelta(seconds=1)


def split_groups(X: pd.DataFrame, metadata, n_intervals: int = TIME_INTERVALS) -> np.ndarray | None:
    """task의 split 정보로 행마다의 group 번호를 만든다. grouped split은 group 열(여러 열이면 값의
    조합), temporal split은 시간 열을 행 수가 비슷한 연속 구간으로 나눈 번호(같은 시각은 같은
    구간, 빈 시각은 -1)다. 둘 다 없으면(무작위 split) None. AutoGluon이 TabArena에서 bagging fold를
    만드는 방식(`resolve_validation_splits`)과 같은 열을 쓴다. 열이 없거나 시간 열에서 구간을 2개
    이상 만들 수 없으면 경고를 내고 None (무작위 fold)."""
    group_on = getattr(metadata, "group_on", None)
    time_on = getattr(metadata, "time_on", None)
    named = [*(group_on if isinstance(group_on, list | tuple) else [group_on]), time_on]
    missing = [c for c in named if c is not None and c not in X.columns]
    if missing:  # 학습 행에 그 열이 없으면 무작위 fold (split을 실패시키지 않는다)
        print(
            f"split_groups: 학습 행에 없는 split 열 {missing}, 무작위 fold를 쓴다", file=sys.stderr
        )
        return None
    if group_on is not None:
        columns = list(group_on) if isinstance(group_on, list | tuple) else [group_on]
        codes = np.stack([pd.factorize(X[c], use_na_sentinel=False)[0] for c in columns], axis=1)
        return np.unique(codes, axis=0, return_inverse=True)[1].reshape(-1)
    if time_on is not None:
        ranks = _time_values(X[time_on]).rank(method="min")
        codes = pd.qcut(ranks, q=n_intervals, labels=False, duplicates="drop")
        if codes.nunique() < 2:
            print(
                f"split_groups: 시간 열 {time_on!r}({X[time_on].dtype})에서 구간을 2개 이상 만들 수"
                " 없어 무작위 fold를 쓴다",
                file=sys.stderr,
            )
            return None
        return codes.fillna(-1).astype(int).to_numpy()
    return None


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
            "TabArena·BeyondArena 평가에는 tabarena 패키지가 필요하다: "
            "uv sync --no-dev --extra arena --group arena-dev"
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
    fingerprint: str = "",
    results: list | None = None,
    results_path: str | Path | None = None,
) -> dict:
    """suite의 split을 돌리고 점수를 낸다. 결과는 `out_dir`에 쓰고 요약을 돌려준다.

    split 결과는 `out_dir/runs/<id>`에 split마다 저장해 두고, 같은 `out_dir`로 다시 부르면 끝난
    split은 다시 돌리지 않는다. `<id>`는 `fingerprint`(estimator를 정하는 것: 체크포인트, 추론
    설정 등. 부르는 쪽이 준다)와 `method`·`bagging`으로 만든다. 그래서 다른 체크포인트의 결과를
    다시 쓰지 않는다. `datasets`를 주면 그 데이터셋만 돌린다(나머지는 imputed). 돌린 split이
    없거나 모두 실패하면 점수는 비우고(None) 나머지 파일은 그대로 쓴다.

    쓰는 파일:
    - `tasks.csv`: split마다 돌렸는지(`skip_reason`, 돌렸는데 실패한 split은 "failed")
    - `results_per_split.csv`: 이 방법의 split 결과 (`metric_error` 등, imputed 포함)
    - `leaderboard.csv`: baseline과 함께 낸 leaderboard (`tabarena` 형식, 모든 방법)
    - `leaderboard_<이름>.csv`: subset별 leaderboard (`breakdown`)
    - `comparison.csv`: 공개 leaderboard(같은 split)와 이 실행의 점수를 방법마다 나란히
    - `summary.json`: 이 방법의 점수, imputed 비율, 승률, 직접 돌린 split의 학습·추론 시간,
      설정 (공개 CSV를 못 받으면
      `comparison.csv`의 공개 쪽 열은 비어 있다)
    - `compare.log`: `tabarena` 집계의 출력

    여러 GPU에 나눠 돌릴 때(`shard_datasets`, 평가 스크립트의 `--shards`):
    - `results_path`를 주면 split을 돌린 뒤 그 결과(`run_jobs`가 낸 dict 목록)를 그 파일에 pickle로
      남기고 점수는 내지 않는다(빈 dict를 돌려준다). 데이터셋을 나눠 맡은 worker가 쓴다
    - `results`를 주면 split을 돌리지 않고 그 결과로 점수만 낸다. worker들의 결과를 모은 것이다.
      `run_jobs`는 캐시가 있어도 task 데이터를 먼저 받으므로(materialize) 캐시를 다시 읽는 대신
      결과를 넘긴다
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

    run_id = _run_id(fingerprint, method, bagging)
    run_dir = out_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "fingerprint.txt").write_text(f"{method}\nbagging={bagging}\n{fingerprint}\n")

    if results is not None:
        results = list(results)
    elif len(todo):
        key = system.register_factory(make_estimator)
        experiments = system.experiments(suite, method, key, bagging)
        jobs = context.build_jobs(
            experiments,
            subset=list(suite.subset),
            dataset_names=sorted(todo["dataset"].unique()),
        )
        wanted = set(zip(todo["dataset"], todo["fold"], todo["repeat"], strict=True))
        jobs = [j for j in jobs if _job_split(j) in wanted]
        prefetch_tasks(context, jobs, what=suite.name)
        # split 하나가 실패해도 멈추지 않는다: 그 split은 결과가 없어 imputed가 된다(`failed`)
        results = context.run_jobs(
            jobs,
            expname=str(run_dir),
            register=False,
            debug_mode=True,
            raise_on_failure=False,
        )
    else:
        results = []
    if results_path is not None:
        with open(results_path, "wb") as f:
            pickle.dump(results, f)
        return {}
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
    name = _new_method_name(leaderboard, method)
    if name is None:
        print(f"{method}: 결과가 있는 split이 없어 점수를 내지 않는다", file=sys.stderr)
    fillna = context.fillna_method
    leaderboard.to_csv(out_dir / "leaderboard.csv", index=False)
    ours = with_imputed(per_split, name or method, fillna)
    ours.drop(columns=["method_metadata"], errors="ignore").to_csv(
        out_dir / "results_per_split.csv", index=False
    )
    # 돌렸는데 결과가 없는 split (results의 fold는 split 번호다)
    filled = ours[ours["imputed"]]
    imputed = set(zip(filled["dataset"], filled["fold"], strict=True))
    was_imputed = pd.Series(
        [(d, s) in imputed for d, s in zip(grid["dataset"], grid["split"], strict=True)],
        index=grid.index,
    )
    failed = grid["skip_reason"].isna() & was_imputed
    grid.loc[failed, "skip_reason"] = "failed"
    grid.to_csv(out_dir / "tasks.csv", index=False)
    if failed.any():
        print(f"실패한 split {int(failed.sum())}개 (imputed): tasks.csv", file=sys.stderr)
    website = context.leaderboard_to_website_format(leaderboard=leaderboard)
    comparison = compare_with_public(website, try_public_leaderboard(suite.public_csv), name)
    comparison.to_csv(out_dir / "comparison.csv", index=False)

    summary = {
        "suite": suite.name,
        "method": name or method,
        "run_id": run_id,
        "bagging": bagging,
        "limits": _limits_dict(limits),
        "splits": len(grid),
        "datasets": int(grid["dataset"].nunique()),
        "run_splits": len(todo),
        "failed_splits": int(failed.sum()),
        "imputed_pct": 100.0 * float(ours["imputed"].mean()),
        "skip_reasons": _reason_counts(grid["skip_reason"]),
        **(_scores(leaderboard, name) if name else dict.fromkeys(SCORE_KEYS)),
        "winrate": winrates(per_split, ours),
        "time": run_times(ours),
        "breakdown": {},
    }
    if breakdown and name:
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


def shard_datasets(grid: pd.DataFrame, n_shards: int) -> list[list[str]]:
    """돌릴 split(`skip_reason`이 없는 것)이 있는 데이터셋을 `n_shards`개 묶음으로. 데이터셋 하나의
    비용은 그 split들의 셀 수(학습 행 × feature) 합이고, 비용이 큰 데이터셋부터 지금까지 비용이 가장
    적은 묶음에 넣는다(LPT, 같으면 번호가 작은 묶음). 같은 grid면 늘 같은 답이다. 빈 묶음은 뺀다."""
    todo = grid[grid["skip_reason"].isna()]
    cells = todo["max_train_rows"].astype(float) * todo["n_features"].clip(lower=1).astype(float)
    cost = cells.groupby(todo["dataset"]).sum()
    order = sorted(cost.index, key=lambda d: (-cost[d], d))
    shards: list[list[str]] = [[] for _ in range(n_shards)]
    loads = [0.0] * n_shards
    for dataset in order:
        i = min(range(n_shards), key=lambda j: (loads[j], j))
        shards[i].append(dataset)
        loads[i] += cost[dataset]
    return [sorted(s) for s in shards if s]


def _run_id(fingerprint: str, method: str, bagging: int) -> str:
    import hashlib

    text = json.dumps([method, bagging, fingerprint])
    return hashlib.sha1(text.encode()).hexdigest()[:12]


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


def prefetch_tasks(context, jobs: list, *, what: str) -> None:
    """split을 돌리기 전에 `jobs`의 task 데이터를 받아 두고, 네트워크 오류면 task마다 다시 한다
    (`benchmarks._retry`).

    `run_jobs`는 split을 돌리는 도중에 task마다 데이터를 받는데(OpenML task), 그 단계의 오류는
    `raise_on_failure=False`가 잡지 않아 평가 전체가 멈춘다. 미리 받아 두면 `run_jobs`는 캐시만
    읽는다. `make_context`는 `cache_config`를 주지 않으므로 둘 다 openml 기본 캐시를 쓴다.
    - OpenML task(TabArena): task·split·dataset을 task마다 받는다(`fetch_openml_task`).
    - Data Foundry task(BeyondArena, id가 정수가 아니다): `materialize`가 받는다. 받은 데이터셋은
      건너뛰므로 다시 불러도 남은 것만 받는다."""
    from minipfn.benchmarks._retry import fetch_openml_task, retry

    collection = context.task_metadata_collection.subset_to_jobs(jobs)
    task_ids: list[int] = []
    for ttm in collection:
        try:
            task_ids.append(int(ttm.task_id_str))
        except (TypeError, ValueError):
            continue
    for task_id in dict.fromkeys(task_ids):
        fetch_openml_task(task_id)
    retry(collection.materialize, what=f"{what} materialize")


def _new_method_name(leaderboard: pd.DataFrame, method: str) -> str | None:
    # 등록한 방법의 leaderboard 이름 (system 설정 이름이 붙는다). 결과가 없으면 None
    names = [m for m in leaderboard["method"] if str(m).startswith(method)]
    if len(names) > 1:
        raise RuntimeError(f"leaderboard에 {method}로 시작하는 방법이 여럿이다: {names}")
    return names[0] if names else None


SCORE_KEYS = (
    "elo",
    "elo_ci_plus",
    "elo_ci_minus",
    "score",
    "improvability",
    "rank",
    "harmonic_rank",
    "imputed_pct_leaderboard",
    "n_methods",
    "position",
)


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


TIME_COLUMNS = ("time_train_s", "time_infer_s")


def run_times(ours: pd.DataFrame) -> dict:
    """직접 돌린 split(imputed가 아닌 것)의 학습·추론 시간(초)의 평균과 합. `ours`는
    `with_imputed`의 결과다. 시간 열이 없으면 빈 dict."""
    run = ours[~ours["imputed"].astype(bool)]
    out = {}
    for col in TIME_COLUMNS:
        if col in run and len(run):
            out[f"{col}_mean"] = float(run[col].mean())
            out[f"{col}_total"] = float(run[col].sum())
    return out


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
        "time": run_times(ours),
    }
    if public_csv:
        website = context.leaderboard_to_website_format(leaderboard=board)
        board = compare_with_public(website, try_public_leaderboard(public_csv), name)
    return board, scores


def load_public_leaderboard(path: str) -> pd.DataFrame:
    """공개 leaderboard CSV (Hugging Face Space `TabArena/leaderboard`의 `path`)."""
    url = PUBLIC_LEADERBOARD + urllib.parse.quote(path)
    df = pd.read_csv(url)
    df["name"] = df["Model"].map(display_name)
    return df


def try_public_leaderboard(path: str) -> pd.DataFrame | None:
    """`load_public_leaderboard`, 받지 못하면(네트워크, 없는 조합) None. 몇 시간 돌린 결과를
    공개 CSV 하나 때문에 잃지 않도록."""
    try:
        return load_public_leaderboard(path)
    except Exception as e:
        print(f"공개 leaderboard {path}를 받지 못했다: {e}", file=sys.stderr, flush=True)
        return None


def display_name(model: str) -> str:
    """leaderboard의 Model 칸(`[이름 (default)](링크) [x% IMPUTED]`)에서 이름만."""
    model = str(model)
    if model.startswith("[") and "](" in model:
        model = model[1 : model.index("](")]
    return model.split(" [")[0].strip()


def compare_with_public(
    website: pd.DataFrame, public: pd.DataFrame | None, name: str | None
) -> pd.DataFrame:
    """방법마다 공개 leaderboard의 Elo·점수와 이 실행에서 다시 낸 값을 나란히 둔다.
    이 실행의 방법(`name`)은 공개 쪽 값이 비어 있다. `public`이 None이면 공개 쪽 열이 모두
    비어 있다. Elo 순."""
    cols = {
        "Elo [⬆️]": "elo",
        "Elo 95% CI": "elo_ci",
        "Score [⬆️]": "score",
        "Improvability (%) [⬇️]": "improvability",
        "Imputed (%) [⬇️]": "imputed_pct",
    }
    ours = website.copy()
    ours["name"] = ours["Model"].map(display_name)
    if name is not None:
        ours.loc[ours["Model"].astype(str).str.contains(name, regex=False), "name"] = name
    ours = ours[["name", *[c for c in cols if c in ours]]].rename(columns=cols)
    if public is None:
        public = pd.DataFrame({"name": pd.Series(dtype=str), **{c: [] for c in cols}})
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
