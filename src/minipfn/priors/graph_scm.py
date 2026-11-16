"""Graph SCM prior (prior 축, `prior_type="graph_scm"`): TabICLv2 graph SCM 생성기에 표 크기 규칙과
결측값을 붙인 prior. 부품(`minipfn.priors.generators.graph_scm`, `transforms`)을 `Pipeline`으로
묶은 것이다. TabPFN v2 recipe(`minipfn.recipes.tabpfn_v2`)가 이 prior를 쓴다(v2 논문의 prior 코드는
공개되지 않아 같은 계열의 공개 prior로 대신한다).

1. 표 크기: batch마다 `sample_table_size`로 학습 행·feature 수를 뽑는다(기본값은 v2 논문 Methods
   "Training details"). GPU에서 묶어 계산하려고 batch 안의 데이터셋은 같은 크기를 쓴다.
2. 데이터셋마다: 클래스 수(`ClassCount`, 분류) → graph SCM → NaN이 생긴 데이터셋 버리기
   (`DropInvalid`) → 검증 행의 클래스가 학습 행에도 나오게 행 순서 바꾸기(`CoverClasses`, 분류).
   버린 데이터셋은 클래스 수부터 다시 뽑는다(최대 `MAX_ATTEMPTS`번).
3. 결측값(v2 논문 "Post-processing", `Missing`): 일부 데이터셋에서 셀마다 같은 확률로 값을
   지운다(MCAR). 비율은 논문에 없어서 여기서 정했다(`missing_dataset_prob`, `max_missing_rate`).

TabICLv2 학습 스크립트에 맞추는 옵션 (기본은 모두 꺼져 예전과 같은 데이터셋):
- `size_rule="train_fraction"`: 표 크기를 행 수(학습 + 검증)와 학습 행 비율로 뽑는다
  (`table_size.sample_split_size`). 검증 행 수가 batch마다 달라진다. `feature_cap_rule="cells"`면
  긴 표의 feature 상한을 TabICLv2 표 대신 셀 상한으로 정한다(TabICLv2에 없음, curriculum stage 4)
- `binary_prob`: 분류 클래스 수를 이 확률로 2로 (`ClassCount`)
- `filter_unpredictable`: 예측할 수 없는 그래프·데이터셋을 다시 뽑는다 (`generators.graph_scm`)

prior 변형 옵션 (issue #92, 기본은 모두 꺼져 예전과 같은 데이터셋):
- `tree_scm_prob`: 데이터셋을 이 확률로 TabICL v1 tree SCM(`generators.tree_scm`, XGBoost 층)으로
  만든다. 나머지는 graph SCM이다. 생성기 선택은 데이터셋마다 `Rng`로 뽑는다(`GeneratorChoice`)
- `graph_scm_corrected`: graph SCM의 TabICL 보정 플래그를 켠다 (`generators.graph_scm`)
- `mixture_root_prob`: graph SCM 데이터셋을 이 확률로 골라 root 노드를 Gaussian mixture(군집)에서
  뽑는다 (`generators.graph_scm`)

    prior = GraphSCMPrior(task="classification", seed=0)
    batch = prior.sample_batch(8)  # x (8, 행, feature), y (8, 행), n_train

TabICL은 numpy·torch의 전역 난수를 쓰고, 같은 시드여도 파이썬 프로세스의 문자열 hash 시드
(`PYTHONHASHSEED`)가 다르면 다른 데이터셋을 만든다. 완전히 같은 데이터가 필요하면 둘 다 고정한다.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch

from minipfn.priors.base import PriorBatch
from minipfn.priors.generators.graph_scm import GraphSCM, check_probability
from minipfn.priors.hyperparameters import ClassCount
from minipfn.priors.pipeline import Mixture, Pipeline, Rng
from minipfn.priors.table_size import (
    MAX_CELLS,
    MAX_FEATURES,
    MAX_TRAIN_ROWS,
    N_EVAL_ROWS,
    SIZE_RULES,
    check_sizes,
    check_split_sizes,
    sample_split_size,
    sample_table_size,
)
from minipfn.priors.transforms import CoverClasses, DropInvalid, Missing

# 분류 데이터셋의 클래스 수 상한 (v2 논문). 분류 모델의 출력 수가 된다(`TrainConfig.n_out`)
MAX_CLASSES = 10
MAX_ATTEMPTS = 100  # 데이터셋 하나를 만들 때 다시 뽑는 최대 횟수 (멈추지 않고 오류를 내도록)


class GraphSCMPrior:
    def __init__(
        self,
        task: Literal["classification", "regression"],
        seed: int = 0,
        *,
        missing_dataset_prob: float = 0.3,
        max_missing_rate: float = 0.3,
        min_train_rows: int = 1,
        max_train_rows: int = MAX_TRAIN_ROWS,
        n_eval_rows: int = N_EVAL_ROWS,
        max_features: int = MAX_FEATURES,
        log_uniform_rows: bool = False,
        max_cells: int = MAX_CELLS,
        feature_scale: int = MAX_FEATURES,
        size_rule: str = "eval_rows",
        min_rows: int = 1024,
        max_rows: int = 1024,
        min_train_fraction: float = 0.3,
        max_train_fraction: float = 0.9,
        feature_cap_rule: str = "rows",
        binary_prob: float = 0.0,
        filter_unpredictable: bool = False,
        tree_scm_prob: float = 0.0,
        graph_scm_corrected: bool = False,
        mixture_root_prob: float = 0.0,
    ) -> None:
        check_probability("tree_scm_prob", tree_scm_prob)
        self._task = task
        if task == "classification":
            min_train_rows = max(min_train_rows, 2)  # 클래스가 2개 이상 학습 행에 나오려면
        _check(
            size_rule=size_rule,
            min_train_rows=min_train_rows,
            max_train_rows=max_train_rows,
            n_eval_rows=n_eval_rows,
            max_features=max_features,
            max_cells=max_cells,
            feature_scale=feature_scale,
            min_rows=min_rows,
            max_rows=max_rows,
            min_train_fraction=min_train_fraction,
            max_train_fraction=max_train_fraction,
            feature_cap_rule=feature_cap_rule,
        )
        # 크기 설정은 batch마다 읽는다. task와 결측값 설정은 부품에 들어가므로 읽기 전용 속성이다
        self.min_train_rows = min_train_rows
        self.max_train_rows = max_train_rows
        self.n_eval_rows = n_eval_rows
        self.max_features = max_features
        self.log_uniform_rows = log_uniform_rows
        self.max_cells = max_cells
        self.feature_scale = feature_scale
        self.size_rule = size_rule
        self.min_rows = min_rows
        self.max_rows = max_rows
        self.min_train_fraction = min_train_fraction
        self.max_train_fraction = max_train_fraction
        self.feature_cap_rule = feature_cap_rule
        self.rng = Rng.from_seed(seed)
        # TabICL의 전역 난수 시드도 여기서 고정한다
        generator = GraphSCM(
            task,
            seed=seed,
            filter_unpredictable=filter_unpredictable,
            corrected=graph_scm_corrected,
            mixture_root_prob=mixture_root_prob,
        )
        classification = task == "classification"
        hyperparameters = ClassCount(MAX_CLASSES, binary_prob) if classification else None
        if tree_scm_prob:
            from minipfn.priors.generators.tree_scm import TreeSCM

            tree = TreeSCM(task, filter_unpredictable=filter_unpredictable)
            generator = Mixture({"graph_scm": generator, "tree_scm": tree}, choose=chosen_generator)
            hyperparameters = GeneratorChoice(hyperparameters, tree_scm_prob)
        # 결측값을 넣기 전까지 (회귀 구간 경계와 `_one_dataset`은 여기까지만 거친다)
        self._accept = [DropInvalid(), CoverClasses()] if classification else [DropInvalid()]
        self._missing = Missing(missing_dataset_prob, max_missing_rate)
        self.pipeline = Pipeline(
            generator,
            rng=self.rng,
            hyperparameters=hyperparameters,
            transforms=[*self._accept, self._missing],
            max_attempts=MAX_ATTEMPTS,
        )

    @property
    def task(self) -> str:
        return self._task

    @property
    def missing_dataset_prob(self) -> float:
        return self._missing.dataset_prob

    @property
    def max_missing_rate(self) -> float:
        return self._missing.max_rate

    def sample_shape(self, rng: np.random.Generator | None = None) -> tuple[int, int, int]:
        """(학습 행 수, feature 수, 검증 행 수) — `sample_batch`의 `size`와 같은 순서. "eval_rows"
        규칙이면 검증 행 수는 늘 `n_eval_rows`다. `rng`를 주면 prior의 난수 대신 그것으로 뽑는다
        (학습 옵션 `balance_tables`: 모든 GPU가 같은 난수로 한 step의 표 크기를 함께 뽑는다)."""
        rng = self.rng.numpy if rng is None else rng
        if self.size_rule == "train_fraction":
            n_train, n_eval, n_features = sample_split_size(
                rng,
                min_rows=self.min_rows,
                max_rows=self.max_rows,
                min_train_fraction=self.min_train_fraction,
                max_train_fraction=self.max_train_fraction,
                # feature 수 U(1, 상한)의 상한. feature_scale이 더 작으면 그것 (TabPFN-2.5 긴 표:
                # 추론의 one-hot 폭 상한은 넓게 두고 학습 표는 TabICLv2처럼 좁게, issue #81)
                max_features=min(self.max_features, self.feature_scale),
                log_uniform_rows=self.log_uniform_rows,
                max_cells=self.max_cells,
                min_train_rows=self.min_train_rows,
                feature_cap_rule=self.feature_cap_rule,
            )
            return n_train, n_features, n_eval
        n_train, n_features = self.sample_size(rng)
        return n_train, n_features, self.n_eval_rows

    def sample_size(self, rng: np.random.Generator | None = None) -> tuple[int, int]:
        """ "eval_rows" 규칙의 (학습 행 수, feature 수). `rng`는 `sample_shape`와 같다."""
        if self.size_rule != "eval_rows":
            raise ValueError("sample_size는 eval_rows 규칙의 것이다 (sample_shape를 쓴다)")
        n_train, n_features = sample_table_size(
            self.rng.numpy if rng is None else rng,
            self.max_train_rows,
            self.n_eval_rows,
            self.max_features,
            min_train_rows=self.min_train_rows,
            log_uniform_rows=self.log_uniform_rows,
            max_cells=self.max_cells,
            feature_scale=self.feature_scale,
        )
        return max(n_train, self.min_train_rows), n_features

    def _one_dataset(
        self, n_train: int, n_features: int, n_eval: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """결측값을 넣기 전의 데이터셋 하나: x (행, feature), y (행)."""
        n_rows = n_train + (self.n_eval_rows if n_eval is None else n_eval)
        group = self.pipeline.sample_group(n_rows, n_train, n_features, transforms=self._accept)
        return group.x[:, 0], group.y[:, 0]

    def sample_batch(self, batch_size: int, size: tuple[int, ...] | None = None) -> PriorBatch:
        """`size`는 (학습 행 수, feature 수) 또는 (학습 행 수, feature 수, 검증 행 수). 주지 않으면
        `sample_shape`로 뽑는다."""
        n_train, n_features, *rest = self.sample_shape() if size is None else size
        n_eval = rest[0] if rest else self.n_eval_rows
        n_train = max(n_train, self.min_train_rows)
        return self.pipeline.sample_batch(batch_size, n_train + n_eval, n_train, n_features)


# 데이터셋 크기 규칙 필드와 기본값 (`GraphSCMPrior`의 크기 인자와 같다). 필드 추가는 여기서만
SIZE_DEFAULTS = {
    "min_train_rows": 1,
    "max_train_rows": MAX_TRAIN_ROWS,
    "n_eval_rows": N_EVAL_ROWS,
    "max_features": MAX_FEATURES,
    "log_uniform_rows": False,
    "max_cells": MAX_CELLS,
    "feature_scale": MAX_FEATURES,
    "size_rule": "eval_rows",
    "min_rows": 1024,
    "max_rows": 1024,
    "min_train_fraction": 0.3,
    "max_train_fraction": 0.9,
    "feature_cap_rule": "rows",
}
SIZE_FIELDS = tuple(SIZE_DEFAULTS)
# 학습 데이터 분포를 바꾸는 prior 옵션
PRIOR_FIELDS = (
    "binary_prob",
    "filter_unpredictable",
    "tree_scm_prob",
    "graph_scm_corrected",
    "mixture_root_prob",
)
# 그 가운데 회귀 target 분포도 바꾸는 것 (회귀 구간 경계가 학습 prior와 같게 따른다. binary_prob는
# 분류만 바꾼다)
BORDER_PRIOR_FIELDS = tuple(name for name in PRIOR_FIELDS if name != "binary_prob")


class GeneratorChoice:
    """하이퍼파라미터 단계: 데이터셋(그룹)마다 생성기를 고른다 (`Mixture`가 읽는 `generator`).
    `tree_scm_prob`의 확률로 "tree_scm", 아니면 "graph_scm"이다. 나머지 하이퍼파라미터는
    `base`(없으면 빈 dict)의 것이다."""

    def __init__(self, base, tree_scm_prob: float) -> None:
        self.base = base
        self.tree_scm_prob = tree_scm_prob

    def __call__(self, rng: Rng, n_train: int, n_features: int) -> dict:
        hp = self.base(rng, n_train, n_features) if self.base else {}
        tree = rng.numpy.random() < self.tree_scm_prob
        return {**hp, "generator": "tree_scm" if tree else "graph_scm"}


def chosen_generator(hp: dict) -> str:
    return hp["generator"]


def _check(*, size_rule: str, **sizes) -> None:
    """크기 규칙과 그 규칙의 설정을 검사한다 (prior와 학습 설정이 같이 쓴다)."""
    if size_rule not in SIZE_RULES:
        raise ValueError(f"size_rule은 {SIZE_RULES} 가운데 하나여야 한다: {size_rule}")
    if size_rule == "train_fraction":
        # feature 수 상한은 min(max_features, feature_scale)이다 (`GraphSCMPrior.sample_shape`)
        if sizes["feature_scale"] < 1:
            raise ValueError(f"feature_scale는 1 이상이어야 한다: {sizes['feature_scale']}")
        check_split_sizes(
            **{
                k: sizes[k]
                for k in (
                    "min_rows",
                    "max_rows",
                    "min_train_fraction",
                    "max_train_fraction",
                    "max_features",
                    "max_cells",
                    "min_train_rows",
                    "feature_cap_rule",
                )
            }
        )
        return
    if sizes.get("feature_cap_rule", "rows") != "rows":
        raise ValueError('feature_cap_rule은 size_rule "train_fraction"에서만 쓴다')
    check_sizes(
        **{
            k: sizes[k]
            for k in (
                "min_train_rows",
                "max_train_rows",
                "n_eval_rows",
                "max_features",
                "max_cells",
                "feature_scale",
            )
        }
    )


def _sizes(config) -> dict:
    # 학습 설정에서 데이터셋 크기 규칙만 (prior와 회귀 구간 경계가 같은 크기 분포를 쓰도록)
    return {name: getattr(config, name) for name in SIZE_FIELDS}


def check_config(config) -> None:
    """학습 설정의 데이터셋 크기가 이 prior로 만들 수 있는지 (`TrainConfig`를 만들 때 부른다)."""
    sizes = {k: v for k, v in _sizes(config).items() if k != "log_uniform_rows"}
    if config.task == "classification":
        sizes["min_train_rows"] = max(sizes["min_train_rows"], 2)
    _check(**sizes)


def make_prior(config, seed: int) -> GraphSCMPrior:
    """학습 설정(`minipfn.training.config.TrainConfig`)의 데이터셋 크기로 prior를 만든다."""
    return GraphSCMPrior(
        config.task,
        seed=seed,
        missing_dataset_prob=config.missing_dataset_prob,
        **_sizes(config),
        **{name: getattr(config, name) for name in PRIOR_FIELDS},
    )


def make_borders(config, n_buckets: int, seed: int, n_jobs: int = 1) -> torch.Tensor:
    """학습 설정과 같은 데이터셋 크기로 뽑은 회귀 구간 경계 (`regression_borders`)."""
    return regression_borders(
        n_buckets,
        config.border_datasets,
        seed,
        n_jobs=n_jobs,
        **{name: getattr(config, name) for name in BORDER_PRIOR_FIELDS},
        **_sizes(config),
    )


def _border_targets(args: tuple[int, int, dict, dict]) -> torch.Tensor:
    # 회귀 데이터셋 n개의 고유한 target 값 (결측값은 넣지 않는다)
    n_datasets, seed, sizes, options = args
    prior = GraphSCMPrior(
        "regression",
        seed=seed,
        missing_dataset_prob=0.0,
        **options,
        **sizes,
    )
    values = []
    for _ in range(n_datasets):
        _, y = prior._one_dataset(*prior.sample_shape())
        values.append(torch.unique(y.float()))
    return torch.cat(values)


def regression_borders(
    n_buckets: int = 5000,
    n_datasets: int = 2000,
    seed: int = 0,
    *,
    n_jobs: int = 1,
    filter_unpredictable: bool = False,
    tree_scm_prob: float = 0.0,
    graph_scm_corrected: bool = False,
    mixture_root_prob: float = 0.0,
    **sizes,
) -> torch.Tensor:
    """회귀 출력 구간의 경계 (Methods "Regression details").

    prior에서 데이터셋을 많이 뽑아 z-정규화한 target을 모으고, 그 분포의 1/n_buckets 분위수들을
    경계로 쓴다. 양 끝은 half-normal 꼬리가 맡는다(`FullSupportBarDistribution`).
    같은 값이 많은 데이터셋(이산화한 target, 4σ에서 잘린 값)이 분위수를 끌어당겨 폭이 0에 가까운
    구간이 생기지 않도록, 데이터셋마다 고유한 값만 모은다(여기서 정함).
    데이터셋 크기 인자는 `SIZE_FIELDS` 이름의 키워드로 주고, 빠지면 `SIZE_DEFAULTS`를 쓴다.
    `GraphSCMPrior`와 같은 뜻이며, 학습에서는 `make_borders`가 학습 prior와 같은 값을 넘긴다.
    `filter_unpredictable`과 prior 변형 옵션(`BORDER_PRIOR_FIELDS`)도 학습 prior와 같게 둔다(걸러진
    데이터셋의 target은 학습에 안 나오고, tree SCM·군집 root는 target 분포를 바꾼다).
    """
    unknown = set(sizes) - set(SIZE_DEFAULTS)
    if unknown:
        raise TypeError(f"regression_borders에 없는 크기 인자: {sorted(unknown)}")
    # n_jobs개 프로세스로 나눠 만든다 (프로세스마다 시드가 다르다)
    counts = [n_datasets // n_jobs + (i < n_datasets % n_jobs) for i in range(n_jobs)]
    table = {**SIZE_DEFAULTS, **sizes}
    options = {
        "filter_unpredictable": filter_unpredictable,
        "tree_scm_prob": tree_scm_prob,
        "graph_scm_corrected": graph_scm_corrected,
        "mixture_root_prob": mixture_root_prob,
    }
    jobs = [(n, seed + i, table, options) for i, n in enumerate(counts)]
    if n_jobs == 1:
        targets = _border_targets(jobs[0])
    else:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(n_jobs) as pool:
            targets = torch.cat(list(pool.map(_border_targets, jobs)))
    targets = targets[torch.randperm(len(targets))[: 2**24]]  # torch.quantile의 입력 크기 한도
    borders = torch.quantile(targets, torch.linspace(0, 1, n_buckets + 1))
    # 같은 값이 겹치면 구간 폭이 0이 되므로 아주 조금씩 벌린다 (경계 수는 그대로)
    for i in range(1, len(borders)):
        borders[i] = torch.maximum(borders[i], borders[i - 1] + 1e-6)
    return borders
