"""표 크기 규칙: batch마다 학습 행 수와 feature 수를 뽑는다.

학습 행 수는 상한까지 균등(또는 log 균등), feature 수는 Beta 분포를 [1, feature_scale]로 편 값이고,
표 하나의 셀 수(검증 행 포함)에 상한을 둔다. 기본값은 TabPFN v2 논문 Methods "Training details"의
수치다(`minipfn.training.config.TrainConfig`의 기본값도 이것이다).
"""

from __future__ import annotations

import numpy as np

MAX_TRAIN_ROWS = 2048  # 학습 행 수 상한
N_EVAL_ROWS = 128  # 정답을 가린 검증 행 수
MAX_FEATURES = 160  # feature 수 상한
MAX_CELLS = 75_000  # 표 하나의 셀 수 상한 (검증 행 포함)
FEATURE_BETA = (0.95, 8.0)  # feature 수 분포 Beta(a, b)


def check_sizes(
    *,
    min_train_rows: int,
    max_train_rows: int,
    n_eval_rows: int,
    max_features: int,
    max_cells: int,
    feature_scale: int,
) -> None:
    """표 크기 설정이 말이 되는지. 틀리면 batch를 만들 때(DataLoader worker 안에서) 터지기
    전에 ValueError를 낸다."""
    if min_train_rows < 1:
        raise ValueError(f"min_train_rows는 1 이상이어야 한다: {min_train_rows}")
    if max_train_rows < min_train_rows:
        raise ValueError(
            f"max_train_rows {max_train_rows}가 min_train_rows {min_train_rows}보다 작다"
        )
    for name, value in (
        ("n_eval_rows", n_eval_rows),
        ("max_features", max_features),
        ("feature_scale", feature_scale),
    ):
        if value < 1:
            raise ValueError(f"{name}는 1 이상이어야 한다: {value}")
    # 가장 작은 표(학습 행 하한 + 검증 행, feature 1개)도 셀 상한 안에 들어야 한다
    if min_train_rows + n_eval_rows > max_cells - 1:
        raise ValueError(
            f"min_train_rows {min_train_rows} + n_eval_rows {n_eval_rows}가 max_cells {max_cells}"
            " 안에 들지 않는다 (feature 1개짜리 표도 만들 수 없다)"
        )


def sample_table_size(
    rng: np.random.Generator,
    max_train_rows: int = MAX_TRAIN_ROWS,
    n_eval_rows: int = N_EVAL_ROWS,
    max_features: int = MAX_FEATURES,
    *,
    min_train_rows: int = 1,
    log_uniform_rows: bool = False,
    max_cells: int = MAX_CELLS,
    feature_scale: int = MAX_FEATURES,
    feature_beta: tuple[float, float] = FEATURE_BETA,
) -> tuple[int, int]:
    """(학습 행 수, feature 수). 셀 수 제한은 실제로 쓰는 feature 수와 검증 행까지 포함해 적용한다.

    기본값에서는 feature 수를 자를 일이 없다. 작은 설정에서 feature 상한을 낮추면
    잘린 feature 수로 셀 수를 센다. `log_uniform_rows`면 학습 행 수를 [min, max]에서 log 균등으로
    뽑는다(v2 논문에 없음. 긴 표로 이어 학습할 때 쓴다, TabICL의 2단계 학습처럼).
    """
    if log_uniform_rows:
        log_n = rng.uniform(np.log(min_train_rows), np.log(max_train_rows))
        n_train = int(round(np.exp(log_n)))
    else:  # 하한은 prior가 나중에 맞춘다 (같은 시드면 예전과 같은 크기를 뽑도록)
        n_train = int(rng.integers(1, max_train_rows + 1))
    # feature 수는 Beta(0.95, 8)을 [1, feature_scale]로 편 것 (v2 논문은 160). feature_scale은 v2
    # 논문에 없음: TabPFN-2.5처럼 feature가 많은 표로 학습할 때 분포 자체를 넓힌다
    n_features = 1 + int(round(rng.beta(*feature_beta) * (feature_scale - 1)))
    # 학습 행 하한(min_train_rows)을 채워도 셀 제한을 넘지 않도록 feature 수를 줄인다
    # (기본값에서는 상한이 581개라 걸리지 않는다)
    n_features = min(n_features, max_features, (max_cells - 1) // (min_train_rows + n_eval_rows))
    max_train = max(1, (max_cells - 1) // n_features - n_eval_rows)
    return min(n_train, max_train), n_features
