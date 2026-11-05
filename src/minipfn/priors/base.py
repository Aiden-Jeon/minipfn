"""prior가 학습 루프에 내는 batch와, 학습 설정이 기본으로 쓰는 데이터셋 크기."""

from __future__ import annotations

from dataclasses import dataclass

import torch

# 데이터셋 크기 기본값 (TabPFN v2 논문 Methods "Training details"). `TrainConfig`의 기본값이고,
# prior는 이 크기 설정을 학습 설정에서 읽는다
MAX_TRAIN_ROWS = 2048  # 학습 행 수 상한
N_EVAL_ROWS = 128  # 정답을 가린 검증 행 수
MAX_FEATURES = 160  # feature 수 상한
MAX_CELLS = 75_000  # 표 하나의 셀 수 상한 (검증 행 포함)


def check_sizes(
    *,
    min_train_rows: int,
    max_train_rows: int,
    n_eval_rows: int,
    max_features: int,
    max_cells: int,
    feature_scale: int,
) -> None:
    """데이터셋 크기 설정이 말이 되는지. 틀리면 batch를 만들 때(DataLoader worker 안에서) 터지기
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


@dataclass
class PriorBatch:
    x: torch.Tensor  # (B, R, F), 결측은 NaN
    y: torch.Tensor  # (B, R). 분류: 클래스 번호, 회귀: z-정규화한 값
    n_train: int  # 앞의 n_train행이 학습 행, 나머지가 정답을 가린 행

    @property
    def y_train(self) -> torch.Tensor:
        return self.y[:, : self.n_train]

    @property
    def y_eval(self) -> torch.Tensor:
        return self.y[:, self.n_train :]
