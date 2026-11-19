"""prior가 학습 루프(`minipfn.training.loop`)에 내는 batch와, prior·손실이 같이 쓰는 상수."""

from __future__ import annotations

from dataclasses import dataclass

import torch

IGNORE_INDEX = -100  # 손실에서 무시할 target (cross-entropy의 ignore_index)


@dataclass
class PriorBatch:
    x: torch.Tensor  # (B, R, F), 결측은 NaN
    y: torch.Tensor  # (B, R). 분류: 클래스 번호, 회귀: z-정규화한 값
    n_train: int  # 앞의 n_train행이 학습 행, 나머지가 정답을 가린 행
    # 범주 열 표시 (B, F) bool. 범주 열 표시를 내는 prior(graph SCM의 `categorical_mask`, issue
    # #110)만 채우고, 그 밖에는 None이다
    categorical: torch.Tensor | None = None

    @property
    def y_train(self) -> torch.Tensor:
        return self.y[:, : self.n_train]

    @property
    def y_eval(self) -> torch.Tensor:
        return self.y[:, self.n_train :]
