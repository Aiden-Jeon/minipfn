"""prior가 학습 루프에 내는 batch."""

from __future__ import annotations

from dataclasses import dataclass

import torch


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
