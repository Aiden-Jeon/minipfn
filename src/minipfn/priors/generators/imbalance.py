"""이진 분류의 클래스 불균형 (issue #103 P2): 생성기를 감싸 소수 클래스 비율을 낮춘다.

graph SCM이 만드는 이진 데이터셋은 소수 클래스 비율 중앙값이 약 0.34이고 10% 미만이 9%뿐이다
(stage 1 설정 실측). 현실의 불균형 이진 표(사기·고장·부도)는 1~10%다.
"""

from __future__ import annotations

import numpy as np
import torch

from minipfn.priors.base import IGNORE_INDEX
from minipfn.priors.pipeline import Generator, Rng


class ImbalancedBinary:
    """생성기 감싸기: 이진(`hp["num_classes"] == 2`) 그룹을 확률 `dataset_prob`로 골라, 행을
    `oversample`배 만든 뒤 소수 클래스 비율이 r ~ logU(`min_ratio`, `max_ratio`)가 되게 덜어 낸다.

    남기는 행: 소수 클래스 max(`min_minority`, round(r × 행 수))개(모자라면 있는 만큼), 나머지는
    다수 클래스. 다수 클래스는 만든 행의 절반 이상이라 2배면 늘 채울 수 있다. 행 순서는
    섞는다(학습·검증 행 나누기는 뒤의 `CoverClasses`가 한다). 클래스가 하나뿐이거나 버린
    데이터셋(y가 모두 `IGNORE_INDEX`)은 앞의 행 수만큼만 잘라 그대로 넘긴다. 생성기 안의 dataset
    filter는 덜어 내기 전의 행에 걸린다. 이진이 아니거나 고르지 않은 그룹은 감싼 생성기를 그대로
    부른다(난수를 더 쓰지 않는다).
    """

    def __init__(
        self,
        generator: Generator,
        dataset_prob: float,
        *,
        min_ratio: float = 0.01,
        max_ratio: float = 0.15,
        min_minority: int = 2,
        oversample: int = 2,
    ) -> None:
        self.generator = generator
        self.dataset_prob = dataset_prob
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.min_minority = min_minority
        self.oversample = oversample

    def __call__(
        self, n_rows: int, n_features: int, n_datasets: int, hp: dict, rng: Rng
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if hp.get("num_classes") != 2 or not rng.numpy.random() < self.dataset_prob:
            return self.generator(n_rows, n_features, n_datasets, hp, rng)
        x, y = self.generator(n_rows * self.oversample, n_features, n_datasets, hp, rng)
        rows = [self.keep_rows(y[:, d], n_rows, rng.numpy) for d in range(n_datasets)]
        xs = [x[rows[d], d] for d in range(n_datasets)]
        ys = [y[rows[d], d] for d in range(n_datasets)]
        return torch.stack(xs, 1), torch.stack(ys, 1)

    def keep_rows(self, y: torch.Tensor, n_rows: int, r: np.random.Generator) -> torch.Tensor:
        """데이터셋 하나 y (만든 행)에서 남길 행 번호 n_rows개."""
        labels = y.cpu().numpy()
        classes, counts = np.unique(labels, return_counts=True)
        if len(classes) != 2 or (labels == IGNORE_INDEX).any():
            return torch.arange(n_rows, device=y.device)
        minority, majority = (classes[0], classes[1]) if counts[0] <= counts[1] else classes[::-1]
        ratio = np.exp(r.uniform(np.log(self.min_ratio), np.log(self.max_ratio)))
        minority_rows = np.flatnonzero(labels == minority)
        majority_rows = np.flatnonzero(labels == majority)
        n_minority = min(max(self.min_minority, round(ratio * n_rows)), len(minority_rows))
        n_majority = n_rows - n_minority
        keep = np.concatenate(
            [
                r.choice(minority_rows, n_minority, replace=False),
                r.choice(majority_rows, n_majority, replace=False),
            ]
        )
        return torch.from_numpy(r.permutation(keep)).to(y.device)
