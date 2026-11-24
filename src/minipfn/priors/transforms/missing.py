"""결측값 (TabPFN v2 논문 "Post-processing"): 일부 데이터셋에서 셀마다 같은 확률로 값을
지운다(MCAR)."""

from __future__ import annotations

import torch

from minipfn.priors.pipeline import Group, Rng


class Missing:
    """데이터셋마다 확률 `dataset_prob`로, 비율 r ~ U(0, max_rate)만큼 x를 NaN으로 바꾼다.

    두 값은 v2 논문에 없어서 graph SCM prior에서 정했다(`missing_dataset_prob`, `max_missing_rate`).
    """

    def __init__(self, dataset_prob: float, max_rate: float) -> None:
        self.dataset_prob = dataset_prob
        self.max_rate = max_rate

    def __call__(self, group: Group, rng: Rng) -> Group:
        r = rng.numpy
        xs = []
        for d in range(group.x.shape[1]):
            x = group.x[:, d]
            if r.random() < self.dataset_prob:
                rate = r.uniform(0.0, self.max_rate)
                mask = torch.from_numpy(r.random(tuple(x.shape)) < rate).to(x.device)
                x = x.masked_fill(mask, float("nan"))
            xs.append(x)
        return Group(torch.stack(xs, 1), group.y, group.n_train, group.hp)
