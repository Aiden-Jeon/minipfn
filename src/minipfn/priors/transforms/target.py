"""회귀 target 다듬기 (issue #103, Kumo-Tabular 블로그의 "꼬리가 긴 회귀 target"을 다시 구현)."""

from __future__ import annotations

import numpy as np
import torch

from minipfn.priors.pipeline import Group, Rng


class LongTailTarget:
    """회귀 데이터셋마다 확률 `dataset_prob`로 target에 꼬리를 붙인다: y ← σ·exp(σ·s·y), 그 뒤
    z-정규화(학습 행과 검증 행을 함께, graph SCM의 target처럼).

    s ~ U(`min_scale`, `max_scale`), σ는 ±1 반반이다(오른쪽 또는 왼쪽 꼬리). 두 경우 모두 y에
    대해 증가 함수라 x와 y의 관계 방향은 그대로다. graph SCM의 target은 4σ에서 잘려 있어 s 1.5면
    가장 먼 값이 중앙값의 수백 배가 된다(가격·건수 같은 log-normal target).
    """

    def __init__(self, dataset_prob: float, *, min_scale: float = 0.3, max_scale: float = 1.5):
        self.dataset_prob = dataset_prob
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, group: Group, rng: Rng) -> Group:
        r = rng.numpy
        ys = []
        for d in range(group.y.shape[1]):
            y = group.y[:, d]
            if r.random() < self.dataset_prob:
                sign = 1.0 if r.random() < 0.5 else -1.0
                s = r.uniform(self.min_scale, self.max_scale)
                v = y.cpu().numpy().astype(np.float64)
                v = sign * np.exp(sign * s * (v - v.mean()))
                v = (v - v.mean()) / max(v.std(), 1e-6)
                y = torch.from_numpy(v).to(y.device, y.dtype)
            ys.append(y)
        return Group(group.x, torch.stack(ys, 1), group.n_train, group.hp)
