"""하이퍼파라미터를 두 단계로 뽑는다 (TabPFN v1 `priors/differentiable_prior.py`의
DifferentiableHyperparameter): meta 분포에서 분포의 모수를 먼저 뽑고, 그 분포에서 값을 뽑는다.

하이퍼파라미터 하나는 `{"distribution": 종류, ...모수 범위}` dict로 적는다. 종류는 `uniform`,
`meta_gamma`, `meta_beta`, `meta_trunc_norm_log_scaled`, `meta_choice`, `meta_choice_mixed`다.
v1 논문이 쓴 표는 `minipfn.recipes.tabpfn_v1`에 있다.

공식 v1 코드(github.com/automl/TabPFN `tabpfn_v1`, Apache 2.0, Copyright 2022 University of
Freiburg)를 따르되, 난수는 전역 `random`·`np.random` 대신 생성기 객체에서 뽑는다.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats


def _truncnorm(mu: float, sigma: float, rng: np.random.Generator) -> float:
    # v1 trunc_norm_sampler_f: [0, 1e6]으로 자른 정규분포
    a, b = (0 - mu) / sigma, (1_000_000 - mu) / sigma
    return float(stats.truncnorm.rvs(a, b, loc=mu, scale=sigma, random_state=rng))


def _softmax_choice(values: list, weight_range: tuple[float, float], rng: np.random.Generator):
    # 첫 선택지의 가중치는 1로 고정하고 나머지는 균등분포에서 뽑아 softmax한다.
    weights = np.array([1.0] + list(rng.uniform(*weight_range, len(values) - 1)))
    probs = np.exp(weights - weights.max())
    return values[rng.choice(len(values), p=probs / probs.sum())]


def sample_hyperparameter(hp: dict, rng: np.random.Generator):
    """meta 분포 하나에서 값을 뽑는다. 분포의 모수를 먼저 뽑고, 그 분포에서 값을 뽑는다."""
    kind = hp["distribution"]
    if kind == "uniform":
        return float(rng.uniform(hp["min"], hp["max"]))
    if kind == "meta_gamma":
        alpha = math.exp(rng.uniform(0.0, math.log(hp["max_alpha"])))
        scale = rng.uniform(0.0, hp["max_scale"])
        value = rng.gamma(alpha, scale / alpha)
        return hp["lower_bound"] + (round(value) if hp["round"] else value)
    if kind == "meta_beta":
        b, k = rng.uniform(hp["min"], hp["max"], 2)
        return hp["scale"] * float(rng.beta(b, k))
    if kind == "meta_trunc_norm_log_scaled":
        mean = math.exp(rng.uniform(math.log(hp["min_mean"]), math.log(hp["max_mean"])))
        std = mean * math.exp(rng.uniform(math.log(0.01), math.log(1.0)))
        value = _truncnorm(mean, std, rng)
        return hp["lower_bound"] + (round(value) if hp["round"] else value)
    if kind == "meta_choice":
        return _softmax_choice(hp["choice_values"], (-3.0, 5.0), rng)
    if kind == "meta_choice_mixed":
        return _softmax_choice(hp["choice_values"], (-5.0, 6.0), rng)
    raise ValueError(f"unknown distribution: {kind}")
