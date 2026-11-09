"""하이퍼파라미터 샘플링 (`Pipeline`의 하이퍼파라미터 단계에서 쓰는 부품).

- `sample_hyperparameter` (TabPFN v1 `differentiable_prior.py` DifferentiableHyperparameter): meta
  분포에서 분포의 모수를 먼저 뽑고, 그 분포에서 값을 뽑는다. 하이퍼파라미터 하나는
  `{"distribution": 종류, ...모수 범위}` dict로 적는다. 종류는 `uniform`, `meta_gamma`,
  `meta_beta`, `meta_trunc_norm_log_scaled`, `meta_choice`, `meta_choice_mixed`다.
- `GroupHyperparameters`: TabPFN v1 prior bag의 그룹 하이퍼파라미터 (생성기 선택, 쓰는 feature 수,
  클래스 수 포함).
- `ClassCount`: 분류 데이터셋의 클래스 수를 학습 행 수에 맞춰 뽑는다.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats

from minipfn.priors.pipeline import Rng


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


class ClassCount:
    """클래스 수 `num_classes` ~ U{2, ..., min(max_classes, 학습 행 수)}.

    모든 클래스가 학습 행에 나와야 하므로 클래스 수는 학습 행 수를 넘지 않는다.
    `binary_prob`가 0보다 크면 그 확률로 먼저 2를 고르고, 아니면 위 균등 분포에서 뽑는다
    (TabICLv2 `PriorDataset`: 0.5. 그러면 binary가 0.5 + 0.5/9 ≈ 56%다). 0이면 난수를 더 쓰지 않아
    예전과 같은 데이터셋이 나온다.
    """

    def __init__(self, max_classes: int, binary_prob: float = 0.0) -> None:
        if not 0.0 <= binary_prob <= 1.0:
            raise ValueError(f"binary_prob는 0 이상 1 이하여야 한다: {binary_prob}")
        self.max_classes = max_classes
        self.binary_prob = binary_prob

    def __call__(self, rng: Rng, n_train: int, n_features: int) -> dict:
        if self.binary_prob and rng.numpy.random() < self.binary_prob:
            return {"num_classes": 2}
        max_classes = min(self.max_classes, n_train)
        return {"num_classes": int(rng.numpy.integers(2, max_classes + 1))}


class GroupHyperparameters:
    """그룹(데이터셋 묶음)이 공유하는 하이퍼파라미터 (TabPFN v1 prior bag).

    MLP·GP 생성기의 하이퍼파라미터를 meta 분포에서 뽑고(`sample_hyperparameter`), 어느 생성기를 쓸지
    (`use_mlp`, P(MLP) = softmax([1, w])[1], w ~ `mlp_weight`), 쓰는 feature 수
    (`num_features_used` = round(U(1, feature 수))), 클래스 수 상한(`num_classes` =
    round(U(min_classes, max_classes)))을 정한다. 분포 표의 v1 논문 값은
    `minipfn.recipes.tabpfn_v1`에 있다.
    """

    def __init__(
        self,
        mlp: dict,
        gp: dict,
        *,
        mlp_weight: dict,
        min_classes: int,
        max_classes: int,
    ) -> None:
        self.mlp = mlp
        self.gp = gp
        self.mlp_weight = mlp_weight
        self.min_classes = min_classes
        self.max_classes = max_classes

    def __call__(self, rng: Rng, n_train: int, n_features: int) -> dict:
        return self.sample(rng.numpy, n_features)

    def sample(self, rng: np.random.Generator, num_features: int) -> dict:
        hp = {name: sample_hyperparameter(d, rng) for name, d in self.mlp.items()}
        hp.update({name: sample_hyperparameter(d, rng) for name, d in self.gp.items()})
        # prior bag: P(MLP) = softmax([1, w])[1]
        w = sample_hyperparameter(self.mlp_weight, rng)
        hp["use_mlp"] = bool(rng.random() < math.exp(w) / (math.e + math.exp(w)))
        # flexible_categorical: 쓰는 feature 수와 클래스 수 상한
        # (v1 uniform_int_sampler_f = round(U(a, b)))
        hp["num_features_used"] = int(round(rng.uniform(1, num_features)))
        hp["num_classes"] = int(round(rng.uniform(self.min_classes, self.max_classes)))
        return hp
