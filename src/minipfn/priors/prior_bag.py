"""Prior bag (TabPFN v1 `priors/prior_bag.py`, 논문 4절·부록 C): 데이터셋 묶음("그룹")마다
GP prior 또는 MLP prior를 고르고, 표 데이터로 다듬는다.

그룹 하나는 하이퍼파라미터를 공유하고(`sample_group_hyperparameters`), 신경망 가중치만 데이터셋마다
새로 뽑는다(v1 `new_mlp_per_example`). 그룹을 만드는 순서는 v1과 같다.

1. GP(`minipfn.priors.gp.sample_gp_tables`) 또는 MLP(`minipfn.priors.mlp_scm`)로 연속값 데이터.
   softmax([1, w])로 고르고, w는 설정의 `mlp_weight` 분포에서 뽑는다.
2. 일부 그룹에서 일부 열을 범주형으로 바꾼다.
3. 이상치 누르기·표준화, target을 클래스로 (`minipfn.priors.tabular`).
4. feature 수 보정과 0 패딩, 학습·테스트 행의 클래스 맞추기, 라벨 정리.

어떤 분포와 확률을 쓸지는 `PriorBagConfig`로 받는다. v1 논문 설정은
`minipfn.recipes.tabpfn_v1.PRIOR`(그 설정을 기본값으로 쓰는 `TabPFNPrior`도 거기에 있다).
내부 텐서는 v1처럼 (행, 데이터셋, feature) 순서이고, 바깥에 내보낼 때 (데이터셋, 행, feature)로
바꾼다.

v1과 다른 점: 난수를 전역 `random`·`np.random` 대신 생성기 객체에서 뽑는다. 결측값 주입은
공개 체크포인트에서 꺼져 있었으므로 구현하지 않았다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from minipfn.priors.gp import sample_gp_tables
from minipfn.priors.hyperparameters import sample_hyperparameter
from minipfn.priors.mlp_scm import sample_mlp_dataset
from minipfn.priors.tabular import (
    make_compatible,
    multiclass_rank,
    normalize_data,
    normalize_labels,
    remove_outliers,
)


@dataclass(frozen=True)
class PriorBagConfig:
    """prior bag이 쓰는 분포와 확률. 하이퍼파라미터 dict의 형식은
    `minipfn.priors.hyperparameters`."""

    mlp_hyperparameters: dict  # MLP prior (mlp_scm)
    gp_hyperparameters: dict  # GP prior (outputscale, lengthscale, noise)
    mlp_weight: dict  # P(MLP) = softmax([1, w])[1]의 w 분포
    min_classes: int  # 클래스 수 상한을 round(U(min_classes, max_classes))로 뽑는다
    categorical_feature_p: float  # 일부 열을 범주형으로 바꾸는 그룹의 비율
    categorical_ordered_p: float  # 범주형 열의 번호를 섞지 않고 둘 확률
    output_multiclass_ordered_p: float  # target 클래스 번호를 섞지 않고 둘 확률
    normalize_by_used_features: bool  # 쓰는 feature 수 k로 나눠 num_features / k배
    # 학습·테스트 행의 클래스 집합을 맞춘다 (아니면 행을 섞고, 10번 실패하면 무시)
    check_is_compatible: bool
    scale_weights_sqrt: bool  # MLP 가중치의 dropout 보정 (mlp_scm.sample_mlp_dataset)
    random_feature_rotation: bool  # MLP 데이터셋의 feature 순서를 무작위로 회전


def sample_group_hyperparameters(
    rng: np.random.Generator, config: PriorBagConfig, max_classes: int, num_features: int
) -> dict:
    """그룹(데이터셋 묶음)이 공유하는 하이퍼파라미터를 한 번에 뽑는다."""
    hp = {name: sample_hyperparameter(d, rng) for name, d in config.mlp_hyperparameters.items()}
    gp = config.gp_hyperparameters
    hp.update({name: sample_hyperparameter(d, rng) for name, d in gp.items()})
    # prior bag: P(MLP) = softmax([1, w])[1]
    w = sample_hyperparameter(config.mlp_weight, rng)
    hp["use_mlp"] = bool(rng.random() < math.exp(w) / (math.e + math.exp(w)))
    # flexible_categorical: 쓰는 feature 수와 클래스 수 상한
    # (v1 uniform_int_sampler_f = round(U(a, b)))
    hp["num_features_used"] = int(round(rng.uniform(1, num_features)))
    hp["num_classes"] = int(round(rng.uniform(config.min_classes, max_classes)))
    return hp


class PriorBag:
    """prior bag 샘플러. `sample_group`이 하이퍼파라미터를 공유하는 데이터셋 묶음을 만든다."""

    def __init__(
        self,
        config: PriorBagConfig,
        *,
        seq_len: int,
        num_features: int,
        max_classes: int,
        group_size: int,
        device: torch.device | str = "cpu",
        seed: int = 0,
    ) -> None:
        self.config = config
        self.seq_len = seq_len
        self.num_features = num_features
        self.max_classes = max_classes
        self.group_size = group_size
        self.device = torch.device(device)
        self.rng = np.random.default_rng(seed)
        self.gen = torch.Generator(device=self.device)
        self.gen.manual_seed(seed)

    @torch.no_grad()
    def sample_group(self, single_eval_pos: int) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """데이터셋 group_size개.

        반환: x (B, T, num_features), y (B, T) (-100 = 무시), 하이퍼파라미터.
        """
        rng, gen, dev, cfg = self.rng, self.gen, self.device, self.config
        T, B = self.seq_len, self.group_size
        hp = sample_group_hyperparameters(rng, cfg, self.max_classes, self.num_features)
        k = hp["num_features_used"]

        # 1) GP 또는 MLP prior로 연속값 데이터 (T, B, k), (T, B)
        if hp["use_mlp"]:
            mlp = {
                "scale_weights_sqrt": cfg.scale_weights_sqrt,
                "random_feature_rotation": cfg.random_feature_rotation,
            }
            xs, ys = zip(
                *(sample_mlp_dataset(T, k, hp, rng, gen, dev, **mlp) for _ in range(B)),
                strict=True,
            )
            x, y = torch.stack(xs, 1), torch.stack(ys, 1)
        else:
            x, y = sample_gp_tables(T, B, k, hp, gen, dev)

        # 2) 일부 열을 범주형으로 (그룹의 categorical_feature_p)
        if rng.random() < cfg.categorical_feature_p:
            p = rng.random()
            for col in range(k):
                n_unique = max(round(rng.gamma(1, 10)), 2)
                if rng.random() < p:
                    x[:, :, col] = multiclass_rank(
                        x[:, :, col], n_unique, cfg.categorical_ordered_p, rng, gen
                    )

        # 3) 이상치 누르기 + 표준화, target을 클래스로
        x = normalize_data(remove_outliers(x))
        y = normalize_data(y)
        y = multiclass_rank(y, hp["num_classes"], cfg.output_multiclass_ordered_p, rng, gen)

        # 4) feature 수 보정(num_features/k배)과 0 패딩
        if cfg.normalize_by_used_features:
            x = x / (k / self.num_features)
        x = torch.cat([x, torch.zeros(T, B, self.num_features - k, device=dev)], dim=-1)

        # 5) 학습/테스트 행 클래스 맞추기, 라벨 정리
        if cfg.check_is_compatible:
            x, y = make_compatible(x, y, single_eval_pos, gen)
        y = normalize_labels(y, gen)
        return x.transpose(0, 1).contiguous(), y.transpose(0, 1).contiguous(), hp
