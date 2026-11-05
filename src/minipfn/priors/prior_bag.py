"""Prior bag (TabPFN v1 `priors/`, 논문 4절·부록 C): 그룹("데이터셋 묶음")마다 GP 생성기 또는 MLP
생성기를 고르고, 표 데이터로 다듬는다. 부품(`minipfn.priors.generators`, `transforms`,
`hyperparameters`)을 `Pipeline`으로 묶은 것이다.

그룹 하나는 하이퍼파라미터를 공유하고(`hyperparameters.GroupHyperparameters`), 신경망 가중치만
데이터셋마다 새로 뽑는다(v1 `new_mlp_per_example`). 그룹을 만드는 순서는 v1과 같다.

1. 하이퍼파라미터: MLP·GP 하이퍼파라미터, 어느 생성기를 쓸지(softmax([1, w]), w는 설정의
   `mlp_weight` 분포), 쓰는 feature 수, 클래스 수.
2. 생성기: GP(`generators.gp.GPTables`) 또는 MLP(`generators.mlp.MLP`)로 연속값 데이터.
3. 후처리: 일부 열을 범주형으로(`Categorical`), 이상치 누르기·표준화(`Standardize`), target을
   클래스로(`ToClasses`), feature 수 보정과 0 패딩(`PadFeatures`), 학습·테스트 행의 클래스
   맞추기(`MatchClasses`), 라벨 정리(`Relabel`).

어떤 분포와 확률을 쓸지는 `PriorBagConfig`로 받는다. v1 논문 설정은
`minipfn.recipes.tabpfn_v1.PRIOR`(그 설정을 기본값으로 쓰는 `TabPFNPrior`도 거기에 있다).

공식 v1 코드(github.com/automl/TabPFN `tabpfn_v1`, Apache 2.0, Copyright 2022 University of
Freiburg)를 따르되, 난수는 전역 `random`·`np.random` 대신 생성기 객체에서 뽑는다. 결측값 주입은
공개 체크포인트에서 꺼져 있었으므로 넣지 않았다(`transforms.missing.Missing`을 붙이면 된다).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from minipfn.priors.generators.gp import GPTables
from minipfn.priors.generators.mlp import MLP
from minipfn.priors.hyperparameters import GroupHyperparameters
from minipfn.priors.pipeline import Mixture, Pipeline, Rng
from minipfn.priors.transforms import (
    Categorical,
    MatchClasses,
    PadFeatures,
    Relabel,
    Standardize,
    ToClasses,
)


@dataclass(frozen=True)
class PriorBagConfig:
    """prior bag이 쓰는 분포와 확률. 하이퍼파라미터 dict의 형식은
    `minipfn.priors.hyperparameters.sample_hyperparameter`."""

    mlp_hyperparameters: dict  # MLP prior (`sample_mlp_dataset`)
    gp_hyperparameters: dict  # GP prior (outputscale, lengthscale, noise)
    mlp_weight: dict  # P(MLP) = softmax([1, w])[1]의 w 분포
    min_classes: int  # 클래스 수 상한을 round(U(min_classes, max_classes))로 뽑는다
    categorical_feature_p: float  # 일부 열을 범주형으로 바꾸는 그룹의 비율
    categorical_ordered_p: float  # 범주형 열의 번호를 섞지 않고 둘 확률
    output_multiclass_ordered_p: float  # target 클래스 번호를 섞지 않고 둘 확률
    normalize_by_used_features: bool  # 쓰는 feature 수 k로 나눠 num_features / k배
    # 학습·테스트 행의 클래스 집합을 맞춘다 (아니면 행을 섞고, 10번 실패하면 무시)
    check_is_compatible: bool
    scale_weights_sqrt: bool  # MLP 가중치의 dropout 보정 (`sample_mlp_dataset`)
    random_feature_rotation: bool  # MLP 데이터셋의 feature 순서를 무작위로 회전


def group_hyperparameters(config: PriorBagConfig, max_classes: int) -> GroupHyperparameters:
    """설정의 분포로 그룹 하이퍼파라미터 부품을 만든다."""
    return GroupHyperparameters(
        config.mlp_hyperparameters,
        config.gp_hyperparameters,
        mlp_weight=config.mlp_weight,
        min_classes=config.min_classes,
        max_classes=max_classes,
    )


def sample_group_hyperparameters(
    rng: np.random.Generator, config: PriorBagConfig, max_classes: int, num_features: int
) -> dict:
    """그룹(데이터셋 묶음)이 공유하는 하이퍼파라미터를 한 번에 뽑는다 (`GroupHyperparameters`)."""
    return group_hyperparameters(config, max_classes).sample(rng, num_features)


def _mlp_or_gp(hp: dict) -> str:
    return "mlp" if hp["use_mlp"] else "gp"


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
        # 아래 설정은 부품을 만들 때 들어가므로 바꿀 수 없다(읽기 전용 속성)
        self._config = config
        self._seq_len = seq_len
        self._num_features = num_features
        self._max_classes = max_classes
        self._device = torch.device(device)
        mlp = MLP(
            scale_weights_sqrt=config.scale_weights_sqrt,
            random_feature_rotation=config.random_feature_rotation,
        )
        transforms = [
            Categorical(config.categorical_feature_p, config.categorical_ordered_p),
            Standardize(),
            ToClasses(config.output_multiclass_ordered_p),
            PadFeatures(num_features, rescale=config.normalize_by_used_features),
        ]
        if config.check_is_compatible:
            transforms.append(MatchClasses())
        transforms.append(Relabel())
        self.pipeline = Pipeline(
            Mixture({"mlp": mlp, "gp": GPTables()}, _mlp_or_gp),
            rng=Rng.from_seed(seed, self._device),
            hyperparameters=group_hyperparameters(config, max_classes),
            transforms=transforms,
            group_size=group_size,
        )

    @property
    def config(self) -> PriorBagConfig:
        return self._config

    @property
    def seq_len(self) -> int:
        return self._seq_len

    @property
    def num_features(self) -> int:
        return self._num_features

    @property
    def max_classes(self) -> int:
        return self._max_classes

    @property
    def group_size(self) -> int:
        return self.pipeline.group_size

    @property
    def device(self) -> torch.device:
        return self._device

    @torch.no_grad()
    def sample_group(self, single_eval_pos: int) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """데이터셋 group_size개.

        반환: x (B, T, num_features), y (B, T) (-100 = 무시), 하이퍼파라미터.
        """
        group = self.pipeline.sample_group(self.seq_len, single_eval_pos, self.num_features)
        x, y = group.x.transpose(0, 1).contiguous(), group.y.transpose(0, 1).contiguous()
        return x, y, group.hp
