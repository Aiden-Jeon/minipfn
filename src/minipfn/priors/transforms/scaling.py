"""값의 크기 맞추기: 표준화와 이상치 누르기, feature 수 보정과 0 패딩 (TabPFN v1)."""

from __future__ import annotations

import torch

from minipfn.priors.pipeline import Group, Rng
from minipfn.tabular import normalize_data, remove_outliers


class Standardize:
    """x는 평균 ± 4σ 밖의 값을 log로 누른 뒤 표준화하고, y도 표준화한다 (그룹 안 데이터셋마다)."""

    def __call__(self, group: Group, rng: Rng) -> Group:
        x = normalize_data(remove_outliers(group.x))
        return Group(x, normalize_data(group.y), group.n_train, group.hp)


class PadFeatures:
    """feature를 `n_features`열까지 0으로 채운다. `rescale`이면 먼저 n_features / k배 한다.

    k는 생성기가 만든 feature 수(하이퍼파라미터 `num_features_used`)다. 쓰는 feature가 적어도 값의
    전체 크기가 비슷하도록 키운다(v1 `normalize_by_used_features`).
    """

    def __init__(self, n_features: int, *, rescale: bool) -> None:
        self.n_features = n_features
        self.rescale = rescale

    def __call__(self, group: Group, rng: Rng) -> Group:
        x = group.x
        n_rows, n_datasets, k = x.shape
        if self.rescale:
            x = x / (k / self.n_features)
        pad = torch.zeros(n_rows, n_datasets, self.n_features - k, device=x.device)
        return Group(torch.cat([x, pad], dim=-1), group.y, group.n_train, group.hp)
