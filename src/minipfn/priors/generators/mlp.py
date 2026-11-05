"""MLP 생성기 (TabPFN v1 `mlp.py`, 논문 4.3·4.4절): 무작위 MLP로 데이터셋을 만든다.

`is_causal`이 True면 SCM prior(계산한 노드 가운데서 x, y를 고른다), False면 BNN prior(입력 = x,
마지막 출력 = y). 하이퍼파라미터(`hp`)는 그룹이 공유하고, 신경망 가중치는 데이터셋마다 새로
뽑는다(v1 `new_mlp_per_example`). 쓰는 하이퍼파라미터 키는 `sample_mlp_dataset`에 있고, v1 논문
분포는 `minipfn.recipes.tabpfn_v1.MLP_HYPERPARAMETERS`다. 내부 텐서는 (행, feature) 순서다.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from minipfn.priors.base import IGNORE_INDEX
from minipfn.priors.pipeline import Rng

ACTIVATIONS = {"tanh": torch.tanh, "identity": lambda z: z, "relu": torch.relu}


def _init_weight(
    n_out: int,
    n_in: int,
    hp: dict,
    apply_dropout: bool,
    rng: np.random.Generator,
    gen: torch.Generator,
    device,
    scale_weights_sqrt: bool,
) -> torch.Tensor:
    """가중치 행렬 하나 (n_out, n_in). v1 MLP.__init__의 초기화 규칙을 그대로 따른다."""
    init_std = hp["init_std"]
    if hp["block_wise_dropout"]:
        # 대각선 블록만 남기고 나머지는 0
        w = torch.zeros(n_out, n_in, device=device)
        n_blocks = int(rng.integers(1, math.ceil(math.sqrt(min(n_out, n_in))) + 1))
        bh, bw = n_out // n_blocks, n_in // n_blocks
        keep_prob = (n_blocks * bh * bw) / (n_out * n_in)
        std = init_std / keep_prob ** (1 / 2 if scale_weights_sqrt else 1)
        for b in range(n_blocks):
            block = torch.randn(bh, bw, generator=gen, device=device) * std
            w[bh * b : bh * (b + 1), bw * b : bw * (b + 1)] = block
        return w
    dropout = min(hp["prior_mlp_dropout_prob"], 0.99) if apply_dropout else 0.0
    # v1 식 그대로: init_std / (1 - dropout ** (1/2))
    std = init_std / (1.0 - dropout ** (1 / 2 if scale_weights_sqrt else 1))
    w = torch.randn(n_out, n_in, generator=gen, device=device) * std
    keep = torch.rand(n_out, n_in, generator=gen, device=device) < (1.0 - dropout)
    return w * keep


def _default_bias(n_out: int, n_in: int, gen: torch.Generator, device) -> torch.Tensor:
    # v1은 bias를 다시 초기화하지 않으므로 nn.Linear 기본값 U(-1/sqrt(n_in), 1/sqrt(n_in))이 남는다.
    bound = 1.0 / math.sqrt(n_in)
    return (torch.rand(n_out, generator=gen, device=device) * 2 - 1) * bound


def sample_mlp_dataset(
    seq_len: int,
    k: int,
    hp: dict,
    rng: np.random.Generator,
    gen: torch.Generator,
    device,
    *,
    scale_weights_sqrt: bool,
    random_feature_rotation: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """데이터셋 하나 (x: (T, k), y: (T,)). 가중치는 데이터셋마다 새로 뽑는다.

    `hp`는 그룹이 공유하는 하이퍼파라미터(`sample_group_hyperparameters`).
    `scale_weights_sqrt`면 dropout으로 지운 만큼 가중치 표준편차를 keep 비율의 제곱근으로 나눈다
    (아니면 keep 비율로). `random_feature_rotation`이면 feature 순서를 무작위로 회전한다.
    """
    is_causal = hp["is_causal"]
    hidden = int(hp["prior_mlp_hidden_dim"])
    num_causes = int(hp["num_causes"]) if is_causal else k
    if is_causal:
        hidden = max(hidden, 1 + 2 * k)  # x와 y로 고를 노드가 충분하도록
    num_layers = int(hp["num_layers"])
    act = ACTIVATIONS[hp["prior_mlp_activations"]]

    # 1) 원인 노드: 평균과 표준편차를 미리 뽑아 둔다 (pre_sample_causes)
    means = rng.normal(0, 1, num_causes)
    stds = np.abs(rng.normal(0, 1, num_causes) * means)
    means_t = torch.tensor(means, dtype=torch.float32, device=device)
    stds_t = torch.tensor(stds, dtype=torch.float32, device=device)
    causes = means_t + stds_t * torch.randn(seq_len, num_causes, generator=gen, device=device)

    # 2) 층 구조: Linear(원인 → hidden), 이후 (활성화 → Linear → 노이즈)를 num_layers-1번
    out_dims = [hidden] * (num_layers - 1) + ([] if is_causal else [1])
    layer0_w = _init_weight(
        hidden,
        num_causes,
        hp,
        apply_dropout=False,
        rng=rng,
        gen=gen,
        device=device,
        scale_weights_sqrt=scale_weights_sqrt,
    )
    h = causes @ layer0_w.T + _default_bias(hidden, num_causes, gen, device)
    outputs = []
    n_in = hidden
    for n_out in out_dims:
        w = _init_weight(
            n_out,
            n_in,
            hp,
            apply_dropout=True,
            rng=rng,
            gen=gen,
            device=device,
            scale_weights_sqrt=scale_weights_sqrt,
        )
        b = _default_bias(n_out, n_in, gen, device)
        if hp["pre_sample_weights"]:
            noise_std = torch.randn(1, n_out, generator=gen, device=device).abs() * hp["noise_std"]
        else:
            noise_std = hp["noise_std"]
        h = act(h) @ w.T + b
        h = h + noise_std * torch.randn(h.shape, generator=gen, device=device)
        outputs.append(h)
        n_in = n_out

    # 3) x, y 고르기
    if is_causal:
        # SCM prior: 계산한 모든 노드 중에서 고른다
        nodes = torch.cat(outputs, dim=-1)
        n_nodes = nodes.shape[-1]
        if hp["in_clique"]:
            start = int(rng.integers(0, n_nodes - 1 - k + 1))
            perm = start + torch.randperm(1 + k, generator=gen, device=device)
        else:
            perm = torch.randperm(n_nodes - 1, generator=gen, device=device)
        y_idx = torch.tensor([n_nodes - 1], device=device) if hp["y_is_effect"] else perm[:1]
        x_idx = perm[1 : 1 + k]
        if hp["sort_features"]:
            x_idx, _ = torch.sort(x_idx)
        x, y = nodes[:, x_idx], nodes[:, y_idx]
    else:
        # BNN prior: 입력이 x, 마지막 출력이 y
        x, y = causes, outputs[-1]

    if torch.isnan(x).any() or torch.isnan(y).any():
        x = torch.zeros_like(x)
        y = torch.full_like(y, float(IGNORE_INDEX))

    if random_feature_rotation:
        shift = int(rng.integers(0, k))
        x = x[:, (torch.arange(k, device=device) + shift) % k]
    return x, y[:, 0]


class MLP:
    """MLP 생성기 (`Generator`). 그룹의 데이터셋마다 `sample_mlp_dataset`을 부른다."""

    def __init__(self, *, scale_weights_sqrt: bool, random_feature_rotation: bool) -> None:
        self.scale_weights_sqrt = scale_weights_sqrt
        self.random_feature_rotation = random_feature_rotation

    def __call__(
        self, n_rows: int, n_features: int, n_datasets: int, hp: dict, rng: Rng
    ) -> tuple[torch.Tensor, torch.Tensor]:
        kw = {
            "scale_weights_sqrt": self.scale_weights_sqrt,
            "random_feature_rotation": self.random_feature_rotation,
        }
        datasets = (
            sample_mlp_dataset(n_rows, n_features, hp, rng.numpy, rng.torch, rng.device, **kw)
            for _ in range(n_datasets)
        )
        xs, ys = zip(*datasets, strict=True)
        return torch.stack(xs, 1), torch.stack(ys, 1)
