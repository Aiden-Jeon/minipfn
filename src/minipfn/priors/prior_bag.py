"""Prior bag (TabPFN v1 `priors/`, 논문 4절·부록 C): 데이터셋 묶음("그룹")마다 GP prior 또는 MLP
prior를 고르고, 표 데이터로 다듬는다. v1 prior가 쓰는 부품을 모두 이 파일에 둔다.

그룹 하나는 하이퍼파라미터를 공유하고(`sample_group_hyperparameters`), 신경망 가중치만 데이터셋마다
새로 뽑는다(v1 `new_mlp_per_example`). 그룹을 만드는 순서는 v1과 같다.

1. GP(`sample_gp_tables`) 또는 MLP(`sample_mlp_dataset`)로 연속값 데이터. softmax([1, w])로
   고르고, w는 설정의 `mlp_weight` 분포에서 뽑는다.
2. 일부 그룹에서 일부 열을 범주형으로 바꾼다.
3. 이상치 누르기·표준화, target을 클래스로.
4. feature 수 보정과 0 패딩, 학습·테스트 행의 클래스 맞추기, 라벨 정리.

파일 안의 순서:

- 하이퍼파라미터 (v1 `differentiable_prior.py`): meta 분포에서 분포의 모수를 먼저 뽑고, 그
  분포에서 값을 뽑는다. 하이퍼파라미터 하나는 `{"distribution": 종류, ...모수 범위}` dict로 적는다.
  종류는 `uniform`, `meta_gamma`, `meta_beta`, `meta_trunc_norm_log_scaled`, `meta_choice`,
  `meta_choice_mixed`다.
- MLP prior (v1 `mlp.py`, 논문 4.3·4.4절): 무작위 MLP로 데이터셋을 만든다. `is_causal`이 True면
  SCM prior(계산한 노드 가운데서 x, y를 고른다), False면 BNN prior(입력 = x, 마지막 출력 = y).
- GP 표 (v1 `fast_gp.py`): RBF 커널 GP에서 k차원 입력의 표.
- 표 데이터 처리 (v1 `flexible_categorical.py`, `tabpfn/utils.py`): 정규화, 이상치 누르기, 연속값을
  클래스로 바꾸기, 학습·테스트 행의 클래스 맞추기. v1 추론(`minipfn.inference.rotation_ensemble`)도
  입력 전처리에 같이 쓴다.
- prior bag (v1 `prior_bag.py`): 위 부품으로 그룹을 만든다.

어떤 분포와 확률을 쓸지는 `PriorBagConfig`로 받는다. v1 논문 설정은
`minipfn.recipes.tabpfn_v1.PRIOR`(그 설정을 기본값으로 쓰는 `TabPFNPrior`도 거기에 있다).
내부 텐서는 v1처럼 (행, 데이터셋, feature) 또는 (행, 데이터셋) 순서이고, `PriorBag`이 바깥에
내보낼 때 (데이터셋, 행, feature)로 바꾼다.

공식 v1 코드(github.com/automl/TabPFN `tabpfn_v1`, Apache 2.0, Copyright 2022 University of
Freiburg)를 따르되, 난수는 전역 `random`·`np.random` 대신 생성기 객체에서 뽑는다. 결측값 주입은
공개 체크포인트에서 꺼져 있었으므로 구현하지 않았다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from scipy import stats

from minipfn.priors.base import IGNORE_INDEX

# ---------------------------------------------------------------------------
# 하이퍼파라미터 (v1 differentiable_prior.py DifferentiableHyperparameter)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# MLP prior (v1 mlp.py). 내부 텐서는 (행, feature) 순서
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# GP 표 (v1 fast_gp.py)
# ---------------------------------------------------------------------------


def sample_gp_tables(
    seq_len: int, n_datasets: int, k: int, hp: dict, gen: torch.Generator, device
) -> tuple[torch.Tensor, torch.Tensor]:
    """RBF 커널 GP에서 표 데이터셋 n_datasets개 (TabPFN v1 `priors/fast_gp.py`). x는 k차원
    표준정규분포(sampling='normal'), 하이퍼파라미터 `hp`의 outputscale·lengthscale·noise를 쓴다.
    반환은 (행, 데이터셋, k), (행, 데이터셋) 순서다."""
    x = torch.randn(n_datasets, seq_len, k, generator=gen, device=device)
    xd = x.double()
    sq_dist = torch.cdist(xd, xd) ** 2
    lengthscale = max(hp["lengthscale"], 1e-6)
    cov = hp["outputscale"] * torch.exp(-0.5 * sq_dist / lengthscale**2)
    cov = cov + (hp["noise"] + 1e-6) * torch.eye(seq_len, dtype=torch.float64, device=device)
    chol = torch.linalg.cholesky(cov)
    z = torch.randn(n_datasets, seq_len, 1, generator=gen, device=device, dtype=torch.float64)
    y = (chol @ z)[..., 0].float()
    return x.transpose(0, 1), y.transpose(0, 1)  # (T, B, k), (T, B)


# ---------------------------------------------------------------------------
# 표 데이터 처리 (v1 flexible_categorical.py, tabpfn/utils.py)
# ---------------------------------------------------------------------------


def _masked_mean_std(x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    num = mask.sum(0).clamp(min=1)
    xm = torch.where(mask, x, torch.zeros_like(x))
    mean = xm.sum(0) / num
    var = (torch.where(mask, (x - mean) ** 2, torch.zeros_like(x))).sum(0) / (num - 1).clamp(min=1)
    return mean, var.sqrt()


def normalization_stats(ref: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """normalize_data가 쓰는 열별 평균·표준편차 (결측값 제외)."""
    return _masked_mean_std(ref, ~torch.isnan(ref))


def apply_normalization(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return torch.clip((x - mean) / (std + 1e-6), -100, 100)


def normalize_data(x: torch.Tensor, n_rows: int = -1) -> torch.Tensor:
    """행 방향(dim 0)으로 표준화하고 ±100으로 자른다. n_rows > 0이면 앞 n_rows행의 통계를 쓴다."""
    ref = x[:n_rows] if n_rows > 0 else x
    return apply_normalization(x, *normalization_stats(ref))


def outlier_bounds(ref: torch.Tensor, n_sigma: float = 4) -> tuple[torch.Tensor, torch.Tensor]:
    """remove_outliers의 경계: 평균 ± nσ 안의 값만으로 평균·표준편차를 다시 구한 ± nσ."""
    mean, std = _masked_mean_std(ref, ~torch.isnan(ref))
    lower, upper = mean - n_sigma * std, mean + n_sigma * std
    mask = (ref <= upper) & (ref >= lower) & ~torch.isnan(ref)
    mean, std = _masked_mean_std(ref, mask)
    return mean - n_sigma * std, mean + n_sigma * std


def squash_outliers(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    x = torch.maximum(-torch.log(1 + torch.abs(x)) + lower, x)
    return torch.minimum(torch.log(1 + torch.abs(x)) + upper, x)


def remove_outliers(x: torch.Tensor, n_sigma: float = 4, n_rows: int = -1) -> torch.Tensor:
    """평균 ± 4σ 밖의 값을 log로 눌러 준다 (v1 remove_outliers)."""
    ref = x[:n_rows] if n_rows > 0 else x
    return squash_outliers(x, *outlier_bounds(ref, n_sigma))


def _randomize_classes(d: torch.Tensor, n_classes: int, gen: torch.Generator) -> torch.Tensor:
    perm = torch.randperm(n_classes, generator=gen, device=d.device).to(d.dtype)
    return perm[d.long()]


def _class_count(max_classes: int, rng: np.random.Generator) -> int:
    # v1 class_sampler_f: 50%는 round(U(2, max)), 50%는 2
    if rng.random() > 0.5:
        return int(round(rng.uniform(2, max_classes)))
    return 2


def multiclass_rank(
    v: torch.Tensor,
    max_classes: int,
    ordered_p: float,
    rng: np.random.Generator,
    gen: torch.Generator,
) -> torch.Tensor:
    """연속값 (T, B)를 클래스 번호로 바꾼다 (v1 MulticlassRank, 논문 4.5절).

    무작위 행의 값을 경계로 써서 구간 번호를 매긴다. 경계 행은 그룹 안 데이터셋이 공유한다.
    """
    n_classes = _class_count(max_classes, rng)
    idx = torch.randint(0, v.shape[0], (n_classes - 1,), generator=gen, device=v.device)
    bounds = v[idx].unsqueeze(1)  # (n-1, 1, B)
    d = (v.unsqueeze(0) > bounds).sum(0)  # (T, B)
    randomized = torch.rand(d.shape[1], generator=gen, device=v.device) > ordered_p
    if randomized.any():
        d[:, randomized] = _randomize_classes(d[:, randomized], n_classes, gen)
    reverse = torch.rand(d.shape[1], generator=gen, device=v.device) > 0.5
    d[:, reverse] = n_classes - 1 - d[:, reverse]
    return d.float()


def make_compatible(
    x: torch.Tensor, y: torch.Tensor, single_eval_pos: int, gen: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    """학습 행과 테스트 행의 클래스 집합이 같아질 때까지 행을 섞는다(최대 10번). 실패하면 -100."""
    for b in range(y.shape[1]):
        ok = False
        for _ in range(10):
            train_cls = torch.unique(y[:single_eval_pos, b])
            test_cls = torch.unique(y[single_eval_pos:, b])
            ok = len(train_cls) > 1 and torch.equal(train_cls, test_cls)
            if ok:
                break
            perm = torch.randperm(x.shape[0], generator=gen, device=x.device)
            x[:, b], y[:, b] = x[perm, b], y[perm, b]
        if not ok:
            y[:, b] = IGNORE_INDEX
    return x, y


def normalize_labels(y: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """클래스 번호를 0..n-1로 다시 매기고 무작위로 회전한다 (v1 normalize_labels)."""
    for b in range(y.shape[1]):
        valid = y[:, b] != IGNORE_INDEX
        if not valid.any():
            continue
        vals = y[valid, b]
        ranks = (vals.unsqueeze(0) > vals.unique().unsqueeze(1)).sum(0).float()
        n_classes = int(ranks.max().item()) + 1
        shift = int(torch.randint(0, n_classes, (1,), generator=gen, device=y.device).item())
        y[valid, b] = (ranks + shift) % n_classes
    return y


# ---------------------------------------------------------------------------
# prior bag (v1 prior_bag.py)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PriorBagConfig:
    """prior bag이 쓰는 분포와 확률. 하이퍼파라미터 dict의 형식은 `sample_hyperparameter`."""

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
