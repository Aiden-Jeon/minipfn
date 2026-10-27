"""TabPFN v1 prior를 torch로 옮긴 것 (논문 4절·부록 C, 수치는 `spec.py`).

공식 v1 코드(github.com/automl/TabPFN `tabpfn_v1`, Apache 2.0, Copyright 2022 University of
Freiburg)의 다음 구조를 따른다.

- `priors/differentiable_prior.py`: 하이퍼파라미터를 두 단계로 뽑는다(분포의 모수 → 값).
- `priors/prior_bag.py`: softmax([1, w])로 GP prior 또는 MLP prior를 고른다.
- `priors/mlp.py`: MLP prior. `is_causal=True`면 SCM prior(그래프 노드에서 x, y를 고름),
  `False`면 BNN prior(입력 = x, 출력 = y).
- `priors/flexible_categorical.py`: feature 수·범주형 변환·정규화·클래스 변환·패딩.

v1에서 데이터셋 8개짜리 forward batch 하나("그룹")는 하이퍼파라미터를 공유하고, 신경망
가중치만 데이터셋마다 새로 뽑는다(`new_mlp_per_example`). 여기서도 그룹 단위로 만든다.
내부 텐서는 v1처럼 (행, 데이터셋, feature) 순서이고, 바깥에 내보낼 때
(데이터셋, 행, feature)로 바꾼다.

v1과 다른 점: 난수를 전역 `random`·`np.random` 대신 생성기 객체에서 뽑는다. 결측값 주입은
공개 체크포인트에서 꺼져 있었으므로 구현하지 않았다.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from scipy import stats
from torch.utils.data import IterableDataset, get_worker_info

from minipfn.tabpfn import spec

ACTIVATIONS = {"tanh": torch.tanh, "identity": lambda z: z, "relu": torch.relu}


# ---------------------------------------------------------------------------
# 하이퍼파라미터 (v1 differentiable_prior.DifferentiableHyperparameter)
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


def sample_group_hyperparameters(
    rng: np.random.Generator, max_classes: int, num_features: int
) -> dict:
    """그룹(데이터셋 8개)이 공유하는 하이퍼파라미터를 한 번에 뽑는다."""
    hp = {name: sample_hyperparameter(d, rng) for name, d in spec.MLP_HYPERPARAMETERS.items()}
    hp.update({name: sample_hyperparameter(d, rng) for name, d in spec.GP_HYPERPARAMETERS.items()})
    # prior bag: P(MLP) = softmax([1, w])[1]
    w = sample_hyperparameter(spec.PRIOR_BAG_MLP_WEIGHT, rng)
    hp["use_mlp"] = bool(rng.random() < math.exp(w) / (math.e + math.exp(w)))
    # flexible_categorical: 쓰는 feature 수와 클래스 수 상한
    # (v1 uniform_int_sampler_f = round(U(a, b)))
    hp["num_features_used"] = int(round(rng.uniform(1, num_features)))
    hp["num_classes"] = int(round(rng.uniform(spec.NUM_CLASSES[0], max_classes)))
    return hp


# ---------------------------------------------------------------------------
# 데이터 생성: MLP prior (v1 priors/mlp.py)
# ---------------------------------------------------------------------------
def _init_weight(
    n_out: int,
    n_in: int,
    hp: dict,
    apply_dropout: bool,
    rng: np.random.Generator,
    gen: torch.Generator,
    device,
) -> torch.Tensor:
    """가중치 행렬 하나 (n_out, n_in). v1 MLP.__init__의 초기화 규칙을 그대로 따른다."""
    init_std = hp["init_std"]
    if hp["block_wise_dropout"]:
        # 대각선 블록만 남기고 나머지는 0
        w = torch.zeros(n_out, n_in, device=device)
        n_blocks = int(rng.integers(1, math.ceil(math.sqrt(min(n_out, n_in))) + 1))
        bh, bw = n_out // n_blocks, n_in // n_blocks
        keep_prob = (n_blocks * bh * bw) / (n_out * n_in)
        std = init_std / keep_prob ** (1 / 2 if spec.PRIOR_MLP_SCALE_WEIGHTS_SQRT else 1)
        for b in range(n_blocks):
            block = torch.randn(bh, bw, generator=gen, device=device) * std
            w[bh * b : bh * (b + 1), bw * b : bw * (b + 1)] = block
        return w
    dropout = min(hp["prior_mlp_dropout_prob"], 0.99) if apply_dropout else 0.0
    # v1 식 그대로: init_std / (1 - dropout ** (1/2))
    std = init_std / (1.0 - dropout ** (1 / 2 if spec.PRIOR_MLP_SCALE_WEIGHTS_SQRT else 1))
    w = torch.randn(n_out, n_in, generator=gen, device=device) * std
    keep = torch.rand(n_out, n_in, generator=gen, device=device) < (1.0 - dropout)
    return w * keep


def _default_bias(n_out: int, n_in: int, gen: torch.Generator, device) -> torch.Tensor:
    # v1은 bias를 다시 초기화하지 않으므로 nn.Linear 기본값 U(-1/sqrt(n_in), 1/sqrt(n_in))이 남는다.
    bound = 1.0 / math.sqrt(n_in)
    return (torch.rand(n_out, generator=gen, device=device) * 2 - 1) * bound


def _sample_mlp_dataset(
    seq_len: int, k: int, hp: dict, rng: np.random.Generator, gen: torch.Generator, device
) -> tuple[torch.Tensor, torch.Tensor]:
    """데이터셋 하나 (x: (T, k), y: (T,)). 가중치는 데이터셋마다 새로 뽑는다."""
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
        hidden, num_causes, hp, apply_dropout=False, rng=rng, gen=gen, device=device
    )
    h = causes @ layer0_w.T + _default_bias(hidden, num_causes, gen, device)
    outputs = []
    n_in = hidden
    for n_out in out_dims:
        w = _init_weight(n_out, n_in, hp, apply_dropout=True, rng=rng, gen=gen, device=device)
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
        y = torch.full_like(y, float(spec.IGNORE_INDEX))

    if spec.RANDOM_FEATURE_ROTATION:
        shift = int(rng.integers(0, k))
        x = x[:, (torch.arange(k, device=device) + shift) % k]
    return x, y[:, 0]


# ---------------------------------------------------------------------------
# 데이터 생성: GP prior (v1 priors/fast_gp.py)
# ---------------------------------------------------------------------------
def _sample_gp_group(
    seq_len: int, n_datasets: int, k: int, hp: dict, gen: torch.Generator, device
) -> tuple[torch.Tensor, torch.Tensor]:
    """RBF 커널 GP에서 데이터셋 n_datasets개. x는 표준정규분포(sampling='normal')."""
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
# 표 데이터 처리 (v1 priors/flexible_categorical.py, tabpfn/utils.py)
# ---------------------------------------------------------------------------
def _masked_mean_std(x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    num = mask.sum(0).clamp(min=1)
    xm = torch.where(mask, x, torch.zeros_like(x))
    mean = xm.sum(0) / num
    var = (torch.where(mask, (x - mean) ** 2, torch.zeros_like(x))).sum(0) / (num - 1).clamp(min=1)
    return mean, var.sqrt()


def normalize_data(x: torch.Tensor, n_rows: int = -1) -> torch.Tensor:
    """행 방향(dim 0)으로 표준화하고 ±100으로 자른다. n_rows > 0이면 앞 n_rows행의 통계를 쓴다."""
    ref = x[:n_rows] if n_rows > 0 else x
    mean, std = _masked_mean_std(ref, ~torch.isnan(ref))
    return torch.clip((x - mean) / (std + 1e-6), -100, 100)


def remove_outliers(x: torch.Tensor, n_sigma: float = 4, n_rows: int = -1) -> torch.Tensor:
    """평균 ± 4σ 밖의 값을 log로 눌러 준다 (v1 remove_outliers)."""
    ref = x[:n_rows] if n_rows > 0 else x
    mean, std = _masked_mean_std(ref, ~torch.isnan(ref))
    lower, upper = mean - n_sigma * std, mean + n_sigma * std
    mask = (ref <= upper) & (ref >= lower) & ~torch.isnan(ref)
    mean, std = _masked_mean_std(ref, mask)
    lower, upper = mean - n_sigma * std, mean + n_sigma * std
    x = torch.maximum(-torch.log(1 + torch.abs(x)) + lower, x)
    return torch.minimum(torch.log(1 + torch.abs(x)) + upper, x)


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


def _make_compatible(
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
            y[:, b] = spec.IGNORE_INDEX
    return x, y


def _normalize_labels(y: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """클래스 번호를 0..n-1로 다시 매기고 무작위로 회전한다 (v1 normalize_labels)."""
    for b in range(y.shape[1]):
        valid = y[:, b] != spec.IGNORE_INDEX
        if not valid.any():
            continue
        vals = y[valid, b]
        ranks = (vals.unsqueeze(0) > vals.unique().unsqueeze(1)).sum(0).float()
        n_classes = int(ranks.max().item()) + 1
        shift = int(torch.randint(0, n_classes, (1,), generator=gen, device=y.device).item())
        y[valid, b] = (ranks + shift) % n_classes
    return y


class TabPFNPrior:
    """TabPFN v1 prior 샘플러. `sample_group`이 하이퍼파라미터를 공유하는 데이터셋 묶음을 만든다."""

    def __init__(
        self,
        seq_len: int = spec.SEQ_LEN,
        num_features: int = spec.MAX_FEATURES,
        max_classes: int = spec.MAX_CLASSES,
        group_size: int = spec.DATASETS_PER_PRIOR_GROUP,
        device: torch.device | str = "cpu",
        seed: int = 0,
    ) -> None:
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
        rng, gen, dev = self.rng, self.gen, self.device
        T, B = self.seq_len, self.group_size
        hp = sample_group_hyperparameters(rng, self.max_classes, self.num_features)
        k = hp["num_features_used"]

        # 1) GP 또는 MLP prior로 연속값 데이터 (T, B, k), (T, B)
        if hp["use_mlp"]:
            xs, ys = zip(
                *(_sample_mlp_dataset(T, k, hp, rng, gen, dev) for _ in range(B)), strict=True
            )
            x, y = torch.stack(xs, 1), torch.stack(ys, 1)
        else:
            x, y = _sample_gp_group(T, B, k, hp, gen, dev)

        # 2) 일부 열을 범주형으로 (그룹의 20%)
        if rng.random() < spec.CATEGORICAL_FEATURE_P:
            p = rng.random()
            for col in range(k):
                n_unique = max(round(rng.gamma(1, 10)), 2)
                if rng.random() < p:
                    x[:, :, col] = multiclass_rank(
                        x[:, :, col], n_unique, spec.CATEGORICAL_ORDERED_P, rng, gen
                    )

        # 3) 이상치 누르기 + 표준화, target을 클래스로
        x = normalize_data(remove_outliers(x))
        y = normalize_data(y)
        y = multiclass_rank(y, hp["num_classes"], spec.OUTPUT_MULTICLASS_ORDERED_P, rng, gen)

        # 4) feature 수 보정(100/k배)과 0 패딩
        if spec.NORMALIZE_BY_USED_FEATURES:
            x = x / (k / self.num_features)
        x = torch.cat([x, torch.zeros(T, B, self.num_features - k, device=dev)], dim=-1)

        # 5) 학습/테스트 행 클래스 맞추기, 라벨 정리
        if spec.CHECK_IS_COMPATIBLE:
            x, y = _make_compatible(x, y, single_eval_pos, gen)
        y = _normalize_labels(y, gen)
        return x.transpose(0, 1).contiguous(), y.transpose(0, 1).contiguous(), hp


def _derive_seed(*keys: int) -> int:
    """정수 여러 개를 섞어 64비트 seed 하나로 만든다. 곱하고 더하는 방식과 달리 겹치지 않는다."""
    return int(np.random.SeedSequence(keys).generate_state(1, np.uint64)[0])


class TabPFNPriorDataset(IterableDataset):
    """forward batch를 끝없이 내보낸다. `DataLoader(batch_size=None)`로 쓴다.

    forward batch 하나는 prior 그룹 `groups_per_batch`개를 이어 붙인 것이다. v1은 그룹 하나
    (데이터셋 8개)가 forward batch 하나이므로 기본값 1이 v1과 같다. batch마다 single_eval_pos를
    Uniform{1, ..., max_eval_pos-1}에서 뽑는다. v1은 0부터 뽑지만, 학습 행이 0개면 데이터셋 전체가
    무시되어(-100) gradient가 0이므로 빼도 학습은 같다. 0이 섞이면 그 GPU에서만 attention 가중치가
    쓰이지 않아 DDP의 gradient 동기화가 어긋난다(8×H100 medium 실행에서 NCCL timeout).
    `seed`(정수 또는 정수 tuple, 예: (config seed, 시작 step, rank))와 worker 번호를
    `SeedSequence`로 섞어 seed를 만든다. 그래서 worker·GPU·resume 시점마다 서로 다른 데이터셋이
    나오고, 정수 seed를 직접 쓰는 검증용 `TabPFNPrior`와도 겹치지 않는다.
    """

    def __init__(
        self,
        seq_len: int = spec.SEQ_LEN,
        max_eval_pos: int = spec.MAX_EVAL_POS,
        num_features: int = spec.MAX_FEATURES,
        max_classes: int = spec.MAX_CLASSES,
        group_size: int = spec.DATASETS_PER_PRIOR_GROUP,
        groups_per_batch: int = 1,
        seed: int | tuple[int, ...] = 0,
    ) -> None:
        super().__init__()
        self.kwargs = dict(
            seq_len=seq_len,
            num_features=num_features,
            max_classes=max_classes,
            group_size=group_size,
        )
        self.max_eval_pos = max_eval_pos
        self.groups_per_batch = groups_per_batch
        self.seed = seed

    def __iter__(self):
        info = get_worker_info()
        worker_id = info.id if info is not None else 0
        if info is not None:
            # worker가 많을 때 각자 CPU 스레드를 여러 개 쓰면 학습 프로세스까지 느려진다.
            torch.set_num_threads(1)
        keys = (self.seed,) if isinstance(self.seed, int) else tuple(self.seed)
        prior = TabPFNPrior(**self.kwargs, seed=_derive_seed(*keys, worker_id, 0))
        rng = np.random.default_rng(_derive_seed(*keys, worker_id, 1))  # single_eval_pos용
        while True:
            single_eval_pos = int(rng.integers(1, self.max_eval_pos))
            groups = [prior.sample_group(single_eval_pos) for _ in range(self.groups_per_batch)]
            x = torch.cat([g[0] for g in groups])
            y = torch.cat([g[1] for g in groups])
            yield x, y, single_eval_pos
