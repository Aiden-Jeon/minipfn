"""클래스 다루기: 연속값 target을 클래스로, 학습·검증 행의 클래스 맞추기, 라벨 정리.

함수는 TabPFN v1 `flexible_categorical.py`(`multiclass_rank`, `make_compatible`,
`normalize_labels`)와 graph SCM prior(`train_rows_cover_classes`)의 것이고, 아래
후처리(`Transform`)가 그룹에 적용한다.
"""

from __future__ import annotations

import numpy as np
import torch

from minipfn.priors.base import IGNORE_INDEX
from minipfn.priors.pipeline import Group, Rng


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


def train_rows_cover_classes(
    y: torch.Tensor, n_train: int, rng: np.random.Generator
) -> torch.Tensor | None:
    """정답을 가린 행의 클래스가 모두 학습 행에도 있도록 행 순서를 정한다.

    클래스마다 한 행씩 먼저 학습 행에 넣고, 나머지 학습 행은 무작위로 채운다. 클래스가 2개보다
    적거나 학습 행 수보다 많으면 None(그 데이터셋은 버린다).
    """
    labels = y.long().numpy()
    classes = np.unique(labels)
    if len(classes) < 2 or len(classes) > n_train:
        return None
    first = np.array([rng.choice(np.flatnonzero(labels == c)) for c in classes])
    rest = rng.permutation(np.setdiff1d(np.arange(len(labels)), first))
    n_more = n_train - len(first)
    train = rng.permutation(np.concatenate([first, rest[:n_more]]))
    return torch.from_numpy(np.concatenate([train, rest[n_more:]]))


class ToClasses:
    """연속값 y를 클래스 `hp["num_classes"]`개로 (`multiclass_rank`, 경계 행은 그룹이 공유한다)."""

    def __init__(self, ordered_p: float) -> None:
        self.ordered_p = ordered_p  # 클래스 번호를 섞지 않고 둘 확률

    def __call__(self, group: Group, rng: Rng) -> Group:
        n_classes = group.hp["num_classes"]
        y = multiclass_rank(group.y, n_classes, self.ordered_p, rng.numpy, rng.torch)
        return Group(group.x, y, group.n_train, group.hp)


class MatchClasses:
    """학습·검증 행의 클래스 집합을 맞춘다 (`make_compatible`, 실패한 데이터셋은 target을 무시)."""

    def __call__(self, group: Group, rng: Rng) -> Group:
        x, y = make_compatible(group.x, group.y, group.n_train, rng.torch)
        return Group(x, y, group.n_train, group.hp)


class Relabel:
    """클래스 번호를 0..n-1로 다시 매기고 무작위로 회전한다 (`normalize_labels`)."""

    def __call__(self, group: Group, rng: Rng) -> Group:
        return Group(group.x, normalize_labels(group.y, rng.torch), group.n_train, group.hp)


class CoverClasses:
    """검증 행의 클래스가 모두 학습 행에도 나오도록 행 순서를 바꾼다 (`train_rows_cover_classes`).

    한 데이터셋이라도 맞출 수 없으면 그룹을 버린다(None).
    """

    def __call__(self, group: Group, rng: Rng) -> Group | None:
        xs, ys = [], []
        for d in range(group.y.shape[1]):
            order = train_rows_cover_classes(group.y[:, d], group.n_train, rng.numpy)
            if order is None:
                return None
            xs.append(group.x[order, d])
            ys.append(group.y[order, d])
        return Group(torch.stack(xs, 1), torch.stack(ys, 1), group.n_train, group.hp)
