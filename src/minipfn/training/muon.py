"""Muon + Adam (issue #28, 학습 옵션 `optimizer="muon"`).

Muon(Jordan et al. 2024)은 2차원 가중치 행렬의 gradient momentum을 Newton–Schulz 반복으로 직교화해
갱신한다. TabICLv2(arXiv 2602.11139), EXAONE Tabular, Mitra-v2가 AdamW 대신(또는 같이) 쓴다.
행렬이 아닌 파라미터(embedding, bias, norm, 출력층)는 Muon 논문 권고대로 Adam이 맡는다.

- Muon이 맡는 것: 반복 블록(`model.repeated_blocks()`) 안의 가중치 행렬. `nn.Linear`의 weight 가운데
  두 차원이 모두 1보다 큰 것과 `nn.MultiheadAttention`의 입력 projection이다. 학습하는 토큰
  (TabPFN-3의 inducing·CLS)과 스칼라를 펴는 층(입력 1개, QASSMax의 `base_mlp.0`)은 행렬 변환이
  아니어서 Adam으로 둔다
- q·k·v를 한 weight로 묶은 것(`qkv` Linear, (3D, D), 그리고 `in_proj_weight`)은 dim 0으로 셋으로
  나눠 따로 직교화한다(Kimi/Moonshot과 같다). 묶은 채로 하면 셋이 한 직교 행렬을 나눠 써서 각자의
  갱신이 작아지고, lr 보정도 (D, D)가 아니라 (3D, D) 크기로 된다
- lr: Moonshot의 "match_rms_adamw"(arXiv 2502.16982, `0.2·sqrt(max(A, B))`, 나눈 행렬이면 나눈 한
  조각의 크기). 갱신 크기를 AdamW의 RMS에 맞춰 Adam과 같은 lr·warmup·cosine을 그대로 쓴다. 그래서 두
  묶음이 기준 lr 하나를 함께 쓴다
- Muon 계산은 `torch.optim.Muon`(PyTorch 2.11)과 같다(Newton–Schulz 5번·bf16, momentum 0.95,
  Nesterov). 나누지 않는 행렬은 비트 단위로 같다(테스트). 나누기만 더하려고 step을 여기 둔다.
  weight decay는 Muon 묶음에만 있고 기본 0이다(지금 Adam 학습에 weight decay가 없어서)

모든 파라미터 Muon(`scope="all"`, TabICLv2 학습 스크립트의 `--muon True`): 블록 밖 파라미터와
2차원이 아닌 파라미터도 Muon이 맡는다. 2차원이 아니면 TabICL `Muon`처럼 (첫 차원, 나머지 곱)
행렬로 펴서 직교화한다(1차원 (n,)은 (n, 1) — 직교화하면 단위 벡터, 0차원은 (1, 1)). lr 보정도 편
행렬 크기로 한다. 그러면 Adam 묶음이 없고 weight decay가 모든 파라미터에 걸린다. q·k·v 묶은
weight는 지금처럼 셋으로 나눈다(TabICL은 나누지 않는다).

`MuonAdam`은 두 optimizer를 하나로 보이게 묶는다. param_groups는 두 optimizer의 group을 이어 붙인
것(Muon, Adam 순서)이라서 scheduler(`LambdaLR`)와 lr 바꾸기(`_set_base_lr`)가 두 묶음을 함께 바꾼다.
DDP에서는 모든 GPU가 평균 낸 같은 gradient로 같은 계산을 한다.

분산 Muon(issue #56, 학습 옵션 `muon_distributed`): 기본은 모든 GPU가 모든 행렬의 Newton–Schulz를
똑같이 계산한다. `process_group`을 주면 그 group의 GPU끼리 행렬 조각(q·k·v를 나눈 조각 단위)을
나눠 맡아(`shard_pieces`) 자기 몫만 직교화하고, 결과(bf16)를 all-gather로 모은다(Moonlight·Kimi의
분산 Muon과 같은 생각). momentum buffer는 지금처럼 모든 GPU에 다 둔다(checkpoint 형식 그대로라서
나누기 전의 checkpoint에서 이어 학습할 수 있다). 받은 조각은 1 GPU에서 계산한 갱신과 값·stride가
같은 tensor로 되돌려 더하므로, 가중치는 1 GPU 계산과 비트 단위로 같다(테스트)
"""

from __future__ import annotations

import math
import os

import torch
import torch.distributed as dist
from torch import nn

MUON_MOMENTUM = 0.95
# Muon이 맡는 파라미터 (`muon_parameters`): 블록 안 행렬만, 또는 모든 파라미터
MUON_SCOPES = ("blocks", "all")
# Newton–Schulz 5차 반복의 계수와 횟수 (Keller Jordan, `torch.optim.Muon`과 같은 값)
NS_COEFFICIENTS = (3.4445, -4.7750, 2.0315)
NS_STEPS = 5
NS_EPS = 1e-7


def muon_parameters(
    model: nn.Module, scope: str = "blocks"
) -> tuple[list[tuple[nn.Parameter, int]], list[nn.Parameter]]:
    """([(Muon이 맡을 파라미터, 나눌 조각 수)], Adam이 맡을 파라미터). 순서는 `model.parameters()`
    순서다. 조각 수는 q·k·v를 묶은 weight가 3, 나머지는 1이다. `scope="all"`이면 모든 파라미터가
    Muon이다(Adam 쪽은 빈 목록)."""
    if scope not in MUON_SCOPES:
        raise ValueError(f"Muon 범위는 {MUON_SCOPES} 가운데 하나다: {scope}")
    blocks = model.repeated_blocks() if hasattr(model, "repeated_blocks") else []
    splits: dict[int, int] = {}
    for block in blocks:
        for name, module in block.named_modules():
            if isinstance(module, nn.Linear) and min(module.weight.shape) > 1:
                qkv = name.rsplit(".", 1)[-1] == "qkv" and module.out_features % 3 == 0
                splits[id(module.weight)] = 3 if qkv else 1
            elif isinstance(module, nn.MultiheadAttention):
                if module.in_proj_weight is not None:
                    splits[id(module.in_proj_weight)] = 3
                else:  # key·value 차원이 query와 다르면 q·k·v weight가 따로 있다
                    for w in (module.q_proj_weight, module.k_proj_weight, module.v_proj_weight):
                        splits[id(w)] = 1
                # out_proj는 nn.Linear라서 위에서 잡힌다
    params = list(model.parameters())
    if scope == "all":
        return [(p, splits.get(id(p), 1)) for p in params], []
    muon = [(p, splits[id(p)]) for p in params if id(p) in splits]
    return muon, [p for p in params if id(p) not in splits]


def _matrix(t: torch.Tensor) -> torch.Tensor:
    """직교화할 2차원 view: (첫 차원, 나머지 곱). 0차원은 (1, 1) (TabICL `Muon`의
    `reshape(len(g), -1)`과 같다). 파라미터를 제자리에서 바꾸므로 view여야 한다."""
    if t.ndim == 2:
        return t
    return t.view(1, 1) if t.ndim == 0 else t.view(t.shape[0], -1)


def _newton_schulz(grad: torch.Tensor) -> torch.Tensor:
    """2차원 행렬을 직교 행렬 쪽으로 (`torch.optim._muon._zeropower_via_newtonschulz`와 같다)."""
    a, b, c = NS_COEFFICIENTS
    x = grad.bfloat16()
    tall = grad.size(0) > grad.size(1)
    if tall:
        x = x.T
    x.div_(x.norm().clamp(min=NS_EPS))  # spectral norm을 1 이하로
    for _ in range(NS_STEPS):
        gram = x @ x.T
        x = torch.addmm(x, torch.addmm(gram, gram, gram, beta=b, alpha=c), x, beta=a)
    return x.T if tall else x


def _nesterov(grad: torch.Tensor, buf: torch.Tensor, group: dict) -> torch.Tensor:
    """직교화할 Nesterov 갱신."""
    return grad.lerp(buf, group["momentum"])


def _apply(p: torch.Tensor, update: torch.Tensor, group: dict) -> None:
    """weight decay와 직교화한 갱신을 더한다. lr은 조각 크기로 보정한다(match_rms_adamw).
    2차원이 아닌 파라미터는 편 행렬(`_matrix`)로 본다."""
    p = _matrix(p)
    rows, cols = p.shape[0] // group["split"], p.shape[1]
    lr = group["lr"]
    p.mul_(1 - lr * group["weight_decay"])
    p.add_(update, alpha=-(lr * (0.2 * math.sqrt(max(rows, cols)))))


def _piece_cost(rows: int, cols: int) -> int:
    """조각 하나의 Newton–Schulz 곱셈 양 (반복 한 번: 짧은 변 m, 긴 변 n이면 m²·(2n + m))."""
    m, n = min(rows, cols), max(rows, cols)
    return m * m * (2 * n + m)


def shard_pieces(shapes: list[tuple[int, int]], n_ranks: int) -> list[int]:
    """조각(행, 열)마다 맡을 rank. 곱셈 양이 큰 조각부터 지금까지 몫이 가장 적은 rank에 준다(LPT,
    같으면 번호가 작은 rank). 정수만 쓰므로 모든 GPU가 같은 답을 낸다."""
    loads, owner = [0] * n_ranks, [0] * len(shapes)
    for i in sorted(range(len(shapes)), key=lambda i: (-_piece_cost(*shapes[i]), i)):
        rank = min(range(n_ranks), key=lambda r: (loads[r], r))
        owner[i] = rank
        loads[rank] += _piece_cost(*shapes[i])
    return owner


def shard_group(scope: str):
    """분산 Muon이 행렬을 나눌 process group. "all"은 모든 GPU, "node"는 노드 안 GPU끼리(노드마다
    group 하나, 노드 사이에는 지금처럼 같은 계산을 한다). 모든 GPU가 같이 불러야 한다. GPU가 하나면
    None(나누지 않는다)."""
    if scope not in ("node", "all"):
        raise ValueError(f"분산 Muon 범위는 node 또는 all이다: {scope}")
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return None
    if scope == "all":
        return dist.group.WORLD
    # torchrun이 넣는 값. 없으면(다른 launcher) 이 노드의 GPU 수로 본다. 전체 GPU 수로 두면
    # node가 조용히 all이 되어 노드 사이로 all-gather가 나간다
    local = int(os.environ.get("LOCAL_WORLD_SIZE") or torch.cuda.device_count() or 1)
    if dist.get_world_size() % local:
        raise ValueError(f"GPU {dist.get_world_size()}개를 노드당 {local}개로 나눌 수 없다")
    if local == dist.get_world_size():
        return dist.group.WORLD
    group, _ = dist.new_subgroups(group_size=local)
    return group


class Muon(torch.optim.Optimizer):
    """`torch.optim.Muon`(Nesterov, match_rms_adamw)과 같은 계산. group의 `split`이 1보다 크면
    weight를 dim 0으로 그만큼 나눠 조각마다 직교화하고 lr도 조각 크기로 보정한다.

    `process_group`을 주면 그 group의 GPU끼리 조각을 나눠 직교화한다(모듈 주석, 분산 Muon)."""

    def __init__(
        self,
        params,
        lr: float,
        *,
        weight_decay: float = 0.0,
        split: int = 1,
        momentum: float = MUON_MOMENTUM,
        any_shape: bool = False,
        process_group=None,
    ) -> None:
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "split": split,
        }
        super().__init__(params, defaults)
        for group in self.param_groups:
            for p in group["params"]:
                # any_shape면 2차원이 아닌 것도 편 행렬로 받는다 (`_matrix`)
                shape = _matrix(p).shape if any_shape else p.shape
                if len(shape) != 2 or shape[0] % group["split"]:
                    raise ValueError(f"Muon은 {group['split']}조각으로 나눌 2차원 행렬만 받는다")
        if process_group is not None and dist.get_world_size(process_group) == 1:
            process_group = None  # 나눌 GPU가 없다
        self.process_group = process_group
        # (group 크기, gradient가 있는 파라미터 묶음)마다 조각 주인
        self._owners: dict[tuple, list[int]] = {}

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError("Muon은 closure를 받지 않는다")
        if self.process_group is not None:
            self._step_sharded()
            return
        for group in self.param_groups:
            split = group["split"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                buf = self._momentum(p, group)
                update = _matrix(_nesterov(p.grad, buf, group))
                rows = update.shape[0] // split
                if split == 1:
                    update = _newton_schulz(update)
                else:
                    update = torch.cat([_newton_schulz(u) for u in update.split(rows)])
                _apply(p, update, group)

    def _momentum(self, p: torch.Tensor, group: dict) -> torch.Tensor:
        """momentum buffer를 이번 gradient로 갱신하고 돌려준다 (두 경로가 같이 쓴다)."""
        state = self.state[p]
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros_like(p.grad)
        buf = state["momentum_buffer"]
        buf.lerp_(p.grad, 1 - group["momentum"])
        return buf

    def _step_sharded(self) -> None:
        """`step`과 같은 갱신을 group의 GPU끼리 조각을 나눠 계산한다.

        1. 모든 GPU가 momentum buffer를 다 갱신한다 (checkpoint 형식 그대로)
        2. 자기가 맡은 조각만 Nesterov 갱신을 직교화해 보낼 buffer에 이어 붙인다
        3. all-gather로 모든 조각을 받는다. 길이는 가장 많이 맡은 GPU에 맞춘다(나머지는 채움)
        4. 받은 조각을 1 GPU의 `_newton_schulz` 결과와 stride까지 같은 모양으로 되돌려 더한다.
           세로로 긴 조각은 직교화가 전치한 채로 계산하므로 그 계산된 쪽(가로로 긴 쪽)을 보낸다
        """
        group_ = self.process_group
        me, n_ranks = dist.get_rank(group_), dist.get_world_size(group_)
        active = [  # (파라미터, group, 조각 행 수) — gradient가 없으면 모든 GPU에서 없다(DDP)
            (p, g, _matrix(p).shape[0] // g["split"])
            for g in self.param_groups
            for p in g["params"]
            if p.grad is not None
        ]
        if not active:
            return
        key = (n_ranks, *(id(p) for p, _, _ in active))
        if key not in self._owners:
            shapes = [
                (rows, _matrix(p).shape[1]) for p, g, rows in active for _ in range(g["split"])
            ]
            self._owners[key] = shard_pieces(shapes, n_ranks)
        owner = self._owners[key]

        sends, sizes, k = [], [0] * n_ranks, 0
        for p, g, rows in active:
            split = g["split"]
            buf = _matrix(self._momentum(p, g))
            grad, cols = _matrix(p.grad), _matrix(p).shape[1]
            for j in range(split):
                if owner[k + j] != me:
                    continue
                # 맡은 조각의 행만 (행 단위 원소별 계산이라 전체를 계산해 자른 것과 같다)
                rows_j = slice(j * rows, (j + 1) * rows)
                orth = _newton_schulz(_nesterov(grad[rows_j], buf[rows_j], g))
                sends.append((orth.T if rows > cols else orth).reshape(-1))
            for j in range(split):
                sizes[owner[k + j]] += rows * cols
            k += split

        width = max(sizes)
        send = torch.zeros(width, dtype=torch.bfloat16, device=active[0][0].device)
        if sends:
            torch.cat(sends, out=send[: sizes[me]])
        received = torch.empty(n_ranks * width, dtype=send.dtype, device=send.device)
        dist.all_gather_into_tensor(received, send, group=group_)

        offsets, k = [r * width for r in range(n_ranks)], 0
        for p, g, rows in active:
            cols, parts = _matrix(p).shape[1], []
            for j in range(g["split"]):
                r = owner[k + j]
                flat = received[offsets[r] : offsets[r] + rows * cols]
                offsets[r] += rows * cols
                parts.append(flat.view(cols, rows).T if rows > cols else flat.view(rows, cols))
            k += g["split"]
            _apply(p, parts[0] if g["split"] == 1 else torch.cat(parts), g)


class MuonAdam(torch.optim.Optimizer):
    """블록 안 가중치 행렬은 Muon, 나머지는 Adam. lr은 두 묶음이 같은 값을 쓴다. `process_group`을
    주면 Muon 쪽 직교화를 그 group의 GPU끼리 나눈다(분산 Muon, 모듈 주석). `scope="all"`이면 모든
    파라미터가 Muon이고 Adam은 없다(`self.adam`이 None).

    묶음과 상태는 안의 두 optimizer가 든다. `torch.optim.Optimizer.__init__`은 부르지 않는다(부르면
    쓰지 않는 묶음·상태가 따로 생긴다). scheduler가 Optimizer인지 보므로 상속만 한다.
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float,
        *,
        weight_decay: float = 0.0,
        adam_kwargs: dict | None = None,
        process_group=None,
        scope: str = "blocks",
        momentum: float = MUON_MOMENTUM,
    ) -> None:
        muon, adam = muon_parameters(model, scope)
        if not muon or (scope == "blocks" and not adam):
            raise ValueError(
                f"{type(model).__name__}에서 Muon·Adam 묶음을 나눌 수 없다"
                f" (Muon {len(muon)}개, Adam {len(adam)}개, `repeated_blocks()`가 있어야 한다)"
            )
        groups = [
            {"params": [p for p, s in muon if s == split], "split": split}
            for split in sorted({s for _, s in muon})
        ]
        self.muon = Muon(
            groups,
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            any_shape=scope == "all",
            process_group=process_group,
        )
        self.adam = torch.optim.Adam(adam, lr=lr, **(adam_kwargs or {})) if adam else None
        self.defaults = {"lr": lr}

    @property
    def param_groups(self) -> list[dict]:
        # 두 optimizer의 group 그대로 (scheduler가 바꾼 lr을 두 optimizer가 바로 본다)
        return [*self.muon.param_groups, *(self.adam.param_groups if self.adam else [])]

    @property
    def state(self) -> dict:
        return {**self.muon.state, **(self.adam.state if self.adam else {})}

    def add_param_group(self, param_group: dict) -> None:
        raise NotImplementedError("MuonAdam은 만들 때 모델에서 묶음을 정한다")

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError("MuonAdam은 closure를 받지 않는다")
        self.muon.step()
        if self.adam:
            self.adam.step()

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.muon.zero_grad(set_to_none=set_to_none)
        if self.adam:
            self.adam.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict:
        # 모든 파라미터 Muon이면 Adam 상태는 None (불러올 때 범위가 같은지 본다)
        return {
            "muon": self.muon.state_dict(),
            "adam": self.adam.state_dict() if self.adam else None,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        if set(state_dict) != {"muon", "adam"}:
            raise ValueError("MuonAdam의 optimizer 상태가 아니다 (Adam으로 학습한 checkpoint인지)")
        # load_state_dict는 group 설정도 저장한 값으로 되돌린다. Muon 쪽(weight decay 등)은 지금
        # 설정을 쓴다. lr은 학습 루프가 다시 맞춘다(`_set_base_lr`). Adam 쪽은 Adam 학습과 같이 둔다
        settings = [
            {k: v for k, v in group.items() if k not in ("params", "lr", "initial_lr")}
            for group in self.muon.param_groups
        ]
        if (state_dict["adam"] is None) != (self.adam is None):
            raise ValueError(
                "checkpoint와 Muon 범위(muon_scope)가 다르다 (원래 run의 범위로 이어 간다)"
            )
        self.muon.load_state_dict(state_dict["muon"])
        for group, kept in zip(self.muon.param_groups, settings, strict=True):
            group.update(kept)
        if self.adam:
            self.adam.load_state_dict(state_dict["adam"])
