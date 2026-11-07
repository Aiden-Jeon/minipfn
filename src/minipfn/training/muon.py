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

`MuonAdam`은 두 optimizer를 하나로 보이게 묶는다. param_groups는 두 optimizer의 group을 이어 붙인
것(Muon, Adam 순서)이라서 scheduler(`LambdaLR`)와 lr 바꾸기(`_set_base_lr`)가 두 묶음을 함께 바꾼다.
DDP에서는 모든 GPU가 평균 낸 같은 gradient로 같은 계산을 한다.
"""

from __future__ import annotations

import math

import torch
from torch import nn

MUON_MOMENTUM = 0.95
# Newton–Schulz 5차 반복의 계수와 횟수 (Keller Jordan, `torch.optim.Muon`과 같은 값)
NS_COEFFICIENTS = (3.4445, -4.7750, 2.0315)
NS_STEPS = 5
NS_EPS = 1e-7


def muon_parameters(model: nn.Module) -> tuple[list[tuple[nn.Parameter, int]], list[nn.Parameter]]:
    """([(Muon이 맡을 파라미터, 나눌 조각 수)], Adam이 맡을 파라미터). 순서는 `model.parameters()`
    순서다. 조각 수는 q·k·v를 묶은 weight가 3, 나머지는 1이다."""
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
    muon = [(p, splits[id(p)]) for p in params if id(p) in splits]
    return muon, [p for p in params if id(p) not in splits]


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


class Muon(torch.optim.Optimizer):
    """`torch.optim.Muon`(Nesterov, match_rms_adamw)과 같은 계산. group의 `split`이 1보다 크면
    weight를 dim 0으로 그만큼 나눠 조각마다 직교화하고 lr도 조각 크기로 보정한다."""

    def __init__(self, params, lr: float, *, weight_decay: float = 0.0, split: int = 1) -> None:
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "momentum": MUON_MOMENTUM,
            "split": split,
        }
        super().__init__(params, defaults)
        for group in self.param_groups:
            for p in group["params"]:
                if p.ndim != 2 or p.shape[0] % group["split"]:
                    raise ValueError(f"Muon은 {group['split']}조각으로 나눌 2차원 행렬만 받는다")

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError("Muon은 closure를 받지 않는다")
        for group in self.param_groups:
            lr, momentum, split = group["lr"], group["momentum"], group["split"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p.grad)
                buf = state["momentum_buffer"]
                buf.lerp_(p.grad, 1 - momentum)
                update = p.grad.lerp(buf, momentum)  # Nesterov
                rows, cols = p.shape[0] // split, p.shape[1]
                if split == 1:
                    update = _newton_schulz(update)
                else:
                    update = torch.cat([_newton_schulz(u) for u in update.split(rows)])
                p.mul_(1 - lr * group["weight_decay"])
                p.add_(update, alpha=-(lr * (0.2 * math.sqrt(max(rows, cols)))))


class MuonAdam(torch.optim.Optimizer):
    """블록 안 가중치 행렬은 Muon, 나머지는 Adam. lr은 두 묶음이 같은 값을 쓴다.

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
    ) -> None:
        muon, adam = muon_parameters(model)
        if not muon or not adam:
            raise ValueError(
                f"{type(model).__name__}에서 Muon·Adam 묶음을 나눌 수 없다"
                f" (Muon {len(muon)}개, Adam {len(adam)}개, `repeated_blocks()`가 있어야 한다)"
            )
        groups = [
            {"params": [p for p, s in muon if s == split], "split": split}
            for split in sorted({s for _, s in muon})
        ]
        self.muon = Muon(groups, lr=lr, weight_decay=weight_decay)
        self.adam = torch.optim.Adam(adam, lr=lr, **(adam_kwargs or {}))
        self.defaults = {"lr": lr}

    @property
    def param_groups(self) -> list[dict]:
        # 두 optimizer의 group 그대로 (scheduler가 바꾼 lr을 두 optimizer가 바로 본다)
        return [*self.muon.param_groups, *self.adam.param_groups]

    @property
    def state(self) -> dict:
        return {**self.muon.state, **self.adam.state}

    def add_param_group(self, param_group: dict) -> None:
        raise NotImplementedError("MuonAdam은 만들 때 모델에서 묶음을 정한다")

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError("MuonAdam은 closure를 받지 않는다")
        self.muon.step()
        self.adam.step()

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adam.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict:
        return {"muon": self.muon.state_dict(), "adam": self.adam.state_dict()}

    def load_state_dict(self, state_dict: dict) -> None:
        if set(state_dict) != {"muon", "adam"}:
            raise ValueError("MuonAdam의 optimizer 상태가 아니다 (Adam으로 학습한 checkpoint인지)")
        # load_state_dict는 group 설정도 저장한 값으로 되돌린다. Muon 쪽(weight decay 등)은 지금
        # 설정을 쓴다. lr은 학습 루프가 다시 맞춘다(`_set_base_lr`). Adam 쪽은 Adam 학습과 같이 둔다
        settings = [
            {k: v for k, v in group.items() if k not in ("params", "lr", "initial_lr")}
            for group in self.muon.param_groups
        ]
        self.muon.load_state_dict(state_dict["muon"])
        for group, kept in zip(self.muon.param_groups, settings, strict=True):
            group.update(kept)
        self.adam.load_state_dict(state_dict["adam"])
