"""Muon + Adam (issue #28, 학습 옵션 `optimizer="muon"`).

Muon(Jordan et al. 2024)은 2차원 가중치 행렬의 gradient momentum을 Newton–Schulz 반복으로 직교화해
갱신한다. TabICLv2(arXiv 2602.11139), EXAONE Tabular, Mitra-v2가 AdamW 대신(또는 같이) 쓴다.
행렬이 아닌 파라미터(embedding, bias, norm, 출력층)는 Muon 논문 권고대로 Adam이 맡는다.

- Muon이 맡는 것: 반복 블록(`model.repeated_blocks()`) 안 `nn.Linear`의 weight 가운데 두 차원이
  모두 1보다 큰 것. 학습하는 토큰(TabPFN-3의 inducing·CLS)과 스칼라를 펴는 층(입력 1개, QASSMax의
  `base_mlp.0`)은 행렬 변환이 아니어서 Adam으로 둔다. qkv는 한 행렬(3d × d)로 묶인 그대로 직교화한다
- lr: `adjust_lr_fn="match_rms_adamw"`(Moonshot, arXiv 2502.16982). 갱신 크기를 AdamW의 RMS에
  맞춰 Adam과 같은 lr·warmup·cosine을 그대로 쓴다. 그래서 두 묶음이 기준 lr 하나를 함께 쓴다
- Muon 계산은 PyTorch의 `torch.optim.Muon`(2.9부터, Newton–Schulz 5번·bf16) 그대로다. momentum 0.95,
  Nesterov. weight decay는 Muon 묶음에만 있고 기본 0이다(지금 Adam 학습에 weight decay가 없어서)

`MuonAdam`은 두 optimizer를 하나로 보이게 묶는다. param_groups는 두 optimizer의 group 그대로(Muon,
Adam 순서)라서 scheduler(`LambdaLR`)와 lr 바꾸기(`_set_base_lr`)가 두 묶음을 함께 바꾼다. DDP에서는
모든 GPU가 평균 낸 같은 gradient로 같은 계산을 한다.
"""

from __future__ import annotations

import torch
from torch import nn

MUON_MOMENTUM = 0.95


def muon_parameters(model: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """(Muon이 맡을 파라미터, Adam이 맡을 파라미터). 순서는 `model.parameters()` 순서다."""
    blocks = model.repeated_blocks() if hasattr(model, "repeated_blocks") else []
    matrices = {
        id(module.weight)
        for block in blocks
        for module in block.modules()
        if isinstance(module, nn.Linear) and min(module.weight.shape) > 1
    }
    params = list(model.parameters())
    return [p for p in params if id(p) in matrices], [p for p in params if id(p) not in matrices]


class MuonAdam(torch.optim.Optimizer):
    """블록 안 가중치 행렬은 Muon, 나머지는 Adam. lr은 두 묶음이 같은 값을 쓴다."""

    def __init__(
        self,
        model: nn.Module,
        lr: float,
        *,
        weight_decay: float = 0.0,
        fused_adam: bool = False,
    ) -> None:
        muon, adam = muon_parameters(model)
        if not muon or not adam:
            raise ValueError(
                f"{type(model).__name__}에서 Muon·Adam 묶음을 나눌 수 없다"
                f" (Muon {len(muon)}개, Adam {len(adam)}개, `repeated_blocks()`가 있어야 한다)"
            )
        self.muon = torch.optim.Muon(
            muon,
            lr=lr,
            weight_decay=weight_decay,
            momentum=MUON_MOMENTUM,
            nesterov=True,
            adjust_lr_fn="match_rms_adamw",
        )
        self.adam = torch.optim.Adam(adam, lr=lr, **({"fused": True} if fused_adam else {}))
        super().__init__([{"params": muon}, {"params": adam}], {"lr": lr})
        # 두 optimizer의 group을 그대로 쓴다 (scheduler가 바꾼 lr을 두 optimizer가 바로 본다)
        self.param_groups = [*self.muon.param_groups, *self.adam.param_groups]

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
        self.muon.load_state_dict(state_dict["muon"])
        self.adam.load_state_dict(state_dict["adam"])
        # load_state_dict는 group dict를 새로 만든다. 묶음 목록을 다시 맞춘다
        self.param_groups = [*self.muon.param_groups, *self.adam.param_groups]
