"""schedule-free AdamW (학습 축). Defazio et al., "The Road Less Scheduled" (NeurIPS 2024)의
방법이고 nanoTabPFN speedrun(arXiv 2606.03681)이 Adam 쪽 파라미터에 쓴다
(`TrainConfig.schedule_free`).

학습률을 cosine처럼 줄이는 대신 평균을 들고 간다. 파라미터마다 세 가지가 있다.

- z: SGD처럼 gradient 방향으로 가는 점 (optimizer 상태)
- x: z의 가중 평균. 검증·저장하는 가중치다
- y = (1 − β1) z + β1 x: gradient를 계산하는 점. 학습 중 모델 파라미터에는 y가 들어 있다

step k에서 c_k = w_k / Σ w (w_k = lr_k², lr_k는 warmup을 거친 lr)로

    y ← (1 − c_k) y + c_k z + lr_k (β1 (1 − c_k) − 1) g̃
    z ← z − lr_k g̃                       (g̃ = g / (√v̂ + ε), v는 Adam의 제곱 평균)

이다. step 수 k와 Σ w는 파라미터마다 센다: gradient가 없는 step(multitask에서 다른 task의 정답
encoder·출력층)은 그 파라미터에게 step이 아니다. 원본 패키지는 param group마다 세서, 한 step 걸러
gradient를 받는 파라미터의 bias correction과 평균이 어긋난다. 늘 gradient를 받는 파라미터에서는
원본과 같다. 검증·저장에는 x를 쓴다(`average`). 학습 checkpoint에도 x를 저장해서 평가 스크립트가
그대로 읽고, 이어 학습할 때 x와 z로 y를 다시 만든다(`average_to_train_point`).
"""

from __future__ import annotations

import torch


class ScheduleFreeAdamW(torch.optim.Optimizer):
    """schedule-free AdamW. lr은 param group의 `lr`을 쓴다(학습 루프의 LambdaLR이 warmup만 곱한다).
    모델 파라미터는 늘 y다. 검증·저장할 평균 x는 `average(p)`로 따로 계산한다(학습 가중치를 바꿨다
    되돌리면 GPU마다 반올림이 달라질 수 있어서). x를 저장한 checkpoint에서 이어 학습할 때는 x를
    파라미터에 읽은 뒤 `average_to_train_point()`로 y를 만든다."""

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)
        self._beta1 = {p: group["betas"][0] for group in self.param_groups for p in group["params"]}

    @torch.no_grad()
    def average(self, p: torch.Tensor) -> torch.Tensor:
        """파라미터 p(= y)의 평균 x = (y − (1 − β1) z) / β1. step 전이면 p 그대로."""
        if "z" not in self.state[p]:
            return p.detach().clone()
        return torch.lerp(p, self.state[p]["z"], 1 - 1 / self._beta1[p])

    @torch.no_grad()
    def average_to_train_point(self) -> None:
        """파라미터에 평균 x가 들어 있을 때 학습 점 y = (1 − β1) z + β1 x로 바꾼다 (이어 학습)."""
        for group in self.param_groups:
            for p in group["params"]:
                if "z" in self.state[p]:
                    p.lerp_(self.state[p]["z"], 1 - group["betas"][0])

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError("ScheduleFreeAdamW는 closure를 받지 않는다")
        for group in self.param_groups:
            lr = group["lr"]
            # 파라미터마다 step 수·lr 최댓값·가중치 합을 갱신하고, 값이 같은 것끼리 한꺼번에
            # 계산한다 (multitask면 공유 파라미터와 task별 파라미터 두 묶음)
            buckets: dict[tuple, list[torch.Tensor]] = {}
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "z" not in state:
                    state.update(
                        z=p.detach().clone(),
                        exp_avg_sq=torch.zeros_like(p),
                        k=0,
                        lr_max=0.0,
                        weight_sum=0.0,
                    )
                state["k"] += 1
                state["lr_max"] = max(state["lr_max"], lr)
                state["weight_sum"] += state["lr_max"] ** 2
                key = (state["k"], state["lr_max"], state["weight_sum"])
                buckets.setdefault(key, []).append(p)
            for (k, lr_max, weight_sum), params in buckets.items():
                self._update(group, params, lr, k, lr_max**2 / weight_sum if weight_sum else 0.0)
        return None

    def _update(self, group: dict, params: list, lr: float, k: int, ckp1: float) -> None:
        # step k(그 파라미터들의 k)의 갱신. ckp1 = w_k / Σ w
        beta1, beta2 = group["betas"]
        bias_correction2 = 1 - beta2**k
        grads = [p.grad for p in params]
        z = [self.state[p]["z"] for p in params]
        v = [self.state[p]["exp_avg_sq"] for p in params]
        torch._foreach_mul_(v, beta2)
        torch._foreach_addcmul_(v, grads, grads, value=1 - beta2)
        denom = torch._foreach_div(v, bias_correction2)
        torch._foreach_sqrt_(denom)
        torch._foreach_add_(denom, group["eps"])
        normed = torch._foreach_div(grads, denom)
        if group["weight_decay"]:
            torch._foreach_add_(normed, params, alpha=group["weight_decay"])
        torch._foreach_lerp_(params, z, ckp1)
        torch._foreach_add_(params, normed, alpha=lr * (beta1 * (1 - ckp1) - 1))
        torch._foreach_add_(z, normed, alpha=-lr)


def warmup_factor(step: int, warmup_steps: int) -> float:
    """schedule-free의 학습률 배율: warmup 동안 선형으로 올리고 그 뒤로는 1 (cosine 없음)."""
    return min(1.0, step / max(1, warmup_steps))
