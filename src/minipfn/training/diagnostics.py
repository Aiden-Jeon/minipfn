"""학습 진단 지표 (학습 축). 가중치·gradient·시간을 읽기만 하고 학습 계산은 바꾸지 않는다.

- 묶음별 norm (issue #65): 파라미터를 단계(모델이 정한다, `parameter_stage`)와 optimizer
  묶음(Muon/Adam)으로 나눠 가중치 norm과 clip 전 gradient norm을 낸다. multitask면 task 하나만
  쓰는 부품(정답 encoder, 출력층)은 `<단계>_<task>` 묶음이 따로 된다. attention logit 배율을 정하는
  작은 parameter(QK-norm gain, QASSMax MLP)도 `<단계>_qk_norm`·`<단계>_qassmax`로 따로 둔다. 긴
  학습에서 gradient가 이쪽으로 몰렸는데(#65 checkpoint 분석) 단계 전체 norm에 묻혀 보이지 않는다
- task gradient cosine (issue #64, 옵션 `grad_cosine`): multitask는 step마다 task를 번갈아 쓰므로
  기록 step 앞의 연속 4 step(분류·회귀·분류·회귀 또는 반대)의 gradient로 두 task가 같이 쓰는
  묶음마다 cosine을 잰다. 다른 task끼리는 1 step 떨어진 세 쌍의 평균, 같은 task끼리는 2 step
  떨어진 쌍이다. 같은 task의 값이 "batch가 달라서 생기는 차이"의 기준이고, 다른 task의 값이 그보다
  낮으면 두 task의 gradient가 공유 부분에서 다른 쪽을 향한다
- rank별 시간·표 크기 (issue #68, 옵션 `rank_timing`): GPU마다 step의 forward·backward 시간과 그
  step의 표 크기(행·학습 행·feature·셀 수)를 재서 기록 step마다 rank 0으로 모은다. backward
  시간에는 DDP all-reduce에서 가장 느린 GPU를 기다리는 시간이 들어 있다
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn

# attention logit 배율을 정하는 부품: 이름에 이 모듈이 들어 있으면 단계 안에서 묶음을 따로 둔다
ATTENTION_SCALE_PARTS: dict[str, str] = {
    "q_norm": "qk_norm",
    "k_norm": "qk_norm",
    "qassmax": "qassmax",
}


def parameter_groups(model: nn.Module, muon_ids: set[int] | None = None) -> dict[str, list]:
    """묶음 이름 → 파라미터 목록 (`model.named_parameters()` 순서).

    이름은 `<단계>[_qk_norm|_qassmax][_<task>][/muon|/adam]`이다. 단계는
    `model.parameter_stage(name)`(없으면 맨 위 모듈 이름)다. `_qk_norm`·`_qassmax`는 attention
    logit 배율 부품(`ATTENTION_SCALE_PARTS`), task는 multitask 모델에서 그 task만 쓰는 부품일 때,
    optimizer는 Muon을 쓸 때(`muon_ids`: Muon이 맡는 파라미터의 id)만 붙는다."""
    tasks = tuple(getattr(model, "tasks", ()))
    stage_of = getattr(model, "parameter_stage", None)
    groups: dict[str, list] = {}
    for name, p in model.named_parameters():
        group = stage_of(name) if stage_of is not None else name.split(".")[0]
        parts = name.split(".")
        scale = next((ATTENTION_SCALE_PARTS[q] for q in parts if q in ATTENTION_SCALE_PARTS), None)
        if scale is not None:
            group += f"_{scale}"
        if len(tasks) > 1:
            task = next((t for t in tasks if t in parts), None)
            if task is not None:
                group += f"_{task}"
        if muon_ids is not None:
            group += "/muon" if id(p) in muon_ids else "/adam"
        groups.setdefault(group, []).append(p)
    return groups


def shared_groups(groups: dict[str, list], tasks: tuple[str, ...]) -> dict[str, list]:
    """두 task가 같이 쓰는 묶음 (이름에 task가 붙지 않은 것)."""
    return {
        name: params
        for name, params in groups.items()
        if not any(name.split("/")[0].endswith(f"_{t}") for t in tasks)
    }


@torch.no_grad()
def group_norms(groups: dict[str, list], grads: bool = False) -> dict[str, torch.Tensor]:
    """묶음마다 가중치(`grads`면 gradient)의 L2 norm. GPU에 둔 채 돌려준다(읽는 것은 기록할 때).
    gradient가 하나도 없는 묶음(이 step에 쓰지 않은 task의 부품)은 빠진다."""
    out = {}
    for name, params in groups.items():
        tensors = [p.grad if grads else p for p in params]
        tensors = [t for t in tensors if t is not None]
        if tensors:
            norms = torch._foreach_norm([t.float() for t in tensors])
            out[name] = torch.linalg.vector_norm(torch.stack(norms))
    return out


class GroupNormWindow:
    """기록 window 동안의 묶음별 gradient norm(clip 전). step마다 더하고 기록할 때 평균·최대를 낸다.

    multitask면 task마다 따로 모은다(지표 `grad_norm/<task>/<묶음>`). 전체 gradient norm
    (`clip_grad_norm_`이 돌려준 clip 전 값)의 window 중앙값·최대도 낸다. 한 step 값인 `grad_norm`은
    튀는 step을 놓친다."""

    def __init__(self, groups: dict[str, list], multitask: bool) -> None:
        self.groups = groups
        self.multitask = multitask
        self.reset()

    def reset(self) -> None:
        self.sums: dict[tuple[str, str], list] = {}  # (task, 묶음) → [합, 최대, step 수]
        self.totals: list[torch.Tensor] = []

    def update(self, task: str, norms: dict[str, torch.Tensor], total: torch.Tensor) -> None:
        """gradient가 유한한 step만 부른다. `norms`는 `group_norms(..., grads=True)`."""
        for name, value in norms.items():
            entry = self.sums.get((task, name))
            if entry is None:
                self.sums[(task, name)] = [value, value, 1]
            else:
                entry[0] = entry[0] + value
                entry[1] = torch.maximum(entry[1], value)
                entry[2] += 1
        self.totals.append(total.detach().float())

    def metrics(self) -> dict[str, float]:
        out: dict[str, float] = {}
        if self.totals:
            totals = torch.stack(self.totals)
            out["grad_norm_median"] = totals.median().item()
            out["grad_norm_max"] = totals.max().item()
        if not self.sums:
            return out
        keys = list(self.sums)
        stacked = torch.stack(
            [torch.stack([self.sums[k][0] / self.sums[k][2], self.sums[k][1]]) for k in keys]
        ).tolist()  # 한 번만 GPU에서 읽는다
        for (task, name), (mean, peak) in zip(keys, stacked, strict=True):
            prefix = f"{task}/" if self.multitask else ""
            out[f"grad_norm/{prefix}{name}"] = mean
            out[f"grad_norm_max/{prefix}{name}"] = peak
        return out


def weight_norm_metrics(groups: dict[str, list]) -> dict[str, float]:
    norms = group_norms(groups)
    values = torch.stack(list(norms.values())).tolist()
    return {f"weight_norm/{name}": v for name, v in zip(norms, values, strict=True)}


class TaskGradientCosine:
    """두 task gradient의 묶음별 cosine (모듈 docstring). rank 0에서만 쓴다(DDP 평균 뒤라 모든 GPU의
    gradient가 같다).

    기록 step s(0부터 센 step 번호)의 앞 3 step(s-3, s-2, s-1)의 gradient를 bf16으로 남겨 두고,
    step마다 새 gradient와 남긴 것의 내적을 묶음별로 쌓는다. 4 step이 다 차면 cosine을 내고 비운다.
    자리는 step 번호로 정한다. 가운데 gradient가 유한하지 않거나 빠진 step이 있으면(이어 학습·
    되돌리기가 window 가운데서 시작) 그 window는 내지 않는다."""

    SPAN = 4

    def __init__(self, groups: dict[str, list], tasks: tuple[str, ...], log_every: int) -> None:
        if len(tasks) != 2:
            raise ValueError(f"task gradient cosine은 multitask(task 둘)에만 쓴다: {tasks}")
        if log_every < self.SPAN:
            raise ValueError(f"log_every는 {self.SPAN} 이상이어야 한다: {log_every}")
        self.log_every = log_every
        self.groups = shared_groups(groups, tasks)
        self.device = next(p for params in groups.values() for p in params).device
        self.reset()

    def reset(self) -> None:
        self.saved: list[tuple[str, list]] = []  # (task, 묶음마다 bf16 gradient 목록)
        self.sq: list[torch.Tensor] = []  # step마다 묶음별 ‖g‖² (G,)
        self.dots: dict[tuple[int, int], torch.Tensor] = {}  # (i, j) → 묶음별 내적 (G,)
        self.tasks: list[str] = []

    def position(self, step: int) -> int | None:
        """step(0부터)의 window 안 자리 (0~3, 3이 기록 step: done = step + 1이 log_every의 배수).
        window 밖이면 None."""
        before_log = -(step + 1) % self.log_every  # 기록 step까지 남은 step 수
        return self.SPAN - 1 - before_log if before_log < self.SPAN else None

    def wants(self, step: int) -> bool:
        """step의 gradient를 볼지 (기록 step과 그 앞 3 step)."""
        return self.position(step) is not None

    @torch.no_grad()
    def update(self, step: int, task: str, finite: bool) -> dict[str, float] | None:
        """step의 gradient(clip 전)를 넣는다. 4 step이 차면 지표를 돌려준다."""
        position = self.position(step)
        if position is None:
            return None
        if position == 0:
            self.reset()
        if not finite or position != len(self.tasks):
            self.reset()  # 이 window는 내지 않는다
            return None
        # 쓰지 않은 파라미터는 None으로 자리를 남긴다 (남긴 gradient와 순서를 맞춘다)
        current = {name: [p.grad for p in params] for name, params in self.groups.items()}
        names = list(self.groups)
        index = len(self.tasks)
        self.sq.append(torch.stack([_dot(current[n], current[n], self.device) for n in names]))
        for j, (_, old) in enumerate(self.saved):
            self.dots[(j, index)] = torch.stack(
                [_dot(old[n], current[n], self.device) for n in names]
            )
        self.tasks.append(task)
        if index + 1 < self.SPAN:
            self.saved.append(
                (
                    task,
                    {
                        n: [None if g is None else g.to(torch.bfloat16) for g in current[n]]
                        for n in names
                    },
                )
            )
            return None
        metrics = self._metrics(names)
        self.reset()
        return metrics

    def _metrics(self, names: list[str]) -> dict[str, float]:
        def cos(i: int, j: int) -> torch.Tensor:
            denom = (self.sq[i] * self.sq[j]).sqrt().clamp(min=1e-30)
            return self.dots[(i, j)] / denom

        cross = torch.stack([cos(i, i + 1) for i in range(self.SPAN - 1)]).mean(0)
        rows = [cross] + [cos(i, i + 2) for i in range(self.SPAN - 2)]
        values = torch.stack(rows).tolist()
        out = {f"grad_cos/{n}": v for n, v in zip(names, values[0], strict=True)}
        for i, row in enumerate(values[1:]):
            task = self.tasks[i]
            out.update({f"grad_cos_same/{task}/{n}": v for n, v in zip(names, row, strict=True)})
        return out


def _dot(a: list, b: list, device: torch.device) -> torch.Tensor:
    """같은 순서의 텐서 목록 두 개의 내적 (float32로 더한다). 한쪽이 None인 자리는 0이다."""
    total = torch.zeros((), device=device)
    for x, y in zip(a, b, strict=True):
        if x is not None and y is not None:
            total = total + torch.dot(x.flatten().float(), y.flatten().float())
    return total


# rank별 시간 기록의 열 (`RankTiming`)
TIMING_FIELDS = ("fwd_ms", "bwd_ms", "rows", "n_train", "features", "cells")


class RankTiming:
    """GPU마다 step의 forward·backward 시간과 표 크기 (모듈 docstring). 모든 GPU에서 쓴다.

    CUDA면 event로 GPU 시간을 재고(값은 모을 때 읽는다. 그 사이 step마다 loss를 읽느라 GPU를
    기다리므로 event는 끝나 있다), CPU면 벽시계로 잰다. gradient accumulation이면 micro batch의
    시간·셀 수는 더하고 행·feature 수는 평균한다. `gather`는 모든 GPU가 같은 step에 부른다."""

    def __init__(self, device: torch.device) -> None:
        self.cuda = device.type == "cuda"
        self.device = device
        self.reset()

    def reset(self) -> None:
        self.records: list[list] = []  # step마다 [step, task 번호, micro 기록 목록]
        self._micros: list = []

    def mark(self):
        """지금 시점 (micro batch의 forward 앞, forward 뒤, backward 뒤에 하나씩)."""
        if self.cuda:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            return event
        return time.perf_counter()

    def add_micro(self, start, fwd, end, x: torch.Tensor, n_train: int) -> None:
        b, rows, features = x.shape
        self._micros.append((start, fwd, end, rows, n_train, features, b * rows * features))

    def step_done(self, step: int, task_index: int) -> None:
        self.records.append([step, task_index, self._micros])
        self._micros = []

    def _elapsed(self, a, b) -> float:
        return a.elapsed_time(b) if self.cuda else (b - a) * 1000.0

    def _rows(self) -> torch.Tensor:
        out = []
        for step, task_index, micros in self.records:
            fwd = sum(self._elapsed(s, f) for s, f, *_ in micros)
            bwd = sum(self._elapsed(f, e) for _, f, e, *_ in micros)
            sizes = torch.tensor([m[3:] for m in micros], dtype=torch.float64)
            rows, n_train, features = sizes[:, :3].mean(0).tolist()
            cells = sizes[:, 3].sum().item()
            out.append([step, task_index, fwd, bwd, rows, n_train, features, cells])
        return torch.tensor(out, dtype=torch.float64).reshape(-1, 2 + len(TIMING_FIELDS))

    def gather(self, world_size: int) -> torch.Tensor | None:
        """모든 GPU의 기록 (rank, step, 2 + 열 수)을 rank 0에 돌려준다(다른 rank는 None). 비운다."""
        mine = self._rows()
        self.reset()
        if world_size == 1:
            return mine[None]
        # NCCL은 GPU 텐서만 모으고 float64를 그대로 받는다 (step 번호를 정확히 남긴다)
        local = mine.to(self.device)
        out = [torch.empty_like(local) for _ in range(world_size)]
        dist.all_gather(out, local)
        if dist.get_rank() != 0:
            return None
        return torch.stack(out).cpu()


def rank_timing_metrics(gathered: torch.Tensor) -> dict[str, float]:
    """모은 기록 (rank R, step N, 열)에서 낸 지표.

    - `rank_fwd_ms`·`rank_bwd_ms`: GPU·step 평균. `rank_fwd_ms_slowest`: step마다 가장 느린 GPU의
      forward 평균. `rank_fwd_imbalance`: 그 둘의 비(1이면 고르다)
    - `rank_cells`·`rank_cells_largest`·`rank_cells_imbalance`: 같은 것을 셀 수로
    - `rank_fwd_cells_corr`: step마다 GPU 평균을 뺀 뒤 forward 시간과 셀 수의 상관. 1에 가까우면
      GPU 사이 forward 차이가 표 크기 차이에서 온다
    - `rank_bwd_fwd_corr`: 같은 방식의 backward와 forward의 상관. 음수면 forward가 빨리 끝난 GPU가
      backward(all-reduce)에서 더 오래 기다린다
    - `rank_slowest_is_largest`: 가장 느린 forward의 GPU가 셀 수도 가장 많은 step의 비율
    GPU가 하나면 GPU 사이 지표(상관·비율)는 내지 않는다."""
    if gathered.shape[1] == 0:
        return {}
    columns = {name: gathered[..., 2 + i] for i, name in enumerate(TIMING_FIELDS)}
    fwd, bwd, cells = columns["fwd_ms"], columns["bwd_ms"], columns["cells"]
    out = {
        "rank_fwd_ms": fwd.mean().item(),
        "rank_bwd_ms": bwd.mean().item(),
        "rank_fwd_ms_slowest": fwd.max(0).values.mean().item(),
        "rank_cells": cells.mean().item(),
        "rank_cells_largest": cells.max(0).values.mean().item(),
        "rank_rows": columns["rows"].mean().item(),
        "rank_features": columns["features"].mean().item(),
    }
    out["rank_fwd_imbalance"] = out["rank_fwd_ms_slowest"] / max(out["rank_fwd_ms"], 1e-12)
    out["rank_cells_imbalance"] = out["rank_cells_largest"] / max(out["rank_cells"], 1e-12)
    if gathered.shape[0] > 1:
        out["rank_fwd_cells_corr"] = _within_step_corr(fwd, cells)
        out["rank_bwd_fwd_corr"] = _within_step_corr(bwd, fwd)
        out["rank_slowest_is_largest"] = (fwd.argmax(0) == cells.argmax(0)).double().mean().item()
    return {k: v for k, v in out.items() if math.isfinite(v)}


def _within_step_corr(a: torch.Tensor, b: torch.Tensor) -> float:
    """(rank, step) 두 값의 상관. step마다 rank 평균을 빼서 step 사이 차이는 뺀다."""
    a = a - a.mean(0, keepdim=True)
    b = b - b.mean(0, keepdim=True)
    denom = (a.square().sum() * b.square().sum()).sqrt()
    return (a * b).sum().item() / denom.item() if denom > 0 else math.nan


def write_rank_timing(path: Path, gathered: torch.Tensor, tasks: tuple[str, ...]) -> None:
    """모은 기록을 step마다 한 줄(JSON)로 덧붙인다. 열마다 rank 순서의 목록."""
    with path.open("a") as f:
        for n in range(gathered.shape[1]):
            line = {
                "step": int(gathered[0, n, 0]),
                "task": tasks[int(gathered[0, n, 1])],
            }
            for i, name in enumerate(TIMING_FIELDS):
                values = gathered[:, n, 2 + i].tolist()
                line[name] = [round(v, 3) for v in values] if name.endswith("_ms") else values
            f.write(json.dumps(line) + "\n")
