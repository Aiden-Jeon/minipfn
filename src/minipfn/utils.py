import math
import os
import random
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Python, NumPy, PyTorch의 난수 시드를 한 번에 고정한다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(prefer: str | None = None) -> torch.device:
    """사용 가능한 가장 빠른 장치를 고른다 (cuda > mps > cpu).

    `prefer`를 지정하면 그 장치를 그대로 쓴다. 예: `get_device("cpu")`.
    """
    if prefer is not None:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _git(*args: str, cwd: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, check=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def log_run_source(script: str | Path, env_prefix: str, artifact_path: str = "code") -> None:
    """실행한 스크립트와 `minipfn` 소스, 실행 명령을 MLflow 활성 run에 남긴다.

    AI Runtime snapshot에는 `.git`이 없어 commit을 알 수 없으므로, 실제로 돈 코드를 그대로
    `artifact_path/`에 올린다. 그러면 run의 코드와 저장소의 코드를 diff로 대조할 수 있다.
    git 저장소 안에서 돌면(로컬) commit과 미커밋 변경 여부도 tag로 남긴다.
    """
    import mlflow

    script = Path(script).resolve()
    package = Path(__file__).resolve().parent
    mlflow.log_artifact(str(script), artifact_path)
    for path in sorted(package.rglob("*.py")):
        rel = path.relative_to(package.parent).parent.as_posix()
        mlflow.log_artifact(str(path), f"{artifact_path}/{rel}")

    tags = {"command": shlex.join([script.name, *sys.argv[1:]])}
    tags.update({f"env.{k}": v for k, v in sorted(os.environ.items()) if k.startswith(env_prefix)})
    commit = _git("rev-parse", "HEAD", cwd=script.parent)
    if commit:
        tags["git_commit"] = commit
        tags["git_dirty"] = str(bool(_git("status", "--porcelain", cwd=script.parent)))
    mlflow.set_tags(tags)


def lr_factor(step: int, warmup_steps: int, schedule_steps: int) -> float:
    """학습률 배율: 0에서 선형으로 올린 뒤(warmup) cosine으로 0까지 (v1 cosine 스케줄)."""
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, schedule_steps - warmup_steps)
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))


@dataclass
class DistContext:
    """torchrun 프로세스 하나의 위치."""

    rank: int = 0
    world_size: int = 1
    local_rank: int = 0

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def setup_distributed() -> tuple[DistContext, torch.device]:
    """torchrun 환경변수가 있으면 NCCL 프로세스 그룹을 연다. 없으면 GPU 하나 또는 CPU."""
    import torch.distributed as dist

    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        d = DistContext(
            int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), int(os.environ["LOCAL_RANK"])
        )
        torch.cuda.set_device(d.local_rank)
        dist.init_process_group("nccl")
        return d, torch.device("cuda", d.local_rank)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return DistContext(), device
