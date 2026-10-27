"""실행 환경 점검: 패키지 버전과 연산 장치를 출력한다 (노트북 00의 스크립트판).

AI Runtime: `databricks air run -f runs/setup_check.yaml --watch`
로컬: `uv run python scripts/setup_check.py`
"""

from __future__ import annotations

import argparse
import sys

import matplotlib
import numpy as np
import scipy
import sklearn
import torch

import minipfn
from minipfn import get_device, set_seed


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--require-cuda", action="store_true", help="cuda가 없으면 실패")
    args = parser.parse_args(argv)

    print("minipfn", minipfn.__file__)
    print(f"python       {sys.version.split()[0]}")
    for mod in (torch, np, scipy, sklearn, matplotlib):
        print(f"{mod.__name__:<12} {mod.__version__}")

    set_seed(42)
    device = get_device()
    print("device:", device)
    if device.type == "cuda":
        print("cuda   :", torch.version.cuda)
        print("gpu    :", torch.cuda.get_device_name(0))

    x = torch.randn(1024, 1024, device=device)
    print((x @ x.T).shape)

    if args.require_cuda and device.type != "cuda":
        sys.exit(f"cuda가 필요하지만 device={device}")


if __name__ == "__main__":
    main()
