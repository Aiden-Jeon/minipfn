"""TabPFN v2를 TabICL prior로 학습하고 MLflow에 기록한다 (Nature 2025 재현).

AI Runtime(`databricks air run -f runs/train_tabpfn_v2.yaml`)과 로컬에서 같은 스크립트를 쓴다.
- GPU 여러 장: `torchrun --standalone --nproc_per_node=8 scripts/train_tabpfn_v2.py`
- 노드 여러 대: 노드마다 `torchrun --nnodes=<노드 수> --node_rank=<번호> --master_addr=<주소>
  --master_port=<포트> --nproc_per_node=8 scripts/train_tabpfn_v2.py`. batch는 그대로이고 GPU당
  micro batch는 batch / GPU 수를 넘지 않게 자동으로 줄어든다.
- 로컬: `uv run python scripts/train_tabpfn_v2.py --config-name smoke --task regression`

규모는 인자 또는 환경변수(`TABPFN_V2_TASK`, `TABPFN_V2_CONFIG_NAME`, `TABPFN_V2_STEPS`,
`TABPFN_V2_MICRO_BATCH`, `TABPFN_V2_NUM_WORKERS`, `TABPFN_V2_LR`,
`TABPFN_V2_CHECKPOINT_EVERY`)로 정한다. 회귀에서 반복되는 target 값을 이웃 구간에 흩으려면
`TABPFN_V2_DEQUANTIZE_BUCKETS`(흩을 구간 수, 0이면 끔)를 준다.
나눠 학습할 때는 `TABPFN_V2_STOP_STEP`(이번에 멈출 step)이나 `TABPFN_V2_DEADLINE`(멈출 시각,
ISO 8601 예: 2026-10-28T08:45+09:00)을 준다. 학습률 스케줄은 `steps` 전체 기준으로 이어진다.
`--resume-run-id`(또는 `TABPFN_V2_RESUME_RUN_ID`)를 주면 그 run의 `checkpoints/latest.pt`에서
이어서 학습한다. `--resume-step`(또는 `TABPFN_V2_RESUME_STEP`)을 같이 주면 그 step에 남긴
`checkpoints/step_XXXXXXX.pt`에서 이어서 학습한다(`TABPFN_V2_KEEP_EVERY` step마다 남는다).
실행한 코드와 명령은 run의 `code/`와 tag(`command`, `env.TABPFN_V2_*`)에 남는다.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402

from minipfn.tabpfn_v2.train import TabPFNv2TrainConfig, train_tabpfn_v2  # noqa: E402
from minipfn.utils import log_run_source  # noqa: E402

CONFIG_NAMES = ("smoke", "medium", "paper")


def _env(name: str, cast=str):
    value = os.environ.get(name)
    return cast(value) if value else None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--task",
        choices=["classification", "regression"],
        default=os.environ.get("TABPFN_V2_TASK", "classification"),
    )
    parser.add_argument(
        "--config-name",
        choices=CONFIG_NAMES,
        default=os.environ.get("TABPFN_V2_CONFIG_NAME", "smoke"),
    )
    parser.add_argument("--steps", type=int, default=_env("TABPFN_V2_STEPS", int))
    parser.add_argument("--micro-batch", type=int, default=_env("TABPFN_V2_MICRO_BATCH", int))
    parser.add_argument("--num-workers", type=int, default=_env("TABPFN_V2_NUM_WORKERS", int))
    parser.add_argument("--lr", type=float, default=_env("TABPFN_V2_LR", float))
    parser.add_argument(
        "--dequantize-buckets", type=float, default=_env("TABPFN_V2_DEQUANTIZE_BUCKETS", float)
    )
    parser.add_argument("--stop-step", type=int, default=_env("TABPFN_V2_STOP_STEP", int))
    parser.add_argument("--deadline", default=os.environ.get("TABPFN_V2_DEADLINE"))
    parser.add_argument(
        "--checkpoint-every", type=int, default=_env("TABPFN_V2_CHECKPOINT_EVERY", int)
    )
    parser.add_argument("--keep-every", type=int, default=_env("TABPFN_V2_KEEP_EVERY", int))
    parser.add_argument("--resume-run-id", default=os.environ.get("TABPFN_V2_RESUME_RUN_ID"))
    parser.add_argument("--resume-step", type=int, default=_env("TABPFN_V2_RESUME_STEP", int))
    parser.add_argument("--out-dir", default=os.environ.get("TABPFN_V2_OUT_DIR", "tabpfn_v2_run"))
    args = parser.parse_args(argv)
    if args.resume_step is not None and not args.resume_run_id:
        parser.error("--resume-step은 --resume-run-id와 같이 줘야 한다")
    return args


def build_config(args: argparse.Namespace) -> TabPFNv2TrainConfig:
    config = getattr(TabPFNv2TrainConfig, args.config_name)(args.task)
    overrides = {
        "steps": args.steps,
        "micro_batch": args.micro_batch,
        "num_workers": args.num_workers,
        "lr": args.lr,
        "dequantize_buckets": args.dequantize_buckets,
        "stop_step": args.stop_step,
        "checkpoint_every": args.checkpoint_every,
        "keep_every": args.keep_every,
        "deadline": datetime.fromisoformat(args.deadline).timestamp() if args.deadline else None,
    }
    return replace(config, **{k: v for k, v in overrides.items() if v is not None})


def plot_history(history: dict, task: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.plot(history["step"], history["train_loss"], color="C0", label="train (prior)")
    if history["val_step"]:
        ax.plot(history["val_step"], history["val_loss"], "o-", color="C1", label="val (prior)")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("cross-entropy" if task == "classification" else "bar distribution NLL")
    ax.set_title(f"TabPFN v2 prior-fitting loss ({task})")
    ax.legend()
    return fig


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = build_config(args)
    rank = int(os.environ.get("RANK", "0"))
    is_main = rank == 0
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    resume_path = None
    if args.resume_run_id:
        name = "latest.pt" if args.resume_step is None else f"step_{args.resume_step:07d}.pt"
        resume_path = mlflow.artifacts.download_artifacts(
            run_id=args.resume_run_id,
            artifact_path=f"checkpoints/{name}",
            dst_path=str(out_dir / f"resume_rank{rank}"),
        )

    if is_main:
        print("config_name:", args.config_name)
        print(config)
        # air가 넘겨준 run이 있으면 거기에, 이어서 학습하면 그 run에, 아니면 새 run에 기록한다.
        # 남긴 step으로 되돌아가면 원래 run의 latest.pt와 뒤쪽 step_*.pt를 덮어쓰지 않도록
        # 새 run에 쓴다
        resume_into = args.resume_run_id if args.resume_step is None else None
        run_id = os.environ.get("MLFLOW_RUN_ID") or resume_into
        run_name = None if run_id else f"tabpfn-v2-{config.task}-{args.config_name}"
        mlflow.start_run(run_id=run_id, run_name=run_name)
        print("MLflow run_id:", mlflow.active_run().info.run_id, flush=True)
        log_run_source(__file__, env_prefix="TABPFN_V2_")

    result = train_tabpfn_v2(config, out_dir, resume_from=resume_path, mlflow_log=is_main)

    if is_main:
        mlflow.log_figure(plot_history(result["history"], config.task), "loss_curve.png")
        mlflow.end_run()
        print("final model:", result["final_path"])


if __name__ == "__main__":
    main()
