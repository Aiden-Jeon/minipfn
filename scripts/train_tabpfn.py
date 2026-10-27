"""TabPFN을 논문 prior로 학습하고 MLflow에 기록한다 (논문 arXiv:2207.01848 재현).

AI Runtime(`databricks air run -f runs/train_tabpfn.yaml`)과 로컬에서 같은
스크립트를 쓴다.
- GPU 여러 장: `torchrun --standalone --nproc_per_node=8 scripts/train_tabpfn.py`
- 로컬: `uv run python scripts/train_tabpfn.py --config-name smoke`

규모는 인자 또는 환경변수(`TABPFN_CONFIG_NAME`, `TABPFN_STEPS`, `TABPFN_NUM_WORKERS`,
`TABPFN_GROUPS_PER_BATCH`, `TABPFN_BATCH_SIZE`, `TABPFN_PROFILE_STEPS`)로 정한다.
`--resume-run-id`(또는 `TABPFN_RESUME_RUN_ID`)를 주면 그 run의 `checkpoints/latest.pt`에서
이어서 학습한다. 실행한 코드와 명령은 run의 `code/`와 tag(`command`, `env.TABPFN_*`)에 남는다.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402

from minipfn.tabpfn.train import TabPFNConfig, train_tabpfn  # noqa: E402
from minipfn.utils import log_run_source  # noqa: E402

CONFIG_NAMES = ("smoke", "medium", "paper")


def _env_int(name: str) -> int | None:
    value = os.environ.get(name)
    return int(value) if value else None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config-name",
        choices=CONFIG_NAMES,
        default=os.environ.get("TABPFN_CONFIG_NAME", "smoke"),
    )
    parser.add_argument("--steps", type=int, default=_env_int("TABPFN_STEPS"))
    parser.add_argument("--num-workers", type=int, default=_env_int("TABPFN_NUM_WORKERS"))
    parser.add_argument("--groups-per-batch", type=int, default=_env_int("TABPFN_GROUPS_PER_BATCH"))
    parser.add_argument("--batch-size", type=int, default=_env_int("TABPFN_BATCH_SIZE"))
    parser.add_argument("--profile-steps", type=int, default=_env_int("TABPFN_PROFILE_STEPS"))
    parser.add_argument("--resume-run-id", default=os.environ.get("TABPFN_RESUME_RUN_ID"))
    parser.add_argument("--out-dir", default=os.environ.get("TABPFN_OUT_DIR", "tabpfn_run"))
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> TabPFNConfig:
    config = getattr(TabPFNConfig, args.config_name)()
    overrides = {
        "steps": args.steps,
        "num_workers": args.num_workers,
        "groups_per_batch": args.groups_per_batch,
        "batch_size": args.batch_size,
        "profile_steps": args.profile_steps,
    }
    return replace(config, **{k: v for k, v in overrides.items() if v is not None})


def plot_history(history: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.plot(history["step"], history["train_ce"], color="C0", label="train CE (prior)")
    if history["val_step"]:
        ax.plot(history["val_step"], history["val_ce"], "o-", color="C1", label="val CE (prior)")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("cross-entropy")
    ax.set_title("TabPFN prior-fitting loss")
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
        resume_path = mlflow.artifacts.download_artifacts(
            run_id=args.resume_run_id,
            artifact_path="checkpoints/latest.pt",
            dst_path=str(out_dir / f"resume_rank{rank}"),
        )

    if is_main:
        print("config_name:", args.config_name)
        print(config)
        # air가 넘겨준 run이 있으면 거기에, 이어서 학습하면 그 run에, 아니면 새 run에 기록한다
        run_id = os.environ.get("MLFLOW_RUN_ID") or args.resume_run_id
        run_name = None if run_id else f"tabpfn-{args.config_name}-{config.steps}steps"
        mlflow.start_run(run_id=run_id, run_name=run_name)
        print("MLflow run_id:", mlflow.active_run().info.run_id, flush=True)
        log_run_source(__file__, env_prefix="TABPFN_")

    result = train_tabpfn(config, out_dir, resume_from=resume_path, mlflow_log=is_main)

    if is_main:
        mlflow.log_figure(plot_history(result["history"]), "loss_curve.png")
        mlflow.end_run()
        print("final model:", result["final_path"])


if __name__ == "__main__":
    main()
