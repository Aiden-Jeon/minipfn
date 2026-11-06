"""TabPFN v2·2.5·3·3.5를 TabICL prior로 학습하고 MLflow에 기록한다 (Nature 2025 재현에서 시작).

AI Runtime(`databricks air run -f scripts/tabpfn_v2/train.yaml`)과 로컬에서 같은 스크립트를 쓴다.
- GPU 여러 장: `torchrun --standalone --nproc_per_node=8 scripts/tabpfn_v2/train.py`
- 노드 여러 대: 노드마다 `torchrun --nnodes=<노드 수> --node_rank=<번호> --master_addr=<주소>
  --master_port=<포트> --nproc_per_node=8 scripts/tabpfn_v2/train.py`. batch는 그대로이고
  GPU당 micro batch는 batch / GPU 수를 넘지 않게 자동으로 줄어든다.
- 로컬: `uv run python scripts/tabpfn_v2/train.py --config-name smoke --task regression`

모델 구조는 `--arch`(또는 `TABPFN_V2_ARCH`)로 고른다. 기본 `v2`, `v2.5`는 TabPFN-2.5
(arXiv 2511.08667)의 구조 변경(층 수 회귀 18·분류 24, 셀 3개 묶기, thinking 행 64개,
회귀 MLP encoder)이다. `v3`·`v3.5`는 TabPFN-3(arXiv 2605.13986)·3.5(arXiv 2609.17895)의
열 → 행 → 행 ICL 모델(`column_row_transformer`, `recipes.tabpfn_v3`)이다. `--task multitask`는
분류·회귀를 한 모델로 같이 학습한다(TabPFN-3.5, `--arch v3.5`만. step마다 task를 번갈아 쓴다).
이어서 학습하거나
가중치를 가져오면 기본은 원래 run의 구조이고, `--arch`를 주면 그 구조가 원래 run과 같은지 확인한다.
규모는 인자 또는 환경변수(`TABPFN_V2_TASK`, `TABPFN_V2_CONFIG_NAME`, `TABPFN_V2_STEPS`,
`TABPFN_V2_MICRO_BATCH`, `TABPFN_V2_NUM_WORKERS`, `TABPFN_V2_LR`,
`TABPFN_V2_CHECKPOINT_EVERY`)로 정한다. 회귀에서 반복되는 target 값을 이웃 구간 몇 개에 흩을지는
`TABPFN_V2_DEQUANTIZE_BUCKETS`(기본 100, 0이면 끔)로 바꾼다.
train loss가 갑자기 뛰어 돌아오지 않으면 메모리에 둔 건강한 상태로 되돌리고 다른 데이터로 다시 간다
(붕괴 감지 문턱 `TABPFN_V2_COLLAPSE_THRESHOLD`, 기본 0.5, `off`면 끔). 그때마다 기준 lr에
`TABPFN_V2_REWIND_LR_FACTOR`를 곱한다(paper 회귀는 0.5, 그 밖에는 1 = 그대로).
`TABPFN_V2_QK_CLIP`(예: 500, `off`면 끔)을 주면 head별 attention logit 상한이 그 값을 넘을 때
Wq·Wk를 줄인다(QK-clip, 학습 절차만 바뀌고 모델 구조·추론은 그대로). 50 step마다 한다.
나눠 학습할 때는 `TABPFN_V2_STOP_STEP`(이번에 멈출 step)이나 `TABPFN_V2_DEADLINE`(멈출 시각,
ISO 8601 예: 2026-10-28T08:45+09:00)을 준다. 학습률 스케줄은 `steps` 전체 기준으로 이어진다.
이어서 학습할 때 `TABPFN_V2_LR`을 checkpoint와 다르게 주면 남은 스케줄을 그 lr 기준으로 바꾸고,
주지 않으면 checkpoint가 쓰던 lr을 이어 간다.
`--resume-run-id`(또는 `TABPFN_V2_RESUME_RUN_ID`)를 주면 그 run의 `checkpoints/latest.pt`에서
이어서 학습한다. `--resume-step`(또는 `TABPFN_V2_RESUME_STEP`)을 같이 주면 그 step에 남긴
`checkpoints/step_XXXXXXX.pt`에서 이어서 학습한다(`TABPFN_V2_KEEP_EVERY` step마다 남는다).
`--init-run-id`(또는 `TABPFN_V2_INIT_RUN_ID`, 남긴 step은 `TABPFN_V2_INIT_STEP`)는 그 run의
checkpoint에서 가중치만 가져와 step 0부터 새로 학습한다(`--config-name long_tables`로 긴 표 학습).
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

from minipfn import recipes  # noqa: E402
from minipfn.hub import run_architecture  # noqa: E402
from minipfn.recipes import tabpfn_v2 as recipe  # noqa: E402
from minipfn.recipes import tabpfn_v3  # noqa: E402
from minipfn.training.config import TrainConfig  # noqa: E402
from minipfn.training.loop import check_source_checkpoint, train  # noqa: E402
from minipfn.utils import log_run_source  # noqa: E402

# 일반 크기(smoke, medium)는 TrainConfig에, 논문 설정(paper, long_tables)은 recipe에 있다
PRESETS = {
    "smoke": TrainConfig.smoke,
    "medium": TrainConfig.medium,
    "paper": recipe.paper,
    "long_tables": recipe.long_tables,
}
CONFIG_NAMES = tuple(PRESETS)
# MLflow run 이름의 머리
RUN_PREFIXES = {"v2": "tabpfn-v2", "v2.5": "tabpfn-v2-5", "v3": "tabpfn-v3", "v3.5": "tabpfn-v3-5"}


def _env(name: str, cast=str):
    value = os.environ.get(name)
    return cast(value) if value else None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--task",
        choices=["classification", "regression", "multitask"],
        default=os.environ.get("TABPFN_V2_TASK", "classification"),
    )
    parser.add_argument(
        "--config-name",
        choices=CONFIG_NAMES,
        default=os.environ.get("TABPFN_V2_CONFIG_NAME", "smoke"),
    )
    parser.add_argument(
        "--arch",
        choices=recipes.architectures(),
        default=os.environ.get("TABPFN_V2_ARCH"),
        help="모델 구조: v2(Nature 2025), v2.5(TabPFN-2.5), v3(TabPFN-3), v3.5(TabPFN-3.5)."
        " 기본: 새로 학습하면 v2,"
        " 이어서 학습하거나 가중치를 가져오면 원래 run의 구조",
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
    parser.add_argument(
        "--collapse-threshold", default=os.environ.get("TABPFN_V2_COLLAPSE_THRESHOLD") or None
    )
    parser.add_argument(
        "--rewind-lr-factor", type=float, default=_env("TABPFN_V2_REWIND_LR_FACTOR", float)
    )
    parser.add_argument("--qk-clip", default=os.environ.get("TABPFN_V2_QK_CLIP") or None)
    parser.add_argument("--resume-run-id", default=os.environ.get("TABPFN_V2_RESUME_RUN_ID"))
    parser.add_argument("--resume-step", type=int, default=_env("TABPFN_V2_RESUME_STEP", int))
    parser.add_argument("--init-run-id", default=os.environ.get("TABPFN_V2_INIT_RUN_ID"))
    parser.add_argument("--init-step", type=int, default=_env("TABPFN_V2_INIT_STEP", int))
    parser.add_argument("--out-dir", default=os.environ.get("TABPFN_V2_OUT_DIR", "tabpfn_v2_run"))
    args = parser.parse_args(argv)
    if args.resume_step is not None and not args.resume_run_id:
        parser.error("--resume-step은 --resume-run-id와 같이 줘야 한다")
    if args.init_step is not None and not args.init_run_id:
        parser.error("--init-step은 --init-run-id와 같이 줘야 한다")
    if args.init_run_id and args.resume_run_id:
        parser.error("--init-run-id와 --resume-run-id는 함께 줄 수 없다")
    if args.config_name == "long_tables" and not (args.init_run_id or args.resume_run_id):
        # 처음부터 학습하면 lr 5e-5·20k step으로는 덜 학습된 모델이 나온다. 회귀 구간 경계는
        # 가져온 checkpoint의 것을 이어 써서 paper 모델의 출력과 맞춘다
        parser.error("long_tables는 --init-run-id(paper run)나 --resume-run-id와 같이 줘야 한다")
    if (
        args.config_name == "long_tables"
        and args.arch is not None
        and args.arch not in recipe.LONG_TABLES_MICRO_BATCH
    ):
        parser.error(f"long_tables는 {tuple(recipe.LONG_TABLES_MICRO_BATCH)} 구조만 쓴다")
    return args


def build_config(
    args: argparse.Namespace, source: dict | None = None
) -> tuple[TrainConfig, str | None]:
    """(학습 설정, 구조 이름). `source`는 이어 가거나 가중치를 가져올 run의 모델 구조
    (`hub.run_architecture`)다. 주면 그 구조를 쓰고, `--arch`도 줬으면 둘이 같은지 본다. 구조 이름은
    `ARCHITECTURES`의 것과 맞지 않으면 None이다."""
    keep_size = args.config_name == "smoke"

    def preset(arch: str) -> TrainConfig:
        if args.config_name == "long_tables":
            return recipe.long_tables(args.task, arch=arch)
        if args.config_name == "paper" and arch in tabpfn_v3.ARCHITECTURES:
            return tabpfn_v3.paper(args.task, arch)  # multitask의 학습 설정도 이 recipe가 정한다
        return recipes.recipe_for_architecture(arch).with_architecture(
            PRESETS[args.config_name](args.task), arch, keep_size=keep_size
        )

    if source is None:
        arch = args.arch or "v2"
        return _overrides(preset(arch), args), arch
    arch = args.arch or recipes.architecture_name(replace(TrainConfig(task=args.task), **source))
    # 원래 run이 긴 표 학습을 하지 않는 구조(TabPFN-3·3.5 등)여도 run을 열기 전에 멈춘다
    if args.config_name == "long_tables" and arch not in recipe.LONG_TABLES_MICRO_BATCH:
        raise SystemExit(
            f"long_tables는 {tuple(recipe.LONG_TABLES_MICRO_BATCH)} 구조만 쓴다: {source}"
        )
    base = preset(arch or "v2")
    if args.arch is not None:
        differ = {k: (getattr(base, k), v) for k, v in source.items() if getattr(base, k) != v}
        if differ:
            raise SystemExit(
                f"--arch {args.arch}가 원래 run의 모델 구조와 다르다 (--arch 값, run 값): {differ}"
            )
    return _overrides(replace(base, **source), args), arch


def _overrides(config: TrainConfig, args: argparse.Namespace) -> TrainConfig:
    """인자·환경변수로 준 값을 설정에 덮어쓴다."""
    overrides = {
        "steps": args.steps,
        "micro_batch": args.micro_batch,
        "num_workers": args.num_workers,
        "lr": args.lr,
        "dequantize_buckets": args.dequantize_buckets,
        "stop_step": args.stop_step,
        "checkpoint_every": args.checkpoint_every,
        "keep_every": args.keep_every,
        "rewind_lr_factor": args.rewind_lr_factor,
        "deadline": datetime.fromisoformat(args.deadline).timestamp() if args.deadline else None,
    }
    config = replace(config, **{k: v for k, v in overrides.items() if v is not None})
    # `off`면 끈다 (CollapseGuard, QK-clip)
    if args.collapse_threshold is not None:
        config = replace(config, collapse_threshold=_optional_float(args.collapse_threshold))
    if args.qk_clip is not None:
        config = replace(config, qk_clip=_optional_float(args.qk_clip))
    return config


def _check_source(path: str, config: TrainConfig, *, resume: bool) -> float:
    # MLflow run을 열기 전에 본다 (원래 run의 tag·코드를 덮어쓰거나 빈 run을 남기지 않도록).
    # 기준 lr도 이 한 번의 읽기로 돌려준다
    try:
        return check_source_checkpoint(path, config, resume=resume)
    except ValueError as e:
        raise SystemExit(str(e)) from None


def _optional_float(value: str) -> float | None:
    return None if value.lower() in ("off", "none") else float(value)


def plot_history(history: dict, task: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.plot(history["step"], history["train_loss"], color="C0", label="train (prior)")
    if history["val_step"]:
        ax.plot(history["val_step"], history["val_loss"], "o-", color="C1", label="val (prior)")
    ax.set_xlabel("optimizer step")
    labels = {"classification": "cross-entropy", "regression": "bar distribution NLL"}
    ax.set_ylabel(labels.get(task, "mean of classification CE and regression NLL"))
    ax.set_title(f"TabPFN prior-fitting loss ({task})")
    ax.legend()
    return fig


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    rank = int(os.environ.get("RANK", "0"))
    is_main = rank == 0
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    source = None
    if args.init_run_id or args.resume_run_id:
        params = mlflow.MlflowClient().get_run(args.init_run_id or args.resume_run_id).data.params
        if params["task"] != args.task:
            raise SystemExit(f"원래 run은 {params['task']} 모델이다 (--task {args.task}).")
        source = run_architecture(params)
    config, arch = build_config(args, source)

    init_path = None
    if args.init_run_id:
        name = "latest.pt" if args.init_step is None else f"step_{args.init_step:07d}.pt"
        init_path = mlflow.artifacts.download_artifacts(
            run_id=args.init_run_id,
            artifact_path=f"checkpoints/{name}",
            dst_path=str(out_dir / f"init_rank{rank}"),
        )
        _check_source(init_path, config, resume=False)

    resume_path = None
    if args.resume_run_id:
        name = "latest.pt" if args.resume_step is None else f"step_{args.resume_step:07d}.pt"
        resume_path = mlflow.artifacts.download_artifacts(
            run_id=args.resume_run_id,
            artifact_path=f"checkpoints/{name}",
            dst_path=str(out_dir / f"resume_rank{rank}"),
        )
        saved_lr = _check_source(resume_path, config, resume=True)
        if args.lr is None:
            # lr을 따로 주지 않으면 설정 기본값이 아니라 checkpoint가 쓰던 lr을 이어 간다
            config = replace(config, lr=saved_lr)
        elif args.lr > saved_lr and config.rewind_lr_factor < 1 and rank == 0:
            print(
                f"주의: checkpoint의 기준 lr {saved_lr:.2e}보다 높은 {args.lr:.2e}로 이어 간다. "
                "붕괴 규칙으로 낮춘 lr이었다면 그 인하가 사라진다",
                flush=True,
            )

    if is_main:
        print("config_name:", args.config_name)
        print(config)
        # air가 넘겨준 run이 있으면 거기에, 이어서 학습하면 그 run에, 아니면 새 run에 기록한다.
        # 남긴 step으로 되돌아가면 원래 run의 latest.pt와 뒤쪽 step_*.pt를 덮어쓰지 않도록
        # 새 run에 쓴다
        resume_into = args.resume_run_id if args.resume_step is None else None
        run_id = os.environ.get("MLFLOW_RUN_ID") or resume_into
        prefix = RUN_PREFIXES.get(arch, "tabpfn-v2")
        run_name = None if run_id else f"{prefix}-{config.task}-{args.config_name}"
        mlflow.start_run(run_id=run_id, run_name=run_name)
        print("MLflow run_id:", mlflow.active_run().info.run_id, flush=True)
        log_run_source(__file__, env_prefix="TABPFN_V2_")

    result = train(
        config, out_dir, resume_from=resume_path, mlflow_log=is_main, init_from=init_path
    )

    if is_main:
        mlflow.log_figure(plot_history(result["history"], config.task), "loss_curve.png")
        mlflow.end_run()
        print("final model:", result["final_path"])


if __name__ == "__main__":
    main()
