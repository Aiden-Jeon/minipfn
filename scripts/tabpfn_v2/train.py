"""TabPFN v2·2.5·3·3.5를 TabICL prior로 학습하고 MLflow에 기록한다 (Nature 2025 재현에서 시작).

AI Runtime(`databricks air run -f scripts/tabpfn_v2/train.yaml`)과 로컬에서 같은 스크립트를 쓴다.
- GPU 여러 장: `torchrun --standalone --nproc_per_node=8 scripts/tabpfn_v2/train.py`
- 노드 여러 대: 노드마다 `torchrun --nnodes=<노드 수> --node_rank=<번호> --master_addr=<주소>
  --master_port=<포트> --nproc_per_node=8 scripts/tabpfn_v2/train.py`. batch는 그대로이고
  GPU당 micro batch는 batch / GPU 수를 넘지 않게 자동으로 줄어든다. GPU를 늘릴 때 GPU당 데이터셋
  수를 유지하려면 `TABPFN_V2_BATCH_SIZE`로 batch를 키운다(이어 학습하면 원래 run의 batch를 따른다).
- 로컬: `uv run python scripts/tabpfn_v2/train.py --config-name smoke --task regression`

모델 구조는 `--arch`(또는 `TABPFN_V2_ARCH`)로 고른다. 기본 `v2`, `v2.5`는 TabPFN-2.5
(arXiv 2511.08667)의 구조 변경(층 수 회귀 18·분류 24, 셀 3개 묶기, thinking 행 64개,
회귀 MLP encoder)이다. `v3`·`v3.5`는 TabPFN-3(arXiv 2605.13986)·3.5(arXiv 2609.17895)의
열 → 행 → 행 ICL 모델(`column_row_transformer`, `recipes.tabpfn_v3`)이다. `--task multitask`는
분류·회귀를 한 모델로 같이 학습한다(TabPFN-3.5, `--arch v3.5`만. step마다 task를 번갈아 쓴다).
이어서 학습하거나
가중치를 가져오면 기본은 원래 run의 구조이고, `--arch`를 주면 그 구조가 원래 run과 같은지 확인한다.
규모는 인자 또는 환경변수(`TABPFN_V2_TASK`, `TABPFN_V2_CONFIG_NAME`, `TABPFN_V2_STEPS`,
`TABPFN_V2_BATCH_SIZE`, `TABPFN_V2_MICRO_BATCH`, `TABPFN_V2_NUM_WORKERS`, `TABPFN_V2_LR`,
`TABPFN_V2_CHECKPOINT_EVERY`, `TABPFN_V2_VAL_EVERY`)로 정한다. 학습 가속 옵션(issue #46,
`TrainConfig` 주석)은 `--compile`·`--fused-adam`·`--tf32`·`--schedule-free`·`--weight-average K`
(또는 `TABPFN_V2_COMPILE=1` 등, `TABPFN_V2_WEIGHT_AVERAGE=K`)로 켠다. optimizer는
`--optimizer muon`(또는 `TABPFN_V2_OPTIMIZER=muon`, issue #28)으로 바꾸고, Muon 묶음의 weight
decay는 `TABPFN_V2_MUON_WEIGHT_DECAY`(기본 0)로 준다. `--muon-distributed node|all`(또는
`TABPFN_V2_MUON_DISTRIBUTED`, issue #56)은 Muon의 Newton–Schulz를 노드 안 GPU끼리 또는 모든
GPU끼리 나눠 계산한다(결과·checkpoint 형식 같음, 이어 학습하면 원래 run의 값을 따르고 `off`로
끈다). 회귀에서 반복되는 target 값을 이웃 구간 몇 개에 흩을지는
`TABPFN_V2_DEQUANTIZE_BUCKETS`(기본 100, 0이면 끔)로 바꾼다.
학습 진단 지표(`minipfn.training.diagnostics`)는 묶음별 가중치·gradient norm을 늘 남기고,
`--grad-cosine`(`TABPFN_V2_GRAD_COSINE=1`, multitask만, issue #64)은 두 task gradient의 묶음별
cosine을, `--rank-timing`(`TABPFN_V2_RANK_TIMING=1`, issue #68)은 GPU마다 forward·backward 시간과
표 크기를 더 남긴다. 학습 계산은 그대로이고 이어 학습해도 따라가지 않는다(chunk마다 켠다).
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
`--config-name paper_run`(`--arch v3.5 --task multitask`)은 실제로 학습한 TabPFN-3.5 경로다
(`recipes.tabpfn_v3.paper_run`: batch 256, Muon, step 구간별 lr·weight decay 변경 `phases`).
구간 설정이 있는 run을 이어 학습하면 그 구간 설정을 따르고, lr은 `--lr`이 아니라 구간이 정한다.
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
from minipfn.training.config import SPEED_OPTIONS, TrainConfig, normalize_phases  # noqa: E402
from minipfn.training.loop import check_source_checkpoint, train  # noqa: E402
from minipfn.utils import log_run_source  # noqa: E402

# 일반 크기(smoke, medium)는 TrainConfig에, 논문 설정(paper, long_tables)은 recipe에 있다
PRESETS = {
    "smoke": TrainConfig.smoke,
    "medium": TrainConfig.medium,
    "paper": recipe.paper,
    "long_tables": recipe.long_tables,
    # 실제로 학습한 TabPFN-3.5 경로 (`recipes.tabpfn_v3.paper_run`, --arch v3.5 --task multitask)
    "paper_run": tabpfn_v3.paper_run,
}
CONFIG_NAMES = tuple(PRESETS)
# MLflow run 이름의 머리
RUN_PREFIXES = {"v2": "tabpfn-v2", "v2.5": "tabpfn-v2-5", "v3": "tabpfn-v3", "v3.5": "tabpfn-v3-5"}


TRUE_VALUES, FALSE_VALUES = ("1", "true", "yes", "on"), ("", "0", "false", "no", "off")


def _env_flag(name: str) -> bool:
    """켜고 끄는 환경변수 (대소문자 무시: 1/true/yes/on, 0/false/no/off, 비우면 끔). 그 밖의 값은
    조용히 꺼지지 않도록 멈춘다."""
    value = os.environ.get(name, "").strip().lower()
    if value in TRUE_VALUES:
        return True
    if value in FALSE_VALUES:
        return False
    raise SystemExit(
        f"{name}={os.environ[name]!r}는 켜고 끄는 값이 아니다 ({TRUE_VALUES} 또는 {FALSE_VALUES})"
    )


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
    parser.add_argument("--batch-size", type=int, default=_env("TABPFN_V2_BATCH_SIZE", int))
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
    parser.add_argument("--val-every", type=int, default=_env("TABPFN_V2_VAL_EVERY", int))
    # 학습 가속 (TrainConfig 주석). 켜기만 한다. 이어 학습하면 원래 run에서 켠 것은 그대로 켜진다
    # (`source_speed_options`)
    for flag in ("compile", "fused-adam", "tf32", "schedule-free"):
        env = "TABPFN_V2_" + flag.upper().replace("-", "_")
        parser.add_argument(f"--{flag}", action="store_true", default=_env_flag(env))
    # 학습 진단 지표 (TrainConfig 주석, issue #64·#68). 학습 계산은 그대로라 이어 학습해도 따라가지
    # 않는다(chunk마다 정한다)
    for flag in ("grad-cosine", "rank-timing"):
        env = "TABPFN_V2_" + flag.upper().replace("-", "_")
        parser.add_argument(f"--{flag}", action="store_true", default=_env_flag(env))
    parser.add_argument("--weight-average", type=int, default=_env("TABPFN_V2_WEIGHT_AVERAGE", int))
    parser.add_argument(
        "--weight-average-every", type=int, default=_env("TABPFN_V2_WEIGHT_AVERAGE_EVERY", int)
    )
    # optimizer (issue #28, TrainConfig 주석). 이어 학습하면 원래 run의 것을 따른다. 다른 종류
    # (adam·muon·schedule-free)를 주면 원래 run의 optimizer 설정은 쓰지 않는다 (`_source_options`)
    parser.add_argument(
        "--optimizer", choices=["adam", "muon"], default=os.environ.get("TABPFN_V2_OPTIMIZER")
    )
    parser.add_argument(
        "--muon-weight-decay", type=float, default=_env("TABPFN_V2_MUON_WEIGHT_DECAY", float)
    )
    parser.add_argument(
        "--muon-distributed",
        choices=["off", "node", "all"],
        default=os.environ.get("TABPFN_V2_MUON_DISTRIBUTED") or None,
    )
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


def resume_phase_lr(config: TrainConfig, args: argparse.Namespace, params: dict) -> TrainConfig:
    """구간 설정이 있는 학습을 이어 갈 때의 lr. lr은 구간이 정하므로 `--lr`은 받지 않고, 첫 구간
    앞의 lr(`config.lr`)은 원래 run의 것을 잇는다(처음에 `--lr`로 바꾼 run도 같은 경로로)."""
    if not config.phases:
        return config
    if args.lr is not None:
        raise SystemExit("구간 설정(phases)이 있는 학습은 --lr로 lr을 바꾸지 않는다")
    return replace(config, lr=float(params["lr"]))


def source_params(params: dict[str, str], tags: dict[str, str]) -> dict[str, str]:
    """원래 run의 param. 구간 설정은 이어 학습하며 바뀔 수 있어 tag(`phases`)가 있으면 그것을
    쓴다(param은 처음 값에서 바꿀 수 없다)."""
    if "phases" in tags:
        return {**params, "phases": tags["phases"]}
    return params


def source_speed_options(params: dict[str, str]) -> dict:
    """원래 run의 학습 가속 옵션 (MLflow param, 문자열). 그 옵션이 생기기 전의 run은 빈 dict."""
    defaults = TrainConfig()
    out = {}
    for name in SPEED_OPTIONS:
        if name in params:
            default = getattr(defaults, name)
            text = params[name]
            if name == "phases":
                # 구간 설정은 있을 때만 잇는다 (없는 run을 이어도 preset의 구간을 지우지 않도록)
                if phases := normalize_phases(text):
                    out[name] = phases
                continue
            out[name] = text == "True" if isinstance(default, bool) else type(default)(text)
    return out


def inherit_batch_size(args: argparse.Namespace, params: dict[str, str]) -> None:
    """이어 학습할 때 `--batch-size`를 주지 않았으면 원래 run의 batch를 쓴다(preset의 64로 조용히
    돌아가지 않도록). 가중치만 가져와 새로 학습하면(`--init-run-id`) preset 값 그대로다."""
    if args.resume_run_id and args.batch_size is None and "batch_size" in params:
        args.batch_size = int(params["batch_size"])


# 구간 설정(phases)에도 Muon weight decay가 들어 있어 optimizer를 바꾸면 같이 버린다
OPTIMIZER_OPTIONS = (
    "optimizer",
    "schedule_free",
    "muon_weight_decay",
    "muon_distributed",
    "phases",
)


def _optimizer_kind(schedule_free: bool, optimizer: str | None) -> str | None:
    return "schedule_free" if schedule_free else optimizer


def _source_options(options: dict, args: argparse.Namespace) -> dict:
    """원래 run의 학습 가속 옵션 가운데 이어 쓸 것. 인자로 준 값은 그 뒤에 덮어쓴다
    (`_override_values`). 원래 run에서 켠 옵션은 켠 채로(플래그는 켜기만 한다), 숫자는 인자로
    주지 않았으면 원래 값이다.

    optimizer 설정(`OPTIMIZER_OPTIONS`)은 한 묶음이다. 원래 run과 다른 종류를 주면(`--optimizer`,
    `--schedule-free`) 묶음을 통째로 쓰지 않는다(preset 값에서 시작). 가중치만 가져와 새로 학습할 때
    optimizer를 바꿀 수 있다. 이어 학습에서 바꾸면 checkpoint 검사(`check_resume_step`)가 멈춘다
    """
    chosen = _optimizer_kind(args.schedule_free, args.optimizer)
    source = _optimizer_kind(options.get("schedule_free", False), options.get("optimizer", "adam"))
    if chosen is not None and chosen != source:
        options = {k: v for k, v in options.items() if k not in OPTIMIZER_OPTIONS}
    return options


def build_config(
    args: argparse.Namespace, source: dict | None = None, speed_options: dict | None = None
) -> tuple[TrainConfig, str | None]:
    """(학습 설정, 구조 이름). `source`는 이어 가거나 가중치를 가져올 run의 모델 구조
    (`hub.run_architecture`)다. 주면 그 구조를 쓰고, `--arch`도 줬으면 둘이 같은지 본다. 구조 이름은
    `ARCHITECTURES`의 것과 맞지 않으면 None이다. `speed_options`는 원래 run의 학습 가속 옵션
    (`source_speed_options`)이다."""
    keep_size = args.config_name == "smoke"

    def preset(arch: str) -> TrainConfig:
        if args.config_name == "long_tables":
            return recipe.long_tables(args.task, arch=arch)
        if args.config_name == "paper_run":
            try:
                return tabpfn_v3.paper_run(args.task, arch)
            except ValueError as e:
                raise SystemExit(str(e)) from None
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
    source_phases = (speed_options or {}).get("phases")
    if source_phases and base.phases and source_phases != base.phases:
        raise SystemExit(
            f"원래 run의 구간 설정 {source_phases}가 --config-name {args.config_name}의 것"
            f" {base.phases}과 다르다"
        )
    # 원래 run의 구조·옵션과 인자를 한 번에 바꾼다 (설정 검사가 중간 상태를 보지 않도록)
    changes = {**source, **_source_options(speed_options or {}, args), **_override_values(args)}
    return _replace(base, changes), arch


def _overrides(config: TrainConfig, args: argparse.Namespace) -> TrainConfig:
    """인자·환경변수로 준 값을 설정에 덮어쓴다."""
    return _replace(config, _override_values(args))


def _replace(config: TrainConfig, changes: dict) -> TrainConfig:
    try:
        return replace(config, **changes)
    except ValueError as e:  # 같이 쓰지 않는 옵션 (`TrainConfig.__post_init__`)
        raise SystemExit(f"학습 설정: {e}") from None


def _override_values(args: argparse.Namespace) -> dict:
    """인자·환경변수로 준 값 (주지 않은 것은 빠진다)."""
    overrides = {
        "steps": args.steps,
        "batch_size": args.batch_size,
        "micro_batch": args.micro_batch,
        "num_workers": args.num_workers,
        "lr": args.lr,
        "dequantize_buckets": args.dequantize_buckets,
        "stop_step": args.stop_step,
        "checkpoint_every": args.checkpoint_every,
        "keep_every": args.keep_every,
        "val_every": args.val_every,
        "weight_average": args.weight_average,
        "weight_average_every": args.weight_average_every,
        "optimizer": args.optimizer,
        "muon_weight_decay": args.muon_weight_decay,
        "muon_distributed": args.muon_distributed,
        **{
            name: True
            for name in (
                "compile",
                "fused_adam",
                "tf32",
                "schedule_free",
                "grad_cosine",
                "rank_timing",
            )
            if getattr(args, name)
        },
        "rewind_lr_factor": args.rewind_lr_factor,
        "deadline": datetime.fromisoformat(args.deadline).timestamp() if args.deadline else None,
    }
    overrides = {k: v for k, v in overrides.items() if v is not None}
    # `off`면 끈다 (CollapseGuard, QK-clip)
    if args.collapse_threshold is not None:
        overrides["collapse_threshold"] = _optional_float(args.collapse_threshold)
    if args.qk_clip is not None:
        overrides["qk_clip"] = _optional_float(args.qk_clip)
    return overrides


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

    source, speed_options = None, None
    if args.init_run_id or args.resume_run_id:
        run = mlflow.MlflowClient().get_run(args.init_run_id or args.resume_run_id)
        params = source_params(run.data.params, run.data.tags)
        if params["task"] != args.task:
            raise SystemExit(f"원래 run은 {params['task']} 모델이다 (--task {args.task}).")
        source = run_architecture(params)
        speed_options = source_speed_options(params)
        if args.init_run_id:
            # 가중치만 가져오면 step 0부터 새 스케줄이라 원래 run의 구간(절대 step)은 잇지 않는다
            speed_options.pop("phases", None)
        inherit_batch_size(args, params)
    config, arch = build_config(args, source, speed_options)
    if args.resume_run_id:
        config = resume_phase_lr(config, args, params)  # checkpoint를 받기 전에 본다

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
        if config.phases:
            pass  # 기준 lr은 구간 lr × checkpoint의 붕괴 배율 (학습 루프. lr은 위에서 정했다)
        elif args.lr is None:
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
