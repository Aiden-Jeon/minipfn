"""step 구간별 설정 변경 (`TrainConfig.phases`, issue #29)."""

import importlib.util
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from minipfn.hub import run_architecture
from minipfn.recipes import tabpfn_v3
from minipfn.training.config import TrainConfig, normalize_phases
from minipfn.training.loop import _phase_lr, train


def _smoke(**changes):
    return replace(
        TrainConfig.smoke("classification"),
        steps=6,
        warmup_steps=1,
        val_every=3,
        log_every=1,
        checkpoint_every=3,
        keep_every=None,
        collapse_threshold=None,
        optimizer="muon",
        **changes,
    )


PHASES = ((3, {"lr": 5e-5, "muon_weight_decay": 0.1}),)


def test_phase_value_follows_the_last_started_phase():
    config = replace(
        TrainConfig(optimizer="muon"),
        phases=((10, {"lr": 1e-4}), (20, {"muon_weight_decay": 0.1})),
    )
    assert config.phases == ((10, (("lr", 1e-4),)), (20, (("muon_weight_decay", 0.1),)))
    assert [config.phase_value("lr", s) for s in (-1, 9, 10, 25)] == [2e-4, 2e-4, 1e-4, 1e-4]
    assert [config.phase_value("muon_weight_decay", s) for s in (9, 19, 20)] == [0.0, 0.0, 0.1]
    # MLflow param의 문자열에서 같은 설정으로 돌아온다
    assert normalize_phases(str(config.phases)) == config.phases
    assert normalize_phases("()") == () == normalize_phases("")


@pytest.mark.parametrize(
    "phases",
    [
        ((10, {"lr": 1e-4}), (10, {"lr": 5e-5})),  # 시작 step이 커지는 순서가 아니다
        ((10, {"warmup_steps": 5}),),  # 바꿀 수 없는 설정
        ((10, {"lr": 0.0}),),
        ((10, {}),),
    ],
)
def test_phases_are_checked(phases):
    with pytest.raises(ValueError):
        TrainConfig(optimizer="muon", phases=phases)


def test_weight_decay_phase_needs_muon():
    with pytest.raises(ValueError):
        TrainConfig(phases=((10, {"muon_weight_decay": 0.1}),))


def test_phase_lr_keeps_the_rewind_factor_across_a_boundary():
    config = TrainConfig(phases=((10, {"lr": 1e-4}),))
    assert _phase_lr(config, 2e-4, 9, 10) == pytest.approx(1e-4)
    # 구간 안에서 붕괴로 절반(5e-5)이 된 뒤 구간 앞으로 되돌리면 앞 구간 lr의 절반
    assert _phase_lr(config, 5e-5, 15, 5) == pytest.approx(1e-4)
    # 같은 구간 안이면 그대로 (비트 동일)
    base = 1.2345e-4
    assert _phase_lr(config, base, 3, 7) is base


def test_training_switches_lr_and_weight_decay_at_the_phase(tmp_path):
    plain = train(_smoke(), tmp_path / "plain", mlflow_log=False)
    phased = train(_smoke(phases=PHASES), tmp_path / "phased", mlflow_log=False)
    # 구간이 시작하는 갱신(step 3) 전까지는 같고, 그 뒤는 다르다
    assert phased["history"]["train_loss"][:3] == plain["history"]["train_loss"][:3]
    assert phased["history"]["train_loss"][3:] != plain["history"]["train_loss"][3:]
    state = torch.load(tmp_path / "phased" / "latest.pt", weights_only=True)
    assert state["scheduler"]["base_lrs"][0] == pytest.approx(5e-5)
    groups = state["optimizer"]["muon"]["param_groups"]
    assert all(g["weight_decay"] == 0.1 for g in groups)


def test_resume_enters_the_phase_at_the_resume_step(tmp_path):
    # 구간이 시작하는 step의 checkpoint에서 이어 가면(실제 chunk02 재시작과 같은 경우) 바로 바뀐다
    train(_smoke(phases=PHASES, stop_step=3), tmp_path / "a", mlflow_log=False)
    before = torch.load(tmp_path / "a" / "latest.pt", weights_only=True)
    assert before["scheduler"]["base_lrs"][0] == pytest.approx(TrainConfig.smoke().lr)
    assert all(g["weight_decay"] == 0.0 for g in before["optimizer"]["muon"]["param_groups"])
    train(
        _smoke(phases=PHASES),
        tmp_path / "b",
        resume_from=tmp_path / "a" / "latest.pt",
        mlflow_log=False,
    )
    after = torch.load(tmp_path / "b" / "latest.pt", weights_only=True)
    assert after["scheduler"]["base_lrs"][0] == pytest.approx(5e-5)
    assert all(g["weight_decay"] == 0.1 for g in after["optimizer"]["muon"]["param_groups"])


def _script():
    path = Path(__file__).parents[1] / "scripts" / "tabpfn_v2" / "train.py"
    spec = importlib.util.spec_from_file_location("train_script_phases", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_paper_run_is_the_trained_v3_5_path(monkeypatch):
    for name in ("TABPFN_V2_ARCH", "TABPFN_V2_CONFIG_NAME", "TABPFN_V2_TASK", "TABPFN_V2_LR"):
        monkeypatch.delenv(name, raising=False)
    config = tabpfn_v3.paper_run()
    assert config.batch_size == 256 and config.optimizer == "muon"
    assert config.phase_value("lr", 124_999) == 2e-4 and config.phase_value("lr", 125_000) == 1e-4
    assert config.phase_value("muon_weight_decay", 125_000) == 0.1
    with pytest.raises(ValueError):
        tabpfn_v3.paper_run("classification", "v3.5")
    script = _script()
    args = ["--config-name", "paper_run", "--arch", "v3.5", "--task", "multitask"]
    built, arch = script.build_config(script.parse_args(args))
    assert arch == "v3.5" and built == config
    with pytest.raises(SystemExit):
        script.build_config(script.parse_args(["--config-name", "paper_run", "--arch", "v3.5"]))
    # 이어 학습하면 원래 run의 구간 설정을 따르고, 구간 설정이 없는 run은 preset의 구간을 지우지
    # 않는다. preset과 다른 구간 설정이면 멈춘다
    params = {k: str(v) for k, v in vars(config).items()}
    options = script.source_speed_options(params)
    assert options["phases"] == config.phases
    assert "phases" not in script.source_speed_options({**params, "phases": "()"})
    resume = script.parse_args(
        ["--config-name", "paper", "--task", "multitask", "--resume-run-id", "r"]
    )
    resumed, _ = script.build_config(resume, run_architecture(params), options)
    assert resumed.phases == config.phases
    other = {**params, "phases": str(((150_000, {"lr": 5e-5}),))}
    resume = script.parse_args(
        ["--config-name", "paper_run", "--task", "multitask", "--resume-run-id", "r"]
    )
    with pytest.raises(SystemExit):
        script.build_config(resume, run_architecture(other), script.source_speed_options(other))
