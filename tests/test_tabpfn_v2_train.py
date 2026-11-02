"""TabPFN v2 학습 루프 테스트 (smoke 크기, CPU)."""

import contextlib
from dataclasses import replace

import pytest
import torch

from minipfn.bar_distribution import FullSupportBarDistribution
from minipfn.tabpfn_v2.model import load_checkpoint
from minipfn.tabpfn_v2.train import (
    CollapseGuard,
    TabPFNv2TrainConfig,
    _make_val_batches,
    _validate,
    checkpoint_lr,
    checkpoint_qk_norm,
    dequantize_targets,
    lr_factor,
    train_tabpfn_v2,
)


def test_lr_schedule_warms_up_then_decays_to_zero():
    assert lr_factor(0, 10, 100) == 0.0
    assert lr_factor(10, 10, 100) == 1.0
    assert 0.0 < lr_factor(55, 10, 100) < 1.0
    assert lr_factor(100, 10, 100) == 0.0


def test_classification_training_writes_checkpoint_and_resumes(tmp_path):
    config = replace(TabPFNv2TrainConfig.smoke("classification"), steps=4, val_every=2)
    result = train_tabpfn_v2(config, tmp_path, mlflow_log=False)
    model, borders = load_checkpoint(result["final_path"])
    assert borders is None and model.config.n_out == 10
    assert result["history"]["step"][-1] == 4

    longer = replace(config, steps=6)
    resumed = train_tabpfn_v2(
        longer, tmp_path / "resumed", resume_from=tmp_path / "latest.pt", mlflow_log=False
    )
    assert resumed["history"]["step"][-1] == 6
    assert resumed["history"]["step"][0] == result["history"]["step"][0]  # 이전 기록을 이어 붙인다


def test_regression_training_stores_borders(tmp_path):
    config = replace(TabPFNv2TrainConfig.smoke("regression"), steps=2, border_datasets=10)
    result = train_tabpfn_v2(config, tmp_path, mlflow_log=False)
    model, borders = load_checkpoint(result["final_path"])
    assert model.config.n_out == 5000
    assert borders.shape == (5001,)
    assert (borders[1:] > borders[:-1]).all()
    assert torch.isfinite(torch.tensor(result["history"]["train_loss"])).all()


def test_chunked_training_continues_the_full_schedule(tmp_path):
    config = replace(TabPFNv2TrainConfig.smoke("classification"), steps=10, log_every=1)
    first = train_tabpfn_v2(replace(config, stop_step=4), tmp_path / "a", mlflow_log=False)
    assert first["history"]["step"][-1] == 4
    state = torch.load(tmp_path / "a" / "latest.pt", weights_only=True)
    assert state["step"] == 4
    # 학습률은 이번 실행(4 step)이 아니라 전체 10 step 스케줄을 따른다
    expected = config.lr * lr_factor(4, config.warmup_steps, config.steps)
    assert abs(state["scheduler"]["_last_lr"][0] - expected) < 1e-12

    second = train_tabpfn_v2(
        config, tmp_path / "b", resume_from=tmp_path / "a" / "latest.pt", mlflow_log=False
    )
    assert second["history"]["step"] == list(range(1, 11))


def test_kept_checkpoints_are_not_overwritten(tmp_path):
    config = replace(
        TabPFNv2TrainConfig.smoke("classification"), steps=6, checkpoint_every=2, keep_every=4
    )
    train_tabpfn_v2(config, tmp_path, mlflow_log=False)
    assert sorted(p.name for p in tmp_path.glob("step_*.pt")) == ["step_0000004.pt"]
    assert torch.load(tmp_path / "step_0000004.pt", weights_only=True)["step"] == 4
    assert torch.load(tmp_path / "latest.pt", weights_only=True)["step"] == 6

    # 남긴 step에서 이어 가면 그 step부터 다시 학습한다
    resumed = train_tabpfn_v2(
        config, tmp_path / "b", resume_from=tmp_path / "step_0000004.pt", mlflow_log=False
    )
    assert resumed["history"]["step"][-1] == 6

    # keep_every가 checkpoint_every의 배수가 아니어도 그 step에 남긴다
    odd = replace(config, checkpoint_every=4, keep_every=3)
    train_tabpfn_v2(odd, tmp_path / "c", mlflow_log=False)
    assert sorted(p.name for p in (tmp_path / "c").glob("step_*.pt")) == [
        "step_0000003.pt",
        "step_0000006.pt",
    ]


def test_dequantize_moves_only_repeated_interior_targets():
    from minipfn.bar_distribution import FullSupportBarDistribution

    bar = FullSupportBarDistribution(torch.linspace(-4.0, 4.0, 101))  # 구간 100개, 폭 0.08
    n_train = 4
    # 데이터셋 0: 1.0이 반복(이산), 데이터셋 1: 모두 다른 값(연속), 꼬리 값 9.0도 반복
    y = torch.tensor(
        [
            [1.0, 1.0, 2.0, 9.0, 1.0, 1.0, 9.0, 0.5],
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, float("nan")],
        ]
    )
    torch.manual_seed(0)
    out = dequantize_targets(y, n_train, bar, n_buckets=10)
    assert out.shape == (2, 4)
    moved = out[0, :2]
    assert not torch.equal(moved, y[0, 4:6])  # 반복된 1.0은 흩어진다
    assert ((moved - 1.0).abs() <= 5 * 0.08 + 1e-6).all()  # 이웃 구간 10개(±5) 안
    assert out[0, 2] == 9.0  # 꼬리 구간 값은 그대로
    assert out[0, 3] == 0.5  # 한 번만 나온 값은 그대로
    assert torch.equal(out[1, :3], y[1, 4:7]) and torch.isnan(
        out[1, 3]
    )  # 연속 target·결측은 그대로


def test_regression_training_with_qk_norm_records_attention_stats(tmp_path):
    config = replace(
        TabPFNv2TrainConfig.smoke("regression"), steps=2, border_datasets=10, qk_norm=True
    )
    result = train_tabpfn_v2(config, tmp_path, mlflow_log=False)
    model, _ = load_checkpoint(result["final_path"])
    assert model.config.qk_norm
    assert torch.isfinite(torch.tensor(result["history"]["train_loss"])).all()

    assert checkpoint_qk_norm(tmp_path / "latest.pt")  # 이어서 학습할 때 구조를 여기서 읽는다
    assert not checkpoint_qk_norm(_plain_checkpoint(tmp_path / "plain"))

    bar = FullSupportBarDistribution(torch.load(tmp_path / "latest.pt")["borders"])
    batches = _make_val_batches(config, n_batches=2)
    val = _validate(model, batches, torch.device("cpu"), contextlib.nullcontext, "regression", bar)
    assert val["attn_logit_max"] > 0 and 0 < val["attn_entropy_min"] <= 1 + 1e-6
    assert model.training  # 검증 뒤 학습 모드로 돌아온다


def _plain_checkpoint(out_dir):
    config = replace(TabPFNv2TrainConfig.smoke("classification"), steps=1)
    train_tabpfn_v2(config, out_dir, mlflow_log=False)
    return out_dir / "latest.pt"


def test_regression_training_with_dequantize(tmp_path):
    config = replace(
        TabPFNv2TrainConfig.smoke("regression"), steps=2, border_datasets=10, dequantize_buckets=10
    )
    result = train_tabpfn_v2(config, tmp_path, mlflow_log=False)
    assert torch.isfinite(torch.tensor(result["history"]["train_loss"])).all()


def test_val_batches_do_not_depend_on_micro_batch():
    config = TabPFNv2TrainConfig.smoke("classification")
    a = _make_val_batches(replace(config, micro_batch=4), n_batches=2)
    b = _make_val_batches(replace(config, micro_batch=1), n_batches=2)
    for batch_a, batch_b in zip(a, b, strict=True):
        assert batch_a.x.shape[0] == config.val_batch
        assert torch.equal(batch_a.x.nan_to_num(), batch_b.x.nan_to_num())
        assert torch.equal(batch_a.y, batch_b.y)


def test_deadline_in_the_past_stops_at_the_next_log_step(tmp_path):
    config = replace(
        TabPFNv2TrainConfig.smoke("classification"), steps=10, log_every=2, deadline=0.0
    )
    result = train_tabpfn_v2(config, tmp_path, mlflow_log=False)
    assert result["history"]["step"][-1] == 2
    assert torch.load(tmp_path / "latest.pt", weights_only=True)["step"] == 2


def test_micro_batch_shrinks_when_gpus_exceed_the_batch():
    config = TabPFNv2TrainConfig.paper("classification")  # batch 64, micro 8
    for world_size, micro, accumulation in [(1, 8, 8), (8, 8, 1), (16, 4, 1), (32, 2, 1)]:
        scaled = config.for_world_size(world_size)
        assert scaled.batch_size == 64
        assert scaled.micro_batch == micro
        assert scaled.grad_accumulation(world_size) == accumulation


def test_collapse_guard_needs_a_sustained_jump():
    guard = CollapseGuard(threshold=0.5, patience=3)
    for v in [0.82, 0.80, 0.81, 0.79, 0.83, 0.80]:
        assert not guard.update(v)
    assert not guard.update(1.5)  # 한 번 튄 건 붕괴가 아니다
    assert not guard.update(0.81) and not guard.suspect
    assert not guard.update(2.0) and not guard.update(2.0)
    assert guard.update(2.0)  # 3번 이어지면 붕괴


def test_training_rewinds_after_a_collapse(tmp_path, monkeypatch):
    calls = {"n": 0}
    original = CollapseGuard.update

    def fake_update(self, window_mean):
        calls["n"] += 1
        if calls["n"] == 7:  # 7번째 window에서 한 번 붕괴했다고 본다
            return True
        return original(self, window_mean)

    monkeypatch.setattr(CollapseGuard, "update", fake_update)
    config = replace(
        TabPFNv2TrainConfig.smoke("classification"), steps=50, log_every=5, snapshot_every=10
    )
    result = train_tabpfn_v2(config, tmp_path, mlflow_log=False)
    steps = result["history"]["step"]
    assert steps[-1] == 50
    assert steps == sorted(set(steps))  # 되돌린 구간의 기록은 지우고 다시 쓴다
    # 35 step에서 붕괴 → 최근 두 snapshot(20, 30) 중 오래된 20으로 되돌아가 다시 50까지
    assert calls["n"] == 7 + (50 - 20) // 5
    # 되돌린 횟수는 checkpoint에 남아 다음 chunk의 데이터 시드에도 섞인다
    assert torch.load(tmp_path / "latest.pt", weights_only=True)["rewinds"] == 1


@pytest.mark.parametrize("min_scale, expected_scale", [(0.0, 0.25), (0.75, 0.75)])
def test_each_collapse_lowers_the_lr_by_the_rewind_factor(
    tmp_path, monkeypatch, min_scale, expected_scale
):
    calls = {"n": 0}
    original = CollapseGuard.update

    def fake_update(self, window_mean):
        calls["n"] += 1
        if calls["n"] in (7, 9):  # 35 step에서 붕괴, 되돌아간 20부터 다시 가다 30 step에서 또 붕괴
            return True
        return original(self, window_mean)

    monkeypatch.setattr(CollapseGuard, "update", fake_update)
    config = replace(
        TabPFNv2TrainConfig.smoke("classification"),
        steps=50,
        log_every=5,
        snapshot_every=10,
        rewind_lr_factor=0.5,
    )
    config = replace(config, rewind_min_lr=config.lr * min_scale)
    train_tabpfn_v2(config, tmp_path, mlflow_log=False)
    state = torch.load(tmp_path / "latest.pt", weights_only=True)
    assert state["rewinds"] == 2
    # 두 번째로 되돌아간 snapshot(20 step)은 첫 인하 전 상태지만 lr은 두 번 모두 반영된다.
    # 하한이 있으면 거기서 멈춘다
    base = config.lr * expected_scale
    assert abs(state["scheduler"]["base_lrs"][0] - base) < 1e-15
    expected = base * lr_factor(50, config.warmup_steps, config.steps)
    assert abs(state["optimizer"]["param_groups"][0]["lr"] - expected) < 1e-12
    assert abs(checkpoint_lr(tmp_path / "latest.pt") - base) < 1e-15  # 다음 chunk가 이어 간다


def test_paper_preset_lowers_the_lr_on_collapse_only_for_regression():
    assert TabPFNv2TrainConfig.paper("regression").rewind_lr_factor == 0.5
    assert TabPFNv2TrainConfig.paper("classification").rewind_lr_factor == 1.0
    for bad in (0.0, -0.5, 5.0):
        with pytest.raises(ValueError, match="rewind_lr_factor"):
            TabPFNv2TrainConfig(rewind_lr_factor=bad)


def test_training_that_ends_suspect_saves_the_newest_healthy_state(tmp_path, monkeypatch):
    calls = {"n": 0}
    original = CollapseGuard.update

    def fake_update(self, window_mean):
        calls["n"] += 1
        if calls["n"] == 10:  # 마지막 window(50 step)만 한 번 튄다
            self.bad += 1
            return False
        return original(self, window_mean)

    monkeypatch.setattr(CollapseGuard, "update", fake_update)
    config = replace(
        TabPFNv2TrainConfig.smoke("classification"),
        steps=50,
        log_every=5,
        snapshot_every=10,
        rewind_lr_factor=0.5,
    )
    result = train_tabpfn_v2(config, tmp_path, mlflow_log=False)
    state = torch.load(tmp_path / "latest.pt", weights_only=True)
    assert state["step"] == 40  # 오래된 30이 아니라 가장 최근 건강한 40
    assert state["scheduler"]["base_lrs"][0] == config.lr  # 붕괴가 아니니 lr은 그대로
    assert result["history"]["step"][-1] == 40


def test_training_stops_after_too_many_rewinds(tmp_path):
    config = replace(
        TabPFNv2TrainConfig.smoke("classification"),
        steps=200,
        log_every=5,
        snapshot_every=10,
        collapse_threshold=-10.0,  # 건강한 기록이 쌓인 뒤로는 모든 window가 나쁘다
        max_rewinds=2,
    )
    with pytest.raises(RuntimeError, match="되돌렸다"):
        train_tabpfn_v2(config, tmp_path, mlflow_log=False)


def test_collapse_guard_counts_windows_without_finite_loss():
    guard = CollapseGuard(threshold=0.5, patience=2)
    assert not guard.update(float("inf")) and guard.suspect  # 건강한 기록이 없어도 나쁘다
    assert guard.update(float("inf"))


def test_resume_with_a_different_lr_rescales_the_rest_of_the_schedule(tmp_path):
    config = replace(TabPFNv2TrainConfig.smoke("classification"), steps=10, log_every=1)
    train_tabpfn_v2(replace(config, stop_step=4), tmp_path / "a", mlflow_log=False)
    assert checkpoint_lr(tmp_path / "a" / "latest.pt") == config.lr  # lr을 안 주면 이 값을 쓴다
    lower = replace(config, lr=config.lr / 2)
    train_tabpfn_v2(
        replace(lower, stop_step=6),
        tmp_path / "b",
        resume_from=tmp_path / "a" / "latest.pt",
        mlflow_log=False,
    )
    state = torch.load(tmp_path / "b" / "latest.pt", weights_only=True)
    assert abs(state["scheduler"]["base_lrs"][0] - lower.lr) < 1e-15
    expected = lower.lr * lr_factor(6, config.warmup_steps, config.steps)
    assert abs(state["scheduler"]["_last_lr"][0] - expected) < 1e-12
    assert abs(state["optimizer"]["param_groups"][0]["lr"] - expected) < 1e-12
