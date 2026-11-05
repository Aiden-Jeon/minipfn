"""TabPFN v2 학습 루프 테스트 (smoke 크기, CPU)."""

from dataclasses import replace

import pytest
import torch

from minipfn.tabpfn_v2.model import TabPFNv2, load_checkpoint
from minipfn.tabpfn_v2.train import (
    CollapseGuard,
    TabPFNv2TrainConfig,
    _make_val_batches,
    apply_qk_clip,
    checkpoint_lr,
    dequantize_targets,
    lr_factor,
    qk_logit_bounds,
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


def test_init_from_takes_only_the_weights(tmp_path):
    config = replace(TabPFNv2TrainConfig.smoke("classification"), steps=4, val_every=2)
    train_tabpfn_v2(config, tmp_path / "first", mlflow_log=False)
    init = tmp_path / "first" / "latest.pt"
    with pytest.raises(ValueError, match="init_from"):
        train_tabpfn_v2(config, tmp_path / "x", resume_from=init, init_from=init, mlflow_log=False)
    # step 0으로 학습하면 가중치가 그대로 남는다 (warmup 첫 step의 lr이 0)
    frozen = replace(config, steps=1, warmup_steps=1, stop_step=1, collapse_threshold=None)
    result = train_tabpfn_v2(frozen, tmp_path / "second", init_from=init, mlflow_log=False)
    assert result["history"]["step"] == [1]  # 이전 기록은 이어 붙이지 않는다
    before = torch.load(init, weights_only=True)["model"]
    after = torch.load(tmp_path / "second" / "latest.pt", weights_only=True)
    assert after["step"] == 1 and after["scheduler"]["last_epoch"] == 1
    for name, tensor in before.items():
        assert torch.equal(after["model"][name], tensor), name


def test_long_tables_preset_keeps_the_paper_model_and_batch():
    paper, long = TabPFNv2TrainConfig.paper(), TabPFNv2TrainConfig.long_tables()
    assert long.model_config() == paper.model_config() and long.batch_size == paper.batch_size
    # v2(12층)는 GPU당 micro batch 2 (batch 64 = 2 × 4 × GPU 8)
    assert long.micro_batch == 2 and long.grad_accumulation(8) == 4
    assert long.max_train_rows == 8192 and long.max_features == long.feature_scale == 500
    assert TabPFNv2TrainConfig.long_tables("regression").rewind_lr_factor == 0.5
    # 2.5 구조는 층이 두 배라 micro batch 1 (init_from은 같은 구조의 가중치여야 한다)
    v25 = TabPFNv2TrainConfig.long_tables("regression", arch="v2.5")
    assert v25.nlayers == 18 and v25.n_thinking_rows == 64 and v25.max_train_rows == 8192
    assert v25.micro_batch == 1 and v25.grad_accumulation(8) == 8


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


def test_paper_preset_clips_qk_only_for_regression():
    assert TabPFNv2TrainConfig.paper("regression").qk_clip == 250.0
    assert TabPFNv2TrainConfig.paper("classification").qk_clip is None


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


def test_qk_logit_bounds_match_the_spectral_norm():
    model = TabPFNv2(TabPFNv2TrainConfig.smoke("regression").model_config())
    torch.manual_seed(0)
    for layer in model.layers:
        torch.nn.init.normal_(layer.sample_attention.qkv.weight, std=0.5)
    bounds = qk_logit_bounds(model)
    e, h = model.config.emsize, model.config.nhead
    d = e // h
    w = model.layers[1].sample_attention.qkv.weight.double()
    for head in range(h):
        q, k = w[head * d : (head + 1) * d], w[e + head * d : e + (head + 1) * d]
        sigma = torch.linalg.matrix_norm(q.T @ k, ord=2).item()
        # 행 순서: 0 = 첫 층 sample, 1 = 둘째 층 feature, 2 = 둘째 층 sample (첫 층 feature는 뺀다)
        assert abs(bounds[2, head].item() - sigma * e / d**0.5) < 1e-3 * sigma * e / d**0.5


def test_qk_clip_caps_the_bound_and_leaves_small_heads_alone():
    model = TabPFNv2(TabPFNv2TrainConfig.smoke("regression").model_config())
    attention = model.layers[0].sample_attention
    with torch.no_grad():
        attention.qkv.weight.normal_(std=0.5)
        attention.qkv.weight[:16].mul_(20.0)  # head 0의 q만 크게 → 그 head만 상한을 넘는다
    before = qk_logit_bounds(model)
    v_before = attention.qkv.weight[2 * 32 :].clone()
    tau = float(before[0, 1:].max()) * 1.5  # head 0만 넘도록
    clipped, worst = apply_qk_clip(model, tau)
    after = qk_logit_bounds(model)
    assert clipped.item() >= 1 and abs(worst.item() - before.max().item()) < 1e-3 * worst.item()
    assert after.max() <= tau * (1 + 1e-4)
    assert torch.allclose(after[0, 1:], before[0, 1:])  # 상한 안쪽 head는 그대로
    assert torch.equal(attention.qkv.weight[2 * 32 :], v_before)  # v는 건드리지 않는다


def test_training_with_qk_clip_keeps_bounds_under_tau(tmp_path):
    config = replace(
        TabPFNv2TrainConfig.smoke("regression"),
        steps=4,
        border_datasets=10,
        qk_clip=5.0,
        qk_clip_every=2,  # 마지막 step(4)에서 clip한다
    )
    result = train_tabpfn_v2(config, tmp_path, mlflow_log=False)
    model, _ = load_checkpoint(result["final_path"])
    assert qk_logit_bounds(model).max() <= 5.0 * (1 + 1e-4)
    with pytest.raises(ValueError, match="qk_clip"):
        TabPFNv2TrainConfig(qk_clip=0.0)


def test_resuming_with_qk_clip_clips_before_the_first_snapshot(tmp_path, monkeypatch):
    config = replace(
        TabPFNv2TrainConfig.smoke("regression"), steps=10, log_every=1, border_datasets=10
    )
    train_tabpfn_v2(replace(config, stop_step=4), tmp_path / "a", mlflow_log=False)
    tau = 1.0  # clip 없이 학습한 checkpoint의 상한보다 훨씬 작다
    state = torch.load(tmp_path / "a" / "latest.pt", weights_only=True)
    plain = TabPFNv2(config.model_config())
    plain.load_state_dict(state["model"])
    assert qk_logit_bounds(plain).max() > tau

    # 마지막 step에서 무너졌다고 보면 첫 snapshot(이어 받은 상태)으로 되돌린 채 끝난다
    monkeypatch.setattr(CollapseGuard, "update", lambda self, window_mean: True)
    train_tabpfn_v2(
        replace(config, stop_step=5, qk_clip=tau),
        tmp_path / "b",
        resume_from=tmp_path / "a" / "latest.pt",
        mlflow_log=False,
    )
    resumed = torch.load(tmp_path / "b" / "latest.pt", weights_only=True)
    assert resumed["step"] == 4
    plain.load_state_dict(resumed["model"])
    assert qk_logit_bounds(plain).max() <= tau * (1 + 1e-4)
