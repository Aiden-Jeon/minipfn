"""학습 가속 옵션 테스트 (issue #46): schedule-free AdamW, LAWA, fused·TF32·compile 설정,
목표 도달 시간."""

from dataclasses import replace

import pytest
import torch

from minipfn import models as registry
from minipfn.training.config import TrainConfig
from minipfn.training.loop import _average_states, train
from minipfn.training.schedule_free import ScheduleFreeAdamW, warmup_factor
from minipfn.training.speed import ValPoint, time_to_target, val_loss_at


def _reference_schedule_free(params, grads_per_step, lrs, betas=(0.9, 0.999), eps=1e-8):
    """Defazio et al. 2024의 AdamW 갱신을 텐서 하나씩 그대로 적은 것 (비교용)."""
    beta1, beta2 = betas
    y = [p.clone() for p in params]
    z = [p.clone() for p in params]
    v = [torch.zeros_like(p) for p in params]
    lr_max, weight_sum = 0.0, 0.0
    for k, (grads, lr) in enumerate(zip(grads_per_step, lrs, strict=True), start=1):
        lr_max = max(lr_max, lr)
        weight_sum += lr_max**2
        c = lr_max**2 / weight_sum if weight_sum else 0.0
        for i, g in enumerate(grads):
            v[i] = beta2 * v[i] + (1 - beta2) * g * g
            normed = g / ((v[i] / (1 - beta2**k)).sqrt() + eps)
            y[i] = (1 - c) * y[i] + c * z[i] + lr * (beta1 * (1 - c) - 1) * normed
            z[i] = z[i] - lr * normed
    x = [(yi - (1 - beta1) * zi) / beta1 for yi, zi in zip(y, z, strict=True)]
    return y, x


def test_schedule_free_matches_the_paper_update():
    torch.manual_seed(0)
    params = [torch.nn.Parameter(torch.randn(3, 4)), torch.nn.Parameter(torch.randn(5))]
    start = [p.detach().clone() for p in params]
    opt = ScheduleFreeAdamW(params, lr=0.1)
    grads_per_step, lrs = [], []
    for step in range(12):
        grads = [torch.randn_like(p) for p in params]
        lr = 0.1 * warmup_factor(step, 4)
        for group in opt.param_groups:
            group["lr"] = lr
        for p, g in zip(params, grads, strict=True):
            p.grad = g.clone()
        opt.step()
        grads_per_step.append(grads)
        lrs.append(lr)
    y, x = _reference_schedule_free(start, grads_per_step, lrs)
    for p, yi, xi in zip(params, y, x, strict=True):
        assert torch.allclose(p, yi, atol=1e-6)
        assert torch.allclose(opt.average(p), xi, atol=1e-6)
    # 평균 x를 파라미터에 넣은 뒤(checkpoint에서 읽은 것처럼) 학습 점 y로 되돌린다
    with torch.no_grad():
        for p in params:
            p.copy_(opt.average(p))
    opt.average_to_train_point()
    for p, yi in zip(params, y, strict=True):
        assert torch.allclose(p, yi, atol=1e-5)


def test_warmup_factor_is_linear_then_flat():
    assert [warmup_factor(s, 4) for s in (0, 2, 4, 100)] == [0.0, 0.5, 1.0, 1.0]


def test_options_default_off_and_bad_combinations():
    config = TrainConfig()
    assert not (config.compile or config.fused_adam or config.tf32 or config.schedule_free)
    assert config.weight_average == 0
    with pytest.raises(ValueError, match="qk_clip"):
        TrainConfig(task="regression", schedule_free=True, qk_clip=250.0)
    with pytest.raises(ValueError, match="weight_average"):
        TrainConfig(weight_average=-1)


def _smoke(**changes):
    return replace(
        TrainConfig.smoke("classification"),
        steps=6,
        val_every=3,
        log_every=3,
        checkpoint_every=3,
        keep_every=None,
        collapse_threshold=None,
        **changes,
    )


def test_schedule_free_trains_saves_the_average_and_resumes(tmp_path):
    config = _smoke(schedule_free=True)
    result = train(config, tmp_path / "a", mlflow_log=False)
    state = torch.load(tmp_path / "a" / "latest.pt", weights_only=True)
    model, _ = registry.load_checkpoint(result["final_path"])
    # 학습 checkpoint와 최종 파일 모두 평균 x를 담는다 (같은 step 6)
    for name, tensor in model.state_dict().items():
        assert torch.equal(state["model"][name], tensor), name
    # step 3 checkpoint에서 이어 6까지
    config3 = replace(config, stop_step=3)
    train(config3, tmp_path / "b", mlflow_log=False)
    resumed = train(
        config, tmp_path / "c", resume_from=tmp_path / "b" / "latest.pt", mlflow_log=False
    )
    assert resumed["history"]["step"][-1] == 6
    assert all(torch.isfinite(torch.tensor(resumed["history"]["train_loss"])))


def test_weight_average_saves_the_mean_of_recent_weights(tmp_path):
    config = _smoke(weight_average=2, weight_average_every=2)
    train(config, tmp_path, mlflow_log=False)
    averaged, _ = registry.load_checkpoint(tmp_path / "final_avg.pt")
    final, _ = registry.load_checkpoint(tmp_path / "final.pt")
    # 평균은 마지막 가중치와 다르다 (step 4·6의 평균)
    assert any(
        not torch.equal(a, b)
        for a, b in zip(averaged.state_dict().values(), final.state_dict().values(), strict=True)
    )


def test_average_states_keeps_dtypes():
    a = {"w": torch.tensor([1.0, 3.0]), "n": torch.tensor(2)}
    b = {"w": torch.tensor([3.0, 5.0]), "n": torch.tensor(2)}
    out = _average_states([a, b])
    assert torch.equal(out["w"], torch.tensor([2.0, 4.0])) and out["n"].dtype == torch.int64


def test_fused_and_tf32_are_no_ops_on_cpu(tmp_path):
    base = train(_smoke(), tmp_path / "a", mlflow_log=False)
    fast = train(_smoke(fused_adam=True, tf32=True), tmp_path / "b", mlflow_log=False)
    assert base["history"]["train_loss"] == fast["history"]["train_loss"]
    torch.set_float32_matmul_precision("highest")  # 다른 테스트에 남기지 않는다


def test_time_to_target():
    points = [ValPoint(1000, 100.0, 1.2), ValPoint(2000, 200.0, 1.0), ValPoint(3000, 300.0, 0.9)]
    assert val_loss_at(points, 2000) == 1.0
    assert time_to_target(points, 1.0).train_seconds == 200.0
    assert time_to_target(points, 0.5) is None
    with pytest.raises(ValueError, match="step 1500"):
        val_loss_at(points, 1500)


def test_train_script_takes_the_speed_options(monkeypatch):
    import importlib.util
    from pathlib import Path

    for name in ("TABPFN_V2_COMPILE", "TABPFN_V2_SCHEDULE_FREE", "TABPFN_V2_WEIGHT_AVERAGE"):
        monkeypatch.delenv(name, raising=False)
    path = Path(__file__).parents[1] / "scripts" / "tabpfn_v2" / "train.py"
    spec = importlib.util.spec_from_file_location("train_script_speed", path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    config, _ = script.build_config(script.parse_args(["--config-name", "smoke"]))
    assert not (config.compile or config.schedule_free) and config.weight_average == 0
    monkeypatch.setenv("TABPFN_V2_SCHEDULE_FREE", "1")
    args = script.parse_args(
        ["--config-name", "smoke", "--compile", "--weight-average", "5", "--val-every", "7"]
    )
    config, _ = script.build_config(args)
    assert config.compile and config.schedule_free and config.weight_average == 5
    assert config.val_every == 7
