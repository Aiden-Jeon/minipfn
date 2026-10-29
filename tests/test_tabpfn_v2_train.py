"""TabPFN v2 학습 루프 테스트 (smoke 크기, CPU)."""

from dataclasses import replace

import torch

from minipfn.tabpfn_v2.model import load_checkpoint
from minipfn.tabpfn_v2.train import (
    TabPFNv2TrainConfig,
    _make_val_batches,
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
