"""학습 진단 지표 테스트 (`training.diagnostics`, issue #64·#65·#68)."""

import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import mlflow
import pytest
import torch
from torch import nn

from minipfn.recipes import tabpfn_v3
from minipfn.training.config import TrainConfig
from minipfn.training.diagnostics import (
    TIMING_FIELDS,
    GroupNormWindow,
    RankTiming,
    TaskGradientCosine,
    group_norms,
    parameter_groups,
    rank_timing_metrics,
    shared_groups,
    write_rank_timing,
)
from minipfn.training.loop import train
from minipfn.training.muon import muon_parameters


def _multitask(**changes):
    config = tabpfn_v3.with_architecture(TrainConfig.smoke("multitask"), "v3.5", keep_size=True)
    return replace(
        config,
        steps=8,
        warmup_steps=1,
        log_every=4,
        val_every=8,
        checkpoint_every=8,
        keep_every=None,
        collapse_threshold=None,
        optimizer="muon",
        **changes,
    )


def test_groups_split_by_stage_task_and_optimizer():
    model = _multitask().build_model()
    muon_ids = {id(p) for p, _ in muon_parameters(model)[0]}
    groups = parameter_groups(model, muon_ids)
    # 모든 파라미터가 한 묶음에 한 번씩
    ids = [id(p) for params in groups.values() for p in params]
    assert sorted(ids) == sorted(id(p) for p in model.parameters())
    assert {"column_stage/muon", "row_stage/muon", "row_icl/muon", "row_icl/adam"} <= set(groups)
    # task 하나만 쓰는 부품(정답 encoder, 출력층)은 따로
    assert {"cell_encoder_classification/adam", "output_regression/adam"} <= set(groups)
    assert all(id(p) in muon_ids for p in groups["row_icl/muon"])
    shared = shared_groups(groups, ("classification", "regression"))
    assert "row_icl/muon" in shared and "output_regression/adam" not in shared
    # Adam만 쓰면 optimizer는 붙지 않는다
    assert "row_icl" in parameter_groups(model)


def test_group_norms_match_a_flat_norm_and_skip_unused():
    model = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 2))
    groups = {"a": list(model[0].parameters()), "b": list(model[1].parameters())}
    flat = torch.cat([p.flatten() for p in model[0].parameters()])
    assert group_norms(groups)["a"].item() == pytest.approx(flat.norm().item(), rel=1e-6)
    model(torch.randn(5, 3))[:, 0].sum().backward()
    model[1].weight.grad = None
    model[1].bias.grad = None
    assert set(group_norms(groups, grads=True)) == {"a"}


def test_grad_norm_window_keeps_mean_and_max_per_task():
    window = GroupNormWindow({}, multitask=True)
    for task, value in [("classification", 1.0), ("regression", 5.0), ("classification", 3.0)]:
        window.update(task, {"g": torch.tensor(value)}, torch.tensor(value))
    m = window.metrics()
    assert m["grad_norm/classification/g"] == pytest.approx(2.0)
    assert m["grad_norm_max/classification/g"] == pytest.approx(3.0)
    assert m["grad_norm/regression/g"] == pytest.approx(5.0)
    assert m["grad_norm_median"] == pytest.approx(3.0) and m["grad_norm_max"] == pytest.approx(5.0)


def _set_grads(params, grads):
    for p, g in zip(params, grads, strict=True):
        p.grad = None if g is None else g.clone()


def test_task_gradient_cosine_over_four_steps():
    a, b = nn.Parameter(torch.zeros(3)), nn.Parameter(torch.zeros(2))
    head = nn.Parameter(torch.zeros(2))
    groups = {"shared": [a, b], "output_regression": [head]}
    cos = TaskGradientCosine(groups, ("classification", "regression"), log_every=10)
    assert list(cos.groups) == ["shared"]
    # 기록 step(step 9, done 10)과 그 앞 3 step
    assert [s for s in range(25) if cos.wants(s)] == [6, 7, 8, 9, 16, 17, 18, 19]
    g_cls = [torch.tensor([1.0, 0.0, 0.0]), torch.tensor([1.0, 0.0])]
    g_reg = [torch.tensor([-1.0, 0.0, 0.0]), torch.tensor([0.0, 1.0])]  # 첫 묶음 반대, cosine −1/2
    out = None
    for step in range(6, 10):
        task = ("classification", "regression")[step % 2]
        _set_grads([a, b], g_cls if task == "classification" else g_reg)
        head.grad = torch.ones(2) if task == "regression" else None
        out = cos.update(step, task, finite=True)
        assert (out is None) == (step < 9)
    assert out["grad_cos/shared"] == pytest.approx(-0.5, abs=1e-3)
    assert out["grad_cos_same/classification/shared"] == pytest.approx(1.0, abs=1e-3)
    assert out["grad_cos_same/regression/shared"] == pytest.approx(1.0, abs=1e-3)
    assert not cos.tasks  # 다음 window를 위해 비웠다


def test_task_gradient_cosine_drops_an_incomplete_window():
    a = nn.Parameter(torch.zeros(2))
    cos = TaskGradientCosine({"shared": [a]}, ("classification", "regression"), log_every=10)
    a.grad = torch.ones(2)
    # 유한하지 않은 step이 끼면 그 window는 내지 않는다
    assert cos.update(6, "classification", finite=True) is None
    assert cos.update(7, "regression", finite=False) is None
    assert cos.update(8, "classification", finite=True) is None
    assert cos.update(9, "regression", finite=True) is None
    # window 가운데서 시작하면(이어 학습) 다음 window까지 모으지 않는다
    assert cos.update(18, "classification", finite=True) is None
    assert cos.update(19, "regression", finite=True) is None
    assert not cos.tasks
    for step in range(26, 30):
        out = cos.update(step, ("classification", "regression")[step % 2], finite=True)
    assert out is not None


def _gathered(fwd, cells):
    """(rank, step) 시간·셀 수로 `RankTiming.gather` 모양의 기록을 만든다."""
    fwd, cells = torch.tensor(fwd, dtype=torch.float64), torch.tensor(cells, dtype=torch.float64)
    ranks, steps = fwd.shape
    out = torch.zeros(ranks, steps, 2 + len(TIMING_FIELDS), dtype=torch.float64)
    out[..., 0] = torch.arange(steps, dtype=torch.float64)
    out[..., 2] = fwd
    out[..., 3] = 100.0 - fwd  # 빨리 끝난 GPU가 더 오래 기다린다
    out[..., 4] = 10.0
    out[..., 5] = 5.0
    out[..., 6] = 3.0
    out[..., 7] = cells
    return out


def test_rank_timing_metrics_find_size_driven_waiting():
    cells = [[100.0, 300.0], [200.0, 100.0], [400.0, 200.0]]
    fwd = [[c / 10 for c in row] for row in cells]  # forward 시간이 셀 수에 비례
    m = rank_timing_metrics(_gathered(fwd, cells))
    assert m["rank_fwd_cells_corr"] == pytest.approx(1.0)
    assert m["rank_bwd_fwd_corr"] == pytest.approx(-1.0)
    assert m["rank_slowest_is_largest"] == pytest.approx(1.0)
    assert m["rank_fwd_ms_slowest"] == pytest.approx(35.0)  # step마다 40, 30의 평균
    assert m["rank_fwd_imbalance"] == pytest.approx(35.0 / (1300 / 6 / 10))
    # GPU가 하나면 GPU 사이 지표는 없다
    single = rank_timing_metrics(_gathered(fwd[:1], cells[:1]))
    assert "rank_fwd_cells_corr" not in single and "rank_fwd_ms" in single


def test_rank_timing_records_and_writes(tmp_path):
    timing = RankTiming(torch.device("cpu"))
    for step in range(3):
        for _ in range(2):  # gradient accumulation 2
            start = timing.mark()
            fwd = timing.mark()
            timing.add_micro(start, fwd, timing.mark(), torch.zeros(4, 10 + step, 3), 6)
        timing.step_done(step, step % 2)
    gathered = timing.gather(world_size=1)
    assert gathered.shape == (1, 3, 2 + len(TIMING_FIELDS))
    assert gathered[0, :, 7].tolist() == [2 * 4 * 10 * 3, 2 * 4 * 11 * 3, 2 * 4 * 12 * 3]
    assert not timing.records
    path = tmp_path / "t.jsonl"
    write_rank_timing(path, gathered, ("classification", "regression"))
    lines = [json.loads(s) for s in path.read_text().splitlines()]
    assert [line["task"] for line in lines] == ["classification", "regression", "classification"]
    assert lines[1]["rows"] == [11.0] and lines[1]["n_train"] == [6.0]


def test_grad_cosine_needs_multitask():
    with pytest.raises(ValueError, match="multitask"):
        TrainConfig(task="classification", grad_cosine=True)


def test_training_logs_diagnostics_without_changing_the_run(tmp_path, monkeypatch):
    # 회귀 구간 경계를 만드는 process pool의 worker도 같은 hash seed를 쓰게 한다 (학습 yaml처럼.
    # 다르면 경계가 run마다 달라져 진단과 상관없이 손실이 달라진다)
    monkeypatch.setenv("PYTHONHASHSEED", "1")
    plain = train(_multitask(), tmp_path / "plain", mlflow_log=False)
    prev_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    try:
        mlflow.set_experiment("test-diagnostics")
        with mlflow.start_run() as run:
            diag = train(_multitask(grad_cosine=True, rank_timing=True), tmp_path / "diag")
        metrics = mlflow.MlflowClient().get_run(run.info.run_id).data.metrics
    finally:
        mlflow.set_tracking_uri(prev_uri)
    # 진단은 읽기만 한다
    assert diag["history"]["train_loss"] == plain["history"]["train_loss"]
    for key in (
        "weight_norm/row_icl/muon",
        "grad_norm/classification/row_icl/muon",
        "grad_norm_max/regression/output_regression/adam",
        "grad_norm_median",
        "grad_cos/row_icl/muon",
        "grad_cos_same/classification/row_icl/muon",
        "rank_fwd_ms",
        "rank_cells",
    ):
        assert key in metrics, key
    assert -1.0 <= metrics["grad_cos/row_icl/muon"] <= 1.0
    lines = (tmp_path / "diag" / "rank_timing_from_0000000.jsonl").read_text().splitlines()
    assert [json.loads(s)["step"] for s in lines] == list(range(8))


def _script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "tabpfn_v2" / "train.py"
    spec = importlib.util.spec_from_file_location("train_script_diagnostics", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_script_flags_turn_on_diagnostics(monkeypatch):
    script = _script()
    base = ["--config-name", "smoke", "--arch", "v3.5", "--task", "multitask"]
    config, _ = script.build_config(script.parse_args(base))
    assert not config.grad_cosine and not config.rank_timing
    config, _ = script.build_config(script.parse_args([*base, "--grad-cosine", "--rank-timing"]))
    assert config.grad_cosine and config.rank_timing
    monkeypatch.setenv("TABPFN_V2_RANK_TIMING", "1")
    config, _ = script.build_config(script.parse_args(base))
    assert config.rank_timing and not config.grad_cosine
