import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import mlflow
import pytest
import torch

from minipfn.tabpfn import spec
from minipfn.tabpfn.train import TabPFNConfig, load_tabpfn, lr_factor, train_tabpfn

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train_tabpfn.py"


def test_paper_config_matches_spec():
    config = TabPFNConfig.paper()
    assert config.batch_size == 512 and config.steps == 18_000
    assert config.grad_accumulation(world_size=8) == 8  # v1 aggregate_k_gradients
    assert config.grad_accumulation(world_size=1) == 64
    with pytest.raises(ValueError):
        replace(config, batch_size=500).grad_accumulation(world_size=8)


def test_lr_schedule_warmup_then_cosine():
    assert lr_factor(0, 10, 100) == 0.0
    assert lr_factor(5, 10, 100) == 0.5
    assert lr_factor(10, 10, 100) == 1.0
    assert abs(lr_factor(55, 10, 100) - 0.5) < 1e-9
    # 논문은 400 epoch 스케줄의 18,000 step에서 멈췄으므로 lr이 0까지 내려가지 않는다.
    assert lr_factor(spec.TOTAL_STEPS, spec.WARMUP_STEPS, spec.SCHEDULE_STEPS) > 0.5


def test_smoke_training_and_resume(tmp_path):
    config = replace(TabPFNConfig.smoke(), steps=6, log_every=2, val_every=3, checkpoint_every=3)
    first = train_tabpfn(config, tmp_path / "a", mlflow_log=False)
    assert len(first["history"]["train_ce"]) == 3
    model, loaded_config = load_tabpfn(first["final_path"])
    assert loaded_config == config
    out = model(torch.randn(1, 5, 100), torch.zeros(1, 5), torch.randn(1, 3, 100))
    assert out.shape == (1, 3, 10)

    # 저장된 상태에서 이어서 학습하면 남은 step만 돌고, 이전 구간의 기록도 이어 붙는다.
    longer = replace(config, steps=8)
    resumed = train_tabpfn(
        longer, tmp_path / "b", resume_from=tmp_path / "a" / "latest.pt", mlflow_log=False
    )
    assert resumed["history"]["step"] == [2, 4, 6, 8]
    assert resumed["history"]["train_ce"][:3] == first["history"]["train_ce"]
    assert resumed["history"]["val_step"] == [3, 6, 8]


def test_train_script_logs_to_mlflow(tmp_path):
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    env = {k: v for k, v in os.environ.items() if not k.startswith(("MLFLOW_", "TABPFN_"))}
    env["MLFLOW_TRACKING_URI"] = uri
    subprocess.run(
        [sys.executable, str(SCRIPT), "--config-name", "smoke", "--steps", "4", "--out-dir", "out"],
        env=env,
        cwd=tmp_path,
        check=True,
    )
    prev_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(uri)
    try:
        runs = mlflow.search_runs(search_all_experiments=True)
        assert list(runs["tags.mlflow.runName"]) == ["tabpfn-smoke-4steps"]
        assert "metrics.train_ce" in runs and "metrics.val_ce" in runs
        client = mlflow.MlflowClient()
        run_id = runs["run_id"].iloc[0]
        artifacts = {a.path for a in client.list_artifacts(run_id)}
        assert {"loss_curve.png", "model", "checkpoints", "code"} <= artifacts
        # 실제로 돈 스크립트와 src가 run에 남는다
        code = {a.path for a in client.list_artifacts(run_id, "code")}
        assert {"code/train_tabpfn.py", "code/minipfn"} <= code
        tabpfn_code = {a.path for a in client.list_artifacts(run_id, "code/minipfn/tabpfn")}
        assert "code/minipfn/tabpfn/train.py" in tabpfn_code
        tags = runs.iloc[0]
        assert tags["tags.command"] == "train_tabpfn.py --config-name smoke --steps 4 --out-dir out"
        assert tags["tags.git_commit"]
    finally:
        mlflow.set_tracking_uri(prev_uri)
