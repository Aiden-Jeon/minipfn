import importlib.util
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import mlflow
import pytest
import torch

from minipfn.recipes import tabpfn_v1 as spec
from minipfn.recipes.tabpfn_v1 import TabPFNConfig, load_tabpfn, lr_factor, train_tabpfn

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "tabpfn" / "train.py"


def _eval_script():
    path = SCRIPT.with_name("eval.py")
    module_spec = importlib.util.spec_from_file_location("eval_tabpfn", path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


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
    args = ["--config-name", "smoke", "--steps", "4", "--keep-every", "2", "--out-dir", "out"]
    subprocess.run([sys.executable, str(SCRIPT), *args], env=env, cwd=tmp_path, check=True)
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
        # latest.pt는 덮어쓰고, keep_every step마다 따로 남긴다
        checkpoints = {a.path for a in client.list_artifacts(run_id, "checkpoints")}
        assert checkpoints == {
            "checkpoints/latest.pt",
            "checkpoints/step_0000002.pt",
            "checkpoints/step_0000004.pt",
        }
        # 실제로 돈 스크립트와 src가 run에 남는다
        code = {a.path for a in client.list_artifacts(run_id, "code")}
        assert {"code/train.py", "code/minipfn"} <= code
        recipe_code = {a.path for a in client.list_artifacts(run_id, "code/minipfn/recipes")}
        assert "code/minipfn/recipes/tabpfn_v1.py" in recipe_code
        tags = runs.iloc[0]
        assert tags["tags.command"] == "train.py " + " ".join(args)
        assert tags["tags.git_commit"]

        # 평가 스크립트는 run의 최종 모델과 학습 중 checkpoint를 runs:/ URI로 읽는다
        eval_script = _eval_script()
        run_uri = f"runs:/{run_id}"
        final = eval_script.load_model(f"{run_uri}/model/tabpfn_final.pt", tmp_path / "e1")
        kept = eval_script.load_model(f"{run_uri}/checkpoints/step_0000004.pt", tmp_path / "e2")
        for a, b in zip(final.state_dict().values(), kept.state_dict().values(), strict=True):
            assert torch.equal(a, b)  # 마지막 step의 checkpoint = 최종 모델
        with pytest.raises(SystemExit, match="runs:/"):
            eval_script.load_model(str(tmp_path / "out" / "step_0000002.pt"), tmp_path / "e3")

        # 모델 설정 param이 없는 run(이어서 학습한 run 등)이면 무엇이 없는지 알리고 멈춘다
        with mlflow.start_run() as bare:
            mlflow.log_artifact(str(tmp_path / "out" / "step_0000002.pt"), "checkpoints")
        with pytest.raises(SystemExit, match="emsize"):
            eval_script.load_model(
                f"runs:/{bare.info.run_id}/checkpoints/step_0000002.pt", tmp_path / "e4"
            )
    finally:
        mlflow.set_tracking_uri(prev_uri)
