import os
import subprocess
import sys
from pathlib import Path

import mlflow

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train_pfn.py"


def test_train_script_logs_run_with_eval_metrics(tmp_path):
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    env = {k: v for k, v in os.environ.items() if not k.startswith(("MLFLOW_", "PFN_"))}
    env["MLFLOW_TRACKING_URI"] = uri
    subprocess.run(
        [sys.executable, str(SCRIPT), "--config-name", "small", "--steps", "5"],
        env=env,
        cwd=tmp_path,
        check=True,
    )

    prev_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(uri)
    try:
        runs = mlflow.search_runs(search_all_experiments=True)
        assert list(runs["tags.mlflow.runName"]) == ["small-5steps"]
        assert runs["params.steps"].iloc[0] == "5"
        assert "metrics.rmse_mean" in runs and "metrics.nll_gap" in runs
        client = mlflow.MlflowClient()
        artifacts = {a.path for a in client.list_artifacts(runs["run_id"].iloc[0])}
        # best/final 모델 로깅은 test_train_checkpoint가 확인한다
        assert {"loss_curve.png", "pfn_overlay.png"} <= artifacts
    finally:
        mlflow.set_tracking_uri(prev_uri)
