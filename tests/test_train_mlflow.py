import mlflow

from minipfn import PFNConfig, build_criterion, build_model, build_prior_sampler, train_pfn
from minipfn.utils import set_seed


def test_train_pfn_logs_params_and_metrics(tmp_path):
    prev_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    try:
        mlflow.set_experiment("test-pfn")
        set_seed(0)
        config = PFNConfig(steps=20, batch_size=4, seq_len=20, max_context=10)
        with mlflow.start_run() as run:
            train_pfn(
                build_model(config),
                build_prior_sampler(config),
                build_criterion(config),
                config,
                progress=False,
                mlflow_log=True,
                metric_every=5,
            )

        client = mlflow.MlflowClient()
        logged = client.get_run(run.info.run_id)
        assert logged.data.params["emsize"] == str(config.emsize)
        assert logged.data.params["steps"] == str(config.steps)
        assert "final_train_nll" in logged.data.metrics
        # step 0, 5, 10, 15 + 마지막 step
        history = client.get_metric_history(run.info.run_id, "train_nll")
        assert sorted(m.step for m in history) == [0, 5, 10, 15, 19]
    finally:
        mlflow.set_tracking_uri(prev_uri)


def test_train_pfn_opens_and_closes_its_own_run(tmp_path):
    prev_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    try:
        mlflow.set_experiment("test-pfn")
        config = PFNConfig(steps=3, batch_size=2, seq_len=12, max_context=6)
        train_pfn(
            build_model(config),
            build_prior_sampler(config),
            build_criterion(config),
            config,
            progress=False,
            mlflow_log=True,
            run_name="own-run",
        )
        assert mlflow.active_run() is None
        runs = mlflow.search_runs(experiment_names=["test-pfn"])
        assert list(runs["tags.mlflow.runName"]) == ["own-run"]
    finally:
        mlflow.set_tracking_uri(prev_uri)
