import pickle

import mlflow
import torch

from minipfn import (
    PFNConfig,
    PFNTransformer,
    build_criterion,
    build_model,
    build_prior_sampler,
    load_checkpoint,
    load_logged_model,
    pfn_predict,
    save_checkpoint,
    train_pfn,
)
from minipfn.train import alias_legacy_package
from minipfn.utils import set_seed


def test_save_load_checkpoint_roundtrip(tmp_path):
    set_seed(0)
    config = PFNConfig(steps=1, batch_size=2, seq_len=12, max_context=6)
    model = build_model(config)
    path = str(tmp_path / "ck.pt")
    save_checkpoint(model, config, path)

    loaded = load_checkpoint(path)
    assert loaded["config"].emsize == config.emsize
    for k, v in model.state_dict().items():
        assert torch.equal(v.cpu(), loaded["model"].state_dict()[k].cpu())
    # 불러온 모델+criterion으로 추론이 된다
    pred = pfn_predict(
        loaded["model"], loaded["criterion"], torch.randn(3, 1), torch.randn(3), torch.randn(4, 1)
    )
    assert pred["mean"].shape == (4,)


def test_train_logs_best_final_and_loads(tmp_path):
    prev = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    try:
        mlflow.set_experiment("test-pfn-models")
        set_seed(0)
        config = PFNConfig(steps=20, batch_size=4, seq_len=20, max_context=10)
        with mlflow.start_run() as run:
            hist = train_pfn(
                build_model(config),
                build_prior_sampler(config),
                build_criterion(config),
                config,
                progress=False,
                mlflow_log=True,
                metric_every=5,
            )
        data = mlflow.MlflowClient().get_run(run.info.run_id).data
        assert "val_nll" in data.metrics
        assert "best_val_nll" in data.metrics
        assert hist["best_step"] >= 0

        # 로깅된 final 모델을 불러와 criterion까지 재구성되고 추론이 된다
        loaded = load_logged_model(f"runs:/{run.info.run_id}/final_model")
        assert loaded["criterion"] is not None
        assert loaded["config"].emsize == config.emsize
        pred = pfn_predict(
            loaded["model"],
            loaded["criterion"],
            torch.randn(3, 1),
            torch.randn(3),
            torch.randn(5, 1),
        )
        assert pred["mean"].shape == (5,)
    finally:
        mlflow.set_tracking_uri(prev)


def test_legacy_package_alias_unpickles_old_module_paths():
    # 이름을 바꾸기 전에 pickle로 로깅한 모델은 pfn_cookbook.* 경로로 클래스를 찾는다.
    # protocol 0은 모듈 경로를 텍스트로 쓰므로 바이트 치환으로 옛 pickle을 만들 수 있다.
    old = pickle.dumps((PFNTransformer, PFNConfig()), protocol=0).replace(
        b"minipfn.", b"pfn_cookbook."
    )
    assert b"pfn_cookbook.model" in old
    alias_legacy_package()
    cls, config = pickle.loads(old)
    assert cls is PFNTransformer and config == PFNConfig()
