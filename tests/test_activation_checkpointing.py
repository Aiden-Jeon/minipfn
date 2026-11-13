"""activation checkpointing (issue #79): `ColumnRowTransformer.activation_checkpointing_rows`와 학습
설정 `TrainConfig.activation_checkpointing_rows`. 켜도 손실·gradient가 같고, 기준보다 작은 표·검증·
추론에서는 쓰지 않는다."""

import inspect
from dataclasses import replace

import numpy as np
import pytest
import torch

from minipfn.models import column_row_transformer as M
from minipfn.recipes import tabpfn_v3
from minipfn.training import loop
from minipfn.training.config import TrainConfig
from minipfn.training.loop import train


def _model(**changes) -> M.ColumnRowTransformer:
    # TabPFN-3.5의 부품(QK-norm, Fourier 셀 encoder, 정답 norm, 출력 앞 MLP, soft-cap) 전부, 작게
    torch.manual_seed(0)
    config = M.ColumnRowTransformerConfig.tiny(
        task="multitask",
        qk_norm=True,
        encoder="fourier",
        target_norm=True,
        head_mlp=True,
        n_buckets=7,
        logit_softcap=5.0,
        **changes,
    )
    return M.ColumnRowTransformer(config).train()


def _batch(task: str, b: int = 2, r: int = 40, c: int = 5, n_train: int = 30):
    g = torch.Generator().manual_seed(1)
    x = torch.randn(b, r, c, generator=g)
    if task == "classification":
        return x, torch.randint(0, 3, (b, n_train), generator=g).float()
    return x, torch.randn(b, n_train, generator=g)


def _grads(model, task, rows):
    model.activation_checkpointing_rows = rows
    model.zero_grad(set_to_none=True)
    x, y = _batch(task)
    loss = model(x, y, task).square().mean()
    loss.backward()
    grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    return loss.detach(), grads


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_checkpointing_same_gradients(task, monkeypatch):
    # float32 CPU: checkpoint는 같은 forward를 다시 계산하므로 손실·gradient가 비트 단위로 같다
    # (분류 decoder가 score 행렬을 만드는 경로). score 행렬을 만들지 않는 decoder(기본)는 계산
    # 순서만 달라서 가깝다
    model = _model()
    loss, grads = _grads(model, task, None)
    monkeypatch.setattr(M, "CHECKPOINT_FUSED_DECODER", False)
    same_loss, same = _grads(model, task, 0)
    assert torch.equal(loss, same_loss) and same.keys() == grads.keys()
    assert all(torch.equal(grads[n], same[n]) for n in grads)
    monkeypatch.setattr(M, "CHECKPOINT_FUSED_DECODER", True)
    close_loss, close = _grads(model, task, 0)
    assert close.keys() == grads.keys()
    assert torch.allclose(loss, close_loss, rtol=1e-5)
    flat = torch.cat([g.flatten() for g in grads.values()])
    diff = torch.cat([(close[n] - grads[n]).flatten() for n in grads])
    assert diff.norm() <= 1e-5 * flat.norm()


def test_checkpointing_only_large_training_tables(monkeypatch):
    # 행 수가 기준보다 많은 학습 forward에서만 쓴다: 블록마다(열 1 + 행 2 + ICL 2) 한 번씩
    calls = []
    real = M.checkpoint

    def counting(block, *args, **kwargs):
        calls.append(type(block).__name__)
        return real(block, *args, **kwargs)

    monkeypatch.setattr(M, "checkpoint", counting)
    model = _model()
    x, y = _batch("regression")  # 행 40개
    model.activation_checkpointing_rows = 40
    model(x, y, "regression").sum().backward()
    assert calls == []
    model.activation_checkpointing_rows = 39
    model(x, y, "regression").sum().backward()
    assert calls == ["InducedBlock", "Block", "Block", "Block", "Block"]
    calls.clear()
    with torch.no_grad():  # gradient가 없으면 다시 계산할 것도 없다
        model(x, y, "regression")
    model.eval()  # 검증·추론
    model(x, y, "regression")
    model(x, y, "regression", inference=True)
    assert calls == []


def test_checkpointing_config():
    config = TrainConfig.smoke("classification")
    assert config.activation_checkpointing_rows is None  # 기본은 끔
    with pytest.raises(ValueError, match="cell_transformer"):  # 그 모델에는 없다
        replace(config, activation_checkpointing_rows=10)
    v35 = tabpfn_v3.with_architecture(config, "v3.5", keep_size=True)
    assert replace(v35, activation_checkpointing_rows=0).activation_checkpointing_rows == 0
    with pytest.raises(ValueError, match="0 이상"):
        replace(v35, activation_checkpointing_rows=-1)
    # stage 3에서만 켠다
    s1, s2, s3 = (tabpfn_v3.curriculum(n) for n in (1, 2, 3))
    assert s1.activation_checkpointing_rows is None and s2.activation_checkpointing_rows is None
    assert s3.activation_checkpointing_rows == s2.max_rows  # stage 2 크기까지는 끈다
    # 모델 설정(checkpoint 파일·config.json)에는 남지 않는다
    assert not hasattr(s3.model_config(), "activation_checkpointing_rows")


def _smoke(**changes):
    config = tabpfn_v3.with_architecture(TrainConfig.smoke("multitask"), "v3.5", keep_size=True)
    small = dict(
        size_rule="train_fraction",
        max_features=6,
        min_rows=48,
        max_rows=80,
        max_cells=80 * 6,
        log_uniform_rows=True,
        steps=4,
        warmup_steps=1,
        val_every=2,
        log_every=1,
        checkpoint_every=2,
        keep_every=None,
        collapse_threshold=None,
        border_datasets=20,
    )
    return replace(config, **{**small, **changes})


def test_checkpointing_trains_same(tmp_path, monkeypatch):
    # 학습 루프: 켠 run(모든 표)이 끈 run과 학습 손실이 같다(score 행렬을 만드는 decoder로 비교).
    # 검증 batch는 추론 경로로 계산한다
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    monkeypatch.setattr(M, "CHECKPOINT_FUSED_DECODER", False)
    inference_calls = []
    real = loop._validate

    signature = inspect.signature(real)

    def recording(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        inference_calls.append(bound.arguments["inference_rows"])
        return real(*args, **kwargs)

    monkeypatch.setattr(loop, "_validate", recording)
    off = train(_smoke(), tmp_path / "off", mlflow_log=False)
    assert set(inference_calls) == {None}
    inference_calls.clear()
    on = train(_smoke(activation_checkpointing_rows=0), tmp_path / "on", mlflow_log=False)
    assert set(inference_calls) == {0}
    assert on["history"]["train_loss"] == off["history"]["train_loss"]
    # 검증 손실은 추론 경로(행 묶음, score 행렬 없는 decoder)라 계산 순서만 다르다
    assert np.allclose(on["history"]["val_loss"], off["history"]["val_loss"], rtol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="블록 compile·flex attention은 GPU")
def test_checkpointing_trains_compiled(tmp_path, monkeypatch):
    # 가속 옵션 compile(블록마다)과 같이 쓴다. 다시 계산하는 forward도 compile한 것이다
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    config = _smoke(activation_checkpointing_rows=60, compile=True, logit_softcap=50.0, stop_step=3)
    out = train(config, tmp_path, mlflow_log=False)
    assert all(np.isfinite(out["history"]["train_loss"]))
