"""분산 Muon 테스트 (issue #56). 여러 process(gloo, CPU)로 나눠 계산한 갱신이 1 process 계산과
비트 단위로 같은지 본다."""

import importlib.util
import os
import socket
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from minipfn.recipes import tabpfn_v3
from minipfn.training.config import TrainConfig
from minipfn.training.loop import train
from minipfn.training.muon import MuonAdam, _piece_cost, muon_parameters, shard_pieces

STEPS = 3


def _model():
    # v3.5: q·k·v를 나누는 weight(3조각), 세로로 긴 행렬(MLP 첫 층), 가로로 긴 행렬이 다 있다
    config = tabpfn_v3.with_architecture(TrainConfig.smoke("multitask"), "v3.5", keep_size=True)
    torch.manual_seed(0)
    return config.build_model()


def _run(process_group=None, scope: str = "blocks") -> dict:
    """같은 gradient로 STEPS번 갱신한 가중치와 optimizer 상태. step 1에서는 Muon 파라미터 하나와
    마지막 파라미터 하나에 gradient가 없다(multitask에서 그 step에 쓰지 않은 층). `scope="all"`이면
    1차원 파라미터(편 행렬 (n, 1))도 나눠 계산한다."""
    model = _model()
    opt = MuonAdam(model, lr=1e-2, weight_decay=0.01, process_group=process_group, scope=scope)
    muon, adam = muon_parameters(model, scope)
    for step in range(STEPS):
        gen = torch.Generator().manual_seed(step)
        for p in model.parameters():
            p.grad = torch.randn(p.shape, generator=gen)
        if step == 1:
            muon[0][0].grad = None
            (adam or [p for p, _ in muon])[-1].grad = None
        opt.step()
    return {"model": model.state_dict(), "optimizer": opt.state_dict()}


def _worker(
    rank: int, world: int, port: int, group_size: int, threads: int, out: str, scope: str
) -> None:
    # CPU matmul은 thread 수에 따라 더하는 순서가 달라진다. 1 process 계산과 같은 수로 맞춘다
    # (앞선 테스트가 prior worker 없이 학습하면 `PriorBatches`가 이 process를 1 thread로 바꾼다)
    torch.set_num_threads(threads)
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        group = dist.group.WORLD
        if group_size < world:
            group, _ = dist.new_subgroups(group_size=group_size)
        torch.save(_run(group, scope), Path(out) / f"rank{rank}.pt")
    finally:
        dist.destroy_process_group()


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _assert_same(a, b, path="") -> None:
    if isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for k in a:
            _assert_same(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, list | tuple):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            _assert_same(x, y, f"{path}[{i}]")
    elif isinstance(a, torch.Tensor):
        assert a.dtype == b.dtype and torch.equal(a, b), path
    else:
        assert a == b, path


@pytest.mark.parametrize(
    ("world", "group_size", "scope"), [(3, 3, "blocks"), (4, 2, "blocks"), (3, 3, "all")]
)
def test_sharded_muon_matches_one_process(tmp_path, world, group_size, scope):
    # 모든 GPU끼리(3개, 조각 수가 나눠떨어지지 않는다) 또는 노드 안끼리(4개를 2개씩) 나눠도, 모든
    # process의 가중치·momentum·Adam 상태가 1 process 계산과 비트 단위로 같다. 모든 파라미터 Muon도
    reference = _run(scope=scope)
    args = (world, _free_port(), group_size, torch.get_num_threads(), str(tmp_path), scope)
    mp.spawn(_worker, args=args, nprocs=world)
    for rank in range(world):
        _assert_same(torch.load(tmp_path / f"rank{rank}.pt", weights_only=True), reference)


def test_shard_pieces_balances_the_work():
    # v3.5 paper의 조각(q·k·v를 나눈 것)을 8·32 GPU에 나누면 곱셈 양이 고르다
    shapes = [
        (p.shape[0] // s, p.shape[1])
        for p, s in muon_parameters(tabpfn_v3.paper("multitask", "v3.5").build_model())[0]
        for _ in range(s)
    ]
    assert len(shapes) == 279
    for n in (8, 32):
        owner = shard_pieces(shapes, n)
        assert owner == shard_pieces(shapes, n)  # 모든 GPU가 같은 답
        loads = [0] * n
        for shape, rank in zip(shapes, owner, strict=True):
            loads[rank] += _piece_cost(*shape)
        assert min(loads) > 0 and max(loads) / (sum(loads) / n) < 1.05
    assert shard_pieces([(4, 4), (8, 8), (2, 2)], 2) == [1, 0, 1]  # 큰 것부터 가벼운 rank에
    assert _piece_cost(4, 2) == _piece_cost(2, 4) == 2 * 2 * (2 * 4 + 2)


def test_one_gpu_ignores_the_option(tmp_path):
    # GPU가 하나면 나누지 않고 지금과 같다 (process group도 만들지 않는다)
    smoke = replace(
        TrainConfig.smoke("classification"),
        steps=4,
        val_every=2,
        log_every=1,
        collapse_threshold=None,
        optimizer="muon",
    )
    off = train(smoke, tmp_path / "off", mlflow_log=False)
    on = train(replace(smoke, muon_distributed="all"), tmp_path / "on", mlflow_log=False)
    assert on["history"] == off["history"]


def test_config_checks():
    assert TrainConfig().muon_distributed == "off"
    with pytest.raises(ValueError, match="muon_distributed"):
        TrainConfig(muon_distributed="node")  # Adam이면 쓰지 않는다
    with pytest.raises(ValueError, match="muon_distributed"):
        TrainConfig(optimizer="muon", muon_distributed="gpu")
    TrainConfig(optimizer="muon", muon_distributed="node")


def _script(name):
    path = Path(__file__).parents[1] / "scripts" / "tabpfn_v2" / "train.py"
    spec = importlib.util.spec_from_file_location(name, path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    return script


def test_train_script_takes_the_option(monkeypatch):
    # 플래그·환경변수로 켜고, 이어 학습하면 원래 run의 값을 따르고(없으면 off), off로 끌 수 있다.
    # optimizer를 바꾸면 버린다
    from minipfn.hub import run_architecture

    for name in ("TABPFN_V2_OPTIMIZER", "TABPFN_V2_MUON_DISTRIBUTED", "TABPFN_V2_CONFIG_NAME"):
        monkeypatch.delenv(name, raising=False)
    script = _script("train_script_muon_distributed")
    monkeypatch.setenv("TABPFN_V2_OPTIMIZER", "muon")
    monkeypatch.setenv("TABPFN_V2_MUON_DISTRIBUTED", "node")
    config, _ = script.build_config(script.parse_args(["--config-name", "smoke"]))
    assert config.muon_distributed == "node"
    monkeypatch.delenv("TABPFN_V2_OPTIMIZER")
    monkeypatch.delenv("TABPFN_V2_MUON_DISTRIBUTED")
    source = run_architecture({k: str(v) for k, v in vars(TrainConfig.smoke()).items()})

    def build(*flags, params):
        args = script.parse_args(["--config-name", "smoke", "--resume-run-id", "r", *flags])
        return script.build_config(args, source, script.source_speed_options(params))[0]

    chunk01 = {"optimizer": "muon", "muon_weight_decay": "0.0"}  # 이 옵션이 생기기 전의 run
    assert build(params=chunk01).muon_distributed == "off"
    assert build("--muon-distributed", "node", params=chunk01).muon_distributed == "node"
    chunk02 = {**chunk01, "muon_distributed": "node"}
    assert build(params=chunk02).muon_distributed == "node"
    assert build("--muon-distributed", "off", params=chunk02).muon_distributed == "off"
    config = build("--optimizer", "adam", params=chunk02)
    assert config.optimizer == "adam" and config.muon_distributed == "off"
