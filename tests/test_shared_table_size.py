"""step마다 GPU끼리 표 크기 맞추기 (`TrainConfig.shared_table_size`, issue #68)."""

import importlib.util
import itertools
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from minipfn.recipes import tabpfn_v3
from minipfn.training.config import TrainConfig
from minipfn.training.data import PriorBatches
from minipfn.training.loop import train


def _batches(config, rank, size_seed, num_workers, n=6):
    loader = DataLoader(
        PriorBatches(config, seed=(0, 0, rank), size_seed=size_seed),
        batch_size=None,
        num_workers=num_workers,
    )
    return list(itertools.islice(iter(loader), n))


@pytest.mark.parametrize("num_workers", [0, 2])
def test_ranks_share_table_sizes_but_not_contents(num_workers):
    config = TrainConfig.smoke("classification")
    a = _batches(config, rank=0, size_seed=(0, 0, 9), num_workers=num_workers)
    b = _batches(config, rank=1, size_seed=(0, 0, 9), num_workers=num_workers)
    sizes_a = [(x.shape, n_train) for x, _, n_train in a]
    assert sizes_a == [(x.shape, n_train) for x, _, n_train in b]
    assert len(set(sizes_a)) > 1  # batch마다는 크기가 바뀐다
    assert not torch.equal(a[0][0], b[0][0])  # 내용은 GPU마다 다르다


def test_worker_count_does_not_change_the_size_sequence():
    config = TrainConfig.smoke("regression")
    one = _batches(config, rank=0, size_seed=(3,), num_workers=0)
    two = _batches(config, rank=0, size_seed=(3,), num_workers=2)
    assert [(x.shape, n) for x, _, n in one] == [(x.shape, n) for x, _, n in two]


def test_without_size_seed_ranks_draw_their_own_sizes():
    config = TrainConfig.smoke("classification")
    a = _batches(config, rank=0, size_seed=None, num_workers=0, n=8)
    b = _batches(config, rank=1, size_seed=None, num_workers=0, n=8)
    assert [(x.shape, n) for x, _, n in a] != [(x.shape, n) for x, _, n in b]


def test_multitask_training_with_shared_sizes(tmp_path):
    config = tabpfn_v3.with_architecture(TrainConfig.smoke("multitask"), "v3.5", keep_size=True)
    config = replace(
        config,
        steps=4,
        warmup_steps=1,
        log_every=2,
        val_every=4,
        checkpoint_every=4,
        keep_every=None,
        collapse_threshold=None,
        shared_table_size=True,
        rank_timing=True,
    )
    out = train(config, tmp_path, mlflow_log=False)
    assert len(out["history"]["train_loss"]) == 2
    assert (tmp_path / "rank_timing_from_0000000.jsonl").exists()


def _script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "tabpfn_v2" / "train.py"
    spec = importlib.util.spec_from_file_location("train_script_shared_size", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_script_flag_and_resume_keep_shared_sizes():
    script = _script()
    base = ["--config-name", "smoke", "--arch", "v3.5", "--task", "multitask"]
    config, _ = script.build_config(script.parse_args([*base, "--shared-table-size"]))
    assert config.shared_table_size
    # 이어 학습하면 원래 run의 값을 따른다 (학습 가속 옵션)
    assert script.source_speed_options({"shared_table_size": "True"}) == {"shared_table_size": True}
