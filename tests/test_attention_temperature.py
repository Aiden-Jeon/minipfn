"""attention 온도 옵션 (issue #102, `column_row_transformer` 모듈 docstring "attention 온도"):
affine 없는 QK-norm, 단계별 qassmax / logscale / gated_logscale / none. 기본값이면 main(7193e1c)과
가중치·출력·gradient가 비트 동일하다."""

import hashlib
import importlib.util
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from minipfn.hub import run_architecture
from minipfn.models import config_dict, model_from_state
from minipfn.models.column_row_transformer import (
    TEMPERATURE_FIELDS,
    Attention,
    ColumnRowTransformer,
    ColumnRowTransformerConfig,
    LogScale,
    QASSMax,
)
from minipfn.recipes import kumo_tabular, tabpfn_v3
from minipfn.training.config import TrainConfig
from minipfn.training.diagnostics import parameter_groups
from minipfn.training.loop import train
from minipfn.training.muon import muon_parameters

KUMO = kumo_tabular.ATTENTION_TEMPERATURE
V35 = {"qk_norm": True, "encoder": "fourier", "target_norm": True, "head_mlp": True}


def _fingerprint(config: ColumnRowTransformerConfig) -> list[str]:
    """가중치·출력·gradient의 sha256 (앞 16자리). 0으로 시작하는 층도 출력에 닿도록 가중치를
    흔든다."""

    def digest(tensors) -> str:
        h = hashlib.sha256()
        for name, t in tensors:
            h.update(name.encode())
            h.update(t.detach().float().contiguous().numpy().tobytes())
        return h.hexdigest()[:16]

    # CPU 스레드 수에 따라 합 순서가 달라 gradient 비트가 바뀐다 (학습 루프가 worker 없이 돌면
    # `training.data`가 1로 바꾼다). 늘 1로 잰다
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        return _fingerprint_one_thread(config, digest)
    finally:
        torch.set_num_threads(threads)


def _fingerprint_one_thread(config: ColumnRowTransformerConfig, digest) -> list[str]:
    torch.manual_seed(0)
    model = ColumnRowTransformer(config)
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn(p.shape, generator=g) * 0.02)
    out = [digest(model.state_dict().items())]
    x = torch.randn(2, 40, 5, generator=g)
    x[0, 3, 1] = float("nan")
    for task in config.tasks:
        if task == "classification":
            y = torch.randint(0, 3, (2, 30), generator=g).float()
        else:
            y = torch.randn(2, 30, generator=g)
        model.eval()
        with torch.no_grad():
            out.append(digest([("out", model(x, y, task))]))
        model.train()
        model.zero_grad()
        model(x, y, task).float().square().mean().backward()
        out.append(digest((n, p.grad) for n, p in model.named_parameters() if p.grad is not None))
    return out


def test_defaults_are_bit_identical_to_main():
    # main 7193e1c에서 같은 함수로 잰 값 (PR 설명의 비교 스크립트, 219M v3.5 stage 1 구조도 같았다)
    tiny = ColumnRowTransformerConfig.tiny()
    assert _fingerprint(tiny) == ["ce8ed1e457dcfd72", "3b100b4a4d27ebd5", "ae62e688d2b21261"]
    multitask = ColumnRowTransformerConfig.tiny(task="multitask", n_buckets=7, **V35)
    assert _fingerprint(multitask) == [
        "ab365cc7a28913da",
        "ce7d6478aae07831",
        "435e15ce3091c672",
        "4c2b286281925312",
        "3c930367cf457a39",
    ]
    # 기본값이면 checkpoint·config.json에 새 필드를 적지 않는다 (예전 코드도 읽는다)
    assert not set(config_dict(multitask)) & {"qk_norm_affine", *TEMPERATURE_FIELDS}
    names = dict(ColumnRowTransformer(multitask).named_parameters())
    assert not any("log_scale" in n for n in names)


def test_log_scale_multiplies_query_by_head_scale_times_log_n():
    torch.manual_seed(0)
    scale = LogScale(nhead=3, head_dim=4, hidden=8)
    assert ((scale.head_scale >= 0) & (scale.head_scale < 1)).all()  # U(0, 1) (Kumo와 같다)
    q = torch.randn(2, 3, 5, 4)
    expected = q * scale.head_scale.view(1, 3, 1, 1) * math.log(37)
    assert torch.allclose(scale(q, 37), expected)
    assert torch.equal(scale(q, 0), torch.zeros_like(q))  # log(max(n, 1))
    gated = LogScale(nhead=3, head_dim=4, hidden=8, gated=True)
    with torch.no_grad():
        gated.head_scale.copy_(scale.head_scale)
    assert torch.allclose(gated(q, 37), expected)  # gate의 마지막 층이 0이라 처음 배율은 1
    torch.nn.init.normal_(gated.gate[2].weight, std=10.0)
    ratio = gated(q, 37) / expected
    assert (ratio >= 0).all() and (ratio <= 2 + 1e-5).all()
    assert not torch.allclose(ratio, torch.ones(1))


def test_affine_free_qk_norm_with_log_scale_bounds_logits():
    # gain이 없으면 |q|=|k|=√d라 logit ≤ √d · a_h · log n이다. Wq·Wk를 아무리 키워도 넘지 않는다
    torch.manual_seed(0)
    dim, nhead, n = 32, 4, 50
    attn = Attention(dim, nhead, qk_norm=True, qk_norm_affine=False, temperature="logscale")
    assert not list(attn.q_norm.parameters()) and attn.qassmax is None
    with torch.no_grad():
        attn.qkv.weight.mul_(1000.0)
    seen = {}

    def spy(q, k, v):
        seen["logits"] = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
        return torch.nn.functional.scaled_dot_product_attention(q, k, v)

    attn._sdpa = spy
    attn(torch.randn(2, n, dim))
    d = dim // nhead
    bound = math.sqrt(d) * attn.log_scale.head_scale.view(1, nhead, 1, 1) * math.log(n)
    assert (seen["logits"].abs() <= bound * (1 + 1e-4)).all()


def _kumo(**changes):
    return ColumnRowTransformerConfig.tiny(task="multitask", n_buckets=7, **V35, **(KUMO | changes))


def test_kumo_temperatures_go_to_the_right_stages():
    model = ColumnRowTransformer(_kumo())
    block = model.column_blocks[0]
    assert (
        isinstance(block.summarize.log_scale, LogScale) and block.summarize.log_scale.gate is None
    )
    assert block.broadcast.log_scale is None and block.broadcast.qassmax is None
    assert model.row_blocks[0].attention.log_scale.gate is not None  # gated_logscale
    assert model.icl_layers[0].attention.log_scale.gate is None
    assert model.heads["classification"].log_scale is not None
    assert not any(isinstance(m, QASSMax) for m in model.modules())
    # gain 없는 QK-norm: q_norm·k_norm에 파라미터가 없다
    assert not any("q_norm" in n or "k_norm" in n for n, _ in model.named_parameters())
    # 단계마다 다른 종류도 고른다 (none이면 모듈이 없다)
    mixed = ColumnRowTransformer(_kumo(icl_temperature="qassmax", decoder_temperature="none"))
    assert isinstance(mixed.icl_layers[0].attention.qassmax, QASSMax)
    assert mixed.heads["classification"].qassmax is None
    assert mixed.heads["classification"].log_scale is None


def test_kumo_model_runs_saves_and_loads():
    torch.manual_seed(0)
    model = ColumnRowTransformer(_kumo()).eval()
    x = torch.randn(2, 40, 5)
    yc, yr = torch.randint(0, 3, (2, 30)).float(), torch.randn(2, 30)
    out = {
        "classification": model(x, yc, "classification"),
        "regression": model(x, yr, "regression"),
    }
    assert all(torch.isfinite(o).all() for o in out.values())
    # 추론 경로 (행을 나눠 계산, fused decoder)도 같은 값
    for task, y in (("classification", yc), ("regression", yr)):
        torch.testing.assert_close(model(x, y, task, inference=True), out[task])
    saved = config_dict(model.config)
    assert saved["qk_norm_affine"] is False and saved["row_temperature"] == "gated_logscale"
    state = {"model_type": model.model_type, "config": saved, "model": model.state_dict()}
    state["borders"] = None
    loaded, _ = model_from_state(state)
    assert loaded.config == model.config
    torch.testing.assert_close(loaded.eval()(x, yc, "classification"), out["classification"])


def test_temperature_validation():
    with pytest.raises(ValueError, match="icl_temperature"):
        ColumnRowTransformerConfig.tiny(icl_temperature="softmax")
    with pytest.raises(ValueError, match="qk_norm_affine"):
        ColumnRowTransformerConfig.tiny(qk_norm_affine=False)  # QK-norm이 없다
    with pytest.raises(ValueError, match="qassmax=True"):
        Attention(16, 2, qassmax=True, temperature="logscale")
    # 그 필드가 없는 모델(cell_transformer)에 기본값이 아닌 온도를 주면 조용히 빠지지 않는다
    with pytest.raises(ValueError, match="row_temperature"):
        TrainConfig(row_temperature="logscale")
    # 학습 설정도 값을 본다 (MLflow run을 열기 전에 멈추도록)
    v3 = tabpfn_v3.curriculum(1)
    with pytest.raises(ValueError, match="row_temperature"):
        replace(v3, row_temperature="gated")
    with pytest.raises(ValueError, match="qk_norm_affine"):
        replace(v3, qk_norm=False, qk_norm_affine=False)


def test_train_config_and_run_params_carry_temperatures():
    config = replace(tabpfn_v3.curriculum(1), logit_softcap=None, **KUMO)
    model_config = config.model_config()
    assert all(getattr(model_config, k) == v for k, v in KUMO.items())
    # MLflow param(문자열)에서 다시 만든 구조도 같다 (이어 학습·가중치 가져오기·평가)
    params = {k: str(v) for k, v in vars(config).items()}
    source = run_architecture(params)
    assert all(source[k] == v for k, v in KUMO.items())
    # 이 필드가 없는 예전 run은 기본값 (지금 구조)
    old = {k: v for k, v in params.items() if k not in KUMO}
    assert run_architecture(old)["icl_temperature"] == "qassmax"
    assert run_architecture(old)["qk_norm_affine"] is True


def test_diagnostics_groups_log_scale_separately():
    config = tabpfn_v3.with_architecture(TrainConfig.smoke("multitask"), "v3.5", keep_size=True)
    model = replace(config, **KUMO).build_model()
    muon_ids = {id(p) for p, _ in muon_parameters(model, scope="all")[0]}
    groups = parameter_groups(model, muon_ids)
    assert {
        "row_icl_log_scale/muon",
        "row_stage_log_scale/muon",
        "output_log_scale_classification/muon",
    } <= set(groups)
    assert not any("qk_norm" in g or "qassmax" in g for g in groups)


def _script(name):
    path = Path(__file__).parents[1] / "scripts" / "tabpfn_v2" / "train.py"
    spec = importlib.util.spec_from_file_location(name, path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    return script


def test_train_script_options(monkeypatch):
    for name in ("TABPFN_V2_CONFIG_NAME", "TABPFN_V2_LOGIT_SOFTCAP", "TABPFN_V2_INIT_RUN_ID"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("TABPFN_V2_ATTENTION_TEMPERATURE", raising=False)
    script = _script("train_script_temperature")
    base = ["--config-name", "stage1", "--arch", "v3.5", "--task", "multitask"]
    config, arch = script.build_config(script.parse_args(base))
    assert config == tabpfn_v3.curriculum(1)  # 주지 않으면 preset 그대로
    args = [*base, "--attention-temperature", "kumo", "--logit-softcap", "none", "--lr", "8e-4"]
    config, arch = script.build_config(script.parse_args(args))
    assert arch == "v3.5"  # 온도는 구조 이름과 따로다
    assert config == replace(tabpfn_v3.curriculum(1), logit_softcap=None, lr=8e-4, **KUMO)
    # 단계별 인자가 묶음을 덮어쓴다, 환경변수로도
    monkeypatch.setenv("TABPFN_V2_ATTENTION_TEMPERATURE", "kumo")
    monkeypatch.setenv("TABPFN_V2_ROW_TEMPERATURE", "none")
    config, _ = script.build_config(script.parse_args([*base, "--qk-norm-affine", "on"]))
    assert config.row_temperature == "none" and config.qk_norm_affine is True
    assert config.icl_temperature == "logscale" and config.logit_softcap == 50.0
    # 틀린 환경변수 값은 조용히 기본 구조로 돌아가지 않고 멈춘다 (argparse는 기본값을 보지 않는다)
    for env, value in (("ATTENTION_TEMPERATURE", "Kumo"), ("QK_NORM_AFFINE", "1")):
        monkeypatch.setenv(f"TABPFN_V2_{env}", value)
        with pytest.raises(SystemExit, match=env):
            script.parse_args(base)
        monkeypatch.setenv(f"TABPFN_V2_{env}", "kumo" if env == "ATTENTION_TEMPERATURE" else "on")
    with pytest.raises(SystemExit, match="logit-softcap"):
        script.build_config(script.parse_args([*base, "--logit-softcap", "fifty"]))
    # 가중치를 가져오면 원래 run의 온도를 쓰고, --arch v3.5여도 구조가 다르다고 멈추지 않는다
    monkeypatch.delenv("TABPFN_V2_ATTENTION_TEMPERATURE")
    monkeypatch.delenv("TABPFN_V2_ROW_TEMPERATURE")
    monkeypatch.delenv("TABPFN_V2_QK_NORM_AFFINE")
    source = run_architecture({k: str(v) for k, v in vars(replace(config, **KUMO)).items()})
    stage2 = ["--config-name", "stage2", "--arch", "v3.5", "--task", "multitask"]
    config, _ = script.build_config(script.parse_args([*stage2, "--init-run-id", "x"]), source)
    assert all(getattr(config, k) == v for k, v in KUMO.items())


def test_kumo_curriculum_smoke_trains(tmp_path, monkeypatch):
    # 작은 크기로 stage 1 코드 경로 (모든 파라미터 Muon, soft-cap 없음)
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    config = tabpfn_v3.with_architecture(TrainConfig.smoke("multitask"), "v3.5", keep_size=True)
    stage = {k: v for k, v in tabpfn_v3.CURRICULUM_COMMON.items() if k != "batch_size"}
    small = dict(
        max_features=8,
        min_rows=48,
        max_rows=80,
        log_uniform_rows=True,
        max_cells=80 * 8,
        steps=3,
        warmup_steps=1,
        val_every=3,
        log_every=1,
        checkpoint_every=3,
        keep_every=None,
        collapse_threshold=None,
        border_datasets=20,
        logit_softcap=None,
        lr=8e-4,
    )
    result = train(replace(config, **{**stage, **small, **KUMO}), tmp_path, mlflow_log=False)
    assert all(np.isfinite(result["history"]["train_loss"]))
    state = torch.load(tmp_path / "latest.pt", weights_only=True)
    assert any(k.endswith("log_scale.head_scale") for k in state["model"])
