"""Kumo-Tabular의 나머지 부품 (issue #110, `column_row_transformer` 모듈 docstring "Kumo-Tabular의
나머지 부품"): decoder q·k norm, 열·행 단계 번갈아 쌓기, 회귀 분위수 출력, 범주를 아는 셀 encoder.
기본값이면 main(75b6992)과 비트 동일하다 (가중치·출력·gradient는 `test_attention_temperature`의
golden, 219M stage 1 구조와 prior batch는 PR 설명의 비교 스크립트)."""

import importlib.util
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from minipfn.hub import run_architecture
from minipfn.inference.ensemble import EnsembleClassifier, EnsembleRegressor
from minipfn.inference.preprocessing import Member, MemberPreprocessor
from minipfn.models import config_dict, model_from_state
from minipfn.models.column_row_transformer import (
    ColumnRowTransformer,
    ColumnRowTransformerConfig,
)
from minipfn.models.quantile_distribution import (
    N_QUANTILES,
    initial_bias,
    pinball_loss,
    quantile_levels,
    quantiles_to_bucket_probs,
)
from minipfn.priors.graph_scm import GraphSCMPrior
from minipfn.recipes import kumo_tabular, tabpfn_v3
from minipfn.training.config import KUMO_OPTION_DEFAULTS, TrainConfig
from minipfn.training.loop import train
from minipfn.training.losses import prior_loss

KUMO = kumo_tabular.ATTENTION_TEMPERATURE
V35 = {"qk_norm": True, "encoder": "fourier", "target_norm": True, "head_mlp": True}
# tiny 크기의 Kumo식 열·행 단계 (블록 2쌍)
TINY_STAGES = {"stage_layout": "interleaved", "column_blocks": 2, "row_blocks": 2}


def _config(**changes) -> ColumnRowTransformerConfig:
    return ColumnRowTransformerConfig.tiny(task="multitask", n_buckets=7, **V35, **(KUMO | changes))


def _inputs(c: int = 5, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(2, 40, c, generator=g)
    x[0, 3, 1] = float("nan")
    x[:, :, 2] = torch.randint(0, 4, (2, 40), generator=g).float()  # 범주 열 (번호)
    yc = torch.randint(0, 3, (2, 30), generator=g).float()
    yr = torch.randn(2, 30, generator=g)
    categorical = torch.zeros(2, c, dtype=torch.bool)
    categorical[:, 2] = True
    return x, yc, yr, categorical


def test_default_config_saves_no_new_fields():
    saved = config_dict(_config())
    assert not set(KUMO_OPTION_DEFAULTS) & set(saved)
    assert TrainConfig().decoder_qk_norm is False
    # 기본 TrainConfig의 열·행 단계 값은 모델 설정 기본값과 같다
    defaults = ColumnRowTransformerConfig()
    for name in ("column_blocks", "row_blocks", "column_heads", "row_heads"):
        assert getattr(TrainConfig(), name) == getattr(defaults, name)


def test_validation():
    with pytest.raises(ValueError, match="stage_layout"):
        _config(stage_layout="mixed")
    with pytest.raises(ValueError, match="열·행 블록 수"):
        _config(stage_layout="interleaved", column_blocks=1, row_blocks=2)
    with pytest.raises(ValueError, match="regression_head"):
        _config(regression_head="gaussian")
    with pytest.raises(ValueError, match="회귀 출력"):
        ColumnRowTransformerConfig.tiny(regression_head="quantile")  # 분류만
    with pytest.raises(ValueError, match="encoder=fourier"):
        ColumnRowTransformerConfig.tiny(categorical_encoder=True)
    with pytest.raises(ValueError, match="categorical_encoder"):
        TrainConfig(categorical_encoder=True)  # cell_transformer에는 없다
    v35 = tabpfn_v3.curriculum(1)
    with pytest.raises(ValueError, match="tree_scm_prob"):
        replace(v35, categorical_encoder=True, tree_scm_prob=0.5)
    with pytest.raises(ValueError, match="회귀 출력"):
        replace(
            tabpfn_v3.paper("classification", "v3.5"), regression_head="quantile"
        ).model_config()
    model = ColumnRowTransformer(_config())
    x, yc, _, categorical = _inputs()
    with pytest.raises(ValueError, match="categorical_encoder"):
        model(x, yc, "classification", categorical=categorical)


# --- D: decoder q·k norm ---------------------------------------------------------------------


def test_decoder_qk_norm_bounds_logits():
    # gain 없는 RMSNorm + logscale: score ≤ √d · a_h · log n (가중치를 아무리 키워도)
    torch.manual_seed(0)
    model = ColumnRowTransformer(_config(decoder_qk_norm=True)).eval()
    decoder = model.heads["classification"]
    assert not any("q_norm" in n for n, _ in decoder.named_parameters())  # 파라미터 없음
    with torch.no_grad():
        decoder.q_projection.weight.mul_(1000.0)
        decoder.k_projection.weight.mul_(1000.0)
    seen = {}
    h = torch.randn(1, 30, model.config.icl_dim)
    y = torch.randint(0, 3, (1, 20)).float()

    def spy(q, k, onehot):
        seen["scores"] = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
        return torch.softmax(seen["scores"], -1) @ onehot[:, None]

    import minipfn.models.column_row_transformer as crt

    saved = crt._retrieve
    crt._retrieve = spy
    try:
        decoder(h, y, 20, fused=True)
    finally:
        crt._retrieve = saved
    d, a = decoder.head_dim, decoder.log_scale.head_scale.view(1, -1, 1, 1)
    bound = math.sqrt(d) * a * math.log(20)
    assert (seen["scores"].abs() <= bound * (1 + 1e-4)).all()
    # 끄면 같은 가중치에서 상한이 없다
    plain = ColumnRowTransformer(_config()).heads["classification"]
    assert not plain.qk_norm


# --- W: 열·행 단계 번갈아 쌓기 ------------------------------------------------------------------


def test_interleaved_layout_runs_and_matches_chunked_inference():
    torch.manual_seed(0)
    config = _config(**TINY_STAGES, categorical_encoder=True)
    model = ColumnRowTransformer(config).eval()
    x, yc, yr, categorical = _inputs()
    for task, y in (("classification", yc), ("regression", yr)):
        full = model(x, y, task, categorical=categorical)
        model.inference_chunk_cells = 37  # 행 묶음 여럿, 열 묶음 여럿
        model.inference_column_chunk = 2
        chunked = model(x, y, task, categorical=categorical, inference=True)
        model.inference_chunk_cells = ColumnRowTransformer.inference_chunk_cells
        model.inference_column_chunk = ColumnRowTransformer.inference_column_chunk
        torch.testing.assert_close(chunked, full)


def test_interleaved_layout_mixes_columns_before_the_next_column_block():
    # 번갈아 쌓으면 둘째 열 블록의 요약이 다른 열을 본다: 다른 열 하나만 바꿔도 첫 열의 열 단계
    # 출력이 바뀐다 (차례대로 쌓으면 열 단계는 열마다 따로다)
    torch.manual_seed(0)
    seen = {}

    def hook(name):
        def f(module, args, out):
            seen[name] = out.detach().clone()

        return f

    x, yc, _, _ = _inputs()
    x2 = x.clone()
    x2[:, :, 4] += 1.0
    for layout in ("sequential", "interleaved"):
        model = ColumnRowTransformer(_config(**{**TINY_STAGES, "stage_layout": layout})).eval()
        model.column_blocks[1].register_forward_hook(hook("a"))
        model(x, yc, "classification")
        first = seen["a"].view(2, 5, 40, -1)[:, 0]
        model(x2, yc, "classification")
        second = seen["a"].view(2, 5, 40, -1)[:, 0]
        assert torch.equal(first, second) == (layout == "sequential")


def test_kumo_stages_parameter_count_and_architecture_name():
    config = replace(tabpfn_v3.curriculum(1), logit_softcap=None, **KUMO, **kumo_tabular.STAGES)
    assert tabpfn_v3.architecture_name(config) == "v3.5"  # 행 ICL 폭이 같다 (1024)
    model_config = config.model_config()
    assert model_config.icl_dim == 1024 and model_config.emsize == 256
    base = replace(tabpfn_v3.curriculum(1), logit_softcap=None, **KUMO).model_config()
    with torch.device("meta"):
        wide = ColumnRowTransformer(model_config).num_parameters_by_stage()
        now = ColumnRowTransformer(base).num_parameters_by_stage()
    assert wide["row_icl"] == now["row_icl"]
    assert wide["column_stage"] + wide["row_stage"] > 5 * (now["column_stage"] + now["row_stage"])
    # 폭이 다르면(행 ICL 512) v3.5가 아니다
    assert tabpfn_v3.architecture_name(replace(config, n_cls=2)) is None


# --- Q: 회귀 분위수 출력 ---------------------------------------------------------------------


def test_pinball_loss_and_initial_quantiles():
    tau = quantile_levels(999)
    assert tau[0] == pytest.approx(0.001) and tau[-1] == pytest.approx(0.999)
    q = torch.tensor([[0.0, 1.0, 2.0]])
    y = torch.tensor([1.5])
    # τ = 0.25, 0.5, 0.75: 0.25·1.5, 0.5·0.5, 0.25·0.5 (y가 q 아래면 (1 − τ)(q − y))
    expected = (0.25 * 1.5 + 0.5 * 0.5 + (1 - 0.75) * 0.5) / 3
    assert pinball_loss(q, y).item() == pytest.approx(expected)
    assert pinball_loss(q, torch.tensor([float("nan")])).item() == 0.0
    bias = initial_bias()
    assert bias[499] == pytest.approx(0.0, abs=1e-6) and bias[-1] == pytest.approx(3.0902, abs=1e-3)
    # 처음 모델은 모든 행에서 표준정규 분위수를 낸다 (weight 0)
    model = ColumnRowTransformer(_config(regression_head="quantile")).eval()
    x, _, yr, _ = _inputs()
    out = model(x, yr, "regression")
    assert out.shape == (2, 10, N_QUANTILES)
    torch.testing.assert_close(out, bias.expand_as(out))
    loss = prior_loss(out, torch.randn(2, 10), "regression", None, "quantile")
    assert torch.isfinite(loss)


def test_quantiles_to_bucket_probs():
    borders = torch.linspace(-6, 6, 2001)
    q = initial_bias()[None].repeat(3, 1)
    q[1] = q[1] * 2 + 1  # N(1, 4)
    q[2] = q[2].flip(0)  # 정렬하지 않은 분위수도 받는다
    p = quantiles_to_bucket_probs(q, borders)
    assert p.shape == (3, 2000) and (p >= 0).all()
    torch.testing.assert_close(p.sum(-1), torch.ones(3))
    centers = (borders[1:] + borders[:-1]) / 2
    means = p @ centers
    assert means[0].item() == pytest.approx(0.0, abs=1e-3)
    assert means[1].item() == pytest.approx(1.0, abs=1e-2)
    torch.testing.assert_close(p[2], p[0])
    # CDF: 경계 0에서 0.5 (N(0, 1))
    cdf = torch.cumsum(p[0], 0)
    assert cdf[999].item() == pytest.approx(0.5, abs=2e-3)
    # 경계 밖 분위수는 끝 구간에 (질량을 잃지 않는다)
    far = quantiles_to_bucket_probs(torch.full((1, 5), 100.0), borders)
    assert far[0, -1].item() == pytest.approx(1.0)


def test_quantile_regressor_predicts_through_the_ensemble():
    torch.manual_seed(0)
    model = ColumnRowTransformer(_config(regression_head="quantile"))
    borders = torch.linspace(-5, 5, 101)
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(60, 4)), rng.normal(size=60) * 3 + 10
    reg = EnsembleRegressor(model, borders, n_estimators=2, member_kinds=("standard",)).fit(X, y)
    pred = reg.predict(X[:7])
    # 처음 모델의 예측은 학습 target 분포의 평균 근처 (표준정규 분위수를 z 공간에서)
    assert pred.shape == (7,) and np.allclose(pred, y.mean(), atol=0.5)
    log_p, out_borders = reg.predict_distribution(X[:7])
    assert log_p.shape == (7, 100)
    torch.testing.assert_close(log_p.exp().sum(-1), torch.ones(7))
    assert out_borders.dtype == torch.float64


# --- C: 범주를 아는 셀 encoder --------------------------------------------------------------


def test_categorical_encoder_uses_the_mask():
    torch.manual_seed(0)
    model = ColumnRowTransformer(_config(categorical_encoder=True)).eval()
    with torch.no_grad():  # 범주 쪽을 수치 쪽과 다르게
        model.cell_encoder.cat_frequencies.mul_(3.0)
    x, yc, _, categorical = _inputs()
    marked = model(x, yc, "classification", categorical=categorical)
    plain = model(x, yc, "classification")
    zeros = model(x, yc, "classification", categorical=torch.zeros_like(categorical))
    torch.testing.assert_close(plain, zeros)  # 주지 않으면 모두 수치 열
    assert not torch.allclose(marked, plain)
    # 추론 경로(행·열 묶음)도 같은 값
    model.inference_chunk_cells, model.inference_column_chunk = 37, 2
    torch.testing.assert_close(
        model(x, yc, "classification", categorical=categorical, inference=True), marked
    )
    with pytest.raises(ValueError, match=r"\(B, C\)"):
        model(x, yc, "classification", categorical=categorical[:, :3])


def test_member_preprocessor_categorical_columns_follow_the_output_columns():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 4))
    X[:, 1] = rng.integers(0, 3, 50)  # one-hot으로 펼치는 범주 (3개)
    X[:, 3] = rng.integers(0, 30, 50)  # 번호로 두는 범주 (30개 > 20)
    for kind, one_hot in (("standard", True), ("standard", False), ("quantile", True)):
        member = Member(
            kind=kind, class_permutation=None, target_power=False, seed=3, one_hot=one_hot
        )
        prep = MemberPreprocessor(member, [1, 3], max_one_hot_width=100).fit(X)
        out = prep.transform(X, np.random.default_rng(1))
        mask = prep.categorical_columns()
        assert mask.shape == (out.shape[1],)
        # 표시한 열은 번호 열(정수 값)이다
        for j in np.flatnonzero(mask):
            assert np.allclose(out[:, j], np.round(out[:, j]))
        expected = (0 if one_hot else 1) + 1  # 열 3은 늘 번호, 열 1은 one-hot이 아니면 번호
        assert mask.sum() == expected


def test_ensemble_passes_member_masks_to_the_model():
    torch.manual_seed(0)
    model = ColumnRowTransformer(_config(categorical_encoder=True))
    seen = []
    model.register_forward_pre_hook(
        lambda m, args, kwargs: seen.append(kwargs.get("categorical")), with_kwargs=True
    )
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 3))
    X[:, 0] = rng.integers(0, 4, 40)
    y = rng.integers(0, 2, 40)
    clf = EnsembleClassifier(
        model, n_estimators=2, categorical_features_indices=[0], one_hot="off"
    ).fit(X, y)
    clf.predict_proba(X[:5])
    assert len(seen) == 2 and all(m is not None and m.dtype == torch.bool for m in seen)
    assert all(m.sum().item() == 1 for m in seen)  # 범주 열 하나 (식별자 열은 수치)
    for i, m in enumerate(seen):
        assert m[0].tolist() == clf.preprocessors_[i].categorical_columns().tolist()


def test_prior_categorical_mask_keeps_the_tables():
    # TabICL은 전역 난수를 쓴다: 두 prior를 차례로 만들어 뽑는다
    kw = dict(size_rule="train_fraction", min_rows=120, max_rows=120, max_features=10)
    kw |= dict(filter_unpredictable=True, binary_prob=0.5)
    for task in ("classification", "regression"):
        plain = GraphSCMPrior(task, seed=7, **kw)
        a = [plain.sample_batch(4) for _ in range(3)]
        marked = GraphSCMPrior(task, seed=7, categorical_mask=True, **kw)
        b = [marked.sample_batch(4) for _ in range(3)]
        for ba, bb in zip(a, b, strict=True):
            assert ba.categorical is None
            torch.testing.assert_close(ba.x, bb.x, equal_nan=True, rtol=0, atol=0)
            assert torch.equal(ba.y, bb.y)
            assert bb.categorical.shape == (4, bb.x.shape[2]) and bb.categorical.dtype == torch.bool
            for d, j in torch.nonzero(bb.categorical).tolist():
                v = bb.x[d, :, j]
                assert v[torch.isfinite(v)].unique().numel() <= 9  # TabICL 범주 크기 2~9
    with pytest.raises(ValueError, match="tree_scm_prob"):
        GraphSCMPrior("classification", categorical_mask=True, tree_scm_prob=0.5)


# --- 학습 설정, 학습 스크립트, smoke 학습 -----------------------------------------------------


def test_train_config_and_run_params_carry_options():
    options = {**kumo_tabular.STAGES, "decoder_qk_norm": True, "regression_head": "quantile"}
    config = replace(tabpfn_v3.curriculum(1), logit_softcap=None, **KUMO, **options)
    model_config = config.model_config()
    assert all(getattr(model_config, k) == v for k, v in options.items())
    params = {k: str(v) for k, v in vars(config).items()}
    source = run_architecture(params)
    assert all(source[k] == v for k, v in options.items())
    old = {k: v for k, v in params.items() if k not in {*options, *KUMO_OPTION_DEFAULTS}}
    assert run_architecture(old)["stage_layout"] == "sequential"
    assert run_architecture(old)["column_blocks"] == 3


def _script(name):
    path = Path(__file__).parents[1] / "scripts" / "tabpfn_v2" / "train.py"
    spec = importlib.util.spec_from_file_location(name, path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    return script


def test_train_script_kumo_options(monkeypatch):
    for name in ("TABPFN_V2_CONFIG_NAME", "TABPFN_V2_LOGIT_SOFTCAP", "TABPFN_V2_KUMO_OPTIONS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("TABPFN_V2_ATTENTION_TEMPERATURE", raising=False)
    script = _script("train_script_kumo_options")
    base = ["--config-name", "stage1", "--arch", "v3.5", "--task", "multitask"]
    b_arm = [*base, "--attention-temperature", "kumo", "--logit-softcap", "none", "--lr", "8e-4"]
    reference, _ = script.build_config(script.parse_args(b_arm))
    for name, values in kumo_tabular.OPTIONS.items():
        config, arch = script.build_config(script.parse_args([*b_arm, "--kumo-options", name]))
        assert arch == "v3.5" and config == replace(reference, **values)
    monkeypatch.setenv("TABPFN_V2_KUMO_OPTIONS", "decoder_qk_norm, quantile_head")
    config, _ = script.build_config(script.parse_args(b_arm))
    assert config.decoder_qk_norm and config.regression_head == "quantile"
    monkeypatch.setenv("TABPFN_V2_KUMO_OPTIONS", "wide")
    with pytest.raises(SystemExit, match="kumo-options"):
        script.parse_args(b_arm)
    monkeypatch.delenv("TABPFN_V2_KUMO_OPTIONS")
    # Kumo식 열·행 단계 run을 이어 학습할 때 --arch v3.5여도 구조가 다르다고 멈추지 않는다
    wide = replace(reference, **kumo_tabular.STAGES)
    source = run_architecture({k: str(v) for k, v in vars(wide).items()})
    config, _ = script.build_config(script.parse_args([*b_arm, "--resume-run-id", "x"]), source)
    assert all(getattr(config, k) == v for k, v in kumo_tabular.STAGES.items())


def _smoke(tmp_path, monkeypatch, **options):
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
    config = replace(config, **{**stage, **small, **KUMO, **options})
    result = train(config, tmp_path, mlflow_log=False)
    assert all(np.isfinite(result["history"]["train_loss"]))
    return torch.load(tmp_path / "latest.pt", weights_only=True)


@pytest.mark.parametrize(
    "options",
    [
        {"decoder_qk_norm": True},
        {"stage_layout": "interleaved", "column_blocks": 2, "row_blocks": 2},
        {"regression_head": "quantile"},
        {"categorical_encoder": True},
    ],
    ids=["decoder_qk_norm", "stages", "quantile_head", "categorical_encoder"],
)
def test_kumo_options_smoke_train(tmp_path, monkeypatch, options):
    state = _smoke(tmp_path, monkeypatch, **options)
    loaded, borders = model_from_state(state)
    assert all(getattr(loaded.config, k) == v for k, v in options.items())
    if "categorical_encoder" in options:
        assert "cell_encoder.cat_frequencies" in state["model"]
    if "regression_head" in options:
        assert borders is not None  # 평가가 구간 확률로 바꿀 때 쓴다


def test_test_time_methods_refuse_categorical_and_quantile_models():
    from minipfn.inference.diag_scale import DiagScale
    from minipfn.inference.ensemble_selection import GreedyEnsembleRegressor

    borders = torch.linspace(-3, 3, 11)
    quantile = ColumnRowTransformer(_config(regression_head="quantile"))
    with pytest.raises(ValueError, match="greedy"):
        GreedyEnsembleRegressor(quantile, borders, n_estimators=2)
    with pytest.raises(ValueError, match="DiagScale"):
        DiagScale(EnsembleRegressor(quantile, borders, n_estimators=2))
    categorical = ColumnRowTransformer(_config(categorical_encoder=True))
    with pytest.raises(ValueError, match="DiagScale"):
        DiagScale(EnsembleClassifier(categorical, n_estimators=2))
    # 분위수 출력 모델의 분류 쪽은 그대로 쓴다
    DiagScale(EnsembleClassifier(quantile, n_estimators=2))


def test_pad_features_pads_the_categorical_mask():
    from minipfn.priors.pipeline import Group, Rng
    from minipfn.priors.transforms.scaling import PadFeatures

    mask = torch.tensor([[True, False], [False, True]])
    group = Group(torch.randn(5, 2, 2), torch.randn(5, 2), 3, {}, mask)
    out = PadFeatures(4, rescale=False)(group, Rng.from_seed(0))
    assert out.x.shape == (5, 2, 4)
    assert out.categorical.tolist() == [[True, False, False, False], [False, True, False, False]]


def test_kumo_curriculum_presets():
    one = tabpfn_v3.kumo_curriculum(1)
    expected = replace(
        tabpfn_v3.curriculum(1),
        logit_softcap=None,
        lr=8e-4,
        micro_batch=4,
        **KUMO,
        **kumo_tabular.STAGES,
        **kumo_tabular.QUANTILE_HEAD,
    )
    assert one == expected
    assert tabpfn_v3.architecture_name(one) == "v3.5"
    assert not one.decoder_qk_norm and not one.categorical_encoder  # 효과 없거나 나빠진 부품은 뺀다
    two, three = tabpfn_v3.kumo_curriculum(2), tabpfn_v3.kumo_curriculum(3)
    assert (two.lr, three.lr) == (1e-4, 2e-5)
    assert two.max_rows == tabpfn_v3.curriculum(2).max_rows and two.micro_batch == 1
    assert three.activation_checkpointing_rows == 10_240 and three.logit_softcap is None
    assert tabpfn_v3.kumo_curriculum(1, steps=1000).warmup_steps == 10
    with pytest.raises(ValueError, match="kumo curriculum"):
        tabpfn_v3.kumo_curriculum(4)


def test_train_script_kumo_curriculum(monkeypatch):
    for name in ("TABPFN_V2_CONFIG_NAME", "TABPFN_V2_LOGIT_SOFTCAP", "TABPFN_V2_KUMO_OPTIONS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("TABPFN_V2_ATTENTION_TEMPERATURE", raising=False)
    monkeypatch.delenv("TABPFN_V2_LR", raising=False)
    monkeypatch.delenv("TABPFN_V2_MICRO_BATCH", raising=False)
    script = _script("train_script_kumo_curriculum")
    common = ["--arch", "v3.5", "--task", "multitask"]
    config, arch = script.build_config(script.parse_args(["--config-name", "stage1_kumo", *common]))
    assert arch == "v3.5" and config == tabpfn_v3.kumo_curriculum(1)
    # stage 2·3은 앞 단계 run의 가중치를 가져온다
    with pytest.raises(SystemExit):
        script.parse_args(["--config-name", "stage2_kumo", *common])
    source = run_architecture({k: str(v) for k, v in vars(config).items()})
    args = script.parse_args(["--config-name", "stage2_kumo", *common, "--init-run-id", "x"])
    two, _ = script.build_config(args, source)
    assert two == tabpfn_v3.kumo_curriculum(2)
