"""TabPFN-3·3.5 (`column_row_transformer` 모델, `recipes.tabpfn_v3`) 테스트."""

import importlib.util
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.datasets import load_breast_cancer, load_diabetes

from minipfn import hub
from minipfn import models as registry
from minipfn.inference.ensemble import EnsembleClassifier, EnsembleRegressor
from minipfn.models.column_row_transformer import (
    ColumnRowTransformer,
    ColumnRowTransformerConfig,
    ecdf_midrank,
    group_offsets,
)
from minipfn.recipes import tabpfn_v2
from minipfn.recipes import tabpfn_v3 as recipe
from minipfn.training.config import TrainConfig
from minipfn.training.loop import check_checkpoint, train


def _model(task="classification", arch="v3", **changes):
    config = recipe.with_architecture(TrainConfig.smoke(task), arch, keep_size=True)
    torch.manual_seed(0)
    model = replace(config, **changes).build_model()
    # 잔차 가지의 마지막 층은 0으로 시작해 처음에는 입력이 출력에 닿지 않는다. 경로를 보려고 채운다
    with torch.no_grad():
        for p in model.parameters():
            if (p == 0).all():
                p.normal_(0, 0.1)
    return model.eval()


def _data(task, b=2, r=40, c=7, n_train=25):
    torch.manual_seed(1)
    x = torch.randn(b, r, c)
    x[0, 3, 2] = float("nan")
    y = (
        torch.randint(0, 3, (b, n_train)).float()
        if task == "classification"
        else torch.randn(b, n_train)
    )
    return x, y


@pytest.mark.parametrize(
    "arch,task,millions",
    # 보고서: TabPFN-3 분류 53M·회귀 58M. TabPFN-3.5는 분류·회귀 한 모델로 219.0M(공개 checkpoint)
    [("v3", "classification", 53), ("v3", "regression", 58), ("v3.5", "multitask", 219)],
)
def test_paper_size_matches_the_reports(arch, task, millions):
    config = recipe.paper(task, arch).model_config()
    with torch.device("meta"):
        model = ColumnRowTransformer(config)
    assert abs(model.num_parameters() / 1e6 - millions) / millions < 0.02
    assert config.icl_dim == (512 if arch == "v3" else 1024)


def test_paper_architecture_and_names():
    config = recipe.paper("regression")
    assert config.model_type == "column_row_transformer"
    assert recipe.architecture_name(config) == "v3"
    assert tabpfn_v2.paper("regression").qk_clip is not None and config.qk_clip is None
    assert config.lr == tabpfn_v2.paper("regression").lr  # 그 밖의 학습 설정은 v2 그대로
    v35 = recipe.paper("regression", "v3.5")
    assert recipe.architecture_name(v35) == "v3.5" and v35.qk_clip is None
    smoke = recipe.with_architecture(TrainConfig.smoke(), "v3.5", keep_size=True)
    assert recipe.architecture_name(smoke) == "v3.5" and smoke.n_inducing == 8
    # 다른 recipe의 구조 이름과 섞이지 않는다
    assert tabpfn_v2.architecture_name(config) is None
    assert recipe.architecture_name(tabpfn_v2.paper()) is None
    with pytest.raises(ValueError, match="model_type"):
        recipe.with_architecture(replace(TrainConfig.smoke(), model_type="tiny_x"), "v3")


def test_group_offsets_and_ecdf():
    assert group_offsets(3) == (0, 1, 3)
    x = torch.tensor([[[1.0], [2.0], [2.0], [float("nan")], [2.0], [9.0]]])
    u = ecdf_midrank(x, n_train=4)[0, :, 0]
    # 학습 행 값 1, 2, 2 (결측 하나는 빼고 3개): 1 → 0.5/3, 2 → (1 + 1)/3, 결측 → 0.5, 9 → 1
    assert torch.allclose(u, torch.tensor([0.5 / 3, 2 / 3, 2 / 3, 0.5, 2 / 3, 1.0]))


@pytest.mark.parametrize("arch", ["v3", "v3.5"])
@pytest.mark.parametrize("task", ["classification", "regression"])
def test_test_rows_do_not_see_each_other(arch, task):
    model = _model(task, arch)
    x, y = _data(task)
    out = model(x, y)
    n_out = 10 if task == "classification" else 5000
    assert out.shape == (2, 15, n_out) and torch.isfinite(out).all()
    changed = x.clone()
    changed[:, 30] += 5.0  # 테스트 행 5번
    other = model(changed, y)
    keep = [i for i in range(15) if i != 5]
    assert torch.equal(other[:, keep], out[:, keep])
    assert not torch.equal(other[:, 5], out[:, 5])
    # 학습 행은 테스트 행 출력을 바꾼다
    changed = x.clone()
    changed[:, 3] += 5.0
    assert not torch.allclose(model(changed, y), out)


def test_retrieval_decoder_gives_probabilities_over_seen_classes():
    model = _model()
    x, y = _data("classification")
    # logit은 log(max(p, 1e-5) + 3e-5)라 합이 1보다 조금 크다 (클래스 10개면 최대 1.0004)
    p = model(x, y).exp()
    assert torch.allclose(p.sum(-1), torch.ones(2, 15), atol=1e-3)
    # 학습 행에 없는 클래스는 clip 값만 남는다 (클래스 수에 따른 파라미터가 없다)
    assert (p[..., 3:] < 1e-4).all()


@pytest.mark.parametrize("arch", ["v3", "v3.5"])
def test_every_parameter_gets_a_gradient(arch):
    # DDP는 forward에 쓰지 않은 파라미터가 있으면 멈춘다
    for task in ("classification", "regression"):
        model = _model(task, arch).train()
        x, y = _data(task)
        # 분류 logit은 log p라 logsumexp가 0이다. 한 클래스의 log 확률을 쓴다
        model(x, y)[..., 0].sum().backward()
        assert [n for n, p in model.named_parameters() if p.grad is None] == []


def test_qk_clip_is_not_offered(tmp_path):
    # QASSMax의 학습 배율과 테스트 행 multi-query 때문에 가중치 상한이 logit을 묶지 못한다
    assert _model().qk_clip_weights() == [] and _model(arch="v3.5").qk_clip_weights() == []
    config = recipe.with_architecture(TrainConfig.smoke(), "v3", keep_size=True)
    with pytest.raises(ValueError, match="QK-clip"):
        train(replace(config, qk_clip=50.0), tmp_path, mlflow_log=False)


def test_test_row_multi_query_matches_expanded_heads():
    # 테스트 행은 head를 query 길이로 접어 계산한다. key·value를 head 수만큼 늘린 것과 같다
    import torch.nn.functional as F

    attention = _model().icl_layers[0].attention
    x = torch.randn(2, 30, attention.qkv.in_features)
    n = 20
    out = attention(x, n_context=n)
    d = x.shape[-1]
    w = attention.qkv.weight
    q = attention._heads(F.linear(x, w[:d]))
    k = attention._heads(F.linear(x[:, :n], w[d : 2 * d]))
    v = attention._heads(F.linear(x[:, :n], w[2 * d :]))
    q = attention.qassmax(q, n)
    test = F.scaled_dot_product_attention(q[:, :, n:], k[:, :1].expand_as(k), v[:, :1].expand_as(v))
    expected = attention.out(test.transpose(1, 2).reshape(2, 10, d))
    assert torch.allclose(out[:, n:], expected, atol=1e-5)


def test_retrieval_decoder_projects_only_what_it_uses():
    model = _model()
    decoder = model.heads["classification"]
    h = torch.randn(2, 30, model.config.icl_dim)
    y = torch.randint(0, 3, (2, 20)).float()
    heads, dh = model.config.decoder_heads, model.config.decoder_head_dim
    # 행마다 query·key를 둘 다 만든 것
    q, k = decoder.q_projection(h), decoder.k_projection(h)
    q = decoder.qassmax(q[:, 20:].reshape(2, -1, heads, dh).transpose(1, 2), 20)
    k = k[:, :20].reshape(2, -1, heads, dh).transpose(1, 2)
    weights = torch.softmax((q @ k.transpose(-1, -2)).float() / dh**0.5, dim=-1)
    onehot = torch.nn.functional.one_hot(y.long(), model.config.n_out).float()
    expected = torch.log((weights @ onehot[:, None]).mean(1).clamp_min(1e-5) + 3e-5)
    assert torch.allclose(decoder(h, y, 20), expected, atol=1e-5)


def test_ecdf_of_columns_constant_on_train_rows_is_flat():
    # 학습 행에서 상수인 열은 z처럼 ECDF도 모든 행에서 같은 값이라 테스트 행에만 있는 신호가 없다
    model = _model(arch="v3.5")
    x = torch.randn(1, 30, 4)
    x[0, :20, 1] = 3.0  # 학습 행 20개에서 상수, 테스트 행에서는 다르다
    x[0, 20:, 1] = torch.linspace(-5, 5, 10)
    seen = {}
    hook = model.cell_encoder.register_forward_hook(lambda m, args, out: seen.update(u=args[2]))
    model._embed_cells(x, 20)
    hook.remove()
    # feature j의 묶음 자리 0은 열 j 자신이다
    assert torch.equal(seen["u"][0, :, 1, 0], torch.full((30,), 0.5))
    assert not torch.equal(seen["u"][0, :, 0, 0], torch.full((30,), 0.5))


def test_bad_options_raise():
    with pytest.raises(ValueError, match="task"):
        ColumnRowTransformerConfig(task="ranking")
    with pytest.raises(ValueError, match="encoder"):
        ColumnRowTransformerConfig(encoder="mlp")
    with pytest.raises(ValueError, match="nhead"):
        ColumnRowTransformerConfig(nhead=7)


@pytest.mark.parametrize("arch", ["v3", "v3.5"])
def test_train_save_load_and_hub(tmp_path, arch):
    config = replace(
        recipe.with_architecture(TrainConfig.smoke("regression"), arch, keep_size=True),
        steps=3,
    )
    result = train(config, tmp_path, mlflow_log=False)
    model, borders = registry.load_checkpoint(result["final_path"])
    assert isinstance(model, ColumnRowTransformer) and borders is not None
    assert model.config == config.model_config()
    state = torch.load(tmp_path / "latest.pt", weights_only=True)
    check_checkpoint(state, config, "resume_from")
    with pytest.raises(ValueError, match="cell_transformer"):
        check_checkpoint(state, replace(TrainConfig.smoke("regression")), "resume_from")

    loaded, loaded_borders = hub.from_pretrained(
        hub.save_pretrained(model, borders, tmp_path / "pretrained")
    )
    assert isinstance(loaded, ColumnRowTransformer) and torch.equal(loaded_borders, borders)
    for name, tensor in model.state_dict().items():
        assert torch.equal(loaded.state_dict()[name], tensor), name
    # 학습 run param에서 구조를 다시 읽는다 (학습 스크립트의 resume). 학습 때 본 feature 수는
    # 구조가 아니라 run 구조에 들어가지 않는다
    params = {k: str(v) for k, v in vars(config).items()}
    source = hub.run_architecture(params)
    assert "max_features" not in source
    rebuilt = replace(TrainConfig(task="regression"), **source, max_features=config.max_features)
    assert rebuilt.model_config() == config.model_config()


def test_ensembles_and_default_inference():
    X, y = load_breast_cancer(return_X_y=True)
    X[::7, 3] = np.nan
    model = _model()
    defaults = recipe.default_inference(model, "classification")
    assert defaults == {"n_estimators": 4, "member_kinds": ("quantile", "robust")}
    clf = EnsembleClassifier(model, **defaults).fit(X[:100], y[:100])
    proba = clf.predict_proba(X[100:130])
    assert proba.shape == (30, 2) and np.allclose(proba.sum(1), 1, atol=1e-5)

    Xr, yr = load_diabetes(return_X_y=True)
    reg_model = _model("regression", "v3.5")
    defaults = recipe.default_inference(reg_model, "regression")
    assert defaults == {"n_estimators": 8, "member_kinds": ("standard",)}
    reg = EnsembleRegressor(reg_model, torch.linspace(-3, 3, 5001), **defaults)
    pred = reg.fit(Xr[:100], yr[:100]).predict(Xr[100:120])
    assert pred.shape == (20,) and np.isfinite(pred).all()
    assert {p.member.kind for p in reg.preprocessors_} == {"standard"}
    # weight와 함께 온 추론 설정이 먼저다
    reg_model.inference_config = {"n_estimators": 2, "member_kinds": ["quantile", "power"]}
    assert recipe.default_inference(reg_model, "regression") == {
        "n_estimators": 2,
        "member_kinds": ("quantile", "power"),
    }


def test_standard_members_keep_values():
    from minipfn.inference.preprocessing import Member, MemberPreprocessor

    X = np.random.default_rng(0).normal(size=(50, 3))
    prep = MemberPreprocessor(Member("standard", None, False, seed=0), [], 160).fit(X)
    out = prep.transform(X, np.random.default_rng(1))
    # 원래 열 3개 + 샘플 식별자 1개를 섞은 것
    assert out.shape == (50, 4)
    kept = out[:, np.argsort(prep.order)[:3]]
    assert np.allclose(kept, X)


def _script(name):
    path = Path(__file__).parents[1] / "scripts" / "tabpfn_v2" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{name}_tabpfn_v2_script_v3", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_train_script_builds_v3_and_reads_its_run(monkeypatch):
    for name in ("TABPFN_V2_ARCH", "TABPFN_V2_CONFIG_NAME", "TABPFN_V2_TASK"):
        monkeypatch.delenv(name, raising=False)
    script = _script("train")
    args = script.parse_args(["--config-name", "paper", "--arch", "v3"])
    config, arch = script.build_config(args)
    assert arch == "v3" and config == recipe.paper("classification")
    params = {k: str(v) for k, v in vars(config).items()}
    args = script.parse_args(["--config-name", "paper", "--resume-run-id", "r"])
    resumed, arch = script.build_config(args, hub.run_architecture(params))
    assert arch == "v3" and resumed.model_config() == config.model_config()
    args = script.parse_args(["--config-name", "smoke", "--arch", "v3.5"])
    config, arch = script.build_config(args)
    assert arch == "v3.5" and config.nlayers == TrainConfig.smoke().nlayers
    with pytest.raises(SystemExit):
        script.parse_args(["--config-name", "long_tables", "--init-run-id", "r", "--arch", "v3"])


def test_train_script_takes_the_batch_size_and_keeps_it_on_resume(monkeypatch):
    # GPU 32장에서 GPU당 데이터셋 8개를 유지하려고 batch를 256으로 키운다. 이어 학습하면 인자를
    # 주지 않아도 원래 run의 batch를 따르고(preset의 64로 돌아가지 않는다), 인자로 주면 그 값이다
    for name in ("ARCH", "CONFIG_NAME", "TASK", "BATCH_SIZE"):
        monkeypatch.delenv(f"TABPFN_V2_{name}", raising=False)
    script = _script("train")
    base = ["--config-name", "paper", "--arch", "v3.5", "--task", "multitask"]
    config, _ = script.build_config(script.parse_args(base))
    assert config.batch_size == 64
    monkeypatch.setenv("TABPFN_V2_BATCH_SIZE", "256")
    config, _ = script.build_config(script.parse_args(base))
    assert config.batch_size == 256 and config.grad_accumulation(32) == 1
    assert config.for_world_size(32).micro_batch == 8
    monkeypatch.delenv("TABPFN_V2_BATCH_SIZE")
    params = {k: str(v) for k, v in vars(config).items()}
    resume = ["--config-name", "paper", "--task", "multitask", "--resume-run-id", "r"]
    for extra, expected in (([], 256), (["--batch-size", "128"], 128)):
        args = script.parse_args(resume + extra)
        script.inherit_batch_size(args, params)
        resumed, _ = script.build_config(args, hub.run_architecture(params))
        assert resumed.batch_size == expected
    # 가중치만 가져와 새로 학습하면 preset 값
    init = ["--config-name", "paper", "--task", "multitask", "--init-run-id", "r"]
    args = script.parse_args(init)
    script.inherit_batch_size(args, params)
    assert args.batch_size is None


def test_eval_script_takes_v3_inference_defaults():
    script = _script("eval")
    assert script.default_inference(_model(arch="v3.5"), "classification") == {
        "n_estimators": 8,
        "member_kinds": ("standard",),
    }
    assert script.resolve_member_kinds("v3.5") == ("standard",)
    assert script.resolve_member_kinds("v2") == tabpfn_v2.MEMBER_KINDS_V2


@pytest.mark.parametrize("arch", ["v3", "v3.5"])
def test_fresh_model_reads_its_input(arch):
    # 처음 만든 모델(0으로 시작하는 층 그대로)에서도 행 벡터가 입력에 따라 달라 gradient가 선다.
    # CLS 토큰이 크거나 행 단계 출력층이 0에서 시작하면 모든 행이 거의 같아 gradient가 1e-8이었다
    config = recipe.with_architecture(TrainConfig.smoke(), arch, keep_size=True)
    torch.manual_seed(0)
    model = config.build_model()
    x, y = _data("classification")
    out = model(x, y)
    assert out.std(dim=1).mean() > 1e-3  # 테스트 행마다 다른 예측
    out[..., 0].sum().backward()
    assert model.row_blocks[0].attention.qkv.weight.grad.norm() > 1e-4
    assert model.heads["classification"].q_projection.weight.grad.norm() > 1e-4


def test_one_recipe_lookup_serves_every_model():
    from minipfn import recipes

    assert recipes.recipe_for_model_type("column_row_transformer") is recipe
    assert recipes.recipe_for_model_type("tabpfn_v2") is tabpfn_v2  # 예전 이름
    assert recipes.recipe_for_architecture("v3.5") is recipe
    assert recipes.architectures() == ("v2", "v2.5", "v3", "v3.5")
    assert recipes.architecture_name(recipe.paper()) == "v3"
    assert recipes.default_inference(_model(arch="v3.5"), "regression")["member_kinds"] == (
        "standard",
    )
    # 묶음 이름은 recipe와 상관없이 다 받는다
    assert tabpfn_v2.resolve_member_kinds("v3.5") == ("standard",)
    assert recipes.resolve_member_kinds("v2") == tabpfn_v2.MEMBER_KINDS_V2


def test_train_script_stops_long_tables_from_a_v3_run(monkeypatch):
    for name in ("TABPFN_V2_ARCH", "TABPFN_V2_CONFIG_NAME", "TABPFN_V2_TASK"):
        monkeypatch.delenv(name, raising=False)
    script = _script("train")
    params = {k: str(v) for k, v in vars(recipe.paper()).items()}
    args = script.parse_args(["--config-name", "long_tables", "--resume-run-id", "r"])
    with pytest.raises(SystemExit, match="long_tables"):
        script.build_config(args, hub.run_architecture(params))


def test_eval_member_kinds_do_not_change_the_member_count(monkeypatch, tmp_path):
    script = _script("eval")
    model = _model(arch="v3")
    monkeypatch.setattr(script, "load_model", lambda path, suite, dst: (model, None))
    args = script.parse_args(["--classifier-path", "x.pt", "--member-kinds", "v3.5"])
    make = script.make_estimator_factory("classification", args)
    assert make.member_kinds == ("standard",) and make.n_estimators == 4
    args = script.parse_args(["--classifier-path", "x.pt", "--n-estimators", "8"])
    make = script.make_estimator_factory("classification", args)
    assert make.member_kinds == ("quantile", "robust") and make.n_estimators == 8


def test_arena_limits_follow_the_model_recipe():
    # TabArena·BeyondArena의 크기 상한은 모델 종류의 recipe가 정한다 (v2 크기로 자르지 않는다)
    import argparse as _argparse

    from minipfn import recipes

    assert (
        recipes.supported_size(_model(arch="v3"), "classification") == recipe.SUPPORTED_SIZE["v3"]
    )
    assert (
        recipes.supported_size(_model("regression", "v3.5"), "regression")["max_features"] == 6_000
    )
    script = _script("eval")
    args = _argparse.Namespace(max_train_rows=None, max_features=None)
    limits = script.arena_limits({"classification": (_model(arch="v3"), None)}, args)
    assert limits["binary"].max_train_rows == 100_000 and limits["binary"].max_classes == 10


# ---------------------------------------------------------------------------
# TabPFN-3.5: 분류·회귀 한 모델 (multitask)
# ---------------------------------------------------------------------------
def _multitask(**changes):
    return _model("multitask", "v3.5", **changes)


def test_multitask_model_needs_a_task_and_shares_everything_but_targets_and_heads():
    model = _multitask()
    assert model.tasks == ("classification", "regression")
    x, yc = _data("classification")
    _, yr = _data("regression")
    assert model(x, yc, task="classification").shape == (2, 15, 10)
    assert model(x, yr, task="regression").shape == (2, 15, model.config.n_buckets)
    with pytest.raises(ValueError, match="task"):
        model(x, yc)
    with pytest.raises(ValueError, match="task"):
        _model()(x, yc, task="regression")  # 한 task 모델은 그 task만
    # task마다 따로인 것은 정답 encoder(셀·ICL), 출력층 앞 MLP, 출력층뿐이다
    per_task = {"cell_target", "icl_target", "pre_head", "heads"}
    for task, other in (("classification", "regression"), ("regression", "classification")):
        model.zero_grad()
        y = yc if task == "classification" else yr
        model(x, y, task=task)[..., 0].sum().backward()
        unused = {n for n, p in model.named_parameters() if p.grad is None}
        assert unused == {
            n
            for n, _ in model.named_parameters()
            if n.split(".")[0] in per_task and f".{other}" in n
        }


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_multitask_test_rows_do_not_see_each_other(task):
    model = _multitask()
    x, y = _data(task)
    out = model(x, y, task=task)
    changed = x.clone()
    changed[:, 30] += 5.0
    other = model(changed, y, task=task)
    keep = [i for i in range(15) if i != 5]
    assert torch.equal(other[:, keep], out[:, keep]) and not torch.equal(other[:, 5], out[:, 5])


def test_labels_reach_the_icl_stage():
    # 정답 embedding은 셀 단계와 행 ICL 입력 두 곳에 들어간다 (공개 추론 코드와 같은 구조)
    model = _multitask()
    assert set(model.icl_target) == {"classification", "regression"}
    x, y = _data("regression")
    out = model(x, y, task="regression")
    with torch.no_grad():
        model.icl_target["regression"].encode.weight.mul_(3.0)
    assert not torch.allclose(model(x, y, task="regression"), out)


def test_qassmax_starts_with_query_independent_scales():
    from minipfn.models.column_row_transformer import QASSMax

    qassmax = QASSMax(nhead=2, head_dim=4, hidden=8)
    q = torch.randn(3, 2, 5, 4)
    base = qassmax.base_mlp(torch.full((1, 1), math.log(7))).view(1, 2, 1, 4)
    assert torch.allclose(qassmax(q, 7), q * base)  # query_mlp가 0에서 시작해 tanh 항이 1


def test_multitask_recipe_and_config():
    config = recipe.paper("multitask", "v3.5")
    assert (
        config.tasks == ("classification", "regression") and config.target_norm and config.head_mlp
    )
    assert config.rewind_lr_factor == tabpfn_v2.paper("regression").rewind_lr_factor
    assert set(config.task_configs()) == {"classification", "regression"}
    with pytest.raises(ValueError, match="multitask"):
        recipe.paper("multitask", "v3")  # 3은 task마다 모델이 따로
    with pytest.raises(ValueError, match="multitask"):
        TrainConfig.smoke("multitask").build_model()  # cell_transformer는 한 task뿐
    with pytest.raises(ValueError, match="task_configs"):
        config.make_prior(0)


def test_multitask_train_save_load_and_resume_checks(tmp_path):
    config = replace(
        recipe.with_architecture(TrainConfig.smoke("multitask"), "v3.5", keep_size=True),
        steps=4,
        val_every=2,
        log_every=2,
    )
    result = train(config, tmp_path, mlflow_log=False)
    model, borders = registry.load_checkpoint(result["final_path"])
    assert model.tasks == ("classification", "regression") and borders is not None
    state = torch.load(tmp_path / "latest.pt", weights_only=True)
    check_checkpoint(state, config, "resume_from")
    single = recipe.with_architecture(TrainConfig.smoke("classification"), "v3.5", keep_size=True)
    with pytest.raises(ValueError):
        check_checkpoint(state, single, "resume_from")  # 경계가 있는 checkpoint를 분류만으로
    assert len(result["history"]["val_loss"]) == 2


def test_multitask_ensembles_and_hub(tmp_path):
    model = _multitask()
    X, y = load_breast_cancer(return_X_y=True)
    clf = EnsembleClassifier(model, **recipe.default_inference(model, "classification"))
    proba = clf.fit(X[:80], y[:80]).predict_proba(X[80:100])
    assert proba.shape == (20, 2) and np.allclose(proba.sum(1), 1, atol=1e-5)
    Xr, yr = load_diabetes(return_X_y=True)
    borders = torch.linspace(-3, 3, model.config.n_buckets + 1)
    reg = EnsembleRegressor(model, borders, **recipe.default_inference(model, "regression"))
    assert np.isfinite(reg.fit(Xr[:80], yr[:80]).predict(Xr[80:100])).all()

    folder = hub.save_pretrained(model, borders, tmp_path / "pretrained")
    import json

    assert json.loads((folder / "config.json").read_text())["task"] == "multitask"
    for task in ("classification", "regression"):
        loaded, loaded_borders = hub.model_or_pretrained(str(folder), "unused", task)
        assert loaded.tasks == model.tasks and torch.equal(loaded_borders, borders)
    assert hub.served_tasks(model, None) == ("classification",)  # 경계 없이는 회귀를 못 낸다


def test_train_script_takes_multitask(monkeypatch):
    for name in ("TABPFN_V2_ARCH", "TABPFN_V2_CONFIG_NAME", "TABPFN_V2_TASK"):
        monkeypatch.delenv(name, raising=False)
    script = _script("train")
    args = script.parse_args(["--config-name", "paper", "--task", "multitask", "--arch", "v3.5"])
    config, arch = script.build_config(args)
    assert arch == "v3.5" and config == recipe.paper("multitask", "v3.5")
    args = script.parse_args(["--config-name", "smoke", "--task", "multitask", "--arch", "v3.5"])
    assert script.build_config(args)[0].task == "multitask"


@pytest.mark.parametrize("arch,task", [("v3", "classification"), ("v3.5", "multitask")])
def test_parameter_counts_by_stage_and_task_add_up(arch, task):
    config = recipe.paper(task, arch).model_config()
    with torch.device("meta"):
        model = ColumnRowTransformer(config)
    by_stage = model.num_parameters_by_stage()
    assert list(by_stage) == ["cell_encoder", "column_stage", "row_stage", "row_icl", "output"]
    assert sum(by_stage.values()) == model.num_parameters()
    by_task = model.num_parameters_by_task()
    assert set(by_task) == {"shared", *config.tasks}
    assert sum(by_task.values()) == model.num_parameters()
    assert all(v > 0 for v in by_task.values())


def test_tiny_config_and_randomize_zero_weights():
    config = ColumnRowTransformerConfig.tiny(n_out=10)
    assert (config.emsize, config.n_cls, config.nlayers, config.n_out) == (16, 2, 2, 10)
    torch.manual_seed(0)
    model = registry.randomize_zero_weights(ColumnRowTransformer(config))
    assert not any(p.dim() > 1 and (p == 0).all() for p in model.parameters())
    norms = [p for n, p in model.named_parameters() if n.endswith("row_norm.weight")]
    assert norms and all((p == 1).all() for p in norms)  # 1차원 파라미터는 그대로
    x, y = _data("classification")
    with torch.no_grad():
        assert model(x, y).shape == (2, 15, 10)
