"""TabPFN-3·3.5 (`column_row_transformer` 모델, `recipes.tabpfn_v3`) 테스트."""

import importlib.util
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
    # 보고서: TabPFN-3 분류 53M·회귀 58M, TabPFN-3.5 220M
    [("v3", "classification", 53), ("v3", "regression", 58), ("v3.5", "regression", 220)],
)
def test_paper_size_matches_the_reports(arch, task, millions):
    config = recipe.paper(task, arch).model_config()
    with torch.device("meta"):
        model = ColumnRowTransformer(config)
    assert abs(model.num_parameters() / 1e6 - millions) / millions < 0.1
    assert config.icl_dim == (512 if arch == "v3" else 1024)


def test_paper_architecture_and_names():
    config = recipe.paper("regression")
    assert config.model_type == "column_row_transformer"
    assert recipe.architecture_name(config) == "v3"
    assert config.qk_clip == tabpfn_v2.paper("regression").qk_clip  # v2 학습 설정 그대로
    v35 = recipe.paper("regression", "v3.5")
    assert recipe.architecture_name(v35) == "v3.5" and v35.qk_clip is None  # q·k RMSNorm이라 끔
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
    p = model(x, y).exp()
    assert torch.allclose(p.sum(-1), torch.ones(2, 15), atol=1e-4)
    # 학습 행에 없는 클래스는 clip 값만 남는다 (클래스 수에 따른 파라미터가 없다)
    assert (p[..., 3:] < 1e-5).all()


@pytest.mark.parametrize("arch", ["v3", "v3.5"])
def test_every_parameter_gets_a_gradient(arch):
    # DDP는 forward에 쓰지 않은 파라미터가 있으면 멈춘다
    for task in ("classification", "regression"):
        model = _model(task, arch).train()
        x, y = _data(task)
        # 분류 logit은 log p라 logsumexp가 0이다. 한 클래스의 log 확률을 쓴다
        model(x, y)[..., 0].sum().backward()
        assert [n for n, p in model.named_parameters() if p.grad is None] == []


def test_qk_clip_sees_the_icl_attention_only_without_qk_norm():
    model = _model()
    weights = model.qk_clip_weights()
    assert len(weights) == model.config.nlayers
    assert all(kind == "sample_attention" for kind, _ in weights)
    assert weights[0][1].shape == (3 * model.config.icl_dim, model.config.icl_dim)
    assert _model(arch="v3.5").qk_clip_weights() == []


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
        qk_clip=None if arch == "v3.5" else 50.0,
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
    assert model.decoder.weight.grad.norm() > 1e-4
