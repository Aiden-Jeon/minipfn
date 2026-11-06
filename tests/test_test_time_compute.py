"""추론에 연산을 더 쓰는 방법 (issue #43): 멤버 pool greedy 선택, DiagScale."""

import copy
import importlib.util
import json
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.datasets import load_breast_cancer, load_diabetes

from minipfn.benchmarks import arena
from minipfn.inference.diag_scale import DiagScale, add_diag_scale, fit_diag_scale
from minipfn.inference.ensemble import EnsembleClassifier, EnsembleRegressor
from minipfn.inference.ensemble_selection import (
    GreedyEnsembleClassifier,
    GreedyEnsembleRegressor,
    greedy_weights,
)
from minipfn.inference.preprocessing import Member, MemberPreprocessor
from minipfn.models.cell_transformer import CellTransformer, CellTransformerConfig

BORDERS = torch.linspace(-3, 3, 17)


def _tiny_model(n_out: int = 10) -> CellTransformer:
    torch.manual_seed(0)
    model = CellTransformer(CellTransformerConfig(emsize=16, nlayers=2, nhead=2, n_out=n_out))
    for p in model.parameters():
        torch.nn.init.normal_(p, std=0.3)
    return model.eval()


def _classification(n=120):
    x, y = load_breast_cancer(return_X_y=True)
    return x[:n, :8], y[:n]


def _regression(n=120):
    x, y = load_diabetes(return_X_y=True)
    return x[:n], y[:n]


def test_greedy_weights_pick_with_replacement_and_break_ties_to_the_first():
    y = np.array([0.0, 1.0])
    preds = np.array([[0.0, 0.0], [0.0, 2.0], [0.0, 2.0]])  # 0과 1의 평균이 정답

    def rmse(p):
        return np.sqrt(np.mean((p - y) ** 2, axis=-1))

    w = greedy_weights(preds, rmse, n_rounds=4)
    assert np.allclose(w, [0.5, 0.5, 0.0])  # 1과 2는 같아서 앞 번호만
    assert greedy_weights(np.zeros((3, 2)), rmse, n_rounds=2)[0] == 1.0  # 모두 같으면 0번


def test_outlier_sigma_of_a_member_sets_the_clip():
    x = np.concatenate([np.zeros((99, 1)), [[100.0]]])

    def clipped(sigma):
        m = Member("power", None, False, seed=0, outlier_sigma=sigma)
        return MemberPreprocessor(m, [], 10).fit(x)._clip(x).max()

    assert clipped(12.0) == 100.0  # 기본: 표준편차 12배(약 119) 안이라 그대로
    assert np.isclose(clipped(3.0), x.mean() + 3 * x.std())


def test_pool_starts_with_the_default_members():
    x, y = _classification()
    greedy = GreedyEnsembleClassifier(_tiny_model(), n_estimators=4, pool_size=7, n_folds=2)
    greedy.fit(x, y)
    default = EnsembleClassifier(_tiny_model(), n_estimators=4).fit(x, y)
    rng = np.random.default_rng(0)
    pool = greedy._pool(rng, greedy._default_members(rng, 2), 2)
    assert len(pool) == 7
    for a, b in zip(pool[:4], [p.member for p in default.preprocessors_], strict=True):
        assert (a.kind, a.seed, a.one_hot) == (b.kind, b.seed, b.one_hot)
        assert np.array_equal(a.class_permutation, b.class_permutation)


def test_pool_members_do_not_expand_one_hot_when_it_is_off():
    rng = np.random.default_rng(0)
    on = GreedyEnsembleClassifier(_tiny_model(), n_estimators=2, pool_size=20)
    off = GreedyEnsembleClassifier(_tiny_model(), n_estimators=2, pool_size=20, one_hot="off")
    assert any(m.one_hot for m in on._pool(rng, [], 2))
    pool = off._pool(np.random.default_rng(0), [], 2)
    assert not any(m.one_hot for m in pool)
    seeds = [m.seed for m in on._pool(np.random.default_rng(0), [], 2)]
    assert seeds == [m.seed for m in pool]  # 나머지 멤버는 같다


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_choosing_only_candidate_zero_is_the_default_ensemble(task, monkeypatch):
    x, y = _classification() if task == "classification" else _regression()
    monkeypatch.setattr(
        GreedyEnsembleClassifier if task == "classification" else GreedyEnsembleRegressor,
        "_select",
        lambda self, oof, loss: np.eye(len(oof))[0],
    )
    if task == "classification":
        greedy = GreedyEnsembleClassifier(_tiny_model(), n_estimators=4, pool_size=6, n_folds=2)
        default = EnsembleClassifier(_tiny_model(), n_estimators=4)
        out, ref = (e.fit(x[:90], y[:90]).predict_proba(x[90:]) for e in (greedy, default))
    else:
        greedy = GreedyEnsembleRegressor(
            _tiny_model(16), BORDERS, n_estimators=4, pool_size=6, n_folds=2
        )
        default = EnsembleRegressor(_tiny_model(16), BORDERS, n_estimators=4)
        out, ref = (e.fit(x[:90], y[:90]).predict(x[90:]) for e in (greedy, default))
        log_probs, _ = greedy.predict_distribution(x[90:])
        assert torch.allclose(log_probs, default.predict_distribution(x[90:])[0], atol=1e-5)
    assert np.allclose(out, ref, atol=1e-5)
    assert greedy.selection_["used_members"] == 4  # 다시 맞춘 것은 기본 멤버뿐


def test_greedy_classifier_outputs_and_out_of_fold_loss():
    x, y = _classification()
    clf = GreedyEnsembleClassifier(_tiny_model(), n_estimators=4, pool_size=8)
    proba = clf.fit(x[:90], y[:90]).predict_proba(x[90:])
    assert proba.shape == (30, 2) and np.allclose(proba.sum(axis=1), 1.0)
    s = clf.selection_
    assert s["n_folds"] == 8 and s["candidates"] == 1 + 8 * 9
    assert s["oof_loss_selected"] <= s["oof_loss_default"] + 1e-12  # 0번도 후보다
    assert s["used_members"] == len(clf.preprocessors_) <= 8
    total = sum(sum(w.values()) for w in clf.member_weights_)
    assert np.isclose(total, 1.0)


def test_greedy_classifier_handles_a_class_with_one_row():
    x, y = _classification(60)
    y = y.copy()
    y[0] = 2  # 학습 행이 하나뿐인 클래스: 그 행이 없는 fold가 생긴다
    clf = GreedyEnsembleClassifier(_tiny_model(), n_estimators=2, pool_size=3).fit(x, y)
    assert clf.selection_["n_folds"] == 2
    assert np.allclose(clf.predict_proba(x[:5]).sum(axis=1), 1.0)


def test_selection_modes():
    x, y = _classification()
    kw = {"n_estimators": 2, "pool_size": 4, "n_folds": 3}
    best = GreedyEnsembleClassifier(_tiny_model(), selection="best", **kw).fit(x, y)
    assert len(best.member_weights_) == 1 or best.selection_["weight_default"] == 1.0
    uniform = GreedyEnsembleClassifier(_tiny_model(), selection="uniform", **kw).fit(x, y)
    assert uniform.selection_["used_members"] == 4
    assert all(len(w) == 9 for w in uniform.member_weights_)
    with pytest.raises(ValueError):
        GreedyEnsembleClassifier(_tiny_model(), selection="mean", **kw)
    with pytest.raises(ValueError):
        GreedyEnsembleClassifier(_tiny_model(), n_estimators=4, pool_size=2)


def test_greedy_regressor_mean_and_distribution_use_the_same_weights():
    x, y = _regression()
    reg = GreedyEnsembleRegressor(_tiny_model(16), BORDERS, n_estimators=4, pool_size=8)
    reg.fit(x[:90], y[:90])
    mean, log_probs, borders = reg.predict_full(x[90:])
    assert np.allclose(mean, reg.predict(x[90:]))
    assert torch.allclose(log_probs.exp().sum(dim=1), torch.ones(30, dtype=log_probs.dtype))
    assert torch.allclose(log_probs, reg.predict_distribution(x[90:])[0])
    assert borders.dtype == torch.float64 and len(borders) == len(BORDERS)


def test_diag_scale_starts_as_the_frozen_model_and_only_scales_queries():
    model = _tiny_model()
    adapted = copy.deepcopy(model)
    deltas = add_diag_scale(adapted)
    # 층 2개 × (feature, sample attention) × E
    assert len(deltas) == 4 and sum(d.numel() for d in deltas) == 4 * 16
    x, y = torch.randn(1, 30, 5), torch.randint(0, 2, (1, 20)).float()
    with torch.no_grad():
        assert torch.equal(model(x, y), adapted(x, y))
        deltas[1].fill_(0.5)
        assert not torch.allclose(model(x, y), adapted(x, y))
    loss = adapted(x, y).square().mean()
    loss.backward()
    assert deltas[0].grad is not None
    assert set(model.state_dict()) == set(adapted.state_dict()) - {
        f"layers.{i}.{k}.diag_scale"
        for i in range(2)
        for k in ("feature_attention", "sample_attention")
    }
    with pytest.raises(ValueError):
        add_diag_scale(adapted)


def test_diag_scale_without_epochs_keeps_the_frozen_predictions():
    x, y = _classification()
    model = _tiny_model()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    clf = DiagScale(EnsembleClassifier(model, n_estimators=2), epochs=0).fit(x[:90], y[:90])
    ref = EnsembleClassifier(model, n_estimators=2).fit(x[:90], y[:90])
    assert np.allclose(clf.predict_proba(x[90:]), ref.predict_proba(x[90:]), atol=1e-6)
    assert clf.adaptation_["best_epoch"] == -1 and clf.adaptation_["delta_norm"] == 0.0
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())  # 원래 모델


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_diag_scale_trains_and_picks_an_epoch_on_validation(task):
    if task == "classification":
        x, y = _classification(160)
        estimator = EnsembleClassifier(_tiny_model(), n_estimators=2)
    else:
        x, y = _regression(160)
        estimator = EnsembleRegressor(_tiny_model(16), BORDERS, n_estimators=2)
    model, info = fit_diag_scale(estimator, x, y, epochs=4, patience=2)
    assert info["val_rows"] == 40 and info["n_params"] == 64
    assert len(info["val_curve"]) == info["epochs_run"] + 1 and info["train_loss"]
    assert info["val_loss_best"] == info["val_curve"][info["best_epoch"] + 1]
    if info["best_epoch"] == -1:
        assert info["delta_norm"] == 0.0
    else:
        assert info["val_loss_best"] < info["val_loss_before"]


def test_diag_scale_wraps_greedy_selection():
    x, y = _classification()
    greedy = GreedyEnsembleClassifier(_tiny_model(), n_estimators=2, pool_size=3, n_folds=2)
    clf = DiagScale(greedy, epochs=2).fit(x[:90], y[:90])
    assert np.allclose(clf.predict_proba(x[90:]).sum(axis=1), 1.0)
    assert list(clf.classes_) == [0, 1] and "used_members" in clf.selection_
    assert clf.estimator_.model is not greedy.model  # 학습한 사본
    with pytest.raises(AttributeError):
        DiagScale(greedy).classes_  # noqa: B018 (fit 전에는 없다)


def test_run_times_average_the_splits_that_ran():
    ours = pd.DataFrame(
        {
            "imputed": [False, False, True],
            "time_train_s": [1.0, 3.0, 100.0],
            "time_infer_s": [0.5] * 3,
        }
    )
    t = arena.run_times(ours)
    assert t["time_train_s_mean"] == 2.0 and t["time_train_s_total"] == 4.0
    assert t["time_infer_s_mean"] == 0.5
    assert arena.run_times(ours.drop(columns=list(arena.TIME_COLUMNS))) == {}


def _eval_script():
    path = Path(__file__).parents[1] / "scripts" / "tabpfn_v2" / "eval.py"
    spec = importlib.util.spec_from_file_location("eval_tabpfn_v2_test_time", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_eval_script_builds_test_time_estimators_and_records_fits(monkeypatch, tmp_path):
    script = _eval_script()
    monkeypatch.setattr(script, "load_model", lambda path, task, cache: (_tiny_model(), None))
    x, y = _classification(60)
    for test_time, kind in [
        ("none", EnsembleClassifier),
        ("greedy", GreedyEnsembleClassifier),
        ("diagscale", DiagScale),
    ]:
        args = script.parse_args(
            ["--classifier-path", "clf.pt", "--test-time", test_time, "--pool-size", "4"]
            + ["--out-dir", str(tmp_path / test_time), "--device", "cpu"]
        )
        make = script.make_estimator_factory("classification", args)
        estimator = make(types.SimpleNamespace(categorical_indices=[]), "all")
        assert isinstance(estimator, kind)
        if test_time == "none":
            assert not (tmp_path / test_time / "test_time.jsonl").exists()
            assert script.test_time_params(args) == {"test_time": "none"}
            continue
        if test_time == "diagscale":
            estimator.kwargs["epochs"] = 1
        estimator.fit(x, y)
        record = json.loads((tmp_path / test_time / "test_time.jsonl").read_text())
        assert record["rows"] == 60 and record["features"] == 8
        assert ("selection" in record) == (test_time == "greedy")
        assert ("adaptation" in record) == (test_time == "diagscale")
    params = script.test_time_params(script.parse_args(["--test-time", "diagscale+greedy"]))
    assert params == {"test_time": "diagscale+greedy", "selection": "greedy", "pool_size": 24}
