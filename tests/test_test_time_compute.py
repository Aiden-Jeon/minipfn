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
    _log_loss,
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


def test_min_improvement_falls_back_to_the_default_ensemble(monkeypatch):
    x, y = _classification()
    kw = {"n_estimators": 4, "pool_size": 6, "n_folds": 2}
    # 조건을 넘을 수 없는 비율: 늘 기본 앙상블과 같은 예측
    strict = GreedyEnsembleClassifier(_tiny_model(), min_improvement=0.999, **kw)
    out = strict.fit(x[:90], y[:90]).predict_proba(x[90:])
    ref = EnsembleClassifier(_tiny_model(), n_estimators=4).fit(x[:90], y[:90])
    assert np.allclose(out, ref.predict_proba(x[90:]), atol=1e-5)
    s = strict.selection_
    assert s["reverted"] and s["weight_default"] == 1.0 and s["used_members"] == 4
    assert s["oof_loss_selected"] == s["oof_loss_default"]
    # 조건이 없으면 되돌리지 않는다 (같은 pool·fold)
    free = GreedyEnsembleClassifier(_tiny_model(), min_improvement=0.0, **kw).fit(x[:90], y[:90])
    assert not free.selection_["reverted"]


def test_guard_needs_an_absolute_drop_in_log_loss_when_the_default_is_near_zero():
    clf = GreedyEnsembleClassifier(_tiny_model(), n_estimators=2, min_improvement=0.01)
    # 기본(0번)의 log loss 1e-4, 고른 후보 0: 비율로는 100%지만 1e-3 nats보다 작다
    oof = np.array([[np.exp(-1e-4)] * 4, [1.0] * 4])
    weights, reverted = clf._guard(np.array([0.0, 1.0]), oof, _log_loss)
    assert reverted and np.array_equal(weights, [1.0, 0.0])
    oof = np.array([[np.exp(-0.1)] * 4, [np.exp(-0.05)] * 4])  # 0.1 → 0.05: 둘 다 넘는다
    assert not clf._guard(np.array([0.0, 1.0]), oof, _log_loss)[1]
    reg = GreedyEnsembleRegressor(_tiny_model(16), BORDERS, n_estimators=2, min_improvement=0.01)
    y = np.zeros(4)

    def rmse(p):
        return np.sqrt(np.mean((p - y) ** 2, axis=-1))

    # 회귀는 비율만 본다: 1e-4 → 0도 넘는다
    oof = np.array([[1e-4] * 4, [0.0] * 4])
    assert not reg._guard(np.array([0.0, 1.0]), oof, rmse)[1]


def test_min_temperature_drops_sharp_temperatures_from_the_pool():
    x, y = _classification()
    clf = GreedyEnsembleClassifier(
        _tiny_model(), n_estimators=2, pool_size=3, n_folds=2, min_temperature=0.5
    ).fit(x, y)
    assert 0.25 not in clf.temperatures and len(clf.temperatures) == 8
    assert clf.selection_["candidates"] == 1 + 3 * 8
    assert all(t >= 0.5 or t == clf.softmax_temperature for w in clf.member_weights_ for t in w)
    with pytest.raises(ValueError):
        GreedyEnsembleClassifier(_tiny_model(), n_estimators=2, min_temperature=10.0)
    with pytest.raises(ValueError):
        GreedyEnsembleClassifier(_tiny_model(), n_estimators=2, min_improvement=1.0)


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_groups_keep_each_group_in_one_out_of_fold_fold(task, monkeypatch):
    x, y = _classification() if task == "classification" else _regression()
    groups = np.repeat(np.arange(12), 10)  # 10행씩 12개 group
    cls = GreedyEnsembleClassifier if task == "classification" else GreedyEnsembleRegressor
    seen = []
    original = cls._out_of_fold

    def capture(self, X, y, members, splits):
        seen.append(splits)
        return original(self, X, y, members, splits)

    monkeypatch.setattr(cls, "_out_of_fold", capture)
    args = (_tiny_model(),) if task == "classification" else (_tiny_model(16), BORDERS)
    est = cls(*args, n_estimators=2, pool_size=3, n_folds=4).fit(
        x, y, groups=[f"g{g}" for g in groups]
    )
    assert est.selection_["grouped"] and est.selection_["n_folds"] == 4
    for train_ix, test_ix in seen[0]:
        assert not set(groups[train_ix]) & set(groups[test_ix])
    # group이 fold 수보다 적으면 group 수만큼, 하나뿐이면 무작위 fold
    est = cls(*args, n_estimators=2, pool_size=3, n_folds=8).fit(x, y, groups=groups % 3)
    assert est.selection_["n_folds"] == 3
    est = cls(*args, n_estimators=2, pool_size=3, n_folds=4).fit(x, y, groups=np.zeros(len(y)))
    assert not est.selection_["grouped"]
    with pytest.raises(ValueError):
        cls(*args, n_estimators=2, pool_size=3).fit(x, y, groups=groups[:5])


def test_split_groups_from_task_split_metadata():
    X = pd.DataFrame(
        {
            "patient": ["a", "b", "a", "c", None],
            "site": [1, 1, 2, 2, 2],
            "when": pd.to_datetime(["2020-01-01", "2020-01-01", "2021-06-01", None, "2022-01-01"]),
        }
    )
    meta = types.SimpleNamespace
    g = arena.split_groups(X, meta(group_on="patient", time_on=None))
    assert g[0] == g[2] and len(set(g)) == 4  # 빈 값도 한 group
    g = arena.split_groups(X, meta(group_on=["patient", "site"], time_on=None))
    assert len(set(g)) == 5
    t = arena.split_groups(X, meta(group_on=None, time_on="when"), n_intervals=2)
    assert t[0] == t[1] and t[3] == -1  # 같은 시각은 같은 구간, 빈 시각은 -1
    assert t[0] <= t[2] <= t[4] and len(set(t[[0, 2, 4]])) == 2  # 시간 순서대로 연속 구간
    assert arena.split_groups(X, meta(group_on=None, time_on=None)) is None
    assert arena.split_groups(X, None) is None
    assert arena.split_groups(X, meta(group_on="missing", time_on=None)) is None


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
    assert params == {
        "test_time": "diagscale+greedy",
        "selection": "greedy",
        "pool_size": 24,
        "min_improvement": 0.01,  # issue #52의 기본값
        "oof_split": "task",
    }
    # 논문 방식(PR #49)으로 되돌리면 그때의 설정과 같다 (arena 결과 cache도 같다)
    paper = ["--test-time", "greedy", "--min-improvement", "0", "--oof-split", "random"]
    params = script.test_time_params(script.parse_args(paper))
    assert params == {"test_time": "greedy", "selection": "greedy", "pool_size": 24}
    args = script.parse_args(["--test-time", "none"])
    assert args.min_improvement == 0.01 and args.oof_split == "task"  # greedy가 아니면 쓰지 않는다
    guarded = ["--test-time", "greedy", "--min-improvement", "0.01", "--min-temperature", "0.5"]
    args = script.parse_args([*guarded, "--oof-split", "task", "--classifier-path", "clf.pt"])
    assert script.test_time_params(args) == {
        "test_time": "greedy",
        "selection": "greedy",
        "pool_size": 24,
        "min_improvement": 0.01,
        "min_temperature": 0.5,
        "oof_split": "task",
    }
    args.out_dir, args.device = str(tmp_path / "guarded"), "cpu"
    make = script.make_estimator_factory("classification", args)
    assert make.fit_groups
    estimator = make(types.SimpleNamespace(categorical_indices=[]), "all")
    assert estimator.min_improvement == 0.01 and min(estimator.temperatures) == 0.5
    estimator.pool_size = 4
    estimator.fit(x, y, groups=np.arange(len(y)) % 5)  # record_fits도 groups를 넘긴다
    record = json.loads((tmp_path / "guarded" / "test_time.jsonl").read_text())
    assert record["selection"]["grouped"] and record["oof_split"] == "task"
    args.min_temperature = 10.0
    with pytest.raises(SystemExit):
        script.make_estimator_factory("classification", args)
    with pytest.raises(SystemExit):
        script.main(["--test-time", "none", "--min-improvement", "0.01"])


def test_arena_factory_passes_groups_only_for_task_out_of_fold(monkeypatch, tmp_path):
    # arena system이 받는 것은 run_arena의 분류·회귀 dispatch 함수다 (fit_groups가 거기 있어야 한다)
    script = _eval_script()
    monkeypatch.setattr(script, "load_model", lambda path, task, cache: (_tiny_model(), None))
    monkeypatch.setattr(script, "arena_suite", lambda name: None)
    monkeypatch.setattr(script, "arena_limits", lambda models, args: {})
    monkeypatch.setattr(script, "print_arena_summary", lambda summary: None)
    seen = {}

    def fake_run(suite, make, **kwargs):
        seen["make"] = make
        return {}

    monkeypatch.setattr(arena, "run", fake_run)
    for oof_split, expected in [("task", True), ("random", False)]:
        args = script.parse_args(
            ["--suite", "beyondarena-lite", "--classifier-path", "clf.pt", "--test-time", "greedy"]
            + ["--oof-split", oof_split, "--out-dir", str(tmp_path), "--device", "cpu"]
        )
        script.run_arena(args, tmp_path, use_mlflow=False)
        assert seen["make"].fit_groups is expected


def test_group_folds_never_leave_an_empty_fold_or_fail_on_small_classes(monkeypatch):
    from minipfn.inference import ensemble_selection

    x, y = _classification(120)
    groups = np.repeat(np.arange(8), 15)
    kw = {"n_estimators": 2, "pool_size": 3, "n_folds": 8}
    clf = GreedyEnsembleClassifier(_tiny_model(), **kw)
    # 층화하다 group이 한 fold에 몰려 검증 행이 없는 fold가 생겨도 그 fold는 버린다
    real = ensemble_selection.StratifiedGroupKFold

    class WithEmptyFold(real):
        def split(self, X, y=None, groups=None):
            yield from super().split(X, y, groups)
            yield np.arange(len(X)), np.array([], dtype=int)

    monkeypatch.setattr(ensemble_selection, "StratifiedGroupKFold", WithEmptyFold)
    splits = clf._group_splits(y, groups, stratified=True)
    assert len(splits) == 8 and all(len(tr) and len(te) for tr, te in splits)
    assert clf.fit(x, y, groups=groups).selection_["n_folds"] == 8
    monkeypatch.setattr(ensemble_selection, "StratifiedGroupKFold", real)
    # 행마다 group, 클래스 행 수 [6, 5] < fold 수 8: 층화 group split이 실패하면 GroupKFold
    x, y = x[:11], np.array([0] * 6 + [1] * 5)
    small = GreedyEnsembleClassifier(_tiny_model(), **kw).fit(x, y, groups=np.arange(11))
    assert small.selection_["grouped"] and small.selection_["n_folds"] == 8


def test_split_groups_reads_string_and_ordered_category_times():
    meta = types.SimpleNamespace(group_on=None, time_on="when")
    strings = pd.DataFrame({"when": ["2020-01-01", "2021-01-01", "2022-01-01", "2023-01-01"]})
    assert list(arena.split_groups(strings, meta, n_intervals=2)) == [0, 0, 1, 1]
    quarters = pd.Categorical(["2020Q2", "2020Q1", "2020Q4", "2020Q3"], ordered=True)
    quarters = quarters.reorder_categories(["2020Q1", "2020Q2", "2020Q3", "2020Q4"])
    t = arena.split_groups(pd.DataFrame({"when": quarters}), meta, n_intervals=2)
    assert list(t) == [0, 0, 1, 1]  # 범주 순서대로
    # 구간을 2개 만들 수 없으면 (상수, 읽을 수 없는 값) 무작위 fold
    assert arena.split_groups(pd.DataFrame({"when": [5, 5, 5]}), meta) is None
    assert arena.split_groups(pd.DataFrame({"when": ["a", "b", "c"]}), meta) is None


def test_members_take_feature_subsets_above_the_cap():
    from minipfn.inference.preprocessing import feature_subset, select_features

    # 상한 안이면 None(전부), 넘으면 멤버 seed로 정한 열 상한 개 (정렬, 중복 없음)
    assert feature_subset(8, None, 0) is None and feature_subset(8, 8, 0) is None
    cols = feature_subset(30, 6, 5)
    assert len(cols) == 6 and np.array_equal(cols, np.unique(cols))
    assert np.array_equal(cols, feature_subset(30, 6, 5))
    assert not np.array_equal(cols, feature_subset(30, 6, 6))
    x, cat = select_features(np.arange(12.0).reshape(2, 6), [1, 4], np.array([0, 4, 5]))
    assert x.tolist() == [[0, 4, 5], [6, 10, 11]] and cat == [1]

    X, y = _classification()
    model = _tiny_model()
    full = EnsembleClassifier(model, n_estimators=3).fit(X[:90], y[:90])
    # 상한이 feature 수 이상이면 예전과 같다
    same = EnsembleClassifier(model, n_estimators=3, max_features_per_member=8).fit(X[:90], y[:90])
    assert np.array_equal(same.predict_proba(X[90:]), full.predict_proba(X[90:]))
    sub = EnsembleClassifier(model, n_estimators=3, max_features_per_member=3).fit(X[:90], y[:90])
    proba = sub.predict_proba(X[90:])
    assert proba.shape == (30, 2) and np.allclose(proba.sum(1), 1)
    assert all(len(c) == 3 for c in sub.feature_subsets_)
    assert len({tuple(c) for c in sub.feature_subsets_}) > 1  # 멤버마다 다르다
    assert not np.allclose(proba, full.predict_proba(X[90:]))
    # greedy: fold와 마지막 fit에서 같은 멤버는 같은 열을 쓴다 (멤버 seed로 정한다)
    greedy = GreedyEnsembleClassifier(
        model, n_estimators=2, pool_size=5, n_folds=2, max_features_per_member=3
    ).fit(X[:90], y[:90])
    for prep, cols in zip(greedy.preprocessors_, greedy.feature_subsets_, strict=True):
        assert np.array_equal(cols, feature_subset(8, 3, prep.member.seed + 4))
    assert greedy.predict_proba(X[90:]).shape == (30, 2)
    # DiagScale의 학습 episode도 같은 상한을 쓴다
    clf = DiagScale(EnsembleClassifier(model, n_estimators=2, max_features_per_member=3), epochs=1)
    assert clf.fit(X[:90], y[:90]).predict_proba(X[90:]).shape == (30, 2)
