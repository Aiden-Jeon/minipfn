"""DiagScale: 데이터셋마다 attention의 query·key 유사도에 곱하는 대각 scale만 학습하는 test-time
적응 ("Test-Time Compute for Tabular Foundation Models", arXiv 2610.12005 "Adaptation"과
Appendix B). 나머지 가중치는 그대로 둔다.

    clf = DiagScale(TabPFNv2Classifier(model)).fit(x_train, y_train)
    proba = clf.predict_proba(x_test)

`DiagScale`은 앙상블 estimator(`minipfn.inference.ensemble`의 클래스, greedy 선택
`minipfn.inference.ensemble_selection` 포함) 하나를 감싼다. fit에서 모델 사본에 scale을 학습하고,
그 사본으로 감싼 estimator를 학습 행 전체에 맞춘다.

논문(과 공개 코드 `ttc.backbones.tabicl_finetune`)이 한 것:

- attention마다 0으로 시작하는 벡터 δ(head × head 차원)를 두고 logit을 qᵀ(I + diag(δ))k로 바꾼다.
  δ = 0이면 원래 모델과 같다. δ만 학습한다
- AdamW, lr 3e-2(TabPFN-3·TabICL v2에 공통으로 고른 값), weight decay 0.01, 전체 step의 10%
  linear warmup 뒤 cosine으로 0, gradient norm 1로 자른다
- 최대 30 epoch, patience 8. 학습 행에서 검증 행(10%, 적어도 min(200, 25%))을 떼고 epoch마다 검증
  손실(분류 log loss, 회귀 RMSE)로 epoch를 고른다. 어떤 epoch도 학습 전보다 낫지 않으면 학습 전
  그대로 둔다
- episode: 나머지 행에서 최대 10,000행을 뽑아 20%를 query, 나머지를 context로 쓴다. epoch마다
  step 수는 max(1, min(10, round(행 수 / 뽑은 행 수)))
- 학습·검증은 멤버 2개, 배포는 기본 멤버 수
- 결과 (TabArena Elo): TabPFN-3 +26, TabICL v2 +82, TabFM +20

여기서 정한 것:

- δ는 qkv Linear 출력의 query 부분에 곱한다(forward hook). `qkv`(query·key·value 순서이고 그 안은
  head마다 연속인 `nn.Linear`)와 `nhead`가 있는 모듈을 attention으로 본다
  (`minipfn.models.cell_transformer.Attention`). 곱셈은 float32로 하고 결과를 계산 dtype으로 돌린다.
  학습은 모델 사본에 하므로 원래 모델은 바뀌지 않는다
- 회귀 손실은 모델이 학습한 bar distribution NLL이다 (TabICL 회귀는 quantile head라 pinball)
- episode 하나의 셀 수(행 × 전처리 뒤 열 수)를 `max_cells`로 줄인다(GPU 메모리. 기본값은
  `recipes.tabpfn_v2.long_tables`가 학습한 표의 셀 상한)
- episode 멤버의 전처리는 context 행으로 맞춘다(추론과 같다). 멤버 2개 이하라 회귀 멤버는 target
  power를 쓰지 않는다(`EnsembleRegressor`는 앞의 max(2, 종류 수)개에 켜지 않는다)
- query에 context에 없는 클래스가 있으면 그 episode는 건너뛴다 (논문 코드와 같다)
"""

from __future__ import annotations

import copy
import functools
import math
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch import nn

from minipfn.inference.ensemble import EnsembleClassifier, EnsembleRegressor, _Base
from minipfn.models.bar_distribution import FullSupportBarDistribution

EPOCHS = 30
LEARNING_RATE = 3e-2
WEIGHT_DECAY = 0.01
PATIENCE = 8
MIN_IMPROVEMENT = 1e-4  # 검증 손실이 이만큼 줄어야 나아진 것으로 본다 (논문 코드)
MAX_ROWS = 10_000
MAX_CELLS = 400_000
MIN_ROWS = 256  # 넓은 표에서도 episode 행은 이만큼 둔다
QUERY_FRACTION = 0.2
N_VIEWS = 2
GRAD_CLIP = 1.0


def _scale_query(attention: nn.Module, linear: nn.Module, inputs, output: torch.Tensor):
    delta = attention.diag_scale
    e = delta.numel()
    q = (output[..., :e].float() * (1.0 + delta)).to(output.dtype)
    return torch.cat([q, output[..., e:]], dim=-1)


def add_diag_scale(model: nn.Module) -> list[nn.Parameter]:
    """attention마다 δ(0으로 시작, head × head 차원 = E개)를 붙이고 그 목록을 돌려준다. `model`을
    바꾼다(사본에 쓴다). δ는 attention 모듈의 parameter `diag_scale`이다."""
    deltas = []
    for module in model.modules():
        qkv = getattr(module, "qkv", None)
        if not (isinstance(qkv, nn.Linear) and hasattr(module, "nhead")):
            continue
        if hasattr(module, "diag_scale"):
            raise ValueError("DiagScale을 이미 붙인 모델이다")
        delta = nn.Parameter(torch.zeros(qkv.out_features // 3, device=qkv.weight.device))
        module.register_parameter("diag_scale", delta)
        qkv.register_forward_hook(functools.partial(_scale_query, module))
        deltas.append(delta)
    if not deltas:
        raise ValueError("qkv와 nhead가 있는 attention 모듈이 없다")
    return deltas


def _views(estimator: _Base, model: nn.Module, n_views: int, random_state: int) -> _Base:
    # 감싼 estimator와 같은 추론 설정에 멤버 수만 다른 기본 앙상블 (학습 episode, 검증)
    kwargs = {
        "n_estimators": n_views,
        "categorical_features_indices": estimator.categorical_features_indices,
        "one_hot": estimator.one_hot,
        "member_kinds": estimator.member_kinds,
        "softmax_temperature": estimator.softmax_temperature,
        "max_one_hot_width": estimator.max_one_hot_width,
        "device": estimator.device,
        "random_state": random_state,
    }
    if isinstance(estimator, EnsembleRegressor):
        return EnsembleRegressor(model, estimator.borders, **kwargs)
    return EnsembleClassifier(model, **kwargs)


def _log_loss(classes: np.ndarray, proba: np.ndarray, y: np.ndarray) -> float:
    # 학습 행에 없던 클래스의 확률은 0으로 본다 (sklearn log_loss는 labels가 맞지 않으면 실패한다)
    ix = np.clip(np.searchsorted(classes, y), 0, len(classes) - 1)
    p = np.where(classes[ix] == y, proba[np.arange(len(y)), ix], 0.0)
    return float(-np.log(np.clip(p, 1e-15, 1.0)).mean())


def _validation_loss(views: _Base, X: pd.DataFrame, y: np.ndarray) -> float:
    if isinstance(views, EnsembleRegressor):
        return float(np.sqrt(np.mean((views.predict(X) - y) ** 2)))
    return _log_loss(views.classes_, views.predict_proba(X), y)


def _episode_backward(views: _Base, model: nn.Module, X_query, y_query, bar) -> float | None:
    """context 행에 맞춘 `views`로 query 행의 손실을 내고 역전파한다(멤버 평균). 손실을 돌려준다.
    query에 context에 없는 클래스가 있으면 None."""
    regression = isinstance(views, EnsembleRegressor)
    if regression:
        target = (y_query - views.y_mean_) / views.y_std_
    else:
        if not np.isin(y_query, views.classes_).all():
            return None
        target = views.label_encoder_.transform(y_query)
    autocast = views.device.type == "cuda"  # 추론과 같은 bfloat16
    target = torch.as_tensor(target, device=views.device)
    total = 0.0
    n = len(views.preprocessors_)
    for i, x, y_train in views._iter_inputs(X_query, views.y_trains_):
        with torch.autocast(views.device.type, dtype=torch.bfloat16, enabled=autocast):
            logits = model(x, y_train)[0].float()
        if regression:
            loss = bar(logits, target.float()).mean()
        else:
            perm = views.preprocessors_[i].member.class_permutation
            logits = logits[:, : len(views.classes_)][:, perm]
            loss = F.cross_entropy(logits, target.long())
        (loss / n).backward()
        total += float(loss.detach()) / n
    return total


def fit_diag_scale(
    estimator: _Base,
    X: Any,
    y: Any,
    *,
    epochs: int = EPOCHS,
    lr: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    patience: int = PATIENCE,
    max_rows: int = MAX_ROWS,
    max_cells: int = MAX_CELLS,
    query_fraction: float = QUERY_FRACTION,
    n_views: int = N_VIEWS,
    grad_clip: float = GRAD_CLIP,
) -> tuple[nn.Module, dict]:
    """`estimator`의 모델 사본에 DiagScale을 붙여 (`X`, `y`)로 학습한다. (모델 사본, 학습 기록)."""
    start = time.perf_counter()
    seed = estimator.random_state
    regression = isinstance(estimator, EnsembleRegressor)
    model = copy.deepcopy(estimator.model).to(estimator.device).eval()
    model.requires_grad_(False)
    deltas = add_diag_scale(model)
    bar = FullSupportBarDistribution(estimator.borders.to(estimator.device)) if regression else None

    X = pd.DataFrame(X).reset_index(drop=True)
    y = np.asarray(y, dtype=np.float64) if regression else np.asarray(y)
    n_val = int(max(round(0.1 * len(y)), min(200, round(0.25 * len(y)))))
    try:
        fit_ix, val_ix = train_test_split(
            np.arange(len(y)),
            test_size=n_val,
            random_state=seed,
            stratify=None if regression else y,
        )
    except ValueError:  # 행이 아주 적은 클래스: 층화 없이
        fit_ix, val_ix = train_test_split(np.arange(len(y)), test_size=n_val, random_state=seed)
    X_fit, y_fit = X.iloc[fit_ix].reset_index(drop=True), y[fit_ix]
    X_val, y_val = X.iloc[val_ix], y[val_ix]
    val_views = _views(estimator, model, n_views, seed).fit(X_fit, y_fit)

    width = max(x.shape[1] for x in val_views.X_trains_)
    take = min(len(y_fit), max_rows, max(MIN_ROWS, max_cells // width))
    steps = max(1, min(10, round(len(y_fit) / take)))
    total_steps = steps * epochs
    warmup = max(1, int(0.1 * total_steps))

    def lr_factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    optimizer = torch.optim.AdamW(deltas, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    before = best = _validation_loss(val_views, X_val, y_val)
    best_state = [d.detach().clone() for d in deltas]
    best_epoch, bad_epochs, curve, losses = -1, 0, [before], []
    for epoch in range(epochs):
        rng = np.random.default_rng(seed * 100_003 + epoch)
        for _ in range(steps):
            idx = rng.permutation(len(y_fit))[:take]
            n_query = int(np.clip(round(len(idx) * query_fraction), 1, max(1, len(idx) - 8)))
            query, context = idx[:n_query], idx[n_query:]
            views = _views(estimator, model, n_views, int(rng.integers(0, 2**31 - 1)))
            views.fit(X_fit.iloc[context], y_fit[context])
            optimizer.zero_grad(set_to_none=True)
            loss = _episode_backward(views, model, X_fit.iloc[query], y_fit[query], bar)
            if loss is None:
                continue
            torch.nn.utils.clip_grad_norm_(deltas, grad_clip)
            optimizer.step()
            scheduler.step()
            losses.append(loss)
        score = _validation_loss(val_views, X_val, y_val)
        curve.append(score)
        if score < best - MIN_IMPROVEMENT:
            best, best_epoch, bad_epochs = score, epoch, 0
            best_state = [d.detach().clone() for d in deltas]
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break
    with torch.no_grad():  # 가장 좋은 epoch (없으면 학습 전, δ = 0)
        for d, s in zip(deltas, best_state, strict=True):
            d.copy_(s)
    info = {
        "n_params": int(sum(d.numel() for d in deltas)),
        "val_rows": len(val_ix),
        "episode_rows": int(take),
        "steps_per_epoch": int(steps),
        "epochs_run": len(curve) - 1,
        "best_epoch": best_epoch,  # -1: 학습 전 그대로
        "val_loss_before": before,
        "val_loss_best": best,
        "val_curve": curve,
        "train_loss": losses,
        "delta_norm": float(torch.cat([d.detach().flatten() for d in deltas]).norm()),
        "seconds": time.perf_counter() - start,
    }
    return model, info


class DiagScale:
    """앙상블 estimator를 감싸 fit에서 DiagScale을 학습한다 (모듈 docstring). 인자는
    `fit_diag_scale`의 것이다. 맞춘 뒤에는 감싼 estimator의 사본(`estimator_`, 모델만 학습한
    사본)으로 예측하고, 그 속성(`classes_`, `predict_full` 등)을 그대로 쓴다. 학습 기록은
    `adaptation_`이다."""

    def __init__(self, estimator: _Base, **kwargs: Any) -> None:
        self.estimator = estimator
        self.kwargs = kwargs

    def fit(self, X: Any, y: Any) -> DiagScale:
        model, self.adaptation_ = fit_diag_scale(self.estimator, X, y, **self.kwargs)
        estimator = copy.copy(self.estimator)
        estimator.model = model
        self.estimator_ = estimator.fit(X, y)
        return self

    def __getattr__(self, name: str):
        # predict·predict_proba·classes_ 등은 맞춘 estimator의 것 (fit 전에는 없다)
        estimator = self.__dict__.get("estimator_")
        if estimator is None:
            raise AttributeError(name)
        return getattr(estimator, name)
