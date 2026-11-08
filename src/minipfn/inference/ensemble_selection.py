"""멤버 pool + greedy 선택 (Caruana et al. 2004의 ensemble selection): 추론에 연산을 더 써서 성능을
올리는 집계. "Test-Time Compute for Tabular Foundation Models"(arXiv 2610.12005)가 TabPFN-3에서
기본 예측보다 오차를 줄인다고 보고한 방법이다.

    clf = GreedyEnsembleClassifier(model, n_estimators=4).fit(x_train, y_train)
    proba = clf.predict_proba(x_test)

논문(Appendix C, 공개 코드 `ttc.aggregation`)이 한 것:

- 설정 96개로 된 pool (0번은 기본 설정). 설정마다 학습 행 안에서 stratified 8-fold의
  out-of-fold 예측을 낸다. 학습 행이 8개보다 적은 클래스가 있으면 fold 수를 줄인다
- greedy: 40번 동안, 지금까지 고른 것의 평균에 더했을 때 out-of-fold 손실(분류 log loss, 회귀
  RMSE)이 가장 작은 후보를 고른다. 이미 고른 것도 다시 고를 수 있고, 일찍 멈추지 않으며, 동점이면
  앞 번호를 고른다. 가중치는 고른 횟수 / 40
- 고른 설정은 학습 행 전체로 다시 맞춰 테스트 행을 예측하고, 그 가중치로 섞는다
- 결과 (TabArena 51개, 기본 대비 오차 변화): greedy −2.44%, 가장 좋은 하나 −1.25%, 96개를 균일
  평균하면 +3.68%

여기서 정한 것 (논문의 설정은 TabPFN-3 전처리 옵션이라 그대로 옮길 수 없다):

- 후보 하나는 멤버 하나(forward 한 번)와 softmax 온도 하나의 조합이다. 논문의 설정 하나는 멤버
  8개짜리 앙상블이다. 온도는 logit을 나누기만 하므로 forward를 더 하지 않는다
- pool 멤버 `pool_size`개: 앞의 `n_estimators`개는 기본 앙상블과 같은 멤버(같은 seed)이고,
  나머지는 무작위로 고른다. 전처리 종류(`pool_kinds`, 기본 `POOL_KINDS`), one-hot 여부
  (`one_hot`이 "off"면 펼치지 않는다), 이상값을 자르는 σ(`OUTLIER_SIGMAS`), (회귀) target power
  여부, 멤버 seed(feature 순서, 클래스 번호, 샘플 식별자 열)
- 온도와 σ의 후보는 TabArena의 TabPFN-2.5 search space(`softmax_temperature`,
  `OUTLIER_REMOVAL_STD`)에서 가져왔다. 논문의 pool도 이 search space에서 만들었다
- 후보 0번은 기본 앙상블이다(기본 멤버를 기본 온도로 균일 평균). 논문의 0번 설정과 같은 역할이다
- out-of-fold 예측은 fold마다 그 fold의 학습 행으로 멤버 전처리와 (회귀) target 정규화를 다시
  맞춘다. 회귀 RMSE는 원래 단위다
- 학습 행 전체로 다시 맞출 때는 가중치가 있는 멤버만 맞춘다 (테스트 시간을 줄인다)
- `selection`: `"greedy"`(논문), `"best"`(out-of-fold 손실이 가장 작은 후보 하나), `"uniform"`(0번을
  뺀 후보 모두 같은 가중치. 논문의 균일 평균에 해당한다)

논문에 없는 것 (issue #52, 기본은 모두 꺼져 있어 논문 방식 그대로다). 논문 방식은 out-of-fold
손실이 조금이라도 낮으면 고른다. 그래서 out-of-fold 손실이 기본부터 거의 0인 데이터셋에서 아주
날카로운 온도 하나를 골라 test에서 지나치게 확신하거나, 무작위 fold가 보지 못하는 grouped·temporal
test의 분포 변화에서 손해를 봤다(BeyondArena-lite):

- `min_improvement`: 고른 조합의 out-of-fold 손실이 기본(후보 0번)보다 이 비율 이상 낮지 않으면
  기본 앙상블을 그대로 쓴다. 분류는 log loss가 `MIN_IMPROVEMENT_NATS` 이상 낮아야 한다는 조건도
  함께 둔다(기본 손실이 거의 0이면 비율로는 아주 작은 차이도 크게 보인다)
- `min_temperature`: 이보다 낮은 온도를 후보에서 뺀다
- `fit(X, y, groups=...)`: 행마다의 group 번호를 주면 out-of-fold를 group 단위로 나눈다(한 group은
  한 fold에만 들어간다. 분류는 `StratifiedGroupKFold`, 나눌 수 없으면 `GroupKFold`, 회귀는
  `GroupKFold`). 빈 fold는 버리고, 남은 fold가 2개보다 적으면 무작위 fold를 쓴다. TabArena·
  BeyondArena의 grouped split은 group 열, temporal split은 시간 구간을 group으로 준다
  (`minipfn.benchmarks.arena.split_groups`). temporal split도 구간 단위 k-fold라 미래 구간으로
  과거를 맞추는 fold가 있다(앞으로만 나누는 split이 아니다. TabArena에서 AutoGluon이 bagging fold를
  만드는 방식과 같다). `DiagScale`로 감싸면 group은 greedy 선택에만 쓰이고 DiagScale의 검증 split은
  무작위 그대로다
"""

from __future__ import annotations

import copy
import warnings
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold, StratifiedKFold

from minipfn.inference.ensemble import EnsembleClassifier, EnsembleRegressor
from minipfn.inference.preprocessing import MEMBER_KINDS, Member

POOL_SIZE = 24
# pool 멤버의 전처리 종류. TabArena-Lite·BeyondArena-lite 판정(PR #49)은 이 세 종류로 했다
# ("standard"는 그 뒤에 더한 종류라 넣지 않았다. 넣으면 pool의 무작위 멤버가 모두 바뀐다)
POOL_KINDS = ("quantile", "power", "robust")
N_FOLDS = 8
N_ROUNDS = 40  # TabArena·AutoGluon ensemble selection의 기본값 (논문도 같다)
SELECTIONS = ("greedy", "best", "uniform")
# TabArena TabPFN-2.5 search space의 softmax_temperature와 OUTLIER_REMOVAL_STD
TEMPERATURES = (0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.5)
OUTLIER_SIGMAS = (3.0, 6.0, 12.0)
# min_improvement를 줄 때 분류 log loss가 함께 낮아져야 하는 최소 차이 (nats)
MIN_IMPROVEMENT_NATS = 1e-3


def greedy_weights(preds: np.ndarray, loss, n_rounds: int = N_ROUNDS) -> np.ndarray:
    """Caruana의 greedy 선택(중복 허용). `preds` (후보 수, n)는 후보마다의 예측, `loss`는
    (b, n) 예측 묶음 → (b,) 손실이다. 가중치는 고른 횟수 / `n_rounds`. 동점과 손실이 모두 NaN이면
    앞 번호를 고른다."""
    total = np.zeros(preds.shape[1])
    counts = np.zeros(len(preds))
    for t in range(n_rounds):
        scores = loss((total + preds) / (t + 1))
        best = int(np.nanargmin(scores)) if np.isfinite(scores).any() else 0
        total += preds[best]
        counts[best] += 1
    return counts / n_rounds


def _log_loss(true_class_probs: np.ndarray) -> np.ndarray:
    # 확률의 합이 1인 후보의 평균은 합이 1이라, 정답 클래스의 확률만으로 log loss가 정해진다
    return -np.log(np.clip(true_class_probs, 1e-15, 1.0)).mean(axis=1)


def check_selection(
    *,
    n_estimators: int,
    pool_size: int = POOL_SIZE,
    selection: str = "greedy",
    pool_kinds: tuple[str, ...] = POOL_KINDS,
    temperatures: tuple[float, ...] = TEMPERATURES,
    min_improvement: float = 0.0,
    min_temperature: float | None = None,
) -> tuple[float, ...]:
    """선택 설정을 확인하고 pool의 온도(`min_temperature` 이상)를 돌려준다. 맞지 않으면 ValueError.
    estimator를 만들기 전에 설정만 확인할 때도 쓴다 (`scripts/tabpfn_v2/eval.py`)."""
    if selection not in SELECTIONS:
        raise ValueError(f"selection은 {SELECTIONS} 가운데 하나여야 한다: {selection}")
    if pool_size < n_estimators:
        raise ValueError(f"pool_size({pool_size})는 n_estimators({n_estimators}) 이상이어야 한다")
    if set(pool_kinds) - set(MEMBER_KINDS):
        raise ValueError(f"pool_kinds는 {MEMBER_KINDS}에서 고른다: {pool_kinds}")
    if not 0.0 <= min_improvement < 1.0:
        raise ValueError(f"min_improvement는 0 이상 1 미만이어야 한다: {min_improvement}")
    if min_temperature is not None:
        temperatures = tuple(t for t in temperatures if t >= min_temperature)
        if not temperatures:
            raise ValueError(f"min_temperature({min_temperature}) 이상인 온도가 없다")
    return tuple(temperatures)


class _Selection:
    """pool·fold·선택 설정과 공통 계산. `EnsembleClassifier`·`EnsembleRegressor`와 함께 상속한다."""

    def __init__(
        self,
        *args: Any,
        pool_size: int = POOL_SIZE,
        pool_kinds: tuple[str, ...] = POOL_KINDS,
        temperatures: tuple[float, ...] = TEMPERATURES,
        n_folds: int = N_FOLDS,
        n_rounds: int = N_ROUNDS,
        selection: str = "greedy",
        min_improvement: float = 0.0,
        min_temperature: float | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.temperatures = check_selection(
            n_estimators=self.n_estimators,
            pool_size=pool_size,
            selection=selection,
            pool_kinds=pool_kinds,
            temperatures=temperatures,
            min_improvement=min_improvement,
            min_temperature=min_temperature,
        )
        self.pool_size = pool_size
        self.pool_kinds = tuple(pool_kinds)
        self.min_improvement = min_improvement
        self.min_temperature = min_temperature
        self.n_folds = n_folds
        self.n_rounds = n_rounds
        self.selection = selection

    def _pool(self, rng: np.random.Generator, defaults: list[Member], n_classes: int | None):
        # 기본 멤버 다음에 무작위 멤버. rng는 기본 멤버를 뽑은 뒤의 것을 이어 쓴다
        members = list(defaults)
        for _ in range(self.pool_size - len(defaults)):
            members.append(
                Member(
                    kind=self.pool_kinds[int(rng.integers(len(self.pool_kinds)))],
                    class_permutation=None if n_classes is None else rng.permutation(n_classes),
                    target_power=n_classes is None and bool(rng.integers(2)),
                    seed=int(rng.integers(0, 2**31 - 1)),
                    # one_hot="off"면 펼치지 않는다 (뽑기는 그대로 해서 나머지 멤버가 같다)
                    one_hot=bool(rng.integers(2)) and self.one_hot != "off",
                    outlier_sigma=OUTLIER_SIGMAS[int(rng.integers(len(OUTLIER_SIGMAS)))],
                )
            )
        return members

    def _group_splits(self, y: np.ndarray, groups: Any, stratified: bool) -> list | None:
        """group 단위 out-of-fold (학습 행, 검증 행) 목록, 또는 group으로 나눌 수 없을 때 None
        (무작위 fold를 쓴다). 빈 fold는 버린다. 쓴 것은 `grouped_`에 남긴다."""
        self.grouped_ = False
        if groups is None:
            return None
        groups = pd.factorize(pd.Series(np.asarray(groups, dtype=object)), use_na_sentinel=False)[0]
        if len(groups) != len(y):
            raise ValueError(f"groups의 길이({len(groups)})가 학습 행 수({len(y)})와 다르다")
        n = int(min(self.n_folds, len(np.unique(groups))))
        if n < 2:
            return None
        rows = np.zeros(len(y))
        splits = None
        if stratified:
            splitter = StratifiedGroupKFold(n, shuffle=True, random_state=self.random_state)
            try:
                with warnings.catch_warnings():
                    # 학습 행이 fold 수보다 적은 클래스: 그 클래스가 없는 fold가 생겨도 그대로 쓴다
                    warnings.simplefilter("ignore", UserWarning)
                    splits = list(splitter.split(rows, y, groups))
            except ValueError:  # 모든 클래스의 행이 fold 수보다 적다
                splits = None
        if splits is None:
            splits = list(GroupKFold(n).split(rows, y, groups))  # group 크기로 나눈다
        # 층화하다 group이 한 fold에 몰리면 검증 행이 없는 fold가 생긴다
        splits = [(tr, te) for tr, te in splits if len(tr) and len(te)]
        if len(splits) < 2:
            return None
        self.grouped_ = True
        return splits

    def _out_of_fold(self, X: Any, y: np.ndarray, members: list[Member], splits) -> np.ndarray:
        """후보마다 out-of-fold 예측 (후보 수, n). 후보 0번은 기본 앙상블, 그다음은 멤버 × 온도
        (멤버 p, 온도 t → 1 + p × 온도 수 + t)."""
        X = pd.DataFrame(X).reset_index(drop=True)
        n_t = len(self.temperatures)
        oof = np.zeros((1 + len(members) * n_t, len(X)))
        for train_ix, test_ix in splits:
            fold = copy.copy(self)
            fold.fit_members(X.iloc[train_ix], y[train_ix], members)
            for p, preds in fold._member_predictions(X.iloc[test_ix], y[test_ix]):
                oof[1 + p * n_t : 1 + (p + 1) * n_t, test_ix] = preds[:n_t]
                if p < self.n_estimators:  # 기본 앙상블: 기본 멤버를 기본 온도로 균일 평균
                    oof[0, test_ix] += preds[n_t] / self.n_estimators
        return oof

    _min_improvement_abs = 0.0  # 분류는 MIN_IMPROVEMENT_NATS

    def _guard(self, weights: np.ndarray, oof: np.ndarray, loss) -> tuple[np.ndarray, bool]:
        """고른 조합의 out-of-fold 손실이 기본보다 충분히 낮지 않으면 기본 앙상블(후보 0번)만 쓴다
        (DiagScale의 "나아진 epoch가 없으면 학습 전 그대로"와 같다). (가중치, 되돌렸는지)."""
        if self.min_improvement <= 0 or weights[0] == 1.0:
            return weights, False
        default, selected = loss(np.stack([oof[0], weights @ oof]))
        need = max(self.min_improvement * default, self._min_improvement_abs)
        if default - selected >= need:
            return weights, False
        return np.eye(len(oof))[0], True  # 손실이 NaN이어도 기본으로

    def _select(self, oof: np.ndarray, loss) -> np.ndarray:
        if self.selection == "greedy":
            return greedy_weights(oof, loss, self.n_rounds)
        weights = np.zeros(len(oof))
        if self.selection == "best":
            scores = loss(oof)
            weights[int(np.nanargmin(scores)) if np.isfinite(scores).any() else 0] = 1.0
        else:  # uniform: 0번(기본 앙상블)을 뺀 후보 모두
            weights[1:] = 1.0 / (len(oof) - 1)
        return weights

    def _fit_selected(self, X: Any, y: np.ndarray, members: list[Member], oof, loss):
        """가중치를 고르고, 가중치가 있는 멤버만 학습 행 전체로 맞춘다."""
        weights, reverted = self._guard(self._select(oof, loss), oof, loss)
        n_t = len(self.temperatures)
        # 멤버마다 {온도: 가중치}. 0번의 가중치는 기본 멤버에 기본 온도로 나눠 준다
        per_member: list[dict[float, float]] = [{} for _ in members]
        for c in np.flatnonzero(weights):
            if c == 0:
                for p in range(self.n_estimators):
                    t = self.softmax_temperature
                    per_member[p][t] = per_member[p].get(t, 0.0) + weights[0] / self.n_estimators
            else:
                p, t = divmod(c - 1, n_t)
                t = self.temperatures[t]
                per_member[p][t] = per_member[p].get(t, 0.0) + weights[c]
        used = [p for p, w in enumerate(per_member) if w]
        self.selection_ = {
            "n_folds": self.n_folds_,
            "candidates": len(oof),
            "pool_members": len(members),
            "used_members": len(used),
            "oof_loss_default": float(loss(oof[:1])[0]),
            "oof_loss_selected": float(loss((weights @ oof)[None])[0]),
            "weight_default": float(weights[0]),
            "reverted": reverted,
            "grouped": self.grouped_,
        }
        self.member_weights_ = [per_member[p] for p in used]
        return self.fit_members(X, y, [members[p] for p in used])


class GreedyEnsembleClassifier(_Selection, EnsembleClassifier):
    """분류: 멤버 × 온도 후보를 out-of-fold log loss로 greedy하게 골라 섞는다 (모듈 docstring).

    `n_estimators`는 기본 앙상블의 멤버 수다(후보 0번과 pool의 앞 멤버)."""

    _min_improvement_abs = MIN_IMPROVEMENT_NATS

    def fit(self, X: Any, y: Any, groups: Any = None) -> GreedyEnsembleClassifier:
        """`groups`(행마다의 group 번호)를 주면 out-of-fold를 group 단위로 나눈다."""
        rng = np.random.default_rng(self.random_state)
        y = self._fit_labels(y)
        n_classes = len(self.classes_)
        members = self._pool(rng, self._default_members(rng, n_classes), n_classes)
        splits = self._group_splits(y, groups, stratified=True)
        if splits is None:
            counts = np.bincount(y, minlength=n_classes)
            n_folds = int(np.clip(counts[counts > 0].min(), 2, self.n_folds))
            splitter = StratifiedKFold(n_folds, shuffle=True, random_state=self.random_state)
            with warnings.catch_warnings():
                # 학습 행이 fold 수보다 적은 클래스: 그 클래스가 없는 fold가 생겨도 그대로 쓴다
                warnings.simplefilter("ignore", UserWarning)
                splits = list(splitter.split(np.zeros(len(y)), y))
        self.n_folds_ = len(splits)
        oof = self._out_of_fold(X, y, members, splits)
        return self._fit_selected(X, y, members, oof, _log_loss)

    def _member_predictions(self, X: Any, y: np.ndarray):
        """멤버마다 (번호, 정답 클래스 확률 (온도 수 + 1, n)). 마지막 행은 기본 온도."""
        temps = (*self.temperatures, self.softmax_temperature)
        for i, logits in self.member_logits(X):
            rows = torch.arange(len(y), device=logits.device)
            target = torch.as_tensor(y, device=logits.device)
            probs = [torch.softmax(logits / t, dim=-1)[rows, target] for t in temps]
            yield i, torch.stack(probs).double().cpu().numpy()

    def predict_proba(self, X: Any) -> np.ndarray:
        total = 0.0
        for i, logits in self.member_logits(X):
            for t, w in self.member_weights_[i].items():
                total = total + w * torch.softmax(logits / t, dim=-1)
        return total.cpu().numpy()


class GreedyEnsembleRegressor(_Selection, EnsembleRegressor):
    """회귀: 멤버 × 온도 후보를 out-of-fold RMSE로 greedy하게 골라 섞는다 (모듈 docstring).
    평균과 예측 분포 모두 같은 가중치로 섞는다."""

    def fit(self, X: Any, y: Any, groups: Any = None) -> GreedyEnsembleRegressor:
        """`groups`(행마다의 group 번호)를 주면 out-of-fold를 group 단위로 나눈다."""
        rng = np.random.default_rng(self.random_state)
        y = np.asarray(y, dtype=np.float64)
        members = self._pool(rng, self._default_members(rng), None)
        splits = self._group_splits(y, groups, stratified=False)
        if splits is None:
            splitter = KFold(
                int(min(self.n_folds, len(y))), shuffle=True, random_state=self.random_state
            )
            splits = list(splitter.split(np.zeros(len(y))))
        self.n_folds_ = len(splits)
        oof = self._out_of_fold(X, y, members, splits)

        def rmse(preds: np.ndarray) -> np.ndarray:
            return np.sqrt(np.mean((preds - y) ** 2, axis=-1))

        return self._fit_selected(X, y, members, oof, rmse)

    def _member_predictions(self, X: Any, y: np.ndarray):
        """멤버마다 (번호, 원래 단위의 평균 (온도 수 + 1, n)). 마지막 행은 기본 온도."""
        temps = (*self.temperatures, self.softmax_temperature)
        for i, logits in self._iter_logits(X, self.y_trains_):
            means = [self._member_mean(i, torch.softmax(logits / t, dim=-1)) for t in temps]
            means = torch.stack(means).double().cpu().numpy()
            yield i, means * self.y_std_ + self.y_mean_

    def _weighted(self, X: Any, mean: bool, distribution: bool):
        # 가중치를 준 (z 평균, 공통 경계 위의 구간 확률)
        total_mean, total_probs = 0.0, 0.0
        for i, logits in self._iter_logits(X, self.y_trains_):
            for t, w in self.member_weights_[i].items():
                p = torch.softmax(logits / t, dim=-1)
                if mean:
                    total_mean = total_mean + w * self._member_mean(i, p)
                if distribution:
                    total_probs = total_probs + w * self._on_common_borders(i, p)
        return total_mean, total_probs

    def predict(self, X: Any) -> np.ndarray:
        # 가중합 하나를 `_mean`·`_mixture`(목록의 평균)에 넘겨 원래 단위·log 확률로 바꾼다
        return self._mean([self._weighted(X, mean=True, distribution=False)[0]])

    def predict_distribution(self, X: Any) -> tuple[torch.Tensor, torch.Tensor]:
        return self._mixture([self._weighted(X, mean=False, distribution=True)[1]])

    def predict_full(self, X: Any) -> tuple[np.ndarray, torch.Tensor, torch.Tensor]:
        total_mean, total_probs = self._weighted(X, mean=True, distribution=True)
        return (self._mean([total_mean]), *self._mixture([total_probs]))
