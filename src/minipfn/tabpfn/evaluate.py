"""OpenML 평가: 논문 Table 1(OpenML-CC18 18개 데이터셋) ROC AUC 재현.

v1 코드의 평가 절차를 따른다(github.com/automl/TabPFN `tabpfn_v1`, Apache 2.0).
- 데이터 로드(`datasets/__init__.py`): seed 13으로 행을 섞고 최대 10,000행, feature는 앞 100개
- split(`scripts/tabular_evaluation.py` evaluate, generate_valid_split): 2,000행 미만이면 50/50,
  split 5개, 학습·테스트 행이 같은 클래스 집합을 갖는 구간을 고른다
- 예측(`scripts/transformer_prediction_interface.py` transformer_predict): 학습 행 기준 표준화,
  상수 feature 제거, power transform 유무, 이상치 누르기, feature 수 보정,
  클래스·feature 회전 앙상블, logits / 0.8을 평균한 뒤 softmax
- 지표(`scripts/tabular_metrics.py`): 다중 클래스 ROC AUC는 OVO
"""

from __future__ import annotations

import itertools
import random
import warnings
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import PowerTransformer

from minipfn.tabpfn import spec
from minipfn.tabpfn.prior import normalize_data, remove_outliers


@dataclass
class OpenMLDataset:
    did: int
    name: str
    x: np.ndarray  # (n, F) float32, 범주형은 정수 코드, 결측값은 nan
    y: np.ndarray  # (n,) int
    categorical: list[int]

    @property
    def is_table1(self) -> bool:
        """논문 Table 1의 조건: 수치형 feature만 있고 결측값이 없다."""
        return not self.categorical and not np.isnan(self.x).any()


def load_openml_dataset(did: int, max_samples: int = spec.EVAL_MAX_SAMPLES) -> OpenMLDataset:
    import openml

    ds = openml.datasets.get_dataset(
        did, download_data=True, download_qualities=False, download_features_meta_data=True
    )
    x_df, y_ser, cat_mask, _ = ds.get_data(target=ds.default_target_attribute)
    categorical = [i for i, c in enumerate(cat_mask) if c]
    cols = []
    for i, col in enumerate(x_df.columns):
        s = x_df[col]
        if i in categorical or not np.issubdtype(s.dtype, np.number):
            codes = s.astype("category").cat.codes.to_numpy().astype(np.float32)
            codes[codes < 0] = np.nan
            cols.append(codes)
        else:
            cols.append(s.to_numpy(dtype=np.float32))
    x = np.stack(cols, axis=1)
    y = np.unique(y_ser.astype(str).to_numpy(), return_inverse=True)[1].astype(np.int64)

    order = np.arange(len(y))
    np.random.RandomState(13).shuffle(order)  # v1: np.random.seed(13); np.random.shuffle(order)
    x, y = x[order][:max_samples], y[order][:max_samples]
    if x.shape[1] > spec.MAX_FEATURES:  # 앞 100개 feature만 쓴다 (논문 부록 F)
        x = x[:, : spec.MAX_FEATURES]
        categorical = [c for c in categorical if c < spec.MAX_FEATURES]
    return OpenMLDataset(did=did, name=ds.name, x=x, y=y, categorical=categorical)


def generate_split(
    x: np.ndarray, y: np.ndarray, split_number: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """v1 evaluate + generate_valid_split. 반환: x_train, y_train, x_test, y_test (없으면 None)."""
    seq_len = min(len(x), spec.EVAL_SEQ_LEN)
    eval_pos = spec.EVAL_TRAIN_SIZE
    if 2 * eval_pos > seq_len:
        eval_pos = int(seq_len * 0.5)
    seq_len = 2 * eval_pos

    gen = torch.Generator().manual_seed(split_number)
    perm = torch.randperm(len(x), generator=gen).numpy() if split_number > 1 else np.arange(len(x))
    x, y = x[perm], y[perm]
    all_classes = np.unique(y)
    for seed in range(13, 21):
        start = random.Random(seed).randint(0, len(x) - seq_len) if len(x) - seq_len > 0 else 0
        yw = y[start : start + seq_len]
        tr, te = np.unique(yw[:eval_pos]), np.unique(yw[eval_pos:])
        if np.array_equal(np.unique(yw), all_classes) and np.array_equal(tr, te):
            xw = x[start : start + seq_len]
            _, yw = np.unique(yw, return_inverse=True)  # 0..C-1로 다시 매긴다
            return xw[:eval_pos], yw[:eval_pos], xw[eval_pos:], yw[eval_pos:]
    return None


def _preprocess(
    x_all: torch.Tensor, n_train: int, transform: str, max_features: int
) -> torch.Tensor:
    """v1 transformer_predict.preprocess_input. x_all: (S, F) → (S, F')."""
    x = normalize_data(x_all.unsqueeze(1), n_rows=n_train)[:, 0]
    keep = [len(torch.unique(x[:n_train, c])) > 1 for c in range(x.shape[1])]
    x = x[:, keep]
    if transform == "power_all":
        x = x.numpy().copy()
        for c in range(x.shape[1]):
            # v1은 경고도 예외로 바꿔서, 변환이 매끄럽지 않은 열은 그대로 둔다.
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                try:
                    pt = PowerTransformer(standardize=True).fit(x[:n_train, c : c + 1])
                    x[:, c : c + 1] = pt.transform(x[:, c : c + 1])
                except Exception:
                    pass
        x = torch.tensor(x).float()
    x = remove_outliers(x.unsqueeze(1), n_rows=n_train)[:, 0]
    return x / (x.shape[-1] / max_features)


@torch.no_grad()
def tabpfn_predict_proba(
    model: torch.nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    n_ensemble: int = spec.N_ENSEMBLE,
    seed: int = 0,
    batch_size: int = 16,
    device: torch.device | str = "cpu",
) -> np.ndarray:
    """테스트 행의 클래스 확률 (m, C). v1 TabPFNClassifier.predict_proba와 같은 앙상블."""
    model.eval()
    n_train = len(x_train)
    num_classes = int(y_train.max()) + 1
    max_features = model.num_features
    x_all = torch.tensor(np.concatenate([x_train, x_test]), dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.float32)

    torch.manual_seed(seed)
    feature_shifts = torch.randperm(x_all.shape[1]).tolist()
    class_shifts = torch.randperm(len(np.unique(y_train))).tolist()
    configs = list(itertools.product(class_shifts, feature_shifts))
    random.Random(seed).shuffle(configs)
    configs = list(itertools.product(configs, ["none", "power_all"]))[:n_ensemble]

    cache: dict[str, torch.Tensor] = {}
    xs, ys = [], []
    for (class_shift, feature_shift), transform in configs:
        if transform not in cache:
            cache[transform] = _preprocess(x_all, n_train, transform, max_features)
        x = cache[transform]
        x = torch.cat([x[:, feature_shift:], x[:, :feature_shift]], dim=1)
        x = torch.cat([x, torch.zeros(len(x), max_features - x.shape[1])], dim=1)
        xs.append(x)
        ys.append((y_tr + class_shift) % num_classes)
    xs, ys = torch.stack(xs), torch.stack(ys)

    logits = []
    for i in range(0, len(xs), batch_size):
        xb, yb = xs[i : i + batch_size].to(device), ys[i : i + batch_size].to(device)
        out = model(xb[:, :n_train], yb, xb[:, n_train:])[..., :num_classes]
        logits.append(out.float().cpu() / spec.SOFTMAX_TEMPERATURE)
    logits = torch.cat(logits)  # (N, m, C)

    total = torch.zeros(logits.shape[1:])
    for i, ((class_shift, _), _) in enumerate(configs):
        out = logits[i]
        total += torch.cat(
            [out[:, class_shift:], out[:, :class_shift]], dim=1
        )  # 클래스 회전 되돌리기
    return torch.softmax(total / len(configs), dim=-1).numpy()


def roc_auc(y_true: np.ndarray, proba: np.ndarray) -> float:
    if proba.shape[1] > 2:
        return float(roc_auc_score(y_true, proba, multi_class=spec.ROC_AUC_MULTI_CLASS))
    return float(roc_auc_score(y_true, proba[:, 1]))


def sklearn_baselines() -> dict:
    """기본값 sklearn baseline.

    논문의 1시간 튜닝 baseline은 직접 돌리지 않고 spec.PAPER_TABLE1_ROC_AUC를 인용한다.
    LogisticRegression만 결측값을 받지 못하므로 평균으로 채운다(Table 1 데이터셋에는 결측값이
    없어 결과가 같다).
    """
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return {
        "RandomForest": lambda: RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1),
        "HistGradientBoosting": lambda: HistGradientBoostingClassifier(random_state=0),
        "LogisticRegression": lambda: make_pipeline(
            SimpleImputer(), StandardScaler(), LogisticRegression(max_iter=2000)
        ),
    }


def evaluate_methods(
    datasets: list[OpenMLDataset],
    methods: dict,
    n_splits: int = spec.N_SPLITS,
) -> list[dict]:
    """methods: 이름 → predict_proba(x_train, y_train, x_test) 함수.

    반환하는 행 하나가 (데이터셋, split, 방법) 하나다.
    """
    rows = []
    for ds in datasets:
        for split in range(1, n_splits + 1):
            s = generate_split(ds.x, ds.y, split)
            if s is None:
                continue
            x_tr, y_tr, x_te, y_te = s
            for name, predict_proba in methods.items():
                proba = predict_proba(x_tr, y_tr, x_te)
                rows.append(
                    {
                        "did": ds.did,
                        "dataset": ds.name,
                        "split": split,
                        "method": name,
                        "n_train": len(y_tr),
                        "n_classes": int(y_tr.max()) + 1,
                        "roc_auc": roc_auc(y_te, proba),
                        "accuracy": float(accuracy_score(y_te, proba.argmax(1))),
                    }
                )
    return rows
