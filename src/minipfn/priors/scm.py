"""표 데이터용 prior: TabPFN(arXiv:2207.01848) 4절의 무작위 SCM 샘플러.

`notebooks/02_tabpfn/01_scm_primer.ipynb` 6절에서 만든 코드를 다른 노트북에서도 쓰도록 옮겼다.
하이퍼파라미터 분포는 논문(부록 E.4)보다 단순하게 잡았다.
"""

from __future__ import annotations

import numpy as np

# 부록 E.4의 활성화 함수 후보
ACTIVATIONS = {
    "tanh": np.tanh,
    "leaky_relu": lambda z: np.where(z > 0, z, 0.01 * z),
    "elu": lambda z: np.where(z > 0, z, np.expm1(np.minimum(z, 0))),
    "identity": lambda z: z,
}


def to_classes(y_cont: np.ndarray, max_classes: int, rng: np.random.Generator) -> np.ndarray:
    """연속값 target을 클래스로 바꾼다 (논문 4.5절).

    클래스 수를 뽑고, target 값 중에서 경계를 무작위로 골라 구간 번호를 클래스로 쓴 뒤,
    순서 정보가 남지 않도록 클래스 번호를 섞는다.
    """
    n_classes = int(rng.integers(2, max_classes + 1))
    bounds = np.sort(rng.choice(y_cont, n_classes - 1, replace=False))
    return rng.permutation(n_classes)[np.searchsorted(bounds, y_cont)]


def sample_random_scm(n_features: int, rng: np.random.Generator) -> dict:
    """MLP 모양의 무작위 SCM 하나를 뽑는다 (논문 부록 C.1).

    층 수와 노드 수를 뽑고, 가중치 일부를 지워(dropout) DAG를 만든 뒤, 전체 노드 중
    `n_features`개를 feature로, 하나를 target으로 고른다. 노드 번호 k * hidden + j는
    k번째 층(0 = 원인 노드)의 j번째 노드다.
    """
    # 층 수와 노드 수는 작은 값이 더 자주 나오도록 뽑는다 (단순성, 4.2절).
    n_layers = int(rng.integers(2, 5))
    hidden = int(np.exp(rng.uniform(np.log(4), np.log(16))))
    hidden = max(hidden, -(-(n_features + 1) // (n_layers + 1)))  # 노드가 feature + target보다 많게
    w_std = np.exp(rng.uniform(np.log(0.3), np.log(3)))
    p_drop = rng.uniform(0, 0.8)
    weights = [rng.normal(0, w_std, (hidden, hidden)) for _ in range(n_layers)]
    masks = [rng.random((hidden, hidden)) > p_drop for _ in range(n_layers)]  # False = 지운 간선
    n_nodes = (n_layers + 1) * hidden
    nodes = rng.choice(n_nodes, n_features + 1, replace=False)
    return {
        "n_layers": n_layers,
        "hidden": hidden,
        "activation": str(rng.choice(list(ACTIVATIONS))),
        "noise_std": np.exp(rng.uniform(np.log(1e-3), np.log(0.3))),
        "weights": [w * m for w, m in zip(weights, masks, strict=True)],
        "masks": masks,
        "feature_nodes": nodes[:-1],
        "target_node": nodes[-1],
    }


def sample_scm_dataset(
    scm: dict, n_samples: int, max_classes: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """`sample_random_scm`으로 뽑은 SCM에서 분류 데이터셋 (x, y)를 만든다.

    모든 노드 값을 층 순서대로 계산한 뒤 feature 열과 target 열만 남긴다.
    x는 표준화하고(부록 C.2.1), y는 `to_classes`로 클래스 번호로 바꾼다.
    """
    act = ACTIVATIONS[scm["activation"]]
    z = rng.standard_normal((n_samples, scm["hidden"]))  # 첫 층: 원인 노드 (노이즈만으로 정해진다)
    layers = [z]
    for w in scm["weights"]:
        z = act(z @ w) + scm["noise_std"] * rng.standard_normal(z.shape)
        layers.append(z)
    all_nodes = np.concatenate(layers, axis=1)

    x = all_nodes[:, scm["feature_nodes"]]
    x = (x - x.mean(0)) / (x.std(0) + 1e-8)
    y = to_classes(all_nodes[:, scm["target_node"]], max_classes, rng)
    return x, y
