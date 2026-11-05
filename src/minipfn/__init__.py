"""minipfn: Prior-Data Fitted Networks(PFN)·TabPFN hands-on 가이드의 공용 코드.

코드는 네 축과 그 조합으로 나눈다. 축의 모듈은 무엇인지로 이름을 붙인 부품이고(일반 명사), 논문
이름은 recipe에만 쓴다. 논문 하나는 각 축의 부품을 논문 수치로 묶은 것(recipe)이다.

- `priors`: 학습 데이터 (GP, 단순 SCM, MLP SCM, prior bag, graph SCM, 표 크기 규칙)
- `models`: 모델과 registry (point / row / cell Transformer, bar distribution)
- `training`: 학습 설정과 공용 학습 루프, 데이터 공급, 손실, 안정화
- `inference`: sklearn 형식 추론 (회전 앙상블, 전처리 멤버 앙상블)
- `benchmarks`: 평가 (AMLB·CTR23, OpenML-CC18, GP 비교)
- `hub`: Hugging Face Hub에 올리고 받기
- `recipes`: 논문별 수치와 조합 (`pfn`, `tabpfn_v1`, `tabpfn_v2`). 논문에만 있는 학습 루프
  (PFN, TabPFN v1)도 여기에 있다

여기서는 PFN 챕터 노트북이 쓰는 이름을 바로 내보낸다. TabPFN 추론에 쓰는 이름(`from_pretrained`와
논문 기본값을 붙인 추론 클래스)도 여기서 쓸 수 있다. 정의는 `hub`·`recipes`에 있고, 처음 쓸 때
불러온다(`import minipfn`이 Hub·추론 코드까지 불러오지 않도록).

    from minipfn import TabPFNv1Classifier

    clf = TabPFNv1Classifier().fit(x_train, y_train)  # 공개 가중치를 받는다
"""

from typing import TYPE_CHECKING

from minipfn.benchmarks.gp import evaluate_against_gp
from minipfn.models.bar_distribution import (
    BarDistribution,
    FullSupportBarDistribution,
    get_bucket_borders,
)
from minipfn.models.point_transformer import PointTransformer, build_attention_mask
from minipfn.priors.gp import GPPriorSampler, analytic_gp_posterior, rbf_kernel
from minipfn.priors.simple_scm import sample_random_scm, sample_scm_dataset, to_classes
from minipfn.recipes.pfn import (
    PFNConfig,
    build_criterion,
    build_model,
    build_prior_sampler,
    load_checkpoint,
    load_logged_model,
    pfn_predict,
    save_checkpoint,
    train_pfn,
)
from minipfn.utils import get_device, log_run_source, set_seed

if TYPE_CHECKING:
    from minipfn.hub import from_pretrained
    from minipfn.recipes.tabpfn_v1 import TabPFNv1Classifier
    from minipfn.recipes.tabpfn_v2 import TabPFNv2Classifier, TabPFNv2Regressor

PFNTransformer = PointTransformer  # 예전 이름

# 처음 쓸 때 불러오는 이름 → 정의한 모듈
_LAZY = {
    "from_pretrained": "minipfn.hub",
    "TabPFNv1Classifier": "minipfn.recipes.tabpfn_v1",
    "TabPFNv2Classifier": "minipfn.recipes.tabpfn_v2",
    "TabPFNv2Regressor": "minipfn.recipes.tabpfn_v2",
}


def __getattr__(name: str):
    if name in _LAZY:
        import importlib

        value = getattr(importlib.import_module(_LAZY[name]), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))


__all__ = [
    "get_device",
    "set_seed",
    "log_run_source",
    "rbf_kernel",
    "analytic_gp_posterior",
    "GPPriorSampler",
    "BarDistribution",
    "FullSupportBarDistribution",
    "get_bucket_borders",
    "PointTransformer",
    "PFNTransformer",
    "build_attention_mask",
    "PFNConfig",
    "build_model",
    "build_criterion",
    "build_prior_sampler",
    "train_pfn",
    "pfn_predict",
    "save_checkpoint",
    "load_checkpoint",
    "load_logged_model",
    "evaluate_against_gp",
    "to_classes",
    "sample_random_scm",
    "sample_scm_dataset",
    "from_pretrained",
    "TabPFNv1Classifier",
    "TabPFNv2Classifier",
    "TabPFNv2Regressor",
]
