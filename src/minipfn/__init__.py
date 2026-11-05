"""minipfn: Prior-Data Fitted Networks(PFN)·TabPFN hands-on 가이드의 공용 코드.

코드는 네 축과 그 조합으로 나눈다. 논문 하나는 각 축의 설정을 고른 조합(recipe)이다.

- `priors`: 학습 데이터 (GP, SCM, TabPFN v1 prior, TabICL prior)
- `models`: 모델과 registry (PFN, TabPFN v1, TabPFN v2·2.5)
- `training`: 학습 설정과 학습 루프, 손실, 안정화
- `inference`: sklearn 형식 추론 (전처리 멤버, 앙상블)
- `benchmarks`: 평가 (AMLB·CTR23, OpenML-CC18, GP 비교)
- `hub`: Hugging Face Hub에 올리고 받기
- `recipes`: 논문별 수치와 설정 조합

여기서는 PFN 챕터 노트북이 쓰는 이름을 바로 내보낸다.
"""

from minipfn.benchmarks.gp import evaluate_against_gp
from minipfn.models.bar_distribution import (
    BarDistribution,
    FullSupportBarDistribution,
    get_bucket_borders,
)
from minipfn.models.pfn import PFNTransformer, build_attention_mask
from minipfn.priors.gp import GPPriorSampler, analytic_gp_posterior, rbf_kernel
from minipfn.priors.scm import sample_random_scm, sample_scm_dataset, to_classes
from minipfn.training.pfn import (
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
]
