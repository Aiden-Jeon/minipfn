"""minipfn: Prior-Data Fitted Networks(PFN)·TabPFN hands-on 가이드의 공용 코드.

각 챕터 노트북에서 반복해서 쓰는 코드(prior, 모델, 학습 루프 등)를 이 패키지에 모아 둔다.
"""

from minipfn.bar_distribution import (
    BarDistribution,
    FullSupportBarDistribution,
    get_bucket_borders,
)
from minipfn.evaluate import evaluate_against_gp
from minipfn.gp_prior import GPPriorSampler, analytic_gp_posterior, rbf_kernel
from minipfn.model import PFNTransformer, build_attention_mask
from minipfn.tabular_prior import sample_random_scm, sample_scm_dataset, to_classes
from minipfn.train import (
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
