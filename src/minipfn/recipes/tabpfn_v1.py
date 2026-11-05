"""TabPFN(arXiv:2207.01848) 재현에 쓰는 논문 수치와 출처.

값은 논문 본문·부록과 공식 v1 코드(github.com/automl/TabPFN, `tabpfn_v1` 브랜치, commit bef0cf2,
Apache 2.0, Copyright 2022 University of Freiburg)에서 가져왔다. 출처 표기:
- "논문 E.1" 등: 논문 절
- "v1 PriorFittingCustomPrior": 공개 체크포인트를 학습한 설정이 적힌 노트북 셀
- "v1 model_configs": `tabpfn/scripts/model_configs.py`의 prior 설정
- "추정": 직접 확인하지 못해 가장 근거 있는 값을 고른 것
논문 표와 v1 코드가 다르면 체크포인트를 만든 v1 코드를 따른다(차이는 주석에 남긴다).
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# 모델 (논문 E.1, v1 PriorFittingCustomPrior)
# ---------------------------------------------------------------------------
EMSIZE = 512
NLAYERS = 12
NHEAD = 4  # emsize // 128
NHID = 1024  # emsize * nhid_factor(2)
DROPOUT = 0.0
MAX_FEATURES = 100  # 부족한 feature는 0으로 채우고 100/k로 스케일 (논문 E.2)
MAX_CLASSES = 10  # 출력 logit 수
PAPER_N_PARAMETERS = 25_820_000  # 논문 E.1: 25.82M

# ---------------------------------------------------------------------------
# 학습 (논문 3절·E.3, v1 PriorFittingCustomPrior + train.py)
# ---------------------------------------------------------------------------
SEQ_LEN = 1152  # bptt = 1024 + 128 (v1)
# single_eval_pos(학습 행 수) ~ Uniform{1, ..., 999}.
# v1은 0부터 뽑는다 (prior.TabPFNPriorDataset 참고).
MAX_EVAL_POS = 1000
# 한 forward batch의 데이터셋 8개가 prior 하이퍼파라미터를 공유한다 (v1)
DATASETS_PER_PRIOR_GROUP = 8
GRAD_ACCUMULATION = 8  # aggregate_k_gradients (v1)
BATCH_SIZE = 512  # optimizer step당 데이터셋 수 = 8 × 8 × GPU 8장 (논문 3절)
TOTAL_STEPS = 18_000  # 논문 3절: 18,000 step × batch 512
STEPS_PER_EPOCH = 128  # num_steps = 1024 // aggregate_k_gradients (v1)
WARMUP_STEPS = 20 * STEPS_PER_EPOCH  # warmup_epochs = 20 (v1 model_builder)
SCHEDULE_STEPS = (
    400 * STEPS_PER_EPOCH
)  # cosine은 400 epoch 기준. 논문은 약 141 epoch(18,000 step)에서 멈췄다.
LR = 1.1555e-4  # 추정: v1 노트북 출력의 lr (탐색 범위 log-uniform [1e-4, 1.5e-4])
WEIGHT_DECAY = 0.0  # AdamW (v1 train.py)
GRAD_CLIP = 1.0  # v1 train.py
# v1은 fp16 + GradScaler. H100에서는 bf16 autocast를 쓴다 (차이: 수치 형식만).

# ---------------------------------------------------------------------------
# Prior (v1 model_configs.get_diff_causal 등, PriorFittingCustomPrior에서 덮어쓴 값 반영)
# 논문 E.4 표와 다른 점: 활성화 함수 후보가 v1에서는 {Tanh, Identity, ReLU}다
# (논문 표: Tanh, LeakyReLU, ELU, Identity).
# ---------------------------------------------------------------------------
# prior bag: softmax([1, w])로 GP 또는 MLP prior를 고른다. w ~ U(2, 10)이라 대부분 MLP다.
PRIOR_BAG_MLP_WEIGHT = {"distribution": "uniform", "min": 2.0, "max": 10.0}

# MLP prior: is_causal=True면 SCM prior, False면 BNN prior (논문 4.3·4.4).
# 둘의 비율은 meta_choice 가중치로 정해진다 (평균 50/50).
MLP_HYPERPARAMETERS = {
    "num_layers": {
        "distribution": "meta_gamma",
        "max_alpha": 2,
        "max_scale": 3,
        "round": True,
        "lower_bound": 2,
    },
    "prior_mlp_hidden_dim": {
        "distribution": "meta_gamma",
        "max_alpha": 3,
        "max_scale": 100,
        "round": True,
        "lower_bound": 4,
    },
    "prior_mlp_dropout_prob": {"distribution": "meta_beta", "scale": 0.6, "min": 0.1, "max": 5.0},
    "noise_std": {
        "distribution": "meta_trunc_norm_log_scaled",
        "max_mean": 0.3,
        "min_mean": 1e-4,
        "round": False,
        "lower_bound": 0.0,
    },
    "init_std": {
        "distribution": "meta_trunc_norm_log_scaled",
        "max_mean": 10.0,
        "min_mean": 0.01,
        "round": False,
        "lower_bound": 0.0,
    },
    "num_causes": {
        "distribution": "meta_gamma",
        "max_alpha": 3,
        "max_scale": 7,
        "round": True,
        "lower_bound": 2,
    },
    "is_causal": {"distribution": "meta_choice", "choice_values": [True, False]},
    "pre_sample_weights": {"distribution": "meta_choice", "choice_values": [True, False]},
    "y_is_effect": {"distribution": "meta_choice", "choice_values": [True, False]},
    "prior_mlp_activations": {
        "distribution": "meta_choice_mixed",
        "choice_values": ["tanh", "identity", "relu"],
    },
    "block_wise_dropout": {"distribution": "meta_choice", "choice_values": [True, False]},
    "sort_features": {"distribution": "meta_choice", "choice_values": [True, False]},
    "in_clique": {"distribution": "meta_choice", "choice_values": [True, False]},
}
PRE_SAMPLE_CAUSES = True  # 원인 노드의 평균·표준편차를 데이터셋마다 미리 뽑는다
PRIOR_MLP_SCALE_WEIGHTS_SQRT = True
RANDOM_FEATURE_ROTATION = True

# GP prior (prior bag의 나머지)
GP_HYPERPARAMETERS = {
    "outputscale": {
        "distribution": "meta_trunc_norm_log_scaled",
        "max_mean": 10.0,
        "min_mean": 1e-5,
        "round": False,
        "lower_bound": 0.0,
    },
    "lengthscale": {
        "distribution": "meta_trunc_norm_log_scaled",
        "max_mean": 10.0,
        "min_mean": 1e-5,
        "round": False,
        "lower_bound": 0.0,
    },
    "noise": {"distribution": "meta_choice", "choice_values": [1e-5, 1e-4, 1e-2]},
}

# 표 데이터 처리 (v1 flexible_categorical, PriorFittingCustomPrior)
NUM_FEATURES_USED = (1, MAX_FEATURES)  # round(U(1, 100))
NUM_CLASSES = (2, MAX_CLASSES)  # 상한 round(U(2, 10)), 그 뒤 50% 확률로 2클래스 (class_sampler_f)
CATEGORICAL_FEATURE_P = 0.2  # 그룹의 20%에서 일부 열을 범주형으로 바꾼다
CATEGORICAL_ORDERED_P = 0.3
NAN_PROB = 0.0  # 공개 체크포인트는 결측값 주입을 끄고 학습했다 (nan_prob_* = 0)
OUTPUT_MULTICLASS_ORDERED_P = 0.0  # 클래스 번호는 항상 섞는다
NORMALIZE_BY_USED_FEATURES = True
CHECK_IS_COMPATIBLE = (
    True  # 학습/테스트 행의 클래스 집합이 같아야 한다. 아니면 행을 섞고, 10번 실패하면 무시(-100)
)
IGNORE_INDEX = -100

# ---------------------------------------------------------------------------
# 평가 (논문 5.2절·부록 F, v1 RunFullDatasetAnalyses, tabular_evaluation,
# transformer_prediction_interface)
# ---------------------------------------------------------------------------
# OpenML-CC18에서 샘플 ≤2000, feature ≤100, 클래스 ≤10인 30개 (v1 datasets/__init__.py open_cc_dids)
OPENML_CC18_TEST_DIDS = [
    11,
    14,
    15,
    16,
    18,
    22,
    23,
    29,
    31,
    37,
    50,
    54,
    188,
    458,
    469,
    1049,
    1050,
    1063,
    1068,
    1510,
    1494,
    1480,
    1462,
    1464,
    6332,
    23381,
    40966,
    40982,
    40994,
    40975,
]
# 논문 Table 1은 이 중 수치형 feature만 있고 결측값이 없는 18개다.
# 평가할 때 OpenML 메타데이터로 걸러낸다.
TABLE1_N_DATASETS = 18
EVAL_SEQ_LEN = 2000  # 학습 + 테스트 행
EVAL_TRAIN_SIZE = 1000  # eval_position
EVAL_MAX_SAMPLES = 10_000  # 로드 시 seed 13으로 섞은 뒤 자른다
N_SPLITS = 5
N_ENSEMBLE = 32  # 논문 5.2절
SOFTMAX_TEMPERATURE = 0.8  # logits / 0.8 (v1)
ROC_AUC_MULTI_CLASS = "ovo"  # v1 tabular_metrics.auc_metric

# 논문 Table 1 (18개 데이터셋, 평균 ROC AUC)
PAPER_TABLE1_ROC_AUC = {
    "TabPFN": 0.934,
    "TabPFN (no ensemble)": 0.932,
    "AutoGluon": 0.930,
    "Auto-sklearn 2.0": 0.929,
    "XGBoost": 0.924,
    "CatBoost": 0.924,
    "LightGBM": 0.920,
}
