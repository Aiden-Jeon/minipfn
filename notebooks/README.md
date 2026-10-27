# 노트북 로드맵

노트북은 논문별 폴더로 나눈 hands-on 가이드입니다. 작은 예제로 개념을 따라가고, GPU에서 학습한 결과는 MLflow run에서 불러와 봅니다. 명령줄에서 돌리는 학습·평가 스크립트는 [`scripts/`](../scripts)에, AI Runtime 설정(yaml)은 [`runs/`](../runs)에 있고, 각 표 아래에 그 논문에서 쓰는 파일을 적었습니다. [`00_setup_check.ipynb`](00_setup_check.ipynb)(환경 점검)는 공통입니다.

## `01_pfn/` — Müller et al., *Transformers Can Do Bayesian Inference* (ICLR 2022)

| # | 노트북 | 핵심 질문 |
|---|--------|-----------|
| 01 | [베이지안 추론과 PPD](01_pfn/01_bayesian_primer.ipynb) | PFN이 근사하려는 대상 $p(y \mid x, D)$는 무엇인가? |
| 02 | [논문 walkthrough](01_pfn/02_pfn_paper_walkthrough.ipynb) | PPD를 "계산"하는 대신 "학습"하면 왜 PPD 근사가 되는가? |
| 03 | [구현 디테일](01_pfn/03_pfn_implementation_details.ipynb) | bar distribution, Transformer encoder, GP prior의 정답 posterior는 어떻게 생겼나? |
| 04 | [GP prior 1D 회귀 재현](01_pfn/04_pfn_gp_reproduction.ipynb) | 직접 학습한 PFN이 정확한 GP posterior와 얼마나 비슷한가? |

실행 파일: `scripts/train_pfn.py`, `runs/train_pfn.yaml`

## `02_tabpfn/` — Hollmann et al., *TabPFN* (ICLR 2023)

| # | 노트북 | 핵심 질문 |
|---|--------|-----------|
| 01 | [구조적 인과 모델(SCM) 기초](02_tabpfn/01_scm_primer.ipynb) | SCM은 무엇이고, 무작위 SCM으로 표 데이터셋을 어떻게 만드는가? |
| 02 | [BNN prior 기초](02_tabpfn/02_bnn_primer.ipynb) | BNN prior는 데이터셋을 어떻게 만들고, SCM prior와 무엇이 다른가? |
| 03 | [논문 walkthrough](02_tabpfn/03_tabpfn_paper_walkthrough.ipynb) | SCM·BNN prior를 섞은 PFN은 어떻게 다양한 표 데이터 분류에 forward pass 한 번으로 적응하는가? |
| 04 | [논문 규모 재현](02_tabpfn/04_tabpfn_reproduction.ipynb) | 논문 수치로 학습한 TabPFN이 OpenML Table 1에서 논문 ROC AUC(0.934)를 다시 내는가? (결과: 0.932) |

실행 파일: `scripts/train_tabpfn.py`, `scripts/eval_tabpfn.py`, `scripts/push_tabpfn_to_hub.py`(+ 모델 카드 `hub/minipfn-tabpfn-v1/README.md`), `runs/train_tabpfn.yaml`
