# 노트북 로드맵

노트북은 논문별 폴더로 나눈 hands-on 가이드입니다. 작은 예제로 개념을 따라가고, GPU에서 학습한 결과는 MLflow run에서 불러와 봅니다. 명령줄에서 돌리는 학습·평가 스크립트와 AI Runtime 설정(yaml)은 [`scripts/`](../scripts)에 논문별로 나란히 있고, 각 표 아래에 그 논문에서 쓰는 파일을 적었습니다. [`00_setup_check.ipynb`](00_setup_check.ipynb)(환경 점검)는 공통입니다.

## TabPFN 버전 비교

v1부터 3.5까지는 한 줄로 이렇게 이어집니다. 1 → 2는 토큰 단위를 행에서 셀로 바꾸고 회귀를 더했고, 2 → 2.5는 같은 구조를 키웠고, 2.5 → 3에서 구조를 통째로 바꿨고, 3 → 3.5는 3의 뼈대를 넓히고 분류·회귀를 한 모델로 합쳤습니다. 3의 마지막 단계(행 ICL)는 행 하나가 토큰 하나인 v1의 행 Transformer와 attention 규칙이 같습니다. 셀 정보를 앞쪽 블록에서 행 벡터로 압축한 뒤 v1처럼 행끼리 in-context learning을 하는 셈입니다.

v1 칸은 [02_tabpfn/04 논문 (1) 아키텍처](02_tabpfn/04_tabpfn_paper_architecture.ipynb)에서, v2 칸은 [03_tabpfn_v2/01](03_tabpfn_v2/01_cell_tokens_and_two_way_attention.ipynb)·[02](03_tabpfn_v2/02_attention_properties_and_cost.ipynb)에서, 2.5 칸은 [04_tabpfn_v2_5/01](04_tabpfn_v2_5/01_layers_and_feature_groups.ipynb)~[04](04_tabpfn_v2_5/04_inference_preprocessing.ipynb)에서, 3 칸은 [05_tabpfn_v3/01](05_tabpfn_v3/01_overview_and_cell_encoder.ipynb)·[02](05_tabpfn_v3/02_column_row_icl_stages.ipynb)에서, 3.5 칸은 [06_tabpfn_v3_5/01](06_tabpfn_v3_5/01_changes_and_multitask.ipynb)에서 실제 값으로 따라갑니다. 모델 클래스는 v1이 `RowTransformer`, v2·2.5가 `CellTransformer`, 3·3.5가 `ColumnRowTransformer`이고, 같은 클래스끼리는 설정만 다릅니다.

|                                    | v1                                                          | v2                                                                 | v2.5                     | v3                                                     | v3.5                              |
| ---------------------------------- | ----------------------------------------------------------- | ------------------------------------------------------------------ | ------------------------ | ------------------------------------------------------ | --------------------------------- |
| Backbone                           | **행 하나 = 토큰 하나**, 행 사이 attention                            | 셀 단위 2방향 attention (feature attention ↔ sample attention을 층마다 번갈아) | v2와 같음                   | **열 → 행 → 행 ICL** 3단계 (TabICL식)                        | 3과 같음                             |
| 셀 묶기                               | 없음 (행 전체가 토큰 하나)                                            | 이웃한 셀 2개 → 토큰 1개                                                   | 이웃한 셀 **3개** → 토큰 1개     | 열 $j$, $j+1$, $j+3$의 셀 3개 → 토큰 1개 (feature마다 토큰 하나)    | 3과 같음                             |
| Embedding dim                      | 512                                                         | 192                                                                | 192 (논문에 없어 v2 값을 씀)     | 셀·열·행 단계 128, 행 ICL 512 (CLS 4 × 128)                  | 행 ICL **1,024** (CLS 8 × 128)     |
| Layers                             | 12                                                          | 12                                                                 | 분류 24, 회귀 18             | 열 3블록 + 행 3블록 + 행 ICL 24층                              | 3과 같음 (24층은 공개되지 않아 여기서 정함)       |
| Head                               | 4                                                           | 6                                                                  | 6                        | 행 ICL 8, 테스트 행은 KV head 1개                             | 행 ICL 16 (head dim 64)            |
| 정답 y                               | 클래스 번호를 실수 하나로 `Linear(1 → E)`해 학습 행 토큰에 더함                 | 행마다 y 토큰을 따로 붙임                                                    | v2와 같음                   | 학습 행의 셀 토큰과 행 벡터에 정답 embedding을 더함 (두 곳)               | + 정답 LayerNorm (두 task 공용)        |
| thinking row                       | 없음                                                          | 없음                                                                 | **64개**                  | 없음                                                     | 없음                                |
| 입력 encoder                         | 행의 feature 100개를 Linear 하나로 (모자라면 0으로 채우고 $100/k$배, 결측은 0)  | 셀 묶음마다 Linear                                                      | 분류 Linear, **회귀 2층 MLP** | 셀 묶음마다 Linear \[값 3, 결측 표시 3\]                         | **Fourier + ECDF 순위** → LayerNorm |
| attention 안정화                      | 없음                                                          | 회귀에 QK-clip 250 (여기서 정함)                                           | v2와 같음                   | **QASSMax**, QK-clip은 끔                                | + **q·k RMSNorm**                 |
| 분류 출력                              | Linear → 클래스 10개                                            | Linear → 클래스 10개                                                   | v2와 같음                   | **retrieval decoder** (닮은 학습 행의 정답을 모음, 보고서 최대 160클래스) | + 출력층 앞 잔차 MLP                    |
| 회귀 출력                              | **없음** (분류만)                                                | bar distribution 5,000                                             | v2와 같음                   | 2층 MLP → bar 5,000                                     | 잔차 MLP → Linear → bar 5,000       |
| checkpoint                         | 분류 하나                                                       | task마다 따로                                                          | task마다 따로                | task마다 따로                                              | **분류·회귀 한 모델**                    |
| Parameter                          | 25.82M                                                      | 분류 7.24M, 회귀 11.08M                                            | 분류 14.34M, 회귀 14.67M   | 분류 53.06M, 회귀 58.27M                               | multitask 218.83M                 |
| 층마다 계산 (행 $R$, 학습 행 $N$, 토큰 열 $C$) | $R \cdot N$ (feature는 100열로 고정)                             | $R \cdot C^2 + C \cdot R \cdot N$                                  | v2와 같고 층만 많음             | 24층 행 ICL은 $R \cdot N$이라 **feature 수와 상관없음**           | 3과 같음                             |
| 권장 표 크기                            | 학습 행 1,000, feature 100, 클래스 10                             | 학습 행 1만, feature 500                                               | 학습 행 5만, feature 2,000 | 100만 행·200 feature 또는 10만 행·2,000 feature              | 100만 행, 6,000 feature             |
| 추론 앙상블 (분류·회귀 멤버 수)                | 클래스 번호 회전 × feature 순서 회전 × (그대로, Yeo-Johnson), 32 (온도 0.8) | quantile + power, 4·8                                              | quantile + robust, 4·8   | quantile + robust, 4·8                                 | **standard 하나**, 8·8              |

2.5의 파라미터가 공식 보고서(11M·10M)보다 많은 것은 2.5 논문에 embedding dim과 head 수가 없어 v2 값(192, 6)을 그대로 썼기 때문입니다. 3.5의 층 수와 학습 설정(step, 학습률, task 섞는 비율)은 보고서에 없어서 이 저장소가 정했습니다.

## 축별로 보기

위 표는 버전마다 무엇이 다른지를 보고, 아래 폴더별 표는 버전 하나를 깊게 봅니다. 축 하나를 버전 순서로 따라가려면 이 표에서 한 줄을 왼쪽부터 읽습니다. 축 하나를 한 노트북에서 다섯 버전으로 나란히 견주는 것은 [`axes/`](#axes--축별-흐름) 폴더에 있습니다(모델 구조, prior, 추론).

| 축 | v1 | v2 | 2.5 | 3 | 3.5 |
|---|---|---|---|---|---|
| 모델 구조 | [02/04](02_tabpfn/04_tabpfn_paper_architecture.ipynb) 2절 | [03/01](03_tabpfn_v2/01_cell_tokens_and_two_way_attention.ipynb), [03/02](03_tabpfn_v2/02_attention_properties_and_cost.ipynb) | [04/01](04_tabpfn_v2_5/01_layers_and_feature_groups.ipynb), [04/02](04_tabpfn_v2_5/02_thinking_rows_and_regression_encoder.ipynb) | [05/01](05_tabpfn_v3/01_overview_and_cell_encoder.ipynb), [05/02](05_tabpfn_v3/02_column_row_icl_stages.ipynb) | [06/01](06_tabpfn_v3_5/01_changes_and_multitask.ipynb) 1~4절 |
| prior (학습 데이터) | [02/01](02_tabpfn/01_scm_primer.ipynb), [02/02](02_tabpfn/02_random_scm_datasets.ipynb), [02/03](02_tabpfn/03_bnn_primer.ipynb), [02/05](02_tabpfn/05_tabpfn_paper_prior.ipynb) | 논문 prior [03/03](03_tabpfn_v2/03_prior_graph_and_edges.ipynb), [03/04](03_tabpfn_v2/04_prior_from_graph_to_table.ipynb), 이 저장소가 쓰는 TabICL prior [03/07](03_tabpfn_v2/07_tabicl_prior.ipynb) | v2와 같은 TabICL prior | v2와 같은 TabICL prior | v2와 같은 TabICL prior |
| 학습 | [02/04](02_tabpfn/04_tabpfn_paper_architecture.ipynb) 2.6절 | [03/05](03_tabpfn_v2/05_paper_walkthrough_model_and_training.ipynb) 5절 | 긴 표 학습 [04/03](04_tabpfn_v2_5/03_long_tables.ipynb) | 학습하지 않음 (3.5만 학습) | multitask [06/01](06_tabpfn_v3_5/01_changes_and_multitask.ipynb) 5절 |
| 추론 | 회전 앙상블 [02/04](02_tabpfn/04_tabpfn_paper_architecture.ipynb) 2.5절 | [03/06](03_tabpfn_v2/06_paper_walkthrough_inference_and_results.ipynb) 1절 | robust·soft clip [04/04](04_tabpfn_v2_5/04_inference_preprocessing.ipynb) | 열 순서 섞기 [05/02](05_tabpfn_v3/02_column_row_icl_stages.ipynb) 5절 | standard 하나 [06/01](06_tabpfn_v3_5/01_changes_and_multitask.ipynb) 2절 |
| 재현 결과 | Table 1 ROC AUC 0.932, Hub 카드 ([분류](../hub/minipfn-tabpfn-v1/README.md)) | Hub 카드 ([분류](../hub/minipfn-tabpfn-v2-classifier/README.md), [회귀](../hub/minipfn-tabpfn-v2-regressor/README.md)) | 학습이 끝난 뒤 | — | 학습이 끝난 뒤 |

## `01_pfn/` — Müller et al., *Transformers Can Do Bayesian Inference* (ICLR 2022)

| # | 노트북 | 핵심 질문 |
|---|--------|-----------|
| 01 | [베이지안 추론과 PPD](01_pfn/01_bayesian_primer.ipynb) | PFN이 근사하려는 대상 $p(y \mid x, D)$는 무엇인가? |
| 02 | [논문 walkthrough](01_pfn/02_pfn_paper_walkthrough.ipynb) | PPD를 "계산"하는 대신 "학습"하면 왜 PPD 근사가 되는가? |
| 03 | [구현 디테일](01_pfn/03_pfn_implementation_details.ipynb) | bar distribution, Transformer encoder, GP prior의 정답 posterior는 어떻게 생겼나? |
| 04 | [GP prior 1D 회귀 재현](01_pfn/04_pfn_gp_reproduction.ipynb) | 논문 크기로 학습한 PFN(H100)이 정확한 GP posterior와 얼마나 비슷한가? |

실행 파일: 없음. 학습은 04 노트북의 8절에서 직접 돌립니다.

## `02_tabpfn/` — Hollmann et al., *TabPFN* (ICLR 2023)

| # | 노트북 | 핵심 질문 |
|---|--------|-----------|
| 01 | [구조적 인과 모델(SCM) 기초](02_tabpfn/01_scm_primer.ipynb) | SCM은 무엇이고, SCM에서 데이터를 어떻게 뽑으며, 개입은 무엇인가? |
| 02 | [무작위 SCM으로 표 데이터셋 만들기](02_tabpfn/02_random_scm_datasets.ipynb) | SCM 자체를 무작위로 뽑으면 어떤 표가 나오고, 그 표는 PFN에 어떻게 들어가는가? |
| 03 | [BNN prior 기초](02_tabpfn/03_bnn_primer.ipynb) | BNN prior는 데이터셋을 어떻게 만들고, SCM prior와 무엇이 다른가? |
| 04 | [논문 (1) PFN 복습과 아키텍처](02_tabpfn/04_tabpfn_paper_architecture.ipynb) | PFN을 표 데이터 분류에 쓰려고 Transformer와 입력·앙상블을 어떻게 바꿨는가? |
| 05 | [논문 (2) 표 데이터를 위한 prior](02_tabpfn/05_tabpfn_paper_prior.ipynb) | SCM·BNN prior를 어떻게 섞고, 연속값을 어떻게 다중 클래스 target으로 바꾸는가? |

실행 파일: `scripts/tabpfn/train.py`, `scripts/tabpfn/eval.py`, `scripts/tabpfn/plot_card.py`, `scripts/tabpfn/train.yaml`, `scripts/tabpfn/eval.yaml` (Hub: `hub/minipfn-tabpfn-v1/README.md` 카드, `hub/push_to_hub.py`)

## `03_tabpfn_v2/` — Hollmann et al., *Accurate predictions on small data with a tabular foundation model* (Nature 2025)

| # | 노트북 | 핵심 질문 |
|---|--------|-----------|
| 01 | [셀 단위 표현과 2방향 attention](03_tabpfn_v2/01_cell_tokens_and_two_way_attention.ipynb) | 표의 셀을 토큰으로 쓰고 행·열 방향으로 번갈아 attend하면 무엇이 달라지는가? |
| 02 | [2방향 attention의 성질과 비용](03_tabpfn_v2/02_attention_properties_and_cost.ipynb) | 이 구조는 무엇을 보장하고, 열 이름표·셀 묶기·상태 캐시로 무엇을 메우며, 계산은 얼마나 드는가? |
| 03 | [v2 prior (1) 그래프, 간선, 초기 데이터](03_tabpfn_v2/03_prior_graph_and_edges.ipynb) | 무작위 그래프와 간선 함수(신경망·나무·이산화)로 노드 값을 어떻게 계산하는가? |
| 04 | [v2 prior (2) 그래프에서 표로](03_tabpfn_v2/04_prior_from_graph_to_table.ipynb) | 노드 값에서 범주형 feature, 결측값, 계단 모양 관계가 있는 표를 어떻게 읽어 내는가? |
| 05 | [논문 (1) 구조, prior, 학습](03_tabpfn_v2/05_paper_walkthrough_model_and_training.ipynb) | v2는 v1의 무엇을 바꿔 행 10,000개·feature 500개의 분류와 회귀를 forward pass 한 번으로 하는가? |
| 06 | [논문 (2) 추론, 실험, 한계](03_tabpfn_v2/06_paper_walkthrough_inference_and_results.ipynb) | 추론 앙상블은 어떻게 하고, benchmark와 강건성 실험에서 무엇을 보였으며, 이 저장소는 무엇을 재현하는가? |
| 07 | [학습 데이터: TabICL prior](03_tabpfn_v2/07_tabicl_prior.ipynb) | 논문 prior 대신 쓰는 TabICL prior는 무엇이 같고 다르며, 학습에 쓸 만큼 빨리 만들어지는가? |

재현 결과(이 prior로 500k step 학습한 뒤 논문 benchmark 숫자와 비교)는 노트북 대신 Hub 모델 카드에 정리했습니다. 분류는 [`hub/minipfn-tabpfn-v2-classifier`](../hub/minipfn-tabpfn-v2-classifier/README.md)로 평균 ROC AUC 0.924(논문 0.932), 회귀는 [`hub/minipfn-tabpfn-v2-regressor`](../hub/minipfn-tabpfn-v2-regressor/README.md)로 논문 대비 RMSE 비율 중앙값 1.023입니다.

실행 파일: `scripts/tabpfn_v2/train.py`, `scripts/tabpfn_v2/eval.py`, `scripts/tabpfn_v2/plot_card.py`, `scripts/tabpfn_v2/train.yaml`, `scripts/tabpfn_v2/eval.yaml` (Hub: `hub/minipfn-tabpfn-v2-classifier/README.md`·`hub/minipfn-tabpfn-v2-regressor/README.md` 카드, `hub/push_to_hub.py`)

## `04_tabpfn_v2_5/` — Grinsztajn et al., *TabPFN-2.5: Advancing the State of the Art in Tabular Foundation Models* (arXiv 2511.08667, 2025)

| # | 노트북 | 핵심 질문 |
|---|--------|-----------|
| 01 | [구조 변경 (1) 층 수와 셀 3개 묶기](04_tabpfn_v2_5/01_layers_and_feature_groups.ipynb) | 층을 두 배로 늘리고 셀을 3개씩 묶으면 파라미터와 계산은 어떻게 바뀌는가? |
| 02 | [구조 변경 (2) thinking 행과 회귀 MLP encoder](04_tabpfn_v2_5/02_thinking_rows_and_regression_encoder.ipynb) | 학습되는 thinking 행 64개와 회귀 MLP encoder는 모델의 무엇을 바꾸는가? |
| 03 | [긴 표 학습](04_tabpfn_v2_5/03_long_tables.ipynb) | 논문이 공개하지 않은 학습 표 크기를 어떻게 정했고, 왜 짧은 표로 오래·긴 표로 잠깐 학습하는가? |
| 04 | [추론 전처리와 여기서 정한 것](04_tabpfn_v2_5/04_inference_preprocessing.ipynb) | robust scaling과 soft clipping은 무엇을 하고, 2.5 재현에서 논문에 없어 여기서 정한 것은 무엇인가? |

재현 결과(v2와 같은 benchmark와 TabArena-Lite에서의 판정)는 학습이 끝난 뒤 이 폴더에 노트북을 더해 정리합니다.

실행 파일: `scripts/tabpfn_v2_5/train.yaml`, `scripts/tabpfn_v2_5/eval.yaml` (스크립트는 v2와 같은 `scripts/tabpfn_v2/train.py`, `scripts/tabpfn_v2/eval.py`)

## `05_tabpfn_v3/` — Prior Labs, *TabPFN-3: Technical Report* (arXiv 2605.13986, 2026)

| # | 노트북 | 핵심 질문 |
|---|--------|-----------|
| 01 | [한눈에 보기와 셀 encoder](05_tabpfn_v3/01_overview_and_cell_encoder.ipynb) | 셀 토큰의 2방향 attention을 세 단계로 바꾸면 파라미터와 계산이 어떻게 달라지고, 셀은 어떻게 토큰이 되는가? |
| 02 | [열 → 행 → 행 ICL 세 단계와 출력](05_tabpfn_v3/02_column_row_icl_stages.ipynb) | 열 요약 → 행 압축 → 행 ICL과 retrieval decoder는 각각 무엇을 받아 무엇을 내고, 구조는 무엇을 보장하는가? |

실행 파일: `scripts/tabpfn_v3/train.yaml`, `scripts/tabpfn_v3/eval.yaml`. 3.5와 같은 파일이고, 3을 학습하려면 `train.yaml`에 `TABPFN_V2_ARCH=v3`와 `TABPFN_V2_TASK=classification`(또는 `regression`)을 override로 줍니다. 평가는 checkpoint에서 구조를 읽습니다. 스크립트는 v2와 같은 `scripts/tabpfn_v2/train.py`, `scripts/tabpfn_v2/eval.py`입니다.

## `06_tabpfn_v3_5/` — Prior Labs, *TabPFN-3.5: Technical Report* (arXiv 2609.17895, 2026)

| # | 노트북 | 핵심 질문 |
|---|--------|-----------|
| 01 | [3.5가 바꾼 것과 multitask 학습](06_tabpfn_v3_5/01_changes_and_multitask.ipynb) | 3의 뼈대를 넓히고 셀 encoder를 Fourier + ECDF로 바꾸면 무엇이 달라지고, 분류·회귀를 한 모델로 어떻게 학습하는가? |

재현 결과(3.5 multitask 학습 뒤 benchmark와 TabArena-Lite·BeyondArena에서의 판정)는 학습이 끝난 뒤 이 폴더에 노트북을 더해 정리합니다.

실행 파일: `scripts/tabpfn_v3/train.yaml`(기본값이 `TABPFN_V2_ARCH=v3.5`, `TABPFN_V2_TASK=multitask`), `scripts/tabpfn_v3/eval.yaml`. 스크립트는 v2와 같은 `scripts/tabpfn_v2/train.py`, `scripts/tabpfn_v2/eval.py`입니다.

## `axes/` — 축별 흐름

| # | 노트북 | 핵심 질문 |
|---|--------|-----------|
| 01 | [모델 구조의 흐름](axes/01_architecture_evolution.ipynb) | 같은 표를 v1·v2·2.5·3·3.5에 넣으면 토큰과 텐서 모양, 파라미터, attention 연산량이 어떻게 달라지는가? |
| 02 | [prior의 흐름](axes/02_prior_evolution.ipynb) | 버전마다 쓰는 prior에서 뽑은 데이터셋은 클래스·범주형 열·결측값·표 크기에서 어떻게 다르고, 만드는 데 얼마나 걸리는가? |
| 03 | [추론의 흐름](axes/03_inference_evolution.ipynb) | 버전마다 앙상블 멤버를 몇 개, 어떤 전처리로 만들고, 멤버 수와 종류가 실제 데이터셋의 점수를 얼마나 바꾸는가? |
