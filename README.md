# minipfn

Prior-Data Fitted Networks(PFN)와 TabPFN(v1, v2)을 밑바닥부터 작게 구현하고, 논문 결과를 직접 재현하면서 이해하는 hands-on 가이드입니다.

PFN은 prior에서 뽑은 합성 데이터셋을 대량으로 학습해, 새 데이터셋이 들어오면 forward pass 한 번으로 posterior predictive distribution(PPD)을 근사하는 모델입니다.

## 1. How to use

[uv](https://docs.astral.sh/uv/)가 필요합니다.

```bash
uv sync             # 가상환경 생성 + 의존성 설치
uv run jupyter lab  # 노트북 실행
```

논문 규모로 학습한 TabPFN 가중치(논문 Table 1 ROC AUC 0.932, 논문 0.934)를 Hugging Face Hub [aiden-jeon/minipfn-tabpfn-v1](https://huggingface.co/aiden-jeon/minipfn-tabpfn-v1)에 올려 두었습니다. GPU 없이 받아서 바로 예측할 수 있습니다([2절](#2-학습한-tabpfn으로-바로-예측하기)).

TabPFN v2는 분류와 회귀를 따로 학습해 올렸습니다. 두 모델 모두 TabICL prior로 500k step을 학습했고, 점수는 논문 benchmark에서 쟀습니다.

- 분류: [aiden-jeon/minipfn-tabpfn-v2-classifier](https://huggingface.co/aiden-jeon/minipfn-tabpfn-v2-classifier). AMLB 29개 데이터셋 평균 ROC AUC 0.924로, 논문은 0.932입니다.
- 회귀: [aiden-jeon/minipfn-tabpfn-v2-regressor](https://huggingface.co/aiden-jeon/minipfn-tabpfn-v2-regressor). 28개 데이터셋에서 논문 대비 RMSE 비율의 중앙값이 1.023, 기하평균이 1.118입니다.

직접 학습하려면 [3절](#3-바닥부터-학습하기)을, 개념부터 따라가려면 [4절](#4-step-by-step)의 노트북을 봅니다. 노트북 순서와 각 노트북의 핵심 질문은 [notebooks/README.md](notebooks/README.md)에 있습니다.

## 2. 학습한 TabPFN으로 바로 예측하기

TabPFN은 가중치를 학습하지 않습니다. 학습 행을 context로 넣고 forward pass 한 번으로 테스트 행의 클래스 확률을 냅니다. 그래서 `fit`은 학습 행을 준비만 하고, 계산은 `predict`에서 일어납니다.

```python
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from minipfn import TabPFNv1Classifier

x, y = load_breast_cancer(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=0)

clf = TabPFNv1Classifier().fit(x_train, y_train)  # 공개 가중치 aiden-jeon/minipfn-tabpfn-v1을 받는다
proba = clf.predict_proba(x_test)  # (테스트 행 수, 클래스 수)
print("ROC AUC:", roc_auc_score(y_test, proba[:, 1]))  # 0.997
```

`fit_mode`는 `fit`에서 어디까지 준비해 둘지 정합니다. 두 모드의 결과는 같고 비용만 다릅니다.

- `"fit_preprocessors"`(기본): 전처리만 캐시합니다. 테스트 행을 한꺼번에 많이 예측하는 batch 작업에 맞습니다.
- `"fit_with_cache"`: 층별 학습 행 key·value까지 캐시합니다. 학습 행을 고정하고 한 행씩 예측하는 서빙에 맞습니다. 메모리를 많이 씁니다(학습 1,000행 기준 약 1.6GB).

다른 가중치는 첫 인자로 줍니다. Hub repo id(`"owner/name"`, branch는 `"owner/name@step-100k"`), `save_pretrained`로 만든 로컬 폴더, 또는 모델을 받습니다.

요청마다 학습 행이 다르면 함수 `minipfn.recipes.tabpfn_v1.tabpfn_predict_proba(model, x_train, y_train, x_test)`를 씁니다. 모델은 `clf.model`을 쓰거나 `minipfn.from_pretrained("aiden-jeon/minipfn-tabpfn-v1")`로 받습니다.

같은 모델로 논문 Table 1(OpenML-CC18 18개 데이터셋 × 5 split)을 측정할 수 있습니다.

```bash
uv run python scripts/tabpfn/eval.py --hub-repo aiden-jeon/minipfn-tabpfn-v1
```

### TabPFN v2

v2도 쓰는 방법은 같습니다. 분류는 `TabPFNv2Classifier`, 회귀는 `TabPFNv2Regressor`를 씁니다. pandas DataFrame의 범주형 열도 그대로 넣을 수 있습니다.

```python
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from minipfn import TabPFNv2Classifier, TabPFNv2Regressor

# 분류: 위 예제와 같은 breast cancer 데이터에서 ROC AUC 0.998
clf = TabPFNv2Classifier()  # .fit(x_train, y_train).predict_proba(x_test)

# 회귀: 가중치와 함께 target 구간 경계(borders)도 받는다
x, y = load_diabetes(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=0)
reg = TabPFNv2Regressor().fit(x_train, y_train)
pred = reg.predict(x_test)  # 예측 분포의 평균
print("RMSE:", mean_squared_error(y_test, pred) ** 0.5)  # 55.4 (y_test 표준편차 74.0)
```

TabPFN-2.5 가중치도 같은 클래스로 씁니다. 앙상블 멤버 수와 멤버 전처리 종류는 가중치가 정합니다. Hub `config.json`의 `inference`에 적힌 값을 쓰고, 그 값이 없으면 모델 구조를 보고 2.5 구조면 2.5 설정을, 아니면 v2 설정을 씁니다. 그래서 따로 줄 것이 없습니다. 다른 설정과 비교해 보고 싶을 때만 `TabPFNv2Classifier(member_kinds="v2.5", n_estimators=8)`처럼 인자로 줍니다.

논문 benchmark(분류 29개·회귀 28개 × 10 fold)는 이렇게 잽니다. OpenML에서 데이터를 받으며, GPU가 있으면 빠릅니다(H100 한 장에서 분류 약 15분, 회귀 약 40분).

```bash
uv run python scripts/tabpfn_v2/eval.py \
  --classifier-path hf://aiden-jeon/minipfn-tabpfn-v2-classifier \
  --regressor-path hf://aiden-jeon/minipfn-tabpfn-v2-regressor
```

평가 스크립트도 멤버 전처리 종류를 가중치에서 정합니다. 다른 종류로 재려면 `--member-kinds v2.5`처럼 줍니다.

`main`에는 500k 가중치가 있고, 학습 중간의 100k~400k 가중치는 같은 repo의 branch `step-100k`~`step-400k`에 있습니다. `TabPFNv2Regressor("aiden-jeon/minipfn-tabpfn-v2-regressor@step-100k")`처럼 repo 뒤에 `@`와 branch를 적고, 평가 스크립트에서는 `hf://aiden-jeon/minipfn-tabpfn-v2-regressor@step-100k`처럼 `@` 뒤에 branch를 적습니다.

## 3. 바닥부터 학습하기

규모는 `--config-name`과 `--steps`로 정합니다. 학습·평가 기록은 MLflow run에 남습니다. PFN(GP prior 1D 회귀)은 학습 스크립트 없이 [`notebooks/01_pfn/04_pfn_gp_reproduction.ipynb`](notebooks/01_pfn/04_pfn_gp_reproduction.ipynb)에서 직접 학습합니다.

### 코드 구성

`src/minipfn/`의 코드는 네 축으로 나눕니다. prior(`priors/`: 학습 데이터), 모델(`models/`: 구조와 모델 registry), 학습(`training/`: 설정, 공용 학습 루프, 데이터 공급, 손실, 안정화), 추론(`inference/`: 앙상블과 전처리)입니다. 여기에 평가(`benchmarks/`)와 Hub 업로드(`hub/`)를 따로 둡니다.

축 안의 모듈은 무엇인지로 이름을 붙인 부품입니다. 예를 들어 prior는 `gp`·`mlp_scm`·`prior_bag`·`graph_scm`, 모델은 토큰 하나가 무엇인지에 따라 `point_transformer`(점)·`row_transformer`(행)·`cell_transformer`(셀 묶음)입니다. 논문 이름은 `recipes/`에만 씁니다. recipe는 논문이 정한 수치와 그 수치로 부품을 묶은 설정이고, 논문 기본값을 붙인 추론 클래스(`TabPFNv1Classifier`, `TabPFNv2Classifier` 등)와 그 논문에만 있는 학습 루프(PFN, TabPFN v1)도 여기에 둡니다. 축은 recipe를 import하지 않습니다. 실행 스크립트와 AI Runtime yaml은 `scripts/<paper>/`에 둡니다.

| recipe | prior | 모델 | 학습 | 추론 |
|---|---|---|---|---|
| `recipes/pfn.py` | `priors/gp.py` | `models/point_transformer.py` | `recipes/pfn.py`의 `train_pfn` | `recipes/pfn.py`의 `pfn_predict` |
| `recipes/tabpfn_v1.py` | `priors/prior_bag.py` (+ `mlp_scm`, `tabular`, `gp`) | `models/row_transformer.py` | `recipes/tabpfn_v1.py`의 `train_tabpfn` | `inference/rotation_ensemble.py` |
| `recipes/tabpfn_v2.py` (v2, 2.5) | `priors/graph_scm.py` (+ `table_size`) | `models/cell_transformer.py` | `training/loop.py` | `inference/ensemble.py` (+ `preprocessing`) |

예를 들어 TabPFN-2.5는 graph SCM prior, cell Transformer의 2.5 구조(`recipes/tabpfn_v2.py`의 `ARCHITECTURES`), 공용 학습 루프, 2.5 멤버 종류 묶음(`MEMBER_KIND_PRESETS["v2.5"]`)을 씁니다. 공용 학습 루프는 `TrainConfig`의 `prior_type`·`model_type`으로 prior와 모델을 고르므로, 한 축만 바꿔 비교할 수 있습니다. 예전 이름(`prior_type="tabicl"`, `model_type="tabpfn_v2"`)은 읽을 때 지금 이름으로 바꿉니다.

예전 경로 `minipfn.tabpfn.hub`·`minipfn.tabpfn.inference`·`minipfn.tabpfn_v2.hub`·`minipfn.tabpfn_v2.inference`는 예전 Hub 모델 카드의 예제가 쓰던 것이라 새 모듈(`hub`, recipe)의 별칭으로 남겨 두었습니다.

### 학습

```bash
# TabPFN (config: smoke / medium / paper). 마지막 가중치는 tabpfn_run/tabpfn_final.pt
uv run python scripts/tabpfn/train.py --config-name smoke

# GPU 여러 장이면 torchrun으로 GPU마다 프로세스를 하나씩 띄운다
uv run torchrun --standalone --nproc_per_node=8 scripts/tabpfn/train.py --config-name paper

# TabPFN v2 (config: smoke / medium / paper, --task classification | regression)
uv run python scripts/tabpfn_v2/train.py --config-name smoke --task regression
```

v2 논문 규모 학습은 AI Runtime에서 `scripts/tabpfn_v2/train.yaml`로 돌렸습니다. 100k step씩 나눠 이어서 학습하는 방법(`TABPFN_V2_STOP_STEP`, `TABPFN_V2_RESUME_RUN_ID`)과 노드 여러 대로 늘리는 방법은 yaml 머리 주석에 있습니다. 회귀 paper 설정에는 논문에 없는 안정화 장치 세 가지(반복 target dequantization, QK-clip τ=250, 붕괴 감지 후 되돌리기)가 기본으로 켜져 있습니다. 왜 필요한지는 [회귀 모델 카드](hub/minipfn-tabpfn-v2-regressor/README.md#training)에 적었습니다.

### 평가

논문 Table 1의 OpenML-CC18 18개 데이터셋 × 5 split에서 ROC AUC를 재고, sklearn 기본값 baseline과 비교합니다. 학습 run이 남긴 모델과 checkpoint는 `runs:/` URI로 줍니다. `--mlflow`를 주면 평가용 run을 따로 만들어 결과를 그 run의 `eval/`에 남깁니다(AI Runtime에서는 `scripts/tabpfn/eval.yaml`).

```bash
uv run python scripts/tabpfn/eval.py --model-path tabpfn_run/tabpfn_final.pt                      # 로컬 가중치
uv run python scripts/tabpfn/eval.py --model-path runs:/<run_id>/model/tabpfn_final.pt --mlflow   # 학습 run의 최종 모델
uv run python scripts/tabpfn/eval.py --model-path runs:/<run_id>/checkpoints/step_0003000.pt      # 학습 중 checkpoint
uv run python scripts/tabpfn/eval.py --baselines-only                                             # baseline만
```

### checkpoint에서 이어서 학습하기

TabPFN 학습은 `checkpoint_every` step마다(paper 1,000 · medium 300 · smoke 10) 학습 상태 전체를 MLflow run의 `checkpoints/latest.pt`에 덮어쓰고, `keep_every` step마다(paper 3,000 · medium 600 · smoke 10) `checkpoints/step_XXXXXXX.pt`로도 남깁니다. 학습이 끊기면 그 run id를 `--resume-run-id`로 주고, 처음과 같은 config와 step 수로 다시 실행합니다. `--resume-step`을 같이 주면 그 step의 checkpoint에서 새 run으로 이어 갑니다.

```bash
uv run python scripts/tabpfn/train.py --config-name smoke --resume-run-id <run_id>
uv run python scripts/tabpfn/train.py --config-name smoke --resume-run-id <run_id> --resume-step 10
```

Hub의 가중치에는 optimizer 상태가 없어서 이어서 학습하는 데 쓸 수 없습니다.

### Hugging Face Hub에 올리기

`hub/`에는 Hub repo마다 모델 카드 폴더(`hub/<repo 이름>/README.md`와 `images/`)를 두고, Hub에 올리는 스크립트도 함께 둡니다. `hub/push_to_hub.py`는 v1·v2 체크포인트를 `model.safetensors`와 `config.json`으로 바꾸고 카드를 붙여 올립니다. 카드의 그림과 수치가 다른 체크포인트의 것이 되지 않도록, 카드에 `--checkpoint`가 백틱(`` ` ``)으로 감싸 그대로 적혀 있어야 올립니다. 가중치를 만든 체크포인트는 Hub에도 남깁니다(v2는 `config.json`의 `source`, v1은 `source.json`). 카드만 동기화할 때도 이 값과 카드를 맞춰 봅니다. 그림은 논문별 `scripts/<paper>/plot_card.py`로 다시 만듭니다. 다른 카드는 `--model-card`로 줍니다.

```bash
uv run python scripts/tabpfn/plot_card.py --run-id <학습 run_id> --eval-run-id <평가 run_id>
uv run python hub/push_to_hub.py --paper tabpfn --repo-id <owner/name> \
  --checkpoint runs:/<학습 run_id>/model/tabpfn_final.pt
# v2는 --task classification 또는 regression
uv run python scripts/tabpfn_v2/plot_card.py --task regression \
  --checkpoint runs:/<학습 run_id>/checkpoints/step_0500000.pt --eval-runs 100000=<평가 run_id>,...
uv run python hub/push_to_hub.py --paper tabpfn_v2 --task regression --repo-id <owner/name> \
  --checkpoint runs:/<학습 run_id>/checkpoints/step_0500000.pt
# 중간 checkpoint는 main을 올린 뒤 branch로 (카드의 중간 checkpoint 표에 --checkpoint가 있어야 한다)
uv run python hub/push_to_hub.py --paper tabpfn_v2 --task regression --repo-id <owner/name> \
  --checkpoint runs:/<학습 run_id>/checkpoints/step_0100000.pt --revision step-100k
```

카드만 고쳤을 때는 가중치를 다시 올리지 않아도 됩니다. main에 `hub/<repo 이름>/`의 변경이 들어오면 GitHub Actions가 `hub/sync_hub_card.py`로 카드만 Hub에 올립니다.

## 4. Step by step

- [`notebooks/01_pfn/`](notebooks/01_pfn) — Müller et al., *[Transformers Can Do Bayesian Inference](https://arxiv.org/pdf/2112.10510)*, ICLR 2022
- [`notebooks/02_tabpfn/`](notebooks/02_tabpfn) — Hollmann et al., *[TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second](https://arxiv.org/pdf/2207.01848)*, ICLR 2023
- [`notebooks/03_tabpfn_v2/`](notebooks/03_tabpfn_v2) — Hollmann et al., *[Accurate predictions on small data with a tabular foundation model](https://doi.org/10.1038/s41586-024-08328-6)*, Nature 2025

공식 구현은 [automl/PFNs](https://github.com/automl/PFNs), [PriorLabs/TabPFN](https://github.com/PriorLabs/TabPFN)입니다.

## 5. 라이선스

코드와 Hugging Face Hub의 가중치는 [Apache-2.0](LICENSE)입니다. TabPFN v1 재현 코드(`src/minipfn/`의 `*/tabpfn_v1.py`, `benchmarks/cc18.py`)는 공식 TabPFN v1 코드([automl/TabPFN](https://github.com/automl/TabPFN) `tabpfn_v1`, Apache-2.0)를 옮긴 부분이 있어 원 저작권 표기를 [NOTICE](NOTICE)에 함께 둡니다.

`notebooks/*/images/`의 논문 그림(`pfn_fig1a.png`, `tabpfn_fig1.png`, `tabpfn_fig2.png`, `tabpfn_fig4.png`)은 각 논문 저자에게 저작권이 있으며 Apache-2.0 대상이 아닙니다. 설명을 위해 출처(Müller et al. 2022, Hollmann et al. 2023)를 밝혀 인용했습니다. `notebooks/03_tabpfn_v2/images/`의 그림(`tabpfn_v2_fig1.png`\~`tabpfn_v2_fig4.png`)은 Hollmann et al. 2025([Nature 637:319](https://doi.org/10.1038/s41586-024-08328-6))의 Fig. 1\~4를 줄인 것으로, 원 논문의 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) 라이선스를 따릅니다. 직접 그린 `pfn_encoder_arch.svg`는 Apache-2.0입니다.
