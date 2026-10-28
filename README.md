# minipfn

Prior-Data Fitted Networks(PFN)와 TabPFN을 밑바닥부터 작게 구현하고, 논문 결과를 직접 재현하면서 이해하는 hands-on 가이드입니다.

PFN은 prior에서 뽑은 합성 데이터셋을 대량으로 학습해, 새 데이터셋이 들어오면 forward pass 한 번으로 posterior predictive distribution(PPD)을 근사하는 모델입니다.

## 1. How to use

[uv](https://docs.astral.sh/uv/)가 필요합니다.

```bash
uv sync             # 가상환경 생성 + 의존성 설치
uv run jupyter lab  # 노트북 실행
```

논문 규모로 학습한 TabPFN 가중치(논문 Table 1 ROC AUC 0.932, 논문 0.934)를 Hugging Face Hub [aiden-jeon/minipfn-tabpfn-v1](https://huggingface.co/aiden-jeon/minipfn-tabpfn-v1)에 올려 두었습니다. GPU 없이 받아서 바로 예측할 수 있습니다([2절](#2-학습한-tabpfn으로-바로-예측하기)).

직접 학습하려면 [3절](#3-바닥부터-학습하기)을, 개념부터 따라가려면 [4절](#4-step-by-step)의 노트북을 봅니다. 노트북 순서와 각 노트북의 핵심 질문은 [notebooks/README.md](notebooks/README.md)에 있습니다.

## 2. 학습한 TabPFN으로 바로 예측하기

TabPFN은 가중치를 학습하지 않습니다. 학습 행을 context로 넣고 forward pass 한 번으로 테스트 행의 클래스 확률을 냅니다. 그래서 `fit`은 학습 행을 준비만 하고, 계산은 `predict`에서 일어납니다.

```python
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from minipfn.tabpfn.hub import from_pretrained
from minipfn.tabpfn.inference import TabPFNClassifier

x, y = load_breast_cancer(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=0)

model, config = from_pretrained("aiden-jeon/minipfn-tabpfn-v1")
clf = TabPFNClassifier(model).fit(x_train, y_train)
proba = clf.predict_proba(x_test)  # (테스트 행 수, 클래스 수)
print("ROC AUC:", roc_auc_score(y_test, proba[:, 1]))  # 0.997
```

`fit_mode`는 `fit`에서 어디까지 준비해 둘지 정합니다. 두 모드의 결과는 같고 비용만 다릅니다.

- `"fit_preprocessors"`(기본): 전처리만 캐시합니다. 테스트 행을 한꺼번에 많이 예측하는 batch 작업에 맞습니다.
- `"fit_with_cache"`: 층별 학습 행 key·value까지 캐시합니다. 학습 행을 고정하고 한 행씩 예측하는 서빙에 맞습니다. 메모리를 많이 씁니다(학습 1,000행 기준 약 1.6GB).

요청마다 학습 행이 다르면 함수 `minipfn.tabpfn.inference.tabpfn_predict_proba(model, x_train, y_train, x_test)`를 씁니다.

같은 모델로 논문 Table 1(OpenML-CC18 18개 데이터셋 × 5 split)을 측정할 수 있습니다.

```bash
uv run python scripts/eval_tabpfn.py --hub-repo aiden-jeon/minipfn-tabpfn-v1
```

## 3. 바닥부터 학습하기

규모는 `--config-name`과 `--steps`로 정합니다. 학습·평가 기록은 MLflow run에 남습니다.

### 학습

```bash
# PFN GP prior 1D 회귀 (config: small / large / paper)
uv run python scripts/train_pfn.py --config-name small --steps 50

# TabPFN (config: smoke / medium / paper). 마지막 가중치는 tabpfn_run/tabpfn_final.pt
uv run python scripts/train_tabpfn.py --config-name smoke

# GPU 여러 장이면 torchrun으로 GPU마다 프로세스를 하나씩 띄운다
uv run torchrun --standalone --nproc_per_node=8 scripts/train_tabpfn.py --config-name paper
```

### 평가

논문 Table 1의 OpenML-CC18 18개 데이터셋 × 5 split에서 ROC AUC를 재고, sklearn 기본값 baseline과 비교합니다. `--run-id`를 주면 결과를 그 학습 run에 함께 기록합니다.

```bash
uv run python scripts/eval_tabpfn.py --model-path tabpfn_run/tabpfn_final.pt   # 로컬 가중치
uv run python scripts/eval_tabpfn.py --run-id <학습 run_id>                    # MLflow run의 가중치
uv run python scripts/eval_tabpfn.py --baselines-only                          # baseline만
```

### checkpoint에서 이어서 학습하기

TabPFN 학습은 `checkpoint_every` step마다(paper 1,000 · medium 300 · smoke 10) 학습 상태 전체를 MLflow run의 `checkpoints/latest.pt`에 올립니다. 학습이 끊기면 그 run id를 `--resume-run-id`로 주고, 처음과 같은 config와 step 수로 다시 실행합니다.

```bash
uv run python scripts/train_tabpfn.py --config-name smoke --resume-run-id <run_id>
```

Hub의 가중치에는 optimizer 상태가 없어서 이어서 학습하는 데 쓸 수 없습니다. PFN 학습(`train_pfn.py`)에는 아직 checkpoint 재개 기능이 없습니다.

### Hugging Face Hub에 올리기

MLflow run의 `tabpfn_final.pt`를 `model.safetensors`와 `config.json`으로 바꾸고, `hub/<repo 이름>/README.md` 모델 카드를 붙여 올립니다. 다른 카드는 `--model-card`로 주고, paper 크기가 아닌 run은 `--allow-non-paper`를 함께 줍니다.

```bash
uv run python scripts/push_tabpfn_to_hub.py --run-id <학습 run_id> --repo-id <owner/name>
```

## 4. Step by step

- [`notebooks/01_pfn/`](notebooks/01_pfn) — Müller et al., *[Transformers Can Do Bayesian Inference](https://arxiv.org/pdf/2112.10510)*, ICLR 2022
- [`notebooks/02_tabpfn/`](notebooks/02_tabpfn) — Hollmann et al., *[TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second](https://arxiv.org/pdf/2207.01848)*, ICLR 2023

공식 구현은 [automl/PFNs](https://github.com/automl/PFNs), [PriorLabs/TabPFN](https://github.com/PriorLabs/TabPFN)입니다. 후속작 TabPFN v2(Hollmann et al., *Accurate predictions on small data with a tabular foundation model*, Nature 2025)는 아직 다루지 않습니다.

## 5. 라이선스

코드와 Hugging Face Hub의 가중치는 [Apache-2.0](LICENSE)입니다. `src/minipfn/tabpfn/`은 공식 TabPFN v1 코드([automl/TabPFN](https://github.com/automl/TabPFN) `tabpfn_v1`, Apache-2.0)를 옮긴 부분이 있어 원 저작권 표기를 [NOTICE](NOTICE)에 함께 둡니다.

`notebooks/*/images/`의 논문 그림(`pfn_fig1a.png`, `tabpfn_fig1.png`, `tabpfn_fig2.png`, `tabpfn_fig4.png`)은 각 논문 저자에게 저작권이 있으며 Apache-2.0 대상이 아닙니다. 설명을 위해 출처(Müller et al. 2022, Hollmann et al. 2023)를 밝혀 인용했습니다. 직접 그린 `pfn_encoder_arch.svg`는 Apache-2.0입니다.
