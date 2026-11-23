"""Kumo-Tabular(NVIDIA, 2026-09)에서 가져온 부품의 설정. 블로그(huggingface.co/blog/nvidia/
kumo-tabular)와 공개 소스(NVIDIA/structured-data-models, Apache 2.0. 동작만 보고 다시 구현했다)에서
확인한 것만 둔다. 논문·기술 보고서는 없다.

Kumo-Tabular 모델 전체(ECOC 분류 출력, 분류·회귀 모델 따로 등)는 옮기지 않았다. 그래서 이 모듈에는
모델 종류가 없고 `TABLE_RECIPES`에 넣지 않는다. 여기 있는 설정은 다른 recipe의 모델(지금은
`tabpfn_v3`의 `column_row_transformer`)에 덧붙여 켠다. 시험에서 고른 부품(attention 온도·`STAGES`·
`QUANTILE_HEAD`)을 v3.5에 얹은 구조는 우리 모델 `minipfn-v1`이다(`recipes.minipfn_v1`).

- `ATTENTION_TEMPERATURE` (issue #102): attention 온도. q·k RMSNorm에 gain이 없고
  (`sdm/models/kumo/tabular/block.py`), 열 단계 요약 쪽과 행 ICL은 `LogScale`, 행 단계는
  `GatedLogScale`, 열 단계 펼치는 쪽은 온도 없음(`row_embedding.py`·`icl.py`, `sdm/nn/scaling.py`).
  QASSMax와 soft-cap은 쓰지 않는다. Kumo에는 retrieval decoder가 없어서 decoder도 LogScale로 둔
  것은 여기서 정했다. soft-cap(`logit_softcap`)은 이 설정과 따로 정한다

    config = replace(tabpfn_v3.curriculum(1), logit_softcap=None, **ATTENTION_TEMPERATURE)

나머지 부품 (issue #110, 이름으로 고르는 묶음 `OPTIONS`, 모델 docstring "Kumo-Tabular의 나머지
부품"). 하나씩 attention 온도 묶음 위에 얹어 시험한다:

- `DECODER_QK_NORM`: retrieval decoder q·k에 gain 없는 RMSNorm (Kumo에는 decoder가 없어 여기서 정함)
- `STAGES`: 열·행 단계를 Kumo-L처럼(`sdm/models/kumo/tabular/model.py`의 large: 셀 폭 256, 열·행
  블록 6쌍 번갈아, head 4, inducing 256, CLS 4개 → 행 벡터 1024). 행 ICL은 v3.5 그대로다(Kumo-L과
  같은 1024 폭·24층·16 head)
- `QUANTILE_HEAD`: 회귀 출력을 분위수 999개 + pinball 손실로 (블로그)
- `CATEGORICAL_ENCODER`: 범주를 아는 셀 encoder (`sdm/models/tabfm/cell_embedding.py`). 학습
  prior(graph SCM)가 범주 열 표시를 넘긴다

    config = replace(config, **STAGES)
"""

from __future__ import annotations

ATTENTION_TEMPERATURE = {
    "qk_norm_affine": False,
    "column_summary_temperature": "logscale",
    "column_spread_temperature": "none",
    "row_temperature": "gated_logscale",
    "icl_temperature": "logscale",
    "decoder_temperature": "logscale",
}

DECODER_QK_NORM = {"decoder_qk_norm": True}
STAGES = {
    "stage_layout": "interleaved",
    "emsize": 256,
    "column_blocks": 6,
    "row_blocks": 6,
    "column_heads": 4,
    "row_heads": 4,
    "n_inducing": 256,
    "n_cls": 4,
}
QUANTILE_HEAD = {"regression_head": "quantile"}
CATEGORICAL_ENCODER = {"categorical_encoder": True}
# 이름으로 고르는 묶음 (학습 스크립트 `--kumo-options`)
OPTIONS = {
    "decoder_qk_norm": DECODER_QK_NORM,
    "stages": STAGES,
    "quantile_head": QUANTILE_HEAD,
    "categorical_encoder": CATEGORICAL_ENCODER,
}
