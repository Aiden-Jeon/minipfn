"""Kumo-Tabular(NVIDIA, 2026-09)에서 가져온 부품의 설정. 블로그(huggingface.co/blog/nvidia/
kumo-tabular)와 공개 소스(NVIDIA/structured-data-models, Apache 2.0. 동작만 보고 다시 구현했다)에서
확인한 것만 둔다. 논문·기술 보고서는 없다.

Kumo-Tabular 모델 전체(범주를 아는 셀 embedding, 열·행 블록 번갈아 쌓기, ECOC·999 분위수 출력)는
옮기지 않았다. 그래서 이 모듈에는 모델 종류가 없고 `TABLE_RECIPES`에 넣지 않는다. 여기 있는 설정은
다른 recipe의 모델(지금은 `tabpfn_v3`의 `column_row_transformer`)에 덧붙여 켠다.

- `ATTENTION_TEMPERATURE` (issue #102): attention 온도. q·k RMSNorm에 gain이 없고
  (`sdm/models/kumo/tabular/block.py`), 열 단계 요약 쪽과 행 ICL은 `LogScale`, 행 단계는
  `GatedLogScale`, 열 단계 펼치는 쪽은 온도 없음(`row_embedding.py`·`icl.py`, `sdm/nn/scaling.py`).
  QASSMax와 soft-cap은 쓰지 않는다. Kumo에는 retrieval decoder가 없어서 decoder도 LogScale로 둔
  것은 여기서 정했다. soft-cap(`logit_softcap`)은 이 설정과 따로 정한다

    config = replace(tabpfn_v3.curriculum(1), logit_softcap=None, **ATTENTION_TEMPERATURE)
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
