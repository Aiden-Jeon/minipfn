"""학습 (학습 축).

- `config`: 학습 설정 `TrainConfig` (모델·prior·최적화·실행을 평평하게. MLflow param이 이
  이름으로 남는다)
- `loop`: prior-fitting 학습 루프 `train` (DDP, checkpoint·이어 학습, 붕괴 되돌리기)
- `losses`: 분류 cross-entropy, 회귀 bar distribution NLL과 target 흩기
- `stability`: QK-clip, 붕괴 감지
- `pfn`, `tabpfn_v1`: PFN·TabPFN v1 재현의 학습 루프 (각자의 설정과 함께 따로 둔다)
"""
