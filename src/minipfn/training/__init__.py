"""학습 (학습 축).

- `config`: 학습 설정 `TrainConfig` (모델·prior·최적화·실행을 평평하게. MLflow param이 이
  이름으로 남는다)
- `loop`: 공용 prior-fitting 학습 루프 `train` (DDP, checkpoint·이어 학습, 붕괴 되돌리기)
- `data`: prior를 DataLoader로 내보내는 IterableDataset (`PriorBatches`, `PriorGroups`)
- `losses`: 분류 cross-entropy, 회귀 bar distribution NLL과 target 흩기
- `stability`: QK-clip, 붕괴 감지

PFN·TabPFN v1 재현의 학습 루프는 그 논문에만 있는 절차라 `minipfn.recipes.pfn`·`recipes.tabpfn_v1`에
있다.
"""
