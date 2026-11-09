"""평가 benchmark.

- `amlb`: TabPFN v2 논문의 AMLB·OpenML-CTR23 suite와 논문 점수
- `cc18`: TabPFN v1 논문 Table 1 (OpenML-CC18)
- `tabarena`, `beyondarena`: TabArena-Lite·BeyondArena (공개 leaderboard와 같은 split·집계,
  공통 부분은 `arena`; extra `arena`의 `tabarena` 패키지가 필요하다)
- `gp`: PFN과 GP 정답 posterior 비교
- `uncertainty`: 예측 분포·불확실성 지표 (회귀 CRPS·구간 coverage·PIT, 분류 ECE·conformal coverage)
- `baselines`: 예측 분포 baseline (GBDT·공식 TabPFN v2·TabICL; extra `baselines`)
"""
