"""TabPFN(arXiv:2207.01848) 논문 규모 재현.

- `spec`: 논문·공식 v1 코드에서 가져온 수치와 출처
- `prior`: v1 prior(SCM·BNN·GP)를 torch로 옮긴 데이터셋 생성기
- `model`: v1 구조의 Transformer (25.82M 파라미터)
- `train`: 다중 GPU prior-fitting 학습 루프
- `evaluate`: OpenML-CC18(논문 Table 1) 평가
"""
