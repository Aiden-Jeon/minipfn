"""추론 (추론 축): 학습한 모델을 sklearn 형식으로 쓴다 (전처리 멤버, 앙상블).

- `rotation_ensemble`: 클래스·feature 회전 앙상블 `RotationEnsembleClassifier` (행 하나가 토큰인
  `RowTransformer`용, TabPFN v1 recipe의 `TabPFNv1Classifier`)
- `preprocessing`: 앙상블 멤버의 전처리 (분위수, power, robust, one-hot)
- `ensemble`: 전처리 멤버 앙상블 `EnsembleClassifier`·`EnsembleRegressor`. `forward(x, y_train)`
  모델이면 클래스와 상관없이 쓴다 (TabPFN v2 recipe의 `TabPFNv2Classifier`·`TabPFNv2Regressor`)
- `ensemble_selection`: 멤버 pool을 학습 행 안의 out-of-fold 손실로 greedy 선택하는 앙상블
  `GreedyEnsembleClassifier`·`GreedyEnsembleRegressor` (추론에 연산을 더 쓰는 집계)
- `diag_scale`: 데이터셋마다 attention의 대각 scale만 학습하는 test-time 적응 `DiagScale`
  (앙상블 estimator를 감싼다)
- `distillation`: 추론 estimator(teacher)를 데이터셋마다 GBDT 학생으로 증류하는
  `DistilledClassifier`·`DistilledRegressor` (예측은 CPU의 학생만 쓴다)

논문의 기본 설정(멤버 수, 온도, 멤버 종류 묶음)을 붙인 클래스는 `minipfn.recipes`에 있다.
"""
