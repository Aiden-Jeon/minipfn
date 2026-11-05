"""추론 (추론 축): 학습한 모델을 sklearn 형식으로 쓴다 (전처리 멤버, 앙상블).

- `rotation_ensemble`: 클래스·feature 회전 앙상블 `RotationEnsembleClassifier` (행 하나가 토큰인
  `RowTransformer`용, TabPFN v1 recipe의 `TabPFNv1Classifier`)
- `preprocessing`: 앙상블 멤버의 전처리 (분위수, power, robust, one-hot)
- `ensemble`: 전처리 멤버 앙상블 `EnsembleClassifier`·`EnsembleRegressor`. `forward(x, y_train)`
  모델이면 클래스와 상관없이 쓴다 (TabPFN v2 recipe의 `TabPFNv2Classifier`·`TabPFNv2Regressor`)

논문의 기본 설정(멤버 수, 온도, 멤버 종류 묶음)을 붙인 클래스는 `minipfn.recipes`에 있다.
"""
