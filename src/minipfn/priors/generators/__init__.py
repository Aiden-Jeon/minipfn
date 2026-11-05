"""생성기: 그룹(하이퍼파라미터를 공유하는 데이터셋 묶음) 하나의 연속값 표 (x, y).

새 prior 논문은 대개 여기에 파일 하나를 더한다. 생성기는 `minipfn.priors.pipeline.Generator`
모양의 호출 가능한 객체다: `(n_rows, n_features, n_datasets, hp, rng) -> (x, y)`, x는
(행, 데이터셋, feature), y는 (행, 데이터셋).

- `mlp.MLP`: 무작위 MLP (SCM prior / BNN prior, TabPFN v1)
- `gp.GPTables`: RBF 커널 GP 표 (TabPFN v1 prior bag)
- `graph_scm.GraphSCM`: TabICLv2 graph SCM (TabPFN v2 recipe가 쓴다. `tabicl` 패키지가 필요하다)

모듈은 쓸 때 import한다(`graph_scm`이 `tabicl`을 불러오지 않도록 여기서는 아무것도 import하지
않는다).
"""
