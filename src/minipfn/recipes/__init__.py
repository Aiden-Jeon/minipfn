"""논문별 recipe (논문 = 각 축 부품의 조합). 각 축(`minipfn.priors`·`models`·`training`·
`inference`)의 모듈은 무엇인지로 이름을 붙인 부품이고, 논문 이름은 여기에만 쓴다. recipe는 논문이
정한 수치와 그 수치로 부품을 묶은 것(설정, 기본값을 붙인 클래스)을 두고, 그 논문에만 있는 학습
루프도 여기에 둔다. 의존 방향은 recipe → 축이다(축은 recipe를 import하지 않는다).

- `pfn`: PFN(arXiv 2112.10510). GP prior + `PointTransformer`, Algorithm 1 학습 루프
- `tabpfn_v1`: TabPFN v1(arXiv 2207.01848)의 수치와 출처. prior bag 설정, `RowTransformer` 크기,
  v1 학습 루프, 회전 앙상블 설정(`TabPFNv1Classifier`)
- `tabpfn_v2`: TabPFN v2·2.5의 학습 설정(`graph_scm` prior + `CellTransformer` + 공용 학습 루프)과
  구조, 전처리 멤버 앙상블 설정(`TabPFNv2Classifier`·`TabPFNv2Regressor`)
- `tabpfn_v3`: TabPFN-3·3.5의 구조(`graph_scm` prior + `ColumnRowTransformer` + 공용 학습 루프)와
  추론 설정(`default_inference`)
"""
