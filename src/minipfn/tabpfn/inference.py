"""옛 경로. v1 추론은 `minipfn.recipes.tabpfn_v1`(`TabPFNv1Classifier`. 예전 이름
`TabPFNClassifier`도 남아 있다. `tabpfn_predict_proba`)로
옮겼다. 예전 Hub 모델 카드의 예제가 이 경로를 썼다.

같은 모듈을 가리키도록 `sys.modules`에 새 모듈을 넣는다(이름 하나하나를 다시 내보내지 않는다).
"""

import sys

from minipfn.recipes import tabpfn_v1 as _module

sys.modules[__name__] = _module
