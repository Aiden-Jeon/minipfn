"""옛 경로. v2 추론은 `minipfn.recipes.tabpfn_v2`(`TabPFNv2Classifier`, `TabPFNv2Regressor`)로
옮겼다. 예전 Hub 모델 카드의 예제가 이 경로를 썼다.

같은 모듈을 가리키도록 `sys.modules`에 새 모듈을 넣는다(이름 하나하나를 다시 내보내지 않는다).
"""

import sys

from minipfn.recipes import tabpfn_v2 as _module

sys.modules[__name__] = _module
