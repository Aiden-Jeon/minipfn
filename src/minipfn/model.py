"""옛 경로. `minipfn.models.point_transformer`로 옮겼다. MLflow에 pickle로 로깅한 PFN 모델이
`minipfn.model.PFNTransformer`를 가리킨다.

같은 모듈을 가리키도록 `sys.modules`에 옛 이름을 남긴 `minipfn.models.pfn`을 넣는다.
"""

import sys

from minipfn.models import pfn as _module

sys.modules[__name__] = _module
