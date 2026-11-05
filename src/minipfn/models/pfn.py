"""옛 경로. `minipfn.models.point_transformer`로 옮겼다(`PFNTransformer` → `PointTransformer`).

MLflow에 pickle로 로깅한 PFN 모델이 `minipfn.models.pfn.PFNTransformer`(또는 그 전의
`minipfn.model.PFNTransformer`)를 가리킨다. unpickle은 이 모듈에서 그 이름을 찾으므로 남겨 둔다.
"""

from minipfn.models.point_transformer import PointTransformer as PFNTransformer
from minipfn.models.point_transformer import build_attention_mask

__all__ = ["PFNTransformer", "build_attention_mask"]
