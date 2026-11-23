"""논문별 recipe (논문 = 각 축 부품의 조합). 각 축(`minipfn.priors`·`models`·`training`·
`inference`)의 모듈은 무엇인지로 이름을 붙인 부품이고, 논문 이름은 여기에만 쓴다. recipe는 논문이
정한 수치와 그 수치로 부품을 묶은 것(설정, 기본값을 붙인 클래스)을 두고, 그 논문에만 있는 학습
루프도 여기에 둔다. 의존 방향은 recipe → 축이다(축은 recipe를 import하지 않는다).

- `pfn`: PFN(arXiv 2112.10510). GP prior + `PointTransformer`, Algorithm 1 학습 루프
- `tabpfn_v1`: TabPFN v1(arXiv 2207.01848)의 수치와 출처. prior bag 설정, `RowTransformer` 크기,
  v1 학습 루프, 회전 앙상블 설정(`TabPFNv1Classifier`)
- `tabpfn_v2`: TabPFN v2·2.5의 학습 설정(`graph_scm` prior + `CellTransformer` + 공용 학습 루프)과
  구조, 전처리 멤버 앙상블 설정(`TabPFNv2Classifier`·`TabPFNv2Regressor`)
- `tabpfn_v3`: TabPFN-3·3.5의 구조(`graph_scm` prior + `ColumnRowTransformer` + 공용 학습 루프),
  추론 설정(`default_inference`), 3.5 공개 가중치를 받는 추론 클래스(`TabPFNv3_5Classifier`·
  `TabPFNv3_5Regressor`)
- `kumo_tabular`: Kumo-Tabular에서 가져온 부품의 설정(attention 온도·열·행 단계·분위수 출력 등).
  모델 종류가 없어 `TABLE_RECIPES`에 넣지 않고, 다른 recipe의 모델에 덧붙여 켠다
- `minipfn_v1`: 우리 자체 모델 minipfn-v1(재현이 아니다). v3.5 구조에 Kumo-Tabular 부품(열·행 단계
  번갈아 쌓기, attention 온도, 회귀 분위수 출력)을 얹은 구조와 그 curriculum. 모델 클래스는
  `tabpfn_v3`와 같은 `column_row_transformer`다
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any

# 공용 학습 루프(`minipfn.training.loop`)와 전처리 멤버 앙상블을 쓰는 recipe. 모듈마다
# `MODEL_TYPE`, `ARCHITECTURES`, `architecture_name`, `model_architecture`, `with_architecture`,
# `default_inference`, `MEMBER_KIND_PRESETS`, `SUPPORTED_SIZE`가 있다. 스크립트·Hub 업로드는
# recipe를 직접 고르지 않고 아래 함수로 찾는다. 모델 종류가 같은 recipe가 여럿이면(`tabpfn_v3`·
# `minipfn_v1`) 먼저 적은 것이 그 모델 종류의 기본이다(`recipe_for_model_type`)
TABLE_RECIPES = ("tabpfn_v2", "tabpfn_v3", "minipfn_v1")


def table_recipes() -> tuple[ModuleType, ...]:
    """`TABLE_RECIPES` 모듈 (쓸 때 불러온다. `import minipfn`이 추론 스택까지 불러오지 않도록)."""
    return tuple(importlib.import_module(f"minipfn.recipes.{name}") for name in TABLE_RECIPES)


def recipe_for_model_type(model_type: str) -> ModuleType:
    """그 모델 종류(`minipfn.models.MODEL_TYPES`)를 쓰는 recipe. 예전 이름도 받는다. 같은 모델
    종류를 쓰는 recipe가 여럿이면 `TABLE_RECIPES`에 먼저 적은 것(column_row_transformer는
    `tabpfn_v3`). 만든 모델의 recipe는 `recipe_for_model`."""
    return _recipes_for_model_type(model_type)[0]


def _recipes_for_model_type(model_type: str) -> list[ModuleType]:
    from minipfn.models import canonical_model_type

    name = canonical_model_type(model_type)
    found = [recipe for recipe in table_recipes() if recipe.MODEL_TYPE == name]
    if not found:
        raise ValueError(f"{model_type} 모델을 쓰는 recipe가 없다")
    return found


def recipe_for_model(model: Any, task: str) -> ModuleType:
    """만든 모델의 recipe: 그 모델 종류를 쓰는 recipe 가운데 구조(`model_architecture`)를 알아보는
    것. 어느 것도 알아보지 못하면 그 모델 종류의 기본 recipe(`recipe_for_model_type`). 예:
    열·행 단계를 번갈아 쌓은 column_row_transformer는 `minipfn_v1`, 그 밖은 `tabpfn_v3`."""
    found = _recipes_for_model_type(getattr(model, "model_type", None) or "")
    for recipe in found:
        if recipe.model_architecture(model, task) is not None:
            return recipe
    return found[0]


def recipe_for_architecture(arch: str) -> ModuleType:
    """그 구조 이름(`ARCHITECTURES`의 키: v2, v2.5, v3, v3.5, minipfn-v1)이 있는 recipe."""
    for recipe in table_recipes():
        if arch in recipe.ARCHITECTURES:
            return recipe
    raise ValueError(f"구조 이름 {arch}는 {architectures()} 가운데 하나여야 한다")


def architectures() -> tuple[str, ...]:
    """모든 recipe의 구조 이름."""
    return tuple(name for recipe in table_recipes() for name in recipe.ARCHITECTURES)


def architecture_name(config: Any) -> str | None:
    """학습 설정의 구조 이름 (어느 recipe의 `ARCHITECTURES`와도 맞지 않으면 None)."""
    for recipe in table_recipes():
        if (name := recipe.architecture_name(config)) is not None:
            return name
    return None


def default_inference(model: Any, task: str) -> dict:
    """모델의 추론 설정 `{"n_estimators", "member_kinds"}`, TabPFN-3·3.5 recipe는 여기에
    `"max_features_per_member"`·`"feature_subsampling"`(멤버당 열 상한, 3.5는 768·balanced)도
    더한다.
    모두 `EnsembleClassifier`·`EnsembleRegressor`의 인자다. 모델의 recipe(`recipe_for_model`)가
    정한다. 어느 recipe에도 없는 모델 종류(registry에 더한 모델)와 종류가 없는 모델은 v2의 것
    (`tabpfn_v2`)."""
    try:
        recipe = recipe_for_model(model, task)
    except ValueError:
        recipe = importlib.import_module("minipfn.recipes.tabpfn_v2")
    return recipe.default_inference(model, task)


def supported_size(model: Any, task: str) -> dict:
    """모델의 논문 권장 크기 `{"max_train_rows", "max_features"}` (recipe의 `SUPPORTED_SIZE`).
    구조를 모르면 그 모델 종류의 기본 recipe의 첫 구조(v2, v3)의 것."""
    recipe = recipe_for_model(model, task)
    arch = recipe.model_architecture(model, task) or next(iter(recipe.ARCHITECTURES))
    return dict(recipe.SUPPORTED_SIZE[arch])


def member_kind_presets() -> dict[str, tuple[str, ...]]:
    """이름으로 고르는 멤버 종류 묶음 (모든 recipe의 `MEMBER_KIND_PRESETS`를 합친 것)."""
    presets: dict[str, tuple[str, ...]] = {}
    for recipe in table_recipes():
        for name, kinds in recipe.MEMBER_KIND_PRESETS.items():
            if presets.get(name, kinds) != kinds:
                raise ValueError(f"멤버 종류 묶음 {name}이 recipe마다 다르다")
            presets[name] = kinds
    return presets


def resolve_member_kinds(value: str | tuple[str, ...] | list[str]) -> tuple[str, ...]:
    """멤버 전처리 종류. 묶음 이름(`member_kind_presets`: v2, v2.5, v3, v3.5), 쉼표로 구분한 문자열,
    종류 목록을 받는다."""
    from minipfn.inference.ensemble import resolve_member_kinds as resolve

    return resolve(value, member_kind_presets())
