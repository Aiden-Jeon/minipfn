import json

import pytest
import torch

from minipfn.hub.tabpfn_v1 import CONFIG_NAME, WEIGHTS_NAME, from_pretrained, save_pretrained
from minipfn.recipes.tabpfn_v1 import TabPFNConfig, build_tabpfn_model


def test_save_and_load_pretrained_roundtrip(tmp_path):
    config = TabPFNConfig.smoke()
    torch.manual_seed(0)
    model = build_tabpfn_model(config)
    for p in model.parameters():
        torch.nn.init.normal_(p, std=0.2)

    save_pretrained(model, config, tmp_path)
    assert {p.name for p in tmp_path.iterdir()} == {WEIGHTS_NAME, CONFIG_NAME}
    assert json.loads((tmp_path / CONFIG_NAME).read_text())["emsize"] == config.emsize

    loaded, loaded_config = from_pretrained(tmp_path)
    assert loaded_config == config
    for (name, a), b in zip(model.state_dict().items(), loaded.state_dict().values(), strict=True):
        assert torch.equal(a, b), name


def test_from_pretrained_rejects_missing_local_paths(tmp_path):
    # 폴더가 아닌 로컬 경로는 Hub 404 대신 FileNotFoundError로 알려 준다.
    (tmp_path / "tabpfn_final.pt").write_bytes(b"")
    for path in (tmp_path / "tabpfn_final.pt", str(tmp_path / "typo"), "tabpfn_hub/uplaod/x"):
        with pytest.raises(FileNotFoundError):
            from_pretrained(path)


def test_hub_inference_imports_do_not_load_the_training_stack():
    # README·Hub 카드의 추론 예제는 mlflow와 학습 설정을 불러오지 않는다 (새 프로세스에서 확인)
    import subprocess
    import sys

    code = (
        "import sys\n"
        "import minipfn.hub.tabpfn_v1, minipfn.recipes.tabpfn_v1\n"
        "import minipfn.hub, minipfn.recipes.tabpfn_v2\n"
        "import minipfn.tabpfn.hub, minipfn.tabpfn.inference\n"
        "import minipfn.tabpfn_v2.hub, minipfn.tabpfn_v2.inference\n"
        "loaded = [m for m in ('mlflow', 'minipfn.training.config') if m in sys.modules]\n"
        "assert not loaded, loaded\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_generic_from_pretrained_reads_v1_folders(tmp_path):
    # `minipfn.from_pretrained`(= `minipfn.hub.from_pretrained`)가 v1 repo도 읽는다. v1
    # config.json은 학습 설정을 평평하게 적은 형식 그대로다 (예전 minipfn도 읽도록 Hub 파일은 바꾸지
    # 않는다)
    from minipfn import from_pretrained as top_level
    from minipfn.hub import from_pretrained as generic
    from minipfn.models.row_transformer import RowTransformer

    assert top_level is generic
    config = TabPFNConfig.smoke()
    torch.manual_seed(0)
    model = build_tabpfn_model(config)
    for p in model.parameters():
        torch.nn.init.normal_(p, std=0.2)
    save_pretrained(model, config, tmp_path)

    loaded, borders = generic(tmp_path)
    assert isinstance(loaded, RowTransformer) and borders is None and not loaded.training
    for (name, a), b in zip(model.state_dict().items(), loaded.state_dict().values(), strict=True):
        assert torch.equal(a, b), name
    x = torch.randn(2, 12, config.num_features)
    y = torch.randint(0, 2, (2, 8)).float()
    model.eval()
    with torch.no_grad():
        assert torch.equal(model(x[:, :8], y, x[:, 8:]), loaded(x[:, :8], y, x[:, 8:]))


def test_top_level_inference_names_load_lazily():
    # `from minipfn import TabPFNv1Classifier, from_pretrained`: 정의는 recipes·hub에 있고, `import
    # minipfn`만으로는 Hub·추론 코드를 불러오지 않는다 (새 프로세스에서 확인)
    import subprocess
    import sys

    code = (
        "import sys, minipfn\n"
        "lazy = ('minipfn.hub', 'minipfn.recipes.tabpfn_v1', 'minipfn.recipes.tabpfn_v2')\n"
        "assert not [m for m in lazy if m in sys.modules]\n"
        "from minipfn import TabPFNv1Classifier, TabPFNv2Classifier, TabPFNv2Regressor\n"
        "from minipfn import from_pretrained\n"
        "from minipfn.hub import from_pretrained as f\n"
        "from minipfn.recipes import tabpfn_v1 as v1, tabpfn_v2 as v2\n"
        "assert from_pretrained is f and TabPFNv1Classifier is v1.TabPFNv1Classifier\n"
        "from minipfn.tabpfn.inference import TabPFNClassifier as old_name\n"
        "assert old_name is TabPFNv1Classifier  # 예전 Hub 카드의 이름\n"
        "assert TabPFNv2Classifier is v2.TabPFNv2Classifier\n"
        "assert TabPFNv2Regressor is v2.TabPFNv2Regressor\n"
        "assert set(minipfn.__all__) <= set(dir(minipfn))\n"
        "try:\n"
        "    minipfn.no_such_name\n"
        "except AttributeError:\n"
        "    pass\n"
        "else:\n"
        "    raise AssertionError\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
