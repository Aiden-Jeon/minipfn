import json

import pytest
import torch

from minipfn.tabpfn.hub import CONFIG_NAME, WEIGHTS_NAME, from_pretrained, save_pretrained
from minipfn.tabpfn.train import TabPFNConfig, build_tabpfn_model


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
