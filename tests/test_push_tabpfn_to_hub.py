import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "push_tabpfn_to_hub.py"
spec = importlib.util.spec_from_file_location("push_tabpfn_to_hub", SCRIPT)
push = importlib.util.module_from_spec(spec)
spec.loader.exec_module(push)


def test_default_model_card_follows_repo_name():
    card = push.resolve_model_card("aiden-jeon/minipfn-tabpfn-v1", None)
    assert card == push.HUB_DIR / "minipfn-tabpfn-v1" / "README.md"


def test_repo_without_card_fails_before_download(tmp_path, monkeypatch):
    # 카드가 없는 repo에 다른 모델의 카드를 붙이지 않고, artifact를 받기 전에 멈춘다.
    def no_download(**kwargs):
        raise AssertionError("카드 확인 전에 artifact를 받으면 안 된다")

    monkeypatch.setattr(push.mlflow.artifacts, "download_artifacts", no_download)
    with pytest.raises(SystemExit, match="모델 카드가 없습니다"):
        push.main(["--run-id", "x", "--repo-id", "someone/minipfn-tabpfn-v2"])
    with pytest.raises(SystemExit, match="모델 카드가 없습니다"):
        push.main(["--run-id", "x", "--repo-id", "a/b", "--model-card", str(tmp_path / "typo.md")])
