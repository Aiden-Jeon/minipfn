import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "hub" / "push_to_hub.py"
spec = importlib.util.spec_from_file_location("push_to_hub", SCRIPT)
push = importlib.util.module_from_spec(spec)
spec.loader.exec_module(push)


@pytest.fixture
def no_download(monkeypatch):
    # 카드를 확인하기 전에 큰 artifact를 받으면 안 된다.
    def fail(**kwargs):
        raise AssertionError("카드 확인 전에 artifact를 받으면 안 된다")

    monkeypatch.setattr(push.mlflow.artifacts, "download_artifacts", fail)


def test_default_model_card_follows_repo_name():
    for name in ("minipfn-tabpfn-v1", "minipfn-tabpfn-v2-classifier"):
        card = push.resolve_model_card(f"aiden-jeon/{name}", None)
        assert card == push.HUB_DIR / name / "README.md"
        assert card.is_file()


def test_repo_without_card_fails_before_download(tmp_path, no_download):
    with pytest.raises(SystemExit, match="모델 카드가 없습니다"):
        push.main(["--paper", "tabpfn", "--checkpoint", "x.pt", "--repo-id", "a/missing"])
    with pytest.raises(SystemExit, match="모델 카드가 없습니다"):
        push.main(
            ["--paper", "tabpfn", "--checkpoint", "x.pt", "--repo-id", "a/b"]
            + ["--model-card", str(tmp_path / "typo.md")]
        )


def test_checkpoint_not_on_card_fails_before_download(no_download):
    # 카드는 다른 체크포인트의 그림과 수치를 적고 있다.
    with pytest.raises(SystemExit, match="모델 카드에 runs:/other"):
        push.main(
            ["--paper", "tabpfn", "--repo-id", "aiden-jeon/minipfn-tabpfn-v1"]
            + ["--checkpoint", "runs:/other/model/tabpfn_final.pt"]
        )


def test_cards_name_the_checkpoint_they_describe():
    v1 = (push.HUB_DIR / "minipfn-tabpfn-v1" / "README.md").read_text()
    assert "runs:/977614dce2104e44915f81b02705a178/model/tabpfn_final.pt" in v1


@pytest.mark.parametrize(
    "extra",
    [
        ["--paper", "tabpfn_v2"],  # v2는 --task가 필요하다
        ["--paper", "tabpfn", "--task", "classification"],
        ["--paper", "tabpfn_v2", "--task", "regression", "--allow-non-paper"],
    ],
)
def test_paper_specific_flags(extra):
    with pytest.raises(SystemExit):
        push.parse_args(["--checkpoint", "x.pt", "--repo-id", "a/b", *extra])
