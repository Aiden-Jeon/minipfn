"""hub/push_to_hub.py·hub/sync_hub_card.py 테스트 (MLflow·Hub 서버 없이)."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

HUB = Path(__file__).resolve().parents[1] / "hub"
# 스크립트로 돌 때처럼 hub/를 sys.path에 넣어야 push_to_hub가 sync_hub_card를 import한다.
sys.path.insert(0, str(HUB))
spec = importlib.util.spec_from_file_location("push_to_hub", HUB / "push_to_hub.py")
push = importlib.util.module_from_spec(spec)
spec.loader.exec_module(push)
import sync_hub_card  # noqa: E402

V1_CHECKPOINT = "runs:/977614dce2104e44915f81b02705a178/model/tabpfn_final.pt"
V2_CHECKPOINT = "runs:/a1df9e34774d49e7869eb381bf65509b/checkpoints/step_0500000.pt"


@pytest.fixture
def no_download(monkeypatch):
    # 카드를 확인하기 전에 큰 artifact를 받으면 안 된다.
    def fail(**kwargs):
        raise AssertionError("카드 확인 전에 artifact를 받으면 안 된다")

    monkeypatch.setattr(push.mlflow.artifacts, "download_artifacts", fail)


def _card(tmp_path, checkpoint):
    card = tmp_path / "card" / "README.md"
    card.parent.mkdir()
    card.write_text(f"Figures come from checkpoint `{checkpoint}`.\n")
    return card


def test_default_model_card_follows_repo_name():
    for name in ("minipfn-tabpfn-v1", "minipfn-tabpfn-v2-classifier"):
        card = push.resolve_model_card(f"aiden-jeon/{name}", None)
        assert card == push.HUB_DIR / name / "README.md"
        assert card.is_file()


def test_cards_name_the_checkpoint_they_describe():
    v1 = push.HUB_DIR / "minipfn-tabpfn-v1" / "README.md"
    v2 = push.HUB_DIR / "minipfn-tabpfn-v2-classifier" / "README.md"
    assert sync_hub_card.names_checkpoint(v1, V1_CHECKPOINT)
    assert sync_hub_card.names_checkpoint(v2, V2_CHECKPOINT)


def test_short_path_inside_the_card_uri_is_not_a_match():
    # 카드의 runs:/.../tabpfn_final.pt 안에 든 짧은 경로로 다른 run의 가중치가 통과하면 안 된다.
    v1 = push.HUB_DIR / "minipfn-tabpfn-v1" / "README.md"
    assert not sync_hub_card.names_checkpoint(v1, "tabpfn_final.pt")
    assert not sync_hub_card.names_checkpoint(v1, "model/tabpfn_final.pt")


def test_repo_without_card_fails_before_download(tmp_path, no_download):
    with pytest.raises(SystemExit, match="모델 카드가 없습니다"):
        push.main(["--paper", "tabpfn", "--checkpoint", "x.pt", "--repo-id", "a/missing"])
    with pytest.raises(SystemExit, match="모델 카드가 없습니다"):
        push.main(
            ["--paper", "tabpfn", "--checkpoint", "x.pt", "--repo-id", "a/b"]
            + ["--model-card", str(tmp_path / "typo.md")]
        )


@pytest.mark.parametrize(
    "argv",
    [
        ["--paper", "tabpfn", "--repo-id", "aiden-jeon/minipfn-tabpfn-v1"],
        ["--paper", "tabpfn_v2", "--task", "classification"]
        + ["--repo-id", "aiden-jeon/minipfn-tabpfn-v2-classifier"],
    ],
)
@pytest.mark.parametrize("checkpoint", ["runs:/other/model/tabpfn_final.pt", "tabpfn_final.pt"])
def test_checkpoint_not_on_card_fails_before_download(argv, checkpoint, no_download):
    # 카드는 다른 체크포인트의 그림과 수치를 적고 있다.
    with pytest.raises(SystemExit, match="모델 카드에"):
        push.main([*argv, "--checkpoint", checkpoint, "--dry-run"])


@pytest.mark.parametrize(
    "extra",
    [
        ["--paper", "tabpfn_v2"],  # v2는 --task가 필요하다
        ["--paper", "tabpfn", "--task", "classification"],
    ],
)
def test_paper_specific_flags(extra):
    with pytest.raises(SystemExit):
        push.parse_args(["--checkpoint", "x.pt", "--repo-id", "a/b", *extra])


def test_v1_dry_run_writes_source_json_and_notice(tmp_path):
    from minipfn.tabpfn.hub import from_pretrained
    from minipfn.tabpfn.train import TabPFNConfig, build_tabpfn_model, save_tabpfn

    config = TabPFNConfig.smoke()
    torch.manual_seed(0)
    model = build_tabpfn_model(config)
    checkpoint = str(tmp_path / "tabpfn_final.pt")
    save_tabpfn(model, config, checkpoint)
    card = _card(tmp_path, checkpoint)
    out = tmp_path / "out"
    push.main(
        ["--paper", "tabpfn", "--checkpoint", checkpoint, "--repo-id", "a/b"]
        + ["--model-card", str(card), "--out-dir", str(out), "--dry-run"]
    )
    upload = out / "upload"
    files = {p.name for p in upload.iterdir()}
    assert files == {"model.safetensors", "config.json", "source.json", "NOTICE", "README.md"}
    assert json.loads((upload / "source.json").read_text()) == {"source": checkpoint}
    # 예전에 설치한 from_pretrained도 읽도록 config.json에는 TabPFNConfig 필드만 둔다
    assert "source" not in json.loads((upload / "config.json").read_text())
    _, loaded_config = from_pretrained(upload)
    assert loaded_config == config


def test_v1_training_checkpoint_without_config_is_rejected(tmp_path):
    checkpoint = str(tmp_path / "step_0000010.pt")
    torch.save({"model": {}, "step": 10}, checkpoint)
    card = _card(tmp_path, checkpoint)
    with pytest.raises(SystemExit, match="모델 설정이 없습니다"):
        push.main(
            ["--paper", "tabpfn", "--checkpoint", checkpoint, "--repo-id", "a/b"]
            + ["--model-card", str(card), "--out-dir", str(tmp_path / "out"), "--dry-run"]
        )


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_v2_dry_run_writes_source_in_config(tmp_path, task):
    from minipfn.tabpfn_v2.hub import from_pretrained
    from minipfn.tabpfn_v2.model import TabPFNv2, save_checkpoint
    from minipfn.tabpfn_v2.train import TabPFNv2TrainConfig

    torch.manual_seed(0)
    model = TabPFNv2(TabPFNv2TrainConfig.smoke(task).model_config()).eval()
    borders = None
    if task == "regression":
        borders = torch.linspace(-3, 3, model.config.n_out + 1)
    checkpoint = str(tmp_path / "model.pt")
    save_checkpoint(checkpoint, model, borders)
    card = _card(tmp_path, checkpoint)
    out = tmp_path / "out"
    push.main(
        ["--paper", "tabpfn_v2", "--task", task, "--checkpoint", checkpoint, "--repo-id", "a/b"]
        + ["--model-card", str(card), "--out-dir", str(out), "--dry-run"]
    )
    upload = out / "upload"
    assert {p.name for p in upload.iterdir()} == {"model.safetensors", "config.json", "README.md"}
    assert json.loads((upload / "config.json").read_text())["source"] == checkpoint
    _, loaded_borders = from_pretrained(upload)
    assert (loaded_borders is None) == (borders is None)


class _Info:
    def __init__(self, private):
        self.private = private


class _Api:
    def __init__(self, exists, private=False):
        self.exists, self.private = exists, private

    def repo_exists(self, repo_id):
        return self.exists

    def model_info(self, repo_id):
        return _Info(self.private)


def test_private_push_to_an_existing_public_repo_is_refused():
    with pytest.raises(SystemExit, match="이미 공개 repo"):
        push.check_visibility(_Api(exists=True, private=False), "a/b", private=True)
    push.check_visibility(_Api(exists=True, private=True), "a/b", private=True)
    push.check_visibility(_Api(exists=False), "a/b", private=True)
    push.check_visibility(_Api(exists=True, private=False), "a/b", private=False)


class _SourceApi:
    def __init__(self, tmp_path, files):
        self.tmp_path, self.files = tmp_path, files

    def file_exists(self, repo_id, name):
        return name in self.files

    def hf_hub_download(self, repo_id, name):
        path = self.tmp_path / name
        path.write_text(json.dumps(self.files[name]))
        return str(path)


def test_hub_source_reads_v2_config_or_v1_source_json(tmp_path):
    v2 = _SourceApi(tmp_path, {"config.json": {"task": "classification", "source": "runs:/a/x"}})
    assert sync_hub_card.hub_source(v2, "a/b") == "runs:/a/x"
    v1 = _SourceApi(tmp_path, {"config.json": {"emsize": 512}, "source.json": {"source": "s"}})
    assert sync_hub_card.hub_source(v1, "a/b") == "s"
    assert sync_hub_card.hub_source(_SourceApi(tmp_path, {"config.json": {}}), "a/b") is None
