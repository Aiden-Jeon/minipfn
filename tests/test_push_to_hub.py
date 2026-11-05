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
    from minipfn.hub.tabpfn_v1 import from_pretrained
    from minipfn.recipes.tabpfn_v1 import TabPFNConfig, build_tabpfn_model, save_tabpfn

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
    from minipfn.hub import from_pretrained
    from minipfn.models import save_checkpoint
    from minipfn.models.cell_transformer import CellTransformer
    from minipfn.training.config import TrainConfig

    torch.manual_seed(0)
    model = CellTransformer(TrainConfig.smoke(task).model_config()).eval()
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
    config = json.loads((upload / "config.json").read_text())
    assert config["source"] == checkpoint
    # 이 weight의 추론 설정: smoke 모델은 v2 구조라 v2 멤버 종류, 멤버 수는 논문 기본값
    from minipfn.recipes.tabpfn_v2 import MEMBER_KINDS_V2, N_ESTIMATORS

    assert config["inference"] == {
        "n_estimators": N_ESTIMATORS[task],
        "member_kinds": list(MEMBER_KINDS_V2),
    }
    loaded, loaded_borders = from_pretrained(upload)
    assert (loaded_borders is None) == (borders is None)
    assert loaded.inference_config == config["inference"]


def test_v2_member_kinds_override_is_written(tmp_path):
    # 카드의 수치를 다른 멤버 종류로 냈으면 --member-kinds로 준 값을 적는다
    from minipfn.models import save_checkpoint
    from minipfn.models.cell_transformer import CellTransformer
    from minipfn.training.config import TrainConfig

    model = CellTransformer(TrainConfig.smoke("classification").model_config()).eval()
    checkpoint = str(tmp_path / "model.pt")
    save_checkpoint(checkpoint, model, None)
    card = _card(tmp_path, checkpoint)
    out = tmp_path / "out"
    args = ["--paper", "tabpfn_v2", "--task", "classification", "--checkpoint", checkpoint]
    args += ["--repo-id", "a/b", "--model-card", str(card), "--out-dir", str(out), "--dry-run"]
    push.main(args + ["--member-kinds", "v2.5"])
    config = json.loads((out / "upload" / "config.json").read_text())
    assert config["inference"]["member_kinds"] == ["quantile", "robust"]
    with pytest.raises(SystemExit):
        push.parse_args(args + ["--member-kinds", "box-cox"])
    with pytest.raises(SystemExit):
        push.parse_args(
            ["--paper", "tabpfn", "--checkpoint", checkpoint, "--repo-id", "a/b"]
            + ["--member-kinds", "v2"]
        )


def test_v2_revision_dry_run_writes_a_short_card(tmp_path, monkeypatch):
    import minipfn.hub
    from minipfn.models import save_checkpoint
    from minipfn.models.cell_transformer import CellTransformer
    from minipfn.training.config import TrainConfig

    model = CellTransformer(TrainConfig.smoke("classification").model_config()).eval()
    checkpoint = str(tmp_path / "model.pt")
    save_checkpoint(checkpoint, model, None)
    card = _card(tmp_path, checkpoint)
    card.write_text(
        "---\nlicense: apache-2.0\n---\n\n# Card\n\n![x](images/x.png)\n"
        + f"| `step-100k` | `{checkpoint}` |\n"
    )
    (card.parent / "images").mkdir()
    (card.parent / "images" / "x.png").write_bytes(b"png")
    out = tmp_path / "out"
    push.main(
        ["--paper", "tabpfn_v2", "--task", "classification", "--checkpoint", checkpoint]
        + ["--repo-id", "a/b", "--revision", "step-100k"]
        + ["--model-card", str(card), "--out-dir", str(out), "--dry-run"]
    )
    upload = out / "upload"
    # branch에는 main 카드의 그림을 올리지 않는다
    assert {p.name for p in upload.iterdir()} == {"model.safetensors", "config.json", "README.md"}
    readme = (upload / "README.md").read_text()
    assert readme.startswith("---\nlicense: apache-2.0\n---\n\n# a/b @ `step-100k`")
    assert f"`{checkpoint}`" in readme and 'revision="step-100k"' in readme

    # 카드의 예제 코드를 그대로 돌린다. Hub 대신 upload 폴더에서 그 branch의 파일을 읽는다
    calls = []

    def download(repo_id, filename, revision=None):
        calls.append((repo_id, revision))
        return str(upload / filename)

    monkeypatch.setattr(minipfn.hub, "hf_hub_download", download)
    snippet = readme.split("```python\n", 1)[1].split("```", 1)[0]
    namespace = {}
    exec(snippet, namespace)
    assert set(calls) == {("a/b", "step-100k")}
    loaded = namespace["model"]
    assert isinstance(loaded, CellTransformer) and namespace["borders"] is None
    for (name, a), b in zip(model.state_dict().items(), loaded.state_dict().values(), strict=True):
        assert torch.equal(a, b), name


def test_revision_must_sit_on_the_same_card_line_as_its_checkpoint(tmp_path, no_download):
    card = tmp_path / "README.md"
    card.write_text("Main `runs:/a/500k.pt`.\n| `step-100k` | `runs:/a/100k.pt` |\n")
    base = ["--paper", "tabpfn_v2", "--task", "classification", "--repo-id", "a/b"]
    base += ["--model-card", str(card), "--dry-run"]
    with pytest.raises(SystemExit, match="함께 적은 줄"):
        push.main(base + ["--checkpoint", "runs:/a/500k.pt", "--revision", "step-100k"])
    with pytest.raises(SystemExit, match="함께 적은 줄"):
        push.main(base + ["--checkpoint", "runs:/a/100k.pt", "--revision", "step-200k"])
    with pytest.raises(SystemExit):
        push.parse_args(
            ["--paper", "tabpfn", "--checkpoint", "x", "--repo-id", "a/b", "--revision", "s"]
        )


def test_front_matter_without_trailing_newline():
    match = push.FRONT_MATTER.match("---\nlicense: x\n---")
    assert match and match.group(0).rstrip() == "---\nlicense: x\n---"
    assert push.FRONT_MATTER.match("# no front matter\n") is None


class _RevisionApi:
    def __init__(self, exists):
        self.exists, self.branches = exists, []

    def repo_exists(self, repo_id):
        return self.exists

    def list_repo_commits(self, repo_id):
        class C:
            def __init__(self, commit_id):
                self.commit_id = commit_id

        return [C("latest"), C("first")]

    def create_branch(self, repo_id, branch, revision, exist_ok):
        self.branches.append((branch, revision, exist_ok))


def test_revision_branches_from_the_first_commit_and_needs_main():
    api = _RevisionApi(exists=True)
    push.create_revision(api, "a/b", "step-100k")
    assert api.branches == [("step-100k", "first", True)]
    with pytest.raises(SystemExit, match="main을 먼저"):
        push.create_revision(_RevisionApi(exists=False), "a/b", "step-100k")


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


def test_v3_inference_comes_from_its_recipe(tmp_path):
    # TabPFN-3·3.5 모델은 v2 recipe가 아니라 자기 recipe의 추론 설정을 적는다 (3.5: standard 8개)
    from minipfn.models import save_checkpoint
    from minipfn.recipes import tabpfn_v3
    from minipfn.training.config import TrainConfig

    config = tabpfn_v3.with_architecture(TrainConfig.smoke(), "v3.5", keep_size=True)
    checkpoint = str(tmp_path / "model.pt")
    save_checkpoint(checkpoint, config.build_model().eval(), None)
    out = tmp_path / "out"
    push.main(
        ["--paper", "tabpfn_v2", "--task", "classification", "--checkpoint", checkpoint]
        + ["--repo-id", "a/b", "--model-card", str(_card(tmp_path, checkpoint))]
        + ["--out-dir", str(out), "--dry-run"]
    )
    config = json.loads((out / "upload" / "config.json").read_text())
    assert config["model_type"] == "column_row_transformer"
    assert config["inference"] == {"n_estimators": 8, "member_kinds": ["standard"]}
