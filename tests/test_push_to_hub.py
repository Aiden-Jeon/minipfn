"""hub/push_to_hub.py·hub/sync_hub_card.py 테스트 (MLflow·Hub 서버 없이)."""

import hashlib
import importlib.util
import json
import re
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

# Hub main에 올라가 있는 model.safetensors의 SHA-256
V1_WEIGHTS = "0a13d4b61510cbeaac035a731fdb751d1a8b1860fdbeda9411d7abf30e465309"
V2_WEIGHTS = "8642571d25045d0aa5cc70e191cbfa61f14a08e04bf8e4262fd757747e4d019b"
CARDS = sorted(push.HUB_DIR.glob("*/README.md"))


@pytest.fixture
def no_download(monkeypatch):
    # 카드를 확인하기 전에 큰 artifact를 받으면 안 된다.
    def fail(**kwargs):
        raise AssertionError("카드 확인 전에 artifact를 받으면 안 된다")

    monkeypatch.setattr(push.mlflow.artifacts, "download_artifacts", fail)


def _card(tmp_path, sha256="0" * 64):
    card = tmp_path / "card" / "README.md"
    card.parent.mkdir(exist_ok=True)
    card.write_text(f"Figures come from weights `{sync_hub_card.weights_id(sha256)}`.\n")
    return card


def _v2_weights(tmp_path, model, borders=None):
    """push_to_hub가 그 모델을 바꿔 만들 model.safetensors의 SHA-256."""
    from minipfn.hub import save_pretrained

    out = save_pretrained(model, borders, tmp_path / "expected")
    return sync_hub_card.file_sha256(out / "model.safetensors")


def test_default_model_card_follows_repo_name():
    for name in ("minipfn-tabpfn-v1", "minipfn-tabpfn-v2-classifier"):
        card = push.resolve_model_card(f"aiden-jeon/{name}", None)
        assert card == push.HUB_DIR / name / "README.md"
        assert card.is_file()


def test_cards_name_the_weights_they_describe():
    v1 = (push.HUB_DIR / "minipfn-tabpfn-v1" / "README.md").read_text()
    v2 = (push.HUB_DIR / "minipfn-tabpfn-v2-classifier" / "README.md").read_text()
    assert sync_hub_card.names_weights(v1, V1_WEIGHTS)
    assert sync_hub_card.names_weights(v2, V2_WEIGHTS)
    assert not sync_hub_card.names_weights(v1, V2_WEIGHTS)


@pytest.mark.parametrize("card", CARDS, ids=lambda p: p.parent.name)
def test_cards_have_no_mlflow_run_id(card):
    # 카드에는 학습 run id(MLflow, 32자리 hex)를 적지 않는다 (#61)
    text = card.read_text()
    assert "runs:/" not in text
    assert not re.search(r"(?<![0-9a-f])[0-9a-f]{32}(?![0-9a-f])", text)


def test_weights_id_is_the_backticked_sha256_prefix(tmp_path):
    path = tmp_path / "w.bin"
    path.write_bytes(b"weights")
    sha = sync_hub_card.file_sha256(path)
    assert sha == hashlib.sha256(b"weights").hexdigest()
    assert sync_hub_card.weights_id(sha) == f"sha256:{sha[:12]}"
    assert sync_hub_card.names_weights(f"from `sha256:{sha[:12]}`.", sha)
    # 백틱 없이 적거나 앞자리가 모자라면 그 가중치를 적은 것이 아니다
    assert not sync_hub_card.names_weights(f"from sha256:{sha[:12]}.", sha)
    assert not sync_hub_card.names_weights(f"from `sha256:{sha[:11]}`.", sha)


def test_repo_without_card_fails_before_download(tmp_path, no_download):
    with pytest.raises(SystemExit, match="모델 카드가 없습니다"):
        push.main(["--paper", "tabpfn", "--checkpoint", "x.pt", "--repo-id", "a/missing"])
    with pytest.raises(SystemExit, match="모델 카드가 없습니다"):
        push.main(
            ["--paper", "tabpfn", "--checkpoint", "x.pt", "--repo-id", "a/b"]
            + ["--model-card", str(tmp_path / "typo.md")]
        )


def test_weights_not_on_card_fail_with_the_id_to_write(tmp_path):
    # 카드는 다른 체크포인트의 그림과 수치를 적고 있다. 오류에 이 가중치의 id를 알려 준다
    from minipfn.models import save_checkpoint
    from minipfn.models.cell_transformer import CellTransformer
    from minipfn.training.config import TrainConfig

    model = CellTransformer(TrainConfig.smoke("classification").model_config()).eval()
    checkpoint = str(tmp_path / "model.pt")
    save_checkpoint(checkpoint, model, None)
    expected = sync_hub_card.weights_id(_v2_weights(tmp_path, model))
    out = tmp_path / "out"
    with pytest.raises(SystemExit, match=f"모델 카드에 가중치 id `{expected}`가 없습니다"):
        push.main(
            ["--paper", "tabpfn_v2", "--task", "classification", "--checkpoint", checkpoint]
            + ["--repo-id", "a/b", "--model-card", str(_card(tmp_path)), "--out-dir", str(out)]
            + ["--dry-run"]
        )
    assert not (out / "upload" / "README.md").exists()


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


def test_v1_dry_run_writes_notice_and_no_source(tmp_path):
    from minipfn.hub.tabpfn_v1 import from_pretrained, save_pretrained
    from minipfn.recipes.tabpfn_v1 import TabPFNConfig, build_tabpfn_model, save_tabpfn

    config = TabPFNConfig.smoke()
    torch.manual_seed(0)
    model = build_tabpfn_model(config)
    checkpoint = str(tmp_path / "tabpfn_final.pt")
    save_tabpfn(model, config, checkpoint)
    expected = save_pretrained(model, config, tmp_path / "expected") / "model.safetensors"
    card = _card(tmp_path, sync_hub_card.file_sha256(expected))
    out = tmp_path / "out"
    push.main(
        ["--paper", "tabpfn", "--checkpoint", checkpoint, "--repo-id", "a/b"]
        + ["--model-card", str(card), "--out-dir", str(out), "--dry-run"]
    )
    upload = out / "upload"
    files = {p.name for p in upload.iterdir()}
    assert files == {"model.safetensors", "config.json", "NOTICE", "README.md"}
    # 예전에 설치한 from_pretrained도 읽도록 config.json에는 TabPFNConfig 필드만 둔다
    assert "source" not in json.loads((upload / "config.json").read_text())
    _, loaded_config = from_pretrained(upload)
    assert loaded_config == config


def test_v1_training_checkpoint_without_config_is_rejected(tmp_path):
    checkpoint = str(tmp_path / "step_0000010.pt")
    torch.save({"model": {}, "step": 10}, checkpoint)
    card = _card(tmp_path)
    with pytest.raises(SystemExit, match="모델 설정이 없습니다"):
        push.main(
            ["--paper", "tabpfn", "--checkpoint", checkpoint, "--repo-id", "a/b"]
            + ["--model-card", str(card), "--out-dir", str(tmp_path / "out"), "--dry-run"]
        )


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_v2_dry_run_writes_inference_but_no_source(tmp_path, task):
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
    card = _card(tmp_path, _v2_weights(tmp_path, model, borders))
    out = tmp_path / "out"
    push.main(
        ["--paper", "tabpfn_v2", "--task", task, "--checkpoint", checkpoint, "--repo-id", "a/b"]
        + ["--model-card", str(card), "--out-dir", str(out), "--dry-run"]
    )
    upload = out / "upload"
    assert {p.name for p in upload.iterdir()} == {"model.safetensors", "config.json", "README.md"}
    config = json.loads((upload / "config.json").read_text())
    assert "source" not in config
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
    card = _card(tmp_path, _v2_weights(tmp_path, model))
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
    wid = sync_hub_card.weights_id(_v2_weights(tmp_path, model))
    card = _card(tmp_path)
    card.write_text(
        "---\nlicense: apache-2.0\n---\n\n# Card\n\n![x](images/x.png)\n"
        + f"| `step-100k` | `{wid}` |\n"
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
    assert f"`{wid}`" in readme and 'revision="step-100k"' in readme
    assert checkpoint not in readme

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


def test_revision_must_sit_on_the_same_card_line_as_its_weights(tmp_path, no_download):
    main, step = "a" * 64, "b" * 64
    card = tmp_path / "README.md"
    card.write_text(
        f"Main `{sync_hub_card.weights_id(main)}`.\n"
        f"| `step-100k` | `{sync_hub_card.weights_id(step)}` |\n"
    )
    # 카드에 없는 branch는 체크포인트를 받기 전에 막는다
    base = ["--paper", "tabpfn_v2", "--task", "classification", "--repo-id", "a/b"]
    base += ["--model-card", str(card), "--dry-run", "--checkpoint", "runs:/a/x.pt"]
    with pytest.raises(SystemExit, match="적은 줄이 없습니다"):
        push.main(base + ["--revision", "step-200k"])
    push.check_card_revision(card, "step-100k")
    push.check_card_revision(card, "step-100k", step)
    with pytest.raises(SystemExit, match="함께 적은 줄"):
        push.check_card_revision(card, "step-100k", main)
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


class _PathsApi:
    """`get_paths_info`만 흉내 낸 HfApi. `sha256`이 None이면 repo에 가중치가 없다."""

    def __init__(self, sha256, exists=True):
        self.sha256, self.exists, self.uploads = sha256, exists, []

    def repo_exists(self, repo_id):
        return self.exists

    def get_paths_info(self, repo_id, paths, revision=None):
        class Lfs:
            sha256 = self.sha256

        class Info:
            path = "model.safetensors"
            lfs = Lfs()

        return [] if self.sha256 is None else [Info()]

    def upload_folder(self, **kwargs):
        self.uploads.append(kwargs)

        class Commit:
            commit_url = "https://example/commit"

        return Commit()


def test_hub_weights_sha256_reads_the_lfs_hash():
    assert sync_hub_card.hub_weights_sha256(_PathsApi("c" * 64), "a/b") == "c" * 64
    assert sync_hub_card.hub_weights_sha256(_PathsApi(None), "a/b") is None


def test_sync_needs_the_hub_weights_on_the_card(tmp_path, monkeypatch):
    card = _card(tmp_path, "c" * 64)
    argv = ["--card-dir", str(card.parent), "--repo-id", "a/b"]
    api = _PathsApi("d" * 64)
    monkeypatch.setattr(sync_hub_card, "HfApi", lambda: api)
    with pytest.raises(SystemExit, match="sha256:dddddddddddd"):
        sync_hub_card.main(argv)
    api.sha256 = "c" * 64
    sync_hub_card.main(argv)
    assert len(api.uploads) == 1
    # 가중치가 아직 없는 repo는 확인할 것이 없다
    api.sha256 = None
    sync_hub_card.main(argv)
    assert len(api.uploads) == 2


def test_v3_inference_comes_from_its_recipe(tmp_path):
    # TabPFN-3·3.5 모델은 v2 recipe가 아니라 자기 recipe의 추론 설정을 적는다 (3.5: standard 8개)
    from minipfn.models import save_checkpoint
    from minipfn.recipes import tabpfn_v3
    from minipfn.training.config import TrainConfig

    config = tabpfn_v3.with_architecture(TrainConfig.smoke(), "v3.5", keep_size=True)
    checkpoint = str(tmp_path / "model.pt")
    m = config.build_model().eval()
    save_checkpoint(checkpoint, m, None)
    out = tmp_path / "out"
    push.main(
        ["--paper", "tabpfn_v2", "--task", "classification", "--checkpoint", checkpoint]
        + ["--repo-id", "a/b", "--model-card", str(_card(tmp_path, _v2_weights(tmp_path, m)))]
        + ["--out-dir", str(out), "--dry-run"]
    )
    config = json.loads((out / "upload" / "config.json").read_text())
    assert config["model_type"] == "column_row_transformer"
    assert config["inference"] == {"n_estimators": 8, "member_kinds": ["standard"]}
