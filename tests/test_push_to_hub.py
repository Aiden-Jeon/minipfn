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

    import mlflow

    monkeypatch.setattr(mlflow.artifacts, "download_artifacts", fail)


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
    assert sync_hub_card.names_main_weights(v1, V1_WEIGHTS)
    assert sync_hub_card.names_main_weights(v2, V2_WEIGHTS)
    assert not sync_hub_card.names_main_weights(v1, V2_WEIGHTS)


def test_branch_rows_do_not_pass_for_main_and_body_lines_do_not_pass_for_branches():
    # main 카드의 표에는 모든 branch의 id가 있다. 중간 step 가중치가 main으로 통과하면 안 된다
    v2 = (push.HUB_DIR / "minipfn-tabpfn-v2-classifier" / "README.md").read_text()
    step_100k = sync_hub_card.branch_rows(v2)["step-100k"][0].split("`sha256:")[1][:12]
    assert not sync_hub_card.names_main_weights(v2, step_100k + "0" * 52)
    assert sync_hub_card.names_branch_weights(v2, "step-100k", step_100k + "0" * 52)
    # 2.5 카드 본문은 `step-500k`와 main id를 한 줄에 적는다. main 가중치가 step-500k로 통과하면
    # 안 된다
    v25 = (push.HUB_DIR / "minipfn-tabpfn-v2-5-classifier" / "README.md").read_text()
    main = "833059512de597c4edf6509e99a4d10765b2c8331fb1e621f8d66f39ad61a137"
    assert any(
        "`step-500k`" in line and "`sha256:833059512de5`" in line for line in v25.split("\n")
    )
    assert sync_hub_card.names_main_weights(v25, main)
    assert not sync_hub_card.names_branch_weights(v25, "step-500k", main)
    assert sync_hub_card.names_branch_weights(v25, "main", main)


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
    with pytest.raises(SystemExit, match=f"main 가중치로 id `{expected}`가 없습니다"):
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


def test_revision_needs_its_weights_on_its_table_row(tmp_path, no_download):
    main, step, step_200k = "a" * 64, "b" * 64, "e" * 64
    card = tmp_path / "README.md"
    card.write_text(
        f"Main `{sync_hub_card.weights_id(main)}` continued `step-200k`"
        f" (`{sync_hub_card.weights_id(step_200k)}`).\n"
        f"| `step-100k` | `{sync_hub_card.weights_id(step)}` |\n"
        f"| `main` (500k) | `{sync_hub_card.weights_id(main)}` |\n"
    )
    # 표에 행이 없는 branch는 체크포인트를 받기 전에 막는다 (본문에만 적힌 branch 포함)
    base = ["--paper", "tabpfn_v2", "--task", "classification", "--repo-id", "a/b"]
    base += ["--model-card", str(card), "--dry-run", "--checkpoint", "runs:/a/x.pt"]
    for revision in ("step-200k", "step-300k"):
        with pytest.raises(SystemExit, match=f"`{revision}` 행이 없습니다"):
            push.main(base + ["--revision", revision])
    push.check_card_revision(card, "step-100k")
    push.check_card_revision(card, "step-100k", step)
    with pytest.raises(SystemExit, match="행에 가중치 id"):
        push.check_card_revision(card, "step-100k", main)
    # main 확인은 다른 branch의 행을 치지 않는다
    push.check_card_weights(card, main, "tabpfn_v2")
    with pytest.raises(SystemExit, match="main 가중치로"):
        push.check_card_weights(card, step, "tabpfn_v2")
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
    """`get_paths_info`·`list_repo_refs`만 흉내 낸 HfApi. `sha256`은 main 가중치(None이면 없음),
    `branches`는 branch마다 가중치의 SHA-256. `lfs=False`면 LFS가 아닌 일반 파일로 올라가 있다."""

    def __init__(self, sha256, exists=True, branches=None, lfs=True):
        self.sha256, self.exists, self.uploads = sha256, exists, []
        self.branches, self.lfs = dict(branches or {}), lfs

    def repo_exists(self, repo_id):
        return self.exists

    def list_repo_refs(self, repo_id):
        class Ref:
            def __init__(self, name):
                self.name = name

        class Refs:
            branches = [Ref(name) for name in ["main", *self.branches]]

        return Refs()

    def get_paths_info(self, repo_id, paths, revision=None):
        sha256 = self.sha256 if revision in (None, "main") else self.branches[revision]

        class Lfs:
            pass

        Lfs.sha256 = sha256

        class Info:
            path = "model.safetensors"
            lfs = Lfs() if self.lfs else None

        return [] if sha256 is None else [Info()]

    def upload_folder(self, **kwargs):
        self.uploads.append(kwargs)

        class Commit:
            commit_url = "https://example/commit"

        return Commit()


def test_hub_weights_sha256_reads_the_lfs_hash():
    assert sync_hub_card.hub_weights_sha256(_PathsApi("c" * 64), "a/b") == "c" * 64
    assert sync_hub_card.hub_weights_sha256(_PathsApi(None), "a/b") is None
    api = _PathsApi("c" * 64, branches={"step-100k": "d" * 64})
    assert sync_hub_card.hub_weights_sha256(api, "a/b", "step-100k") == "d" * 64
    # 가중치가 있는데 LFS 정보가 없으면 확인 없이 넘어가지 않는다
    with pytest.raises(SystemExit, match="LFS SHA-256이 없어"):
        sync_hub_card.hub_weights_sha256(_PathsApi("c" * 64, lfs=False), "a/b")


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


def test_sync_checks_branch_rows_against_hub_branches(tmp_path, monkeypatch):
    main, step = "c" * 64, "d" * 64
    card = _card(tmp_path, main)
    card.write_text(
        card.read_text()
        + f"| `step-100k` | `{sync_hub_card.weights_id(step)}` | 0.9 |\n"
        + f"| `step-200k` | `{sync_hub_card.weights_id('e' * 64)}` | 0.9 |\n"
        + f"| `main` (500k) | `{sync_hub_card.weights_id(main)}` | 0.9 |\n"
    )
    argv = ["--card-dir", str(card.parent), "--repo-id", "a/b", "--dry-run"]
    # step-200k는 Hub에 아직 없어 건너뛴다 (카드에 행을 먼저 넣고 나중에 올린다)
    api = _PathsApi(main, branches={"step-100k": step})
    monkeypatch.setattr(sync_hub_card, "HfApi", lambda: api)
    sync_hub_card.main(argv)
    assert sync_hub_card.check_branch_rows(api, "a/b", card.read_text()) == ["step-100k"]
    # 카드의 step-100k 행이 Hub의 그 branch와 다르다 (오타, 다른 step의 가중치)
    api.branches["step-100k"] = "f" * 64
    with pytest.raises(SystemExit, match="`step-100k` 행이 Hub step-100k branch"):
        sync_hub_card.main(argv)


def test_runs_checkpoint_is_downloaded_once_per_out_dir(tmp_path, monkeypatch):
    # 카드에 id가 없어 멈춘 뒤 카드를 고쳐 다시 돌려도 큰 checkpoint를 또 받지 않는다
    import mlflow

    from minipfn.hub.tabpfn_v1 import save_pretrained
    from minipfn.recipes.tabpfn_v1 import TabPFNConfig, build_tabpfn_model, save_tabpfn

    config = TabPFNConfig.smoke()
    torch.manual_seed(0)
    model = build_tabpfn_model(config)
    source = tmp_path / "remote" / "tabpfn_final.pt"
    source.parent.mkdir()
    save_tabpfn(model, config, str(source))
    calls = []

    def download(artifact_uri, dst_path):
        calls.append(artifact_uri)
        dst = Path(dst_path) / source.name
        dst.write_bytes(source.read_bytes())
        return str(dst)

    monkeypatch.setattr(mlflow.artifacts, "download_artifacts", download)
    card = _card(tmp_path)
    argv = ["--paper", "tabpfn", "--checkpoint", "runs:/r/model/tabpfn_final.pt"]
    argv += ["--repo-id", "a/b", "--model-card", str(card), "--out-dir", str(tmp_path / "out")]
    argv += ["--dry-run"]
    with pytest.raises(SystemExit, match="main 가중치로 id"):
        push.main(argv)
    expected = save_pretrained(model, config, tmp_path / "expected") / "model.safetensors"
    card.write_text(f"weights `{sync_hub_card.weights_id(sync_hub_card.file_sha256(expected))}`\n")
    push.main(argv)
    assert calls == ["runs:/r/model/tabpfn_final.pt"]
    # 다른 체크포인트는 새로 받는다 (이 가짜 download는 같은 파일을 주므로 카드 확인은 통과한다)
    push.main([*argv[:3], "runs:/other/model/tabpfn_final.pt", *argv[4:]])
    assert calls == ["runs:/r/model/tabpfn_final.pt", "runs:/other/model/tabpfn_final.pt"]


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
