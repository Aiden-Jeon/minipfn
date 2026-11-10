"""학습한 TabPFN(v1·v2) 가중치를 Hugging Face Hub에 올린다.

예) 저장소 루트에서 (.env.local을 불러온 셸)
  MLFLOW_TRACKING_URI=databricks uv run python hub/push_to_hub.py --paper tabpfn \
    --checkpoint runs:/<학습 run_id>/model/tabpfn_final.pt \
    --repo-id aiden-jeon/minipfn-tabpfn-v1 --dry-run
  MLFLOW_TRACKING_URI=databricks uv run python hub/push_to_hub.py --paper tabpfn_v2 \
    --checkpoint runs:/<학습 run_id>/checkpoints/step_0500000.pt --task classification \
    --repo-id aiden-jeon/minipfn-tabpfn-v2-classifier --dry-run

`--paper`는 `scripts/<paper>/` 폴더 이름이다.

- `tabpfn`: `save_tabpfn`으로 저장한 `tabpfn_final.pt`(파일 또는
  `runs:/<run_id>/model/tabpfn_final.pt`). 저장소 루트의 `NOTICE`(공식 TabPFN v1 코드의 저작권
  표기)를 함께 올린다.
- `tabpfn_v2`: `minipfn.hub.load_model`이 읽는 체크포인트면 무엇이든(v2·2.5, TabPFN-3·3.5 모델
  포함). `--task`가 필요하다. 이 weight의 추론 설정(멤버 수, 멤버 전처리 종류)은 `config.json`의
  `inference`에 적는다. 추론 설정은 모델 종류의 recipe가 구조로 정하고
  (`minipfn.recipes.default_inference`: 2.5 구조면 v2.5, TabPFN-3.5면 standard 8개 등), 카드의
  수치를 다른 멤버 종류로 냈으면 `--member-kinds`로 준다.

체크포인트를 `model.safetensors` + `config.json`으로 바꾸고 모델 카드를 붙여 올린다. 모델 카드는
`--model-card`로 주거나, 주지 않으면 repo 이름으로 `hub/<name>/README.md`를 찾는다. 카드가
`](images/...)`로 링크한 그림만 카드 옆에서 복사하고, Hub의 `images/`에서 카드가 더는 쓰지 않는
파일은 지운다. `--dry-run`이면 폴더만 만들고 올리지 않는다.

공개 repo를 실수로 덮어쓰지 않도록 `--repo-id`는 기본값 없이 받는다. 카드의 그림과 수치가 다른
체크포인트의 것이 되지 않도록, 바꾼 `model.safetensors`의 가중치 id(`` `sha256:<SHA-256 앞
12자리>` ``)가 카드에 적혀 있어야 올린다(그림은 `scripts/<paper>/plot_card.py`로 다시 만든다).
id는 체크포인트를 바꿔 봐야 알 수 있으므로 처음에는 `--dry-run`으로 돌려 출력된 id를 카드에
적는다. `runs:/` 체크포인트는 `--out-dir`에 받아 두고 다음 실행에서 다시 쓰므로, 카드를 고친 뒤
다시 돌려도 또 받지 않는다. 중간 checkpoint 표의 다른 branch 행에 적힌 id는 main의 확인에 치지
않는다. 카드·Hub 파일·Hub commit 메시지에는 학습 run id(MLflow)를 남기지 않는다. 이미 있는
repo의 공개 여부는 바꾸지 않으므로, `--private`인데 repo가 공개면 올리지 않는다.

학습 중간 checkpoint는 `--revision step-100k`처럼 같은 repo의 branch로 올린다. `main`(모델 카드가
설명하는 최종 가중치)을 먼저 올려 repo가 있어야 한다. branch는 repo의 첫 commit에서 만들어
`main`의 카드와 그림을 물려받지 않고, 가중치·설정과 함께 그 step을 설명하는 짧은 README를 둔다.
이때는 `main` 카드의 한 줄(중간 checkpoint 표의 행)에 branch와 그 가중치 id가 함께 적혀
있어야 한다 — branch 이름과 다른 step의 가중치가 올라가지 않도록. v2만 지원한다.
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import torch
from huggingface_hub import HfApi

# 스크립트 폴더(hub/)가 sys.path에 들어가므로 같은 폴더의 카드 동기화 스크립트를 그대로 쓴다.
from sync_hub_card import (
    WEIGHTS_ID_HEX,
    WEIGHTS_NAME,
    branch_rows,
    card_images,
    file_sha256,
    names_branch_weights,
    names_main_weights,
    weights_id,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
HUB_DIR = Path(__file__).resolve().parent
NOTICE = REPO_ROOT / "NOTICE"
PAPERS = ("tabpfn", "tabpfn_v2")
TITLE = {"tabpfn": "TabPFN", "tabpfn_v2": "TabPFN v2"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--paper", choices=PAPERS, required=True, help="scripts/<paper>/ 이름")
    parser.add_argument("--checkpoint", required=True, help="체크포인트 (파일, runs:/ URI)")
    parser.add_argument("--repo-id", required=True, help="Hub repo id (owner/name)")
    parser.add_argument(
        "--task", choices=["classification", "regression"], help="tabpfn_v2만 (필수)"
    )
    parser.add_argument(
        "--member-kinds",
        help="tabpfn_v2만: config.json에 적을 멤버 전처리 종류 (v2, v2.5, v3, v3.5 또는 쉼표로"
        " 구분한 목록)."
        " 기본: 모델 구조로 정한다",
    )
    parser.add_argument(
        "--model-card",
        type=Path,
        help="Hub README로 올릴 모델 카드 (기본: hub/<repo 이름>/README.md)",
    )
    parser.add_argument(
        "--revision", help="중간 checkpoint를 올릴 branch (예: step-100k). 없으면 main"
    )
    parser.add_argument("--out-dir", default="hub_upload")
    parser.add_argument("--private", action="store_true", help="새로 만드는 repo를 비공개로")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.paper == "tabpfn_v2" and args.task is None:
        parser.error("--paper tabpfn_v2에는 --task가 필요합니다")
    if args.paper == "tabpfn" and args.task is not None:
        parser.error("--task는 --paper tabpfn_v2에서만 씁니다 (v1은 분류만)")
    if args.paper == "tabpfn" and args.member_kinds is not None:
        parser.error("--member-kinds는 --paper tabpfn_v2에서만 씁니다")
    if args.member_kinds is not None:
        from minipfn.recipes import resolve_member_kinds

        try:
            args.member_kinds = resolve_member_kinds(args.member_kinds)
        except ValueError as e:
            parser.error(str(e))
    if args.revision == "main":
        args.revision = None
    if args.paper == "tabpfn" and args.revision is not None:
        parser.error("--revision은 --paper tabpfn_v2에서만 씁니다")
    return args


def resolve_model_card(repo_id: str, model_card: Path | None) -> Path:
    """올릴 모델 카드를 정한다. 다른 모델의 카드가 붙지 않도록 repo 이름별 카드만 기본으로 쓴다."""
    card = model_card or HUB_DIR / repo_id.split("/")[-1] / "README.md"
    if not card.is_file():
        raise SystemExit(
            f"모델 카드가 없습니다: {card}. hub/<repo 이름>/README.md를 만들거나 --model-card로"
            " 주세요."
        )
    return card


def check_card_weights(card: Path, sha256: str, paper: str) -> None:
    """카드의 그림과 수치가 올리는 가중치의 것인지 확인한다. 중간 checkpoint 표에서 다른 branch의
    행에 적힌 id는 치지 않는다 (중간 step의 가중치가 main으로 올라가지 않도록)."""
    if not names_main_weights(card.read_text(), sha256):
        raise SystemExit(
            f"모델 카드에 main 가중치로 id `{weights_id(sha256)}`가 없습니다 (다른 branch의 행은"
            " 치지 않습니다). 이 체크포인트의 결과를 적은 카드라면 그 id를 카드에 적어 주세요."
            " 다른 체크포인트의 결과라면"
            f" scripts/{paper}/plot_card.py로 그림을 다시 만들고 카드의 수치와 id를 고쳐 주세요."
        )


def check_visibility(api: HfApi, repo_id: str, private: bool) -> None:
    """`create_repo(exist_ok=True)`는 이미 있는 repo의 공개 여부를 바꾸지 않는다.

    `--private`으로 올렸는데 실제로는 공개 repo에 올라가는 일을 막는다."""
    if private and api.repo_exists(repo_id) and not api.model_info(repo_id).private:
        raise SystemExit(
            f"{repo_id}는 이미 공개 repo라 --private으로 올려도 공개됩니다. 다른 repo id를 주거나"
            " Hub에서 먼저 비공개로 바꿔 주세요."
        )


def prepare_tabpfn(args: argparse.Namespace, out_dir: Path, upload_dir: Path):
    """v1: `tabpfn_final.pt`를 변환하고, 원본 모델과 변환한 파일에서 다시 읽은 모델을 돌려준다."""
    from minipfn.hub import download_artifact
    from minipfn.hub.tabpfn_v1 import from_pretrained, save_pretrained
    from minipfn.recipes.tabpfn_v1 import tabpfn_from_state

    path = args.checkpoint
    if path.startswith("runs:/"):
        path = download_artifact(path, out_dir / "checkpoint")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if "config" not in state:
        raise SystemExit(
            f"{args.checkpoint}에 모델 설정이 없습니다 (학습 중 checkpoint). 설정이 함께 저장된"
            " runs:/<run_id>/model/tabpfn_final.pt를 주세요."
        )
    model, config = tabpfn_from_state(state["config"], state["state_dict"])
    save_pretrained(model, config, upload_dir)
    shutil.copy(NOTICE, upload_dir / "NOTICE")
    reloaded, _ = from_pretrained(upload_dir)
    return model, reloaded


def prepare_tabpfn_v2(args: argparse.Namespace, out_dir: Path, upload_dir: Path):
    """v2: 체크포인트를 변환하고 원본과 다시 읽은 모델을 돌려준다. 회귀 구간 경계도 확인한다."""
    from minipfn.hub import from_pretrained, load_model, save_pretrained
    from minipfn.recipes import default_inference

    model, borders = load_model(args.checkpoint, args.task, out_dir / "checkpoint")
    inference = default_inference(model, args.task)
    if args.member_kinds is not None:
        inference["member_kinds"] = args.member_kinds
    inference["member_kinds"] = list(inference["member_kinds"])
    print("inference:", inference)
    save_pretrained(model, borders, upload_dir, inference=inference)
    reloaded, reloaded_borders = from_pretrained(upload_dir)
    if (borders is None) != (reloaded_borders is None) or (
        borders is not None and not torch.equal(borders.cpu(), reloaded_borders)
    ):
        raise RuntimeError("변환한 회귀 구간 경계가 원본과 다릅니다")
    if default_inference(reloaded, args.task) != default_inference(model, args.task) | {
        "member_kinds": tuple(inference["member_kinds"])
    }:
        raise RuntimeError("다시 읽은 추론 설정이 적은 것과 다릅니다")
    return model, reloaded


PREPARE = {"tabpfn": prepare_tabpfn, "tabpfn_v2": prepare_tabpfn_v2}
FRONT_MATTER = re.compile(r"\A---\r?\n.*?^---[ \t]*(?:\r?\n|\Z)", re.S | re.M)


def check_card_revision(card: Path, revision: str, sha256: str | None = None) -> None:
    """main 카드의 중간 checkpoint 표에서 branch 행(`` | `step-100k` | ``로 시작하는 줄)이 그
    가중치 id를 적고 있는지. 본문에서 branch와 다른 가중치를 함께 적은 줄은 치지 않는다 — 다른
    step의 가중치가 그 branch로 올라가지 않도록.

    `sha256`이 없으면(체크포인트를 받기 전) branch의 행이 있는지만 본다."""
    text = card.read_text()
    if revision not in branch_rows(text):
        raise SystemExit(
            f"모델 카드의 중간 checkpoint 표에 `{revision}` 행이 없습니다. 그 branch의 행"
            f" (| `{revision}` | `sha256:...` | ...)을 추가하거나 branch 이름을 확인해 주세요."
        )
    if sha256 is not None and not names_branch_weights(text, revision, sha256):
        raise SystemExit(
            f"모델 카드의 `{revision}` 행에 가중치 id `{weights_id(sha256)}`가 없습니다. 그 행에"
            " 이 id를 적거나 branch 이름·체크포인트를 확인해 주세요."
        )


def revision_card(model_card: Path, repo_id: str, revision: str, sha256: str) -> str:
    """중간 checkpoint branch의 README. main 카드의 front matter를 그대로 쓰고 본문은 짧게 둔다."""
    match = FRONT_MATTER.match(model_card.read_text())
    front = match.group(0).rstrip() + "\n\n" if match else ""
    url = f"https://huggingface.co/{repo_id}"
    return (
        f"{front}# {repo_id} @ `{revision}`\n\n"
        f"An intermediate training checkpoint, kept for comparison. The model card, benchmark"
        f" numbers for every checkpoint, and usage are on the [main branch]({url}).\n\n"
        f"Weights: `{weights_id(sha256)}` (first {WEIGHTS_ID_HEX} hex digits of the SHA-256 of"
        f" `{WEIGHTS_NAME}`).\n\n"
        "```python\n"
        "from minipfn import from_pretrained\n\n"
        f'model, borders = from_pretrained("{repo_id}", revision="{revision}")\n'
        "```\n"
    )


def create_revision(api: HfApi, repo_id: str, revision: str) -> None:
    """repo의 첫 commit에서 branch를 만든다(이미 있으면 그대로). main을 먼저 올려야 한다."""
    if not api.repo_exists(repo_id):
        raise SystemExit(f"{repo_id}가 없습니다. --revision 없이 main을 먼저 올려 주세요.")
    first = api.list_repo_commits(repo_id)[-1].commit_id
    api.create_branch(repo_id, branch=revision, revision=first, exist_ok=True)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    # 큰 artifact를 받기 전에 카드부터 확인한다. 가중치 id는 체크포인트를 바꾼 뒤에 본다.
    model_card = resolve_model_card(args.repo_id, args.model_card)
    images = card_images(model_card)
    if args.revision is not None:
        check_card_revision(model_card, args.revision)
    api = None if args.dry_run else HfApi()
    if api is not None:
        check_visibility(api, args.repo_id, args.private)
        if args.revision is not None and not api.repo_exists(args.repo_id):
            raise SystemExit(f"{args.repo_id}가 없습니다. --revision 없이 main을 먼저 올려 주세요.")

    out_dir = Path(args.out_dir)
    upload_dir = out_dir / "upload"
    # 지난 실행의 파일이 섞여 올라가지 않도록 매번 빈 폴더에서 시작한다.
    shutil.rmtree(upload_dir, ignore_errors=True)
    upload_dir.mkdir(parents=True)
    model, reloaded = PREPARE[args.paper](args, out_dir, upload_dir)
    # 올리기 전에 변환한 파일만으로 같은 가중치가 다시 만들어지는지 확인한다.
    for (name, a), b in zip(
        model.state_dict().items(), reloaded.state_dict().values(), strict=True
    ):
        if not torch.equal(a.cpu(), b.cpu()):
            raise RuntimeError(f"변환한 가중치가 원본과 다릅니다: {name}")
    sha256 = file_sha256(upload_dir / WEIGHTS_NAME)
    print("weights:", weights_id(sha256))
    if args.revision is None:
        check_card_weights(model_card, sha256, args.paper)
    else:
        check_card_revision(model_card, args.revision, sha256)
    if args.revision is None:
        shutil.copy(model_card, upload_dir / "README.md")
        for rel in images:
            (upload_dir / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(model_card.parent / rel, upload_dir / rel)
    else:
        card = revision_card(model_card, args.repo_id, args.revision, sha256)
        (upload_dir / "README.md").write_text(card)
    prepared = sorted(str(p.relative_to(upload_dir)) for p in upload_dir.rglob("*") if p.is_file())
    print("prepared:", prepared)
    if api is None:
        return

    if args.revision is None:
        api.create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    else:
        create_revision(api, args.repo_id, args.revision)
    # 같은 commit에서 다시 올리는 파일은 지우지 않으므로, 카드가 더는 쓰지 않는 그림만 지워진다.
    commit = api.upload_folder(
        repo_id=args.repo_id,
        folder_path=upload_dir,
        revision=args.revision,
        commit_message=f"Upload {TITLE[args.paper]} weights {weights_id(sha256)}",
        delete_patterns=["images/*"],
    )
    print("uploaded:", commit.commit_url)


if __name__ == "__main__":
    main()
