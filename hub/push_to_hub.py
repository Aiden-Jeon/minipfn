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
  표기)를 함께 올리고, 체크포인트는 `source.json`에 적는다(`minipfn.hub.tabpfn_v1` 참고).
- `tabpfn_v2`: `minipfn.hub.load_model`이 읽는 체크포인트면 무엇이든. `--task`가 필요하다.
  체크포인트는 `config.json`의 `source`에 적는다.

체크포인트를 `model.safetensors` + `config.json`으로 바꾸고 모델 카드를 붙여 올린다. 모델 카드는
`--model-card`로 주거나, 주지 않으면 repo 이름으로 `hub/<name>/README.md`를 찾는다. 카드가
`](images/...)`로 링크한 그림만 카드 옆에서 복사하고, Hub의 `images/`에서 카드가 더는 쓰지 않는
파일은 지운다. `--dry-run`이면 폴더만 만들고 올리지 않는다.

공개 repo를 실수로 덮어쓰지 않도록 `--repo-id`는 기본값 없이 받는다. 카드의 그림과 수치가 다른
체크포인트의 것이 되지 않도록, 카드에 `--checkpoint`가 `` `...` ``로 그대로 적혀 있어야 올린다
(그림은 `scripts/<paper>/plot_card.py`로 다시 만든다). 이미 있는 repo의 공개 여부는 바꾸지
않으므로, `--private`인데 repo가 공개면 올리지 않는다.

학습 중간 checkpoint는 `--revision step-100k`처럼 같은 repo의 branch로 올린다. `main`(모델 카드가
설명하는 최종 가중치)을 먼저 올려 repo가 있어야 한다. branch는 repo의 첫 commit에서 만들어
`main`의 카드와 그림을 물려받지 않고, 가중치·설정과 함께 그 step을 설명하는 짧은 README를 둔다.
이때는 `main` 카드의 한 줄(중간 checkpoint 표의 행)에 branch와 `--checkpoint`가 함께 적혀
있어야 한다 — branch 이름과 다른 step의 가중치가 올라가지 않도록. v2만 지원한다.
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import mlflow
import torch
from huggingface_hub import HfApi

# 스크립트 폴더(hub/)가 sys.path에 들어가므로 같은 폴더의 카드 동기화 스크립트를 그대로 쓴다.
from sync_hub_card import card_images, names_checkpoint

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


def check_card_checkpoint(card: Path, checkpoint: str, paper: str) -> None:
    """카드의 그림과 수치가 올리는 체크포인트의 것인지 확인한다."""
    if not names_checkpoint(card, checkpoint):
        raise SystemExit(
            f"모델 카드에 `{checkpoint}`가 없습니다. 카드는 다른 체크포인트의 결과를 적고 있을 수"
            f" 있으니 scripts/{paper}/plot_card.py로 그림을 다시 만들고 카드의 수치와 체크포인트를"
            " 고쳐 주세요."
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
    from minipfn.hub.tabpfn_v1 import from_pretrained, save_pretrained
    from minipfn.recipes.tabpfn_v1 import tabpfn_from_state

    path = args.checkpoint
    if path.startswith("runs:/"):
        path = mlflow.artifacts.download_artifacts(
            artifact_uri=path, dst_path=str(out_dir / "checkpoint")
        )
    state = torch.load(path, map_location="cpu", weights_only=True)
    if "config" not in state:
        raise SystemExit(
            f"{args.checkpoint}에 모델 설정이 없습니다 (학습 중 checkpoint). 설정이 함께 저장된"
            " runs:/<run_id>/model/tabpfn_final.pt를 주세요."
        )
    model, config = tabpfn_from_state(state["config"], state["state_dict"])
    save_pretrained(model, config, upload_dir, source=args.checkpoint)
    shutil.copy(NOTICE, upload_dir / "NOTICE")
    reloaded, _ = from_pretrained(upload_dir)
    return model, reloaded


def prepare_tabpfn_v2(args: argparse.Namespace, out_dir: Path, upload_dir: Path):
    """v2: 체크포인트를 변환하고 원본과 다시 읽은 모델을 돌려준다. 회귀 구간 경계도 확인한다."""
    from minipfn.hub import from_pretrained, load_model, save_pretrained

    model, borders = load_model(args.checkpoint, args.task, out_dir / "checkpoint")
    save_pretrained(model, borders, upload_dir, source=args.checkpoint)
    reloaded, reloaded_borders = from_pretrained(upload_dir)
    if (borders is None) != (reloaded_borders is None) or (
        borders is not None and not torch.equal(borders.cpu(), reloaded_borders)
    ):
        raise RuntimeError("변환한 회귀 구간 경계가 원본과 다릅니다")
    return model, reloaded


PREPARE = {"tabpfn": prepare_tabpfn, "tabpfn_v2": prepare_tabpfn_v2}
FRONT_MATTER = re.compile(r"\A---\r?\n.*?^---[ \t]*(?:\r?\n|\Z)", re.S | re.M)


def check_card_revision(card: Path, revision: str, checkpoint: str) -> None:
    """main 카드의 한 줄에 branch와 체크포인트가 함께 있는지. 다른 step의 가중치를 막는다."""
    if not any(
        f"`{revision}`" in line and f"`{checkpoint}`" in line
        for line in card.read_text().splitlines()
    ):
        raise SystemExit(
            f"모델 카드에 `{revision}`와 `{checkpoint}`를 함께 적은 줄이 없습니다. 카드의 중간"
            " checkpoint 표에 그 branch의 행을 추가하거나 branch 이름을 확인해 주세요."
        )


def revision_card(model_card: Path, repo_id: str, revision: str, checkpoint: str) -> str:
    """중간 checkpoint branch의 README. main 카드의 front matter를 그대로 쓰고 본문은 짧게 둔다."""
    match = FRONT_MATTER.match(model_card.read_text())
    front = match.group(0).rstrip() + "\n\n" if match else ""
    url = f"https://huggingface.co/{repo_id}"
    return (
        f"{front}# {repo_id} @ `{revision}`\n\n"
        f"An intermediate training checkpoint, kept for comparison. The model card, benchmark"
        f" numbers for every checkpoint, and usage are on the [main branch]({url}).\n\n"
        f"Weights come from checkpoint `{checkpoint}`.\n\n"
        "```python\n"
        "from minipfn.hub import from_pretrained\n\n"
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
    # 큰 artifact를 받기 전에 카드부터 확인한다.
    model_card = resolve_model_card(args.repo_id, args.model_card)
    images = card_images(model_card)
    check_card_checkpoint(model_card, args.checkpoint, args.paper)
    if args.revision is not None:
        check_card_revision(model_card, args.revision, args.checkpoint)
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
    if args.revision is None:
        shutil.copy(model_card, upload_dir / "README.md")
        for rel in images:
            (upload_dir / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(model_card.parent / rel, upload_dir / rel)
    else:
        card = revision_card(model_card, args.repo_id, args.revision, args.checkpoint)
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
        commit_message=f"Upload {TITLE[args.paper]} weights from {args.checkpoint}",
        delete_patterns=["images/*"],
    )
    print("uploaded:", commit.commit_url)


if __name__ == "__main__":
    main()
