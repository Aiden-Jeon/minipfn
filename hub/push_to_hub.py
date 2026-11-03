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
  표기)를 함께 올린다. 모델 카드는 paper run 결과를 적고 있어서, paper 크기가 아닌 모델은
  `--allow-non-paper`를 줄 때만 올린다.
- `tabpfn_v2`: `minipfn.tabpfn_v2.hub.load_model`이 읽는 체크포인트면 무엇이든. `--task`가 필요하다.

체크포인트를 `model.safetensors` + `config.json`으로 바꾸고 모델 카드를 붙여 올린다. 모델 카드는
`--model-card`로 주거나, 주지 않으면 repo 이름으로 `hub/<name>/README.md`를 찾는다. 카드가
`](images/...)`로 링크한 그림만 카드 옆에서 복사하고, Hub의 `images/`에서 카드가 더는 쓰지 않는
파일은 지운다. `--dry-run`이면 폴더만 만들고 올리지 않는다.

공개 repo를 실수로 덮어쓰지 않도록 `--repo-id`는 기본값 없이 받는다. 카드의 그림과 수치가 다른
체크포인트의 것이 되지 않도록, 카드에 `--checkpoint` 문자열이 그대로 적혀 있어야 올린다(그림은
`scripts/<paper>/plot_card.py`로 다시 만든다).
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import mlflow
import torch
from huggingface_hub import HfApi

REPO_ROOT = Path(__file__).resolve().parents[1]
HUB_DIR = Path(__file__).resolve().parent
NOTICE = REPO_ROOT / "NOTICE"
IMAGE_LINK = re.compile(r"\]\((images/[^)\s]+)\)")
PAPERS = ("tabpfn", "tabpfn_v2")


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
    parser.add_argument("--out-dir", default="hub_upload")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--allow-non-paper", action="store_true", help="tabpfn만: paper 크기가 아닌 모델도 올린다"
    )
    args = parser.parse_args(argv)
    if args.paper == "tabpfn_v2" and args.task is None:
        parser.error("--paper tabpfn_v2에는 --task가 필요합니다")
    if args.paper == "tabpfn" and args.task is not None:
        parser.error("--task는 --paper tabpfn_v2에서만 씁니다 (v1은 분류만)")
    if args.paper == "tabpfn_v2" and args.allow_non_paper:
        parser.error("--allow-non-paper는 --paper tabpfn에서만 씁니다")
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


def card_images(card: Path) -> list[str]:
    """카드가 링크한 `images/` 아래 그림의 상대 경로. 없는 파일이나 폴더 밖 경로는 막는다."""
    images_dir = (card.parent / "images").resolve()
    paths = sorted(set(IMAGE_LINK.findall(card.read_text())))
    for rel in paths:
        path = (card.parent / rel).resolve()
        if not path.is_relative_to(images_dir) or not path.is_file():
            raise SystemExit(f"카드가 링크한 그림이 {images_dir} 안에 없습니다: {rel}")
    return paths


def check_card_checkpoint(card: Path, checkpoint: str, paper: str) -> None:
    """카드의 그림과 수치가 올리는 체크포인트의 것인지 확인한다."""
    if checkpoint not in card.read_text():
        raise SystemExit(
            f"모델 카드에 {checkpoint}가 없습니다. 카드는 다른 체크포인트의 결과를 적고 있을 수"
            f" 있으니 scripts/{paper}/plot_card.py로 그림을 다시 만들고 카드의 수치와 체크포인트를"
            " 고쳐 주세요."
        )


def prepare_tabpfn(args: argparse.Namespace, out_dir: Path, upload_dir: Path):
    """v1: `tabpfn_final.pt`를 변환하고, 다시 읽은 모델과 원본을 돌려준다."""
    from minipfn.tabpfn.hub import from_pretrained, save_pretrained
    from minipfn.tabpfn.train import TabPFNConfig, load_tabpfn

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
    model, config = load_tabpfn(path)
    if not args.allow_non_paper:
        paper = TabPFNConfig.paper()
        got = (config.nlayers, config.emsize, config.steps)
        if got != (paper.nlayers, paper.emsize, paper.steps):
            raise SystemExit(
                f"paper run이 아닙니다 (nlayers, emsize, steps = {got}). 모델 카드는 paper 결과를"
                " 적고 있으니 --allow-non-paper를 줄 때만 올립니다."
            )
    save_pretrained(model, config, upload_dir)
    shutil.copy(NOTICE, upload_dir / "NOTICE")
    reloaded, _ = from_pretrained(upload_dir)
    return model, reloaded


def prepare_tabpfn_v2(args: argparse.Namespace, out_dir: Path, upload_dir: Path):
    """v2: 체크포인트를 변환하고, 다시 읽은 모델과 원본을 돌려준다. 회귀 구간 경계도 확인한다."""
    from minipfn.tabpfn_v2.hub import from_pretrained, load_model, save_pretrained

    model, borders = load_model(args.checkpoint, args.task, out_dir / "checkpoint")
    save_pretrained(model, borders, upload_dir, source=args.checkpoint)
    reloaded, reloaded_borders = from_pretrained(upload_dir)
    if (borders is None) != (reloaded_borders is None) or (
        borders is not None and not torch.equal(borders.cpu(), reloaded_borders)
    ):
        raise RuntimeError("변환한 회귀 구간 경계가 원본과 다릅니다")
    return model, reloaded


PREPARE = {"tabpfn": prepare_tabpfn, "tabpfn_v2": prepare_tabpfn_v2}
TITLE = {"tabpfn": "TabPFN", "tabpfn_v2": "TabPFN v2"}


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    # 큰 artifact를 받기 전에 카드부터 확인한다.
    model_card = resolve_model_card(args.repo_id, args.model_card)
    images = card_images(model_card)
    check_card_checkpoint(model_card, args.checkpoint, args.paper)

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
    shutil.copy(model_card, upload_dir / "README.md")
    for rel in images:
        (upload_dir / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(model_card.parent / rel, upload_dir / rel)
    prepared = sorted(str(p.relative_to(upload_dir)) for p in upload_dir.rglob("*") if p.is_file())
    print("prepared:", prepared)
    if args.dry_run:
        return

    api = HfApi()
    api.create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    # 같은 commit에서 다시 올리는 파일은 지우지 않으므로, 카드가 더는 쓰지 않는 그림만 지워진다.
    commit = api.upload_folder(
        repo_id=args.repo_id,
        folder_path=upload_dir,
        commit_message=f"Upload {TITLE[args.paper]} weights from {args.checkpoint}",
        delete_patterns=["images/*"],
    )
    print("uploaded:", commit.commit_url)


if __name__ == "__main__":
    main()
