"""학습한 TabPFN v2 가중치를 Hugging Face Hub에 올린다.

예) 저장소 루트에서 (.env.local을 불러온 셸)
  MLFLOW_TRACKING_URI=databricks uv run python scripts/tabpfn_v2/push_to_hub.py \
    --checkpoint runs:/<학습 run_id>/checkpoints/step_0500000.pt --task classification \
    --repo-id aiden-jeon/minipfn-tabpfn-v2-classifier --dry-run

체크포인트(`hub.load_model`이 읽는 형식이면 무엇이든)를 `model.safetensors` + `config.json`으로
바꾸고 모델 카드를 붙여 올린다. 모델 카드는 `--model-card`로 주거나, 주지 않으면 repo 이름으로
`hub/<name>/README.md`를 찾는다. 카드가 `](images/...)`로 링크한 그림만 카드 옆에서 복사하고,
Hub의 `images/`에서 카드가 더는 쓰지 않는 파일은 지운다. `--dry-run`이면 폴더만 만들고 올리지
않는다.

공개 repo를 실수로 덮어쓰지 않도록 `--repo-id`는 기본값 없이 받는다. 카드의 그림과 수치가 다른
체크포인트의 것이 되지 않도록, 카드에 `--checkpoint` 문자열이 그대로 적혀 있어야 올린다(그림은
`plot_card.py`로 다시 만든다).
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import torch
from huggingface_hub import HfApi

from minipfn.tabpfn_v2.hub import from_pretrained, load_model, save_pretrained

REPO_ROOT = Path(__file__).resolve().parents[2]
HUB_DIR = REPO_ROOT / "hub"
IMAGE_LINK = re.compile(r"\]\((images/[^)\s]+)\)")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True, help="체크포인트 (파일, runs:/ URI)")
    parser.add_argument("--task", choices=["classification", "regression"], required=True)
    parser.add_argument("--repo-id", required=True, help="Hub repo id (owner/name)")
    parser.add_argument(
        "--model-card",
        type=Path,
        help="Hub README로 올릴 모델 카드 (기본: hub/<repo 이름>/README.md)",
    )
    parser.add_argument("--out-dir", default="tabpfn_v2_hub")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


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


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    # 큰 artifact를 받기 전에 카드부터 확인한다.
    model_card = resolve_model_card(args.repo_id, args.model_card)
    images = card_images(model_card)
    if args.checkpoint not in model_card.read_text():
        raise SystemExit(
            f"모델 카드에 {args.checkpoint}가 없습니다. 카드는 다른 체크포인트의 결과를 적고"
            " 있을 수 있으니 plot_card.py로 그림을 다시 만들고 카드의 수치와"
            " 체크포인트를 고쳐 주세요."
        )
    out_dir = Path(args.out_dir)
    model, borders = load_model(args.checkpoint, args.task, out_dir / "checkpoint")
    # 지난 실행의 파일이 섞여 올라가지 않도록 매번 빈 폴더에서 시작한다.
    shutil.rmtree(out_dir / "upload", ignore_errors=True)
    upload_dir = save_pretrained(model, borders, out_dir / "upload", source=args.checkpoint)
    shutil.copy(model_card, upload_dir / "README.md")
    for rel in images:
        (upload_dir / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(model_card.parent / rel, upload_dir / rel)

    # 올리기 전에 변환한 파일만으로 같은 가중치가 다시 만들어지는지 확인한다.
    reloaded, reloaded_borders = from_pretrained(upload_dir)
    for (name, a), b in zip(
        model.state_dict().items(), reloaded.state_dict().values(), strict=True
    ):
        if not torch.equal(a, b):
            raise RuntimeError(f"변환한 가중치가 원본과 다릅니다: {name}")
    if (borders is None) != (reloaded_borders is None) or (
        borders is not None and not torch.equal(borders.cpu(), reloaded_borders)
    ):
        raise RuntimeError("변환한 회귀 구간 경계가 원본과 다릅니다")
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
        commit_message=f"Upload TabPFN v2 weights from {args.checkpoint}",
        delete_patterns=["images/*"],
    )
    print("uploaded:", commit.commit_url)


if __name__ == "__main__":
    main()
