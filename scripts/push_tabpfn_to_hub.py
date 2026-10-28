"""학습한 TabPFN 가중치를 Hugging Face Hub에 올린다.

예) 저장소 루트에서 (.env.local을 불러온 셸)
  MLFLOW_TRACKING_URI=databricks \
    uv run python scripts/push_tabpfn_to_hub.py --run-id <학습 run_id> \
    --repo-id aiden-jeon/minipfn-tabpfn-v1
  uv run python scripts/push_tabpfn_to_hub.py --model-path tabpfn_final.pt \
    --repo-id <owner/name> --dry-run

MLflow run의 `model/tabpfn_final.pt`를 `model.safetensors` + `config.json`으로 바꾸고, 모델
카드와 저장소 루트의 `NOTICE`를 붙여 올린다. 모델 카드는 `--model-card`로 주거나, 주지 않으면
repo 이름으로 `hub/<name>/README.md`를 찾는다. 카드가 `](images/...)`로 링크한 그림만 카드
옆에서 복사하고, Hub의 `images/`에서 카드가 더는 쓰지 않는 파일은 지운다. `--dry-run`이면
폴더만 만들고 올리지 않는다.

공개 repo를 실수로 덮어쓰지 않도록 `--repo-id`는 기본값 없이 받는다. 모델 카드는 paper run
결과를 적고 있어서, paper 크기가 아닌 모델은 `--allow-non-paper`를 줄 때만 올린다. 카드의
그림과 수치가 다른 run의 것이 되지 않도록, `--run-id`로 올릴 때는 카드에 그 run id가 적혀
있어야 한다(그림은 `plot_tabpfn_card.py --run-id`로 다시 만든다).
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import mlflow
import torch
from huggingface_hub import HfApi

from minipfn.tabpfn.hub import from_pretrained, save_pretrained
from minipfn.tabpfn.train import TabPFNConfig, load_tabpfn

REPO_ROOT = Path(__file__).resolve().parents[1]
HUB_DIR = REPO_ROOT / "hub"
NOTICE = REPO_ROOT / "NOTICE"
IMAGE_LINK = re.compile(r"\]\((images/[^)\s]+)\)")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--run-id", help="학습 run (artifact model/tabpfn_final.pt)")
    src.add_argument("--model-path", help="로컬 tabpfn_final.pt")
    parser.add_argument("--repo-id", required=True, help="Hub repo id (owner/name)")
    parser.add_argument(
        "--model-card",
        type=Path,
        help="Hub README로 올릴 모델 카드 (기본: hub/<repo 이름>/README.md)",
    )
    parser.add_argument("--out-dir", default="tabpfn_hub")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--allow-non-paper", action="store_true", help="paper 크기가 아닌 모델도 올린다"
    )
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


def check_card_run(card: Path, run_id: str) -> None:
    """카드의 그림과 수치가 올리는 run의 것인지 확인한다."""
    if run_id not in card.read_text():
        raise SystemExit(
            f"모델 카드에 run {run_id}가 없습니다. 카드는 다른 run의 결과를 적고 있을 수 있으니"
            " plot_tabpfn_card.py --run-id로 그림을 다시 만들고 카드의 수치와 run id를 고쳐 주세요."
        )


def check_paper_config(config: TabPFNConfig) -> None:
    """모델 카드의 결과(18,000 step, 12층)와 맞는 run인지 확인한다."""
    paper = TabPFNConfig.paper()
    got = (config.nlayers, config.emsize, config.steps)
    if got != (paper.nlayers, paper.emsize, paper.steps):
        raise SystemExit(
            f"paper run이 아닙니다 (nlayers, emsize, steps = {got}). 모델 카드는 paper 결과를"
            " 적고 있으니 --allow-non-paper를 줄 때만 올립니다."
        )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    # 큰 artifact를 받기 전에 카드부터 확인한다.
    model_card = resolve_model_card(args.repo_id, args.model_card)
    images = card_images(model_card)
    if args.run_id:
        check_card_run(model_card, args.run_id)
    out_dir = Path(args.out_dir)
    path = args.model_path or mlflow.artifacts.download_artifacts(
        run_id=args.run_id, artifact_path="model/tabpfn_final.pt", dst_path=str(out_dir / "mlflow")
    )
    model, config = load_tabpfn(path)
    if not args.allow_non_paper:
        check_paper_config(config)
    # 지난 실행의 파일이 섞여 올라가지 않도록 매번 빈 폴더에서 시작한다.
    shutil.rmtree(out_dir / "upload", ignore_errors=True)
    upload_dir = save_pretrained(model, config, out_dir / "upload")
    shutil.copy(model_card, upload_dir / "README.md")
    shutil.copy(NOTICE, upload_dir / "NOTICE")
    for rel in images:
        (upload_dir / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(model_card.parent / rel, upload_dir / rel)

    # 올리기 전에 변환한 파일만으로 같은 가중치가 다시 만들어지는지 확인한다.
    reloaded, _ = from_pretrained(upload_dir)
    for (name, a), b in zip(
        model.state_dict().items(), reloaded.state_dict().values(), strict=True
    ):
        if not torch.equal(a, b):
            raise RuntimeError(f"변환한 가중치가 원본과 다릅니다: {name}")
    prepared = sorted(str(p.relative_to(upload_dir)) for p in upload_dir.rglob("*") if p.is_file())
    print("prepared:", prepared)
    if args.dry_run:
        return

    api = HfApi()
    api.create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    message = f"Upload TabPFN weights from MLflow run {args.run_id}" if args.run_id else None
    # 같은 commit에서 다시 올리는 파일은 지우지 않으므로, 카드가 더는 쓰지 않는 그림만 지워진다.
    commit = api.upload_folder(
        repo_id=args.repo_id,
        folder_path=upload_dir,
        commit_message=message,
        delete_patterns=["images/*"],
    )
    print("uploaded:", commit.commit_url)


if __name__ == "__main__":
    main()
