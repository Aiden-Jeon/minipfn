"""학습한 TabPFN 가중치를 Hugging Face Hub에 올린다.

예) 저장소 루트에서 (.env.local을 불러온 셸)
  MLFLOW_TRACKING_URI=databricks \
    uv run python scripts/push_tabpfn_to_hub.py --run-id <학습 run_id> \
    --repo-id aiden-jeon/minipfn-tabpfn-v1
  uv run python scripts/push_tabpfn_to_hub.py --model-path tabpfn_final.pt \
    --repo-id <owner/name> --dry-run

MLflow run의 `model/tabpfn_final.pt`를 `model.safetensors` + `config.json`으로 바꾸고, 모델
카드(옆에 `images/`가 있으면 함께)와 저장소 루트의 `NOTICE`를 붙여 올린다. 모델 카드는
`--model-card`로 주거나, 주지 않으면 repo 이름으로 `hub/<name>/README.md`를 찾는다.
`--dry-run`이면 폴더만 만들고 올리지 않는다.

공개 repo를 실수로 덮어쓰지 않도록 `--repo-id`는 기본값 없이 받는다. 모델 카드는 paper run
결과를 적고 있어서, paper 크기가 아닌 모델은 `--allow-non-paper`를 줄 때만 올린다.
"""

from __future__ import annotations

import argparse
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
    out_dir = Path(args.out_dir)
    path = args.model_path or mlflow.artifacts.download_artifacts(
        run_id=args.run_id, artifact_path="model/tabpfn_final.pt", dst_path=str(out_dir / "mlflow")
    )
    model, config = load_tabpfn(path)
    if not args.allow_non_paper:
        check_paper_config(config)
    upload_dir = save_pretrained(model, config, out_dir / "upload")
    shutil.copy(model_card, upload_dir / "README.md")
    shutil.copy(NOTICE, upload_dir / "NOTICE")
    # 카드가 쓰는 그림은 카드 옆 images/에 둔다.
    if (model_card.parent / "images").is_dir():
        shutil.copytree(model_card.parent / "images", upload_dir / "images", dirs_exist_ok=True)

    # 올리기 전에 변환한 파일만으로 같은 가중치가 다시 만들어지는지 확인한다.
    reloaded, _ = from_pretrained(upload_dir)
    for (name, a), b in zip(
        model.state_dict().items(), reloaded.state_dict().values(), strict=True
    ):
        if not torch.equal(a, b):
            raise RuntimeError(f"변환한 가중치가 원본과 다릅니다: {name}")
    print("prepared:", sorted(p.name for p in upload_dir.iterdir()))
    if args.dry_run:
        return

    api = HfApi()
    api.create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    message = f"Upload TabPFN weights from MLflow run {args.run_id}" if args.run_id else None
    commit = api.upload_folder(repo_id=args.repo_id, folder_path=upload_dir, commit_message=message)
    print("uploaded:", commit.commit_url)


if __name__ == "__main__":
    main()
