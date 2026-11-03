"""모델 카드(README와 카드가 링크한 그림)만 Hugging Face Hub repo에 올린다.

예) 저장소 루트에서
  uv run --no-project --with huggingface_hub python hub/sync_hub_card.py \
    --card-dir hub/minipfn-tabpfn-v2-classifier --repo-id aiden-jeon/minipfn-tabpfn-v2-classifier

main에 카드가 바뀌어 들어오면 GitHub Actions(`.github/workflows/sync-hub-cards.yml`)가 돌린다.
가중치는 올리지 않는다(새로 학습했을 때 `hub/push_to_hub.py`로 올린다). 그래서 repo가
아직 없으면 카드만 먼저 올리지 않고 건너뛴다. repo에 `source`(가중치를 만든 체크포인트, v2는
`config.json`, v1은 `source.json`)가 있으면 카드에도 그 체크포인트가 `` `...` ``로 그대로 적혀
있어야 올린다 — 카드의 수치가 올라가 있는 가중치와 다른 체크포인트의 것이 되지 않도록. Hub의
`images/`에서 카드가 더는 쓰지 않는 파일은 지운다. torch 없이 huggingface_hub만으로 돈다.
`push_to_hub.py`도 이 파일의 `card_images`·`names_checkpoint`를 쓴다.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

IMAGE_LINK = re.compile(r"\]\((images/[^)\s]+)\)")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--card-dir", type=Path, required=True, help="README.md가 든 hub/<name>")
    parser.add_argument("--repo-id", required=True, help="Hub repo id (owner/name)")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def card_images(card: Path) -> list[str]:
    """카드가 링크한 `images/` 아래 그림의 상대 경로. 없는 파일이나 폴더 밖 경로는 막는다."""
    images_dir = (card.parent / "images").resolve()
    paths = sorted(set(IMAGE_LINK.findall(card.read_text())))
    for rel in paths:
        path = (card.parent / rel).resolve()
        if not path.is_relative_to(images_dir) or not path.is_file():
            raise SystemExit(f"카드가 링크한 그림이 {images_dir} 안에 없습니다: {rel}")
    return paths


def names_checkpoint(card: Path, checkpoint: str) -> bool:
    """카드가 그 체크포인트를 `` `...` ``로 통째로 적고 있는지.

    그냥 부분 문자열로 찾으면 `tabpfn_final.pt` 같은 짧은 경로가 카드의
    `runs:/<run_id>/model/tabpfn_final.pt` 안에서 찾아져 다른 run의 가중치가 통과한다."""
    return f"`{checkpoint}`" in card.read_text()


def hub_source(api: HfApi, repo_id: str) -> str | None:
    """Hub에 올라가 있는 가중치를 만든 체크포인트. v2는 `config.json`, v1은 `source.json`에 있다.

    v1이 `config.json`에 두지 않는 까닭은 `minipfn.tabpfn.hub`에 적었다. 둘 다 없으면 None."""
    for name in ("config.json", "source.json"):
        if not api.file_exists(repo_id, name):
            continue
        source = json.loads(Path(api.hf_hub_download(repo_id, name)).read_text()).get("source")
        if source is not None:
            return source
    return None


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    card = args.card_dir / "README.md"
    if not card.is_file():
        raise SystemExit(f"모델 카드가 없습니다: {card}")
    images = card_images(card)
    api = HfApi()
    if not api.repo_exists(args.repo_id):
        # GitHub Actions 로그에 경고로 남기고 실패로는 치지 않는다
        print(f"::warning::{args.repo_id}가 아직 없어 카드를 올리지 않았습니다 (가중치가 먼저)")
        return
    source = hub_source(api, args.repo_id)
    if source is not None and not names_checkpoint(card, source):
        raise SystemExit(
            f"카드에 Hub 가중치의 체크포인트 {source}가 없습니다. 카드가 다른 체크포인트의 결과를"
            " 적고 있으면 가중치를 함께 다시 올려 주세요."
        )
    with tempfile.TemporaryDirectory() as tmp:
        upload = Path(tmp)
        shutil.copy(card, upload / "README.md")
        for rel in images:
            (upload / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(card.parent / rel, upload / rel)
        print("prepared:", ["README.md", *images])
        if args.dry_run:
            return
        commit = api.upload_folder(
            repo_id=args.repo_id,
            folder_path=upload,
            commit_message=f"Sync model card from {args.card_dir}",
            delete_patterns=["images/*"],
        )
        print("uploaded:", commit.commit_url)


if __name__ == "__main__":
    main()
