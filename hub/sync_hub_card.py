"""모델 카드(README와 카드가 링크한 그림)만 Hugging Face Hub repo에 올린다.

예) 저장소 루트에서
  uv run --no-project --with huggingface_hub python hub/sync_hub_card.py \
    --card-dir hub/minipfn-tabpfn-v2-classifier --repo-id aiden-jeon/minipfn-tabpfn-v2-classifier

main에 카드가 바뀌어 들어오면 GitHub Actions(`.github/workflows/sync-hub-cards.yml`)가 돌린다.
가중치는 올리지 않는다(새로 학습했을 때 `hub/push_to_hub.py`로 올린다). 그래서 repo가
아직 없으면 카드만 먼저 올리지 않고 건너뛴다. 카드에는 Hub `main`의 `model.safetensors`가
가중치 id(`` `sha256:<SHA-256 앞 12자리>` ``)로 적혀 있어야 올린다 — 카드의 수치가 올라가 있는
가중치와 다른 체크포인트의 것이 되지 않도록. 중간 checkpoint 표의 branch 행(`` | `step-100k` | ``로
시작하는 줄)에 적은 id는 그 branch의 것이라 main의 확인에 쓰지 않고, Hub에 그 branch가 있으면 그
branch의 가중치와 맞춰 본다. 해시는 Hub가 LFS 파일마다 기록한 SHA-256을 읽으므로 가중치를 받지
않는다. 카드와 Hub 어디에도 학습 run id(MLflow)는 적지 않는다. Hub의 `images/`에서 카드가 더는
쓰지 않는 파일은 지운다. torch 없이 huggingface_hub만으로 돈다.
`push_to_hub.py`도 이 파일의 카드 확인 함수를 쓴다.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

IMAGE_LINK = re.compile(r"\]\((images/[^)\s]+)\)")
WEIGHTS_NAME = "model.safetensors"
# 카드에 적는 SHA-256 앞자리 수. 잘못 고른 체크포인트를 알아보는 데는 이 정도로 충분하다
WEIGHTS_ID_HEX = 12
# 중간 checkpoint 표의 행: 첫 칸이 `` `branch` ``다 (`main` 행은 `` | `main` (500k) | ``)
BRANCH_ROW = re.compile(r"^\|\s*`([^`]+)`")


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


def weights_id(sha256: str) -> str:
    """카드에 적는 가중치 id: `sha256:` 뒤에 `model.safetensors`의 SHA-256 앞
    `WEIGHTS_ID_HEX`자리."""
    return f"sha256:{sha256[:WEIGHTS_ID_HEX]}"


def file_sha256(path: Path) -> str:
    """파일의 SHA-256 (Hub가 LFS 파일에 기록하는 값과 같다)."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def names_weights(text: str, sha256: str) -> bool:
    """`text`(카드 또는 그 한 줄)가 그 가중치의 id를 `` `...` ``로 적고 있는지."""
    return f"`{weights_id(sha256)}`" in text


def branch_rows(text: str) -> dict[str, list[str]]:
    """카드의 중간 checkpoint 표에서 branch마다 그 행."""
    rows: dict[str, list[str]] = {}
    for line in text.splitlines():
        if match := BRANCH_ROW.match(line):
            rows.setdefault(match.group(1), []).append(line)
    return rows


def names_main_weights(text: str, sha256: str) -> bool:
    """카드가 main 가중치로 그 id를 적고 있는지. 다른 branch의 행에 적힌 id는 치지 않는다
    (main 카드는 표에 모든 branch의 id를 적으므로, 그대로 찾으면 중간 step의 가중치가 통과한다)."""
    return any(
        names_weights(line, sha256)
        for line in text.splitlines()
        if (match := BRANCH_ROW.match(line)) is None or match.group(1) == "main"
    )


def names_branch_weights(text: str, branch: str, sha256: str) -> bool:
    """카드의 `branch` 행이 그 id를 적고 있는지. 본문에서 branch와 다른 가중치를 함께 적은 줄
    (예: "main은 `step-500k`에서 이어 학습했다 (`sha256:...`)")은 치지 않는다."""
    return any(names_weights(line, sha256) for line in branch_rows(text).get(branch, []))


def hub_weights_sha256(api: HfApi, repo_id: str, revision: str | None = None) -> str | None:
    """Hub에 올라가 있는 `model.safetensors`의 SHA-256. 가중치가 없으면 None.

    파일은 있는데 LFS 정보가 없으면 확인할 수 없으니 멈춘다 (확인 없이 올리지 않도록)."""
    for info in api.get_paths_info(repo_id, [WEIGHTS_NAME], revision=revision):
        if info.path != WEIGHTS_NAME:
            continue
        lfs = getattr(info, "lfs", None)
        if lfs is None or not getattr(lfs, "sha256", None):
            where = f"{repo_id}@{revision}" if revision else repo_id
            raise SystemExit(
                f"{where}의 {WEIGHTS_NAME}에 LFS SHA-256이 없어 카드와 맞춰 볼 수 없습니다"
                " (LFS가 아닌 일반 파일로 올라갔을 수 있습니다)."
            )
        return lfs.sha256
    return None


def check_branch_rows(api: HfApi, repo_id: str, text: str) -> list[str]:
    """카드 표의 branch 행이 적은 id가 Hub의 그 branch 가중치와 같은지. 맞춰 본 branch를 돌려준다.

    Hub에 아직 없는 branch(카드에 행을 먼저 넣고 나중에 올린다)는 건너뛴다."""
    hub_branches = {ref.name for ref in api.list_repo_refs(repo_id).branches}
    checked = []
    for branch, lines in branch_rows(text).items():
        if branch == "main" or not any("`sha256:" in line for line in lines):
            continue
        if branch not in hub_branches:
            print(f"{repo_id}에 {branch} branch가 아직 없어 그 행은 맞춰 보지 않았습니다")
            continue
        sha256 = hub_weights_sha256(api, repo_id, branch)
        if sha256 is None or not names_branch_weights(text, branch, sha256):
            found = "가중치가 없습니다" if sha256 is None else f"id는 `{weights_id(sha256)}`입니다"
            raise SystemExit(
                f"카드의 `{branch}` 행이 Hub {branch} branch의 가중치와 다릅니다 (Hub {found})."
            )
        checked.append(branch)
    return checked


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
    text = card.read_text()
    sha256 = hub_weights_sha256(api, args.repo_id)
    if sha256 is not None and not names_main_weights(text, sha256):
        raise SystemExit(
            f"카드에 Hub 가중치의 id `{weights_id(sha256)}`가 없습니다 (다른 branch의 행은 치지"
            " 않습니다). 카드가 다른 체크포인트의 결과를 적고 있으면 가중치를 함께 다시 올려"
            " 주세요."
        )
    if sha256 is not None:
        print("branches checked:", check_branch_rows(api, args.repo_id, text))
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
