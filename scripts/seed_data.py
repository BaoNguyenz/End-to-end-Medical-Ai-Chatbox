"""
seed_data.py
Copy sample documents from sample_data/sample_data/ into data/ directory.
Flattens the nested folder structure.
"""

import shutil
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
SOURCE_DIR = ROOT_DIR / "sample_data" / "sample_data"
TARGET_DIR = ROOT_DIR / "data"

FOLDERS = ["technical_docs", "policy_docs", "product_catalog"]


def seed() -> None:
    """Copy sample data folders into data/ directory."""
    if not SOURCE_DIR.exists():
        print(f"[ERROR] Source directory not found: {SOURCE_DIR}")
        print("  Make sure sample_data/sample_data/ exists with the document folders.")
        return

    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    for folder in FOLDERS:
        src = SOURCE_DIR / folder
        dst = TARGET_DIR / folder

        if not src.exists():
            print(f"[WARN] Source folder missing: {src}")
            continue

        if dst.exists():
            shutil.rmtree(dst)
            print(f"  [CLEAN] Removed existing {dst}")

        shutil.copytree(src, dst)
        file_count = len(list(dst.glob("*.md")))
        print(f"  [OK] {folder}: {file_count} files -> {dst}")

    total = len(list(TARGET_DIR.rglob("*.md")))
    print(f"\n[DONE] Seeded {total} documents into {TARGET_DIR}")


if __name__ == "__main__":
    seed()
