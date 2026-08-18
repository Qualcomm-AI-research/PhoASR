# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Download the fine-tuned punctuation/capitalization checkpoint from GitHub releases.

Usage:
    python download_checkpoints.py

The checkpoint is downloaded and extracted into the ``checkpoints/`` directory
at the repository root. The directory is created if it does not exist.
"""

from __future__ import annotations

import os
import tarfile
import urllib.request
from pathlib import Path

CHECKPOINTS = {
    "bartpho-punc-cap": "https://github.com/qualcomm-ai-research/PhoASR/releases/download/v1.0/bartpho-punc-cap.tar.gz",
}

REPO_ROOT = Path(__file__).resolve().parent
CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"


def download_and_extract(name: str, url: str, dest_dir: Path) -> None:
    """Download a checkpoint archive and extract it into dest_dir."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    archive_path = dest_dir / f"{name}.tar.gz"
    print(f"Downloading {name} from {url} ...")
    urllib.request.urlretrieve(url, archive_path)
    print(f"Extracting {name} ...")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=dest_dir)
    os.remove(archive_path)
    print(f"  -> {dest_dir / name}")


def main() -> None:
    """Download all checkpoints listed in CHECKPOINTS."""
    for name, url in CHECKPOINTS.items():
        target = CHECKPOINTS_DIR / name
        if target.exists():
            print(f"Skipping {name}: already present at {target}")
            continue
        download_and_extract(name, url, CHECKPOINTS_DIR)
    print("Done.")


if __name__ == "__main__":
    main()
