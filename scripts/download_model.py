"""Download the Phi-3 Mini 4k instruct GGUF (Q4) model.

    python scripts/download_model.py
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

URL = (
    "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/"
    "resolve/main/Phi-3-mini-4k-instruct-q4.gguf"
)
DEST = Path(__file__).resolve().parent.parent / "models" / "Phi-3-mini-4k-instruct-q4.gguf"


def _progress(block_num: int, block_size: int, total_size: int) -> None:
    if total_size <= 0:
        return
    downloaded = min(block_num * block_size, total_size)
    pct = downloaded * 100 / total_size
    mb = downloaded / 1024 / 1024
    total_mb = total_size / 1024 / 1024
    sys.stdout.write(f"\r  {pct:5.1f}%  {mb:7.1f}/{total_mb:.1f} MB")
    sys.stdout.flush()


def main() -> int:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    if DEST.exists() and DEST.stat().st_size > 100 * 1024 * 1024:
        print(f"Model already present: {DEST}")
        return 0
    print(f"Downloading Phi-3 Mini Q4 to {DEST}")
    urllib.request.urlretrieve(URL, DEST, _progress)
    print(f"\nDone: {DEST} ({DEST.stat().st_size / 1024 / 1024:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
