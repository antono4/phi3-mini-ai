"""Download Phi-3 Mini GGUF weights.

    python scripts/download_model.py          # Q4 (~2.3 GB, default)
    python scripts/download_model.py q8       # Q8 (~4.1 GB)

Q4 comes from Microsoft's official GGUF repo. Microsoft does not publish a Q8,
so that one comes from the bartowski repo.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

MODELS = {
    "q4": {
        "url": (
            "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/"
            "resolve/main/Phi-3-mini-4k-instruct-q4.gguf"
        ),
        "dest": "Phi-3-mini-4k-instruct-q4.gguf",
    },
    "q8": {
        "url": (
            "https://huggingface.co/bartowski/Phi-3-mini-4k-instruct-old-GGUF/"
            "resolve/main/Phi-3-mini-4k-instruct-Q8_0.gguf"
        ),
        "dest": "Phi-3-mini-4k-instruct-q8.gguf",
    },
}

ROOT = Path(__file__).resolve().parent.parent


def _progress(block_num: int, block_size: int, total_size: int) -> None:
    if total_size <= 0:
        return
    downloaded = min(block_num * block_size, total_size)
    pct = downloaded * 100 / total_size
    mb = downloaded / 1024 / 1024
    total_mb = total_size / 1024 / 1024
    sys.stdout.write(f"\r  {pct:5.1f}%  {mb:7.1f}/{total_mb:.1f} MB")
    sys.stdout.flush()


def download(name: str) -> int:
    spec = MODELS[name]
    dest = ROOT / "models" / spec["dest"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 100 * 1024 * 1024:
        print(f"Model already present: {dest}")
        return 0
    print(f"Downloading Phi-3 Mini {name.upper()} to {dest}")
    urllib.request.urlretrieve(spec["url"], dest, _progress)
    print(f"\nDone: {dest} ({dest.stat().st_size / 1024 / 1024:.1f} MB)")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    name = (args[0] if args else "q4").lower()
    if name not in MODELS:
        print(f"Unknown model '{name}'. Choose one of: {', '.join(MODELS)}")
        return 2
    return download(name)


if __name__ == "__main__":
    raise SystemExit(main())
