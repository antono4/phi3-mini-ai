#!/usr/bin/env python
"""Download the image model used by the Foto mode.

Defaults to SD-Turbo, a 1-step distilled Stable Diffusion model: a single
denoising step is enough, which is what makes image generation viable on CPU.
Only the fp16 weights are fetched (the repo also ships fp32 copies and a 5 GB
single-file checkpoint we do not need), which keeps the download near 2.6 GB.

Usage:
    python scripts/download_media_model.py [--repo ID] [--dest DIR] [--all-weights]
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

# Configs, tokenizer and scheduler are tiny; the weights are fp16-only so the
# fp32 duplicates and the combined .safetensors checkpoint stay off the disk.
CONFIG_PATTERNS = ["*.json", "*.txt", "*.model", "tokenizer/*", "scheduler/*"]
FP16_PATTERNS = ["*/model.fp16.safetensors", "*/diffusion_pytorch_model.fp16.safetensors"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default=os.environ.get("PHI3_MEDIA_REPO", "stabilityai/sd-turbo"))
    ap.add_argument("--dest", default=None, help="target directory (default: models/media/<name>)")
    ap.add_argument("--all-weights", action="store_true",
                    help="download fp32 weights too (about 13 GB)")
    args = ap.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("huggingface_hub is missing. Install it with:\n"
              "  .venv/bin/pip install -r requirements-media.txt", file=sys.stderr)
        return 1

    root = Path(__file__).resolve().parent.parent
    dest = Path(args.dest) if args.dest else root / "models" / "media" / args.repo.split("/")[-1]
    dest.mkdir(parents=True, exist_ok=True)

    patterns = CONFIG_PATTERNS + (["*.safetensors"] if args.all_weights else FP16_PATTERNS)

    free = shutil.disk_usage(dest).free / 1e9
    needed = 14 if args.all_weights else 3
    if free < needed:
        print(f"warning: only {free:.1f} GB free at {dest}; about {needed} GB is needed.",
              file=sys.stderr)

    print(f"downloading {args.repo} -> {dest}")
    print(f"patterns: {', '.join(patterns)}")
    path = snapshot_download(
        repo_id=args.repo,
        local_dir=str(dest),
        allow_patterns=patterns,
        max_workers=4,
    )
    print(f"done: {path}")
    print(f"point the server at it with PHI3_MEDIA_MODEL={dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
