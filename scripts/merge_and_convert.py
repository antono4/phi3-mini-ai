"""Merge a LoRA adapter into the base model, then export a quantized GGUF.

    python scripts/merge_and_convert.py --adapter models/adapters/phi3-lora

Produces the merged HF model under ``--merged`` and a quantized GGUF at
``--gguf`` that the llama.cpp server can load. The llama.cpp checkout (with
``convert_hf_to_gguf.py`` and a built ``llama-quantize``) is required.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
BASE_MODEL = ROOT / "models" / "phi3-hf-base"

# save_pretrained writes the HF tokenizer files but not the SentencePiece
# tokenizer.model, which convert_hf_to_gguf.py requires for Phi-3.
TOKENIZER_FILES = (
    "tokenizer.model",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
)


def _copy_tokenizer_files(base: Path, merged: Path) -> None:
    for name in TOKENIZER_FILES:
        src = base / name
        if src.exists():
            shutil.copy2(src, merged / name)


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge LoRA adapter and export GGUF.")
    parser.add_argument("--base", type=Path, default=BASE_MODEL)
    parser.add_argument("--adapter", type=Path, default=ROOT / "models" / "adapters" / "phi3-lora")
    parser.add_argument("--merged", type=Path, default=ROOT / "models" / "merged" / "phi3-lora-merged")
    parser.add_argument("--gguf", type=Path, default=ROOT / "models" / "phi3-lora-q4.gguf")
    parser.add_argument("--quant", default="Q4_K_M")
    parser.add_argument("--skip-gguf", action="store_true")
    parser.add_argument("--keep-f16", action="store_true", help="keep the f16 GGUF after quantizing")
    args = parser.parse_args()

    if not args.adapter.exists():
        raise FileNotFoundError(f"Adapter not found at {args.adapter}. Train first.")

    print("Loading base + adapter")
    tokenizer = AutoTokenizer.from_pretrained(str(args.base))
    base = AutoModelForCausalLM.from_pretrained(
        str(args.base), dtype=torch.bfloat16
    )
    model = PeftModel.from_pretrained(base, str(args.adapter))
    merged = model.merge_and_unload()

    args.merged.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(args.merged), safe_serialization=True)
    tokenizer.save_pretrained(str(args.merged))
    _copy_tokenizer_files(args.base, args.merged)
    print(f"Merged model saved to {args.merged}")

    if args.skip_gguf:
        return 0

    converter = _find_converter()
    if converter is None:
        print(
            "llama.cpp convert script not found. Install llama.cpp to export GGUF:\n"
            "  git clone https://github.com/ggerganov/llama.cpp",
            file=sys.stderr,
        )
        return 1

    f16_gguf = args.merged.parent / f"{args.merged.name}.f16.gguf"
    print("Converting to GGUF (f16)...")
    subprocess.run(
        [sys.executable, str(converter), str(args.merged), "--outfile", str(f16_gguf), "--outtype", "f16"],
        check=True,
    )
    print(f"GGUF written: {f16_gguf}")

    quantizer = _find_quantizer()
    if quantizer is None:
        print(
            "llama-quantize not found; leaving the f16 GGUF as the output.\n"
            "Build it with: cmake -B llama.cpp/build && "
            "cmake --build llama.cpp/build --target llama-quantize",
            file=sys.stderr,
        )
        shutil.move(str(f16_gguf), str(args.gguf))
        return 0

    args.gguf.parent.mkdir(parents=True, exist_ok=True)
    print(f"Quantizing to {args.quant}...")
    subprocess.run(
        [str(quantizer), str(f16_gguf), str(args.gguf), args.quant],
        check=True,
    )
    print(f"Quantized GGUF written: {args.gguf}")

    if not args.keep_f16:
        f16_gguf.unlink(missing_ok=True)
    return 0


def _find_converter() -> Path | None:
    candidates = [
        ROOT / "llama.cpp" / "convert_hf_to_gguf.py",
        ROOT / "llama.cpp" / "convert-hf-to-gguf.py",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _find_quantizer() -> Path | None:
    candidates = [
        ROOT / "llama.cpp" / "build" / "bin" / "llama-quantize",
        ROOT / "llama.cpp" / "llama-quantize",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


if __name__ == "__main__":
    raise SystemExit(main())
