"""Test a trained LoRA adapter by generating a reply with it.

    python scripts/test_adapter.py --adapter models/adapters/phi3-lora \
        --prompt "What is the capital of Indonesia?"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
BASE_MODEL = ROOT / "models" / "phi3-hf-base"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate with a LoRA adapter.")
    parser.add_argument("--base", type=Path, default=BASE_MODEL)
    parser.add_argument("--adapter", type=Path, default=ROOT / "models" / "adapters" / "phi3-lora")
    parser.add_argument("--prompt", default="What is the capital of Indonesia?")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    print("Loading model + adapter")
    tokenizer = AutoTokenizer.from_pretrained(str(args.base))
    base = AutoModelForCausalLM.from_pretrained(
        str(args.base), dtype=torch.bfloat16
    )
    model = PeftModel.from_pretrained(base, str(args.adapter))
    model.eval()

    chat = f"<|user|>\n{args.prompt}<|end|>\n<|assistant|>\n"
    inputs = tokenizer(chat, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print("--- prompt ---")
    print(args.prompt)
    print("--- response ---")
    print(text.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
