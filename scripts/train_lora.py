"""LoRA fine-tuning for Phi-3 Mini on CPU.

Loads the HuggingFace base weights, trains a small LoRA adapter on an
instruction dataset, and saves the adapter. The adapter can then be
merged into the base and converted to GGUF for use with the llama.cpp
server.

    python scripts/train_lora.py --data data/sample_instructions.jsonl --epochs 1

Because this environment is CPU-only, keep the dataset tiny and epochs low.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

ROOT = Path(__file__).resolve().parent.parent
BASE_MODEL = ROOT / "models" / "phi3-hf-base"
DEFAULT_OUT = ROOT / "models" / "adapters" / "phi3-lora"

PROMPT_TEMPLATE = (
    "<|user|>\n{instruction}<|end|>\n<|assistant|>\n{response}<|end|>\n"
)


def load_dataset(path: Path) -> Dataset:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return Dataset.from_list(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="LoRA fine-tune Phi-3 Mini.")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "sample_instructions.jsonl")
    parser.add_argument("--base", type=Path, default=BASE_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--rank", type=int, default=8)
    args = parser.parse_args()

    if not args.base.exists():
        raise FileNotFoundError(
            f"Base model not found at {args.base}. Download it first (see README)."
        )

    print(f"Loading tokenizer from {args.base}")
    tokenizer = AutoTokenizer.from_pretrained(str(args.base))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw = load_dataset(args.data)
    print(f"Dataset size: {len(raw)} examples")

    def format_example(ex):
        text = PROMPT_TEMPLATE.format(
            instruction=ex["instruction"], response=ex["response"]
        )
        return tokenizer(
            text,
            truncation=True,
            max_length=args.max_length,
            padding="max_length",
        )

    tokenized = raw.map(format_example, remove_columns=raw.column_names)
    tokenized = tokenized.map(lambda ex: {"labels": ex["input_ids"]})

    print(f"Loading base model from {args.base} (CPU, bfloat16)")
    model = AutoModelForCausalLM.from_pretrained(
        str(args.base), torch_dtype=torch.bfloat16
    )
    model.config.use_cache = False
    # Trades compute for memory: recompute activations in the backward pass.
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    lora = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank * 2,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=str(args.output),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_steps=1,
        save_strategy="epoch",
        report_to=[],
        use_cpu=True,
        fp16=False,
        bf16=True,
    )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=collator,
    )
    trainer.train()

    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.output))
    tokenizer.save_pretrained(str(args.output))
    print(f"Adapter saved to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
