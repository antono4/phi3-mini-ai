# AGENTS.md

## Project
Local Phi-3 Mini chatbot (Microsoft Phi-3 Mini 4k Instruct) running on CPU via
llama.cpp. Provides a CLI, a FastAPI REST API (custom + OpenAI-compatible), and
a small web chat UI.

## Layout
- `phi3/config.py` — central settings + model presets (`q4`, `q8`, `lora`, `marbel`).
- `phi3/engine.py` — thread-safe llama.cpp wrapper (chat + streaming).
- `phi3/server.py` — FastAPI app (`/health`, `/v1/chat`, `/v1/chat/stream`, `/`).
- `phi3/openai_api.py` — OpenAI-compatible router (`/v1/chat/completions`,
  `/v1/completions`, `/v1/models`).
- `phi3/cli.py` — terminal chat (`python -m phi3.cli`).
- `scripts/` — `download_model.py`, `smoke_test.py`, LoRA pipeline
  (`train_lora.py`, `merge_and_convert.py`, `test_adapter.py`).
- `static/index.html` — web UI.

## Environment / build
- Use the project venv: `.venv/bin/python` (Python 3.13).
- Runtime deps: `pip install -r requirements.txt`. LoRA deps:
  `pip install -r requirements-lora.txt`.
- `llama-cpp-python` is installed from the CPU wheel index
  (`--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu`).
- Keep `numpy>=2` — the GGUF converter's deps (`datasets`/`pyarrow`) require it,
  and `llama-cpp-python` 0.3.36 works with it.
- Models live in `models/` and are gitignored (`.gguf`). Download with
  `python scripts/download_model.py` (Q4) — Q8 comes from the bartowski repo
  (microsoft's repo has no Q8).

## Running / testing
- Start the server: `./run.sh` (port 12000; respects `PHI3_PORT`,
  `PHI3_MODEL_PATH`). Public preview: `https://work-1-<host>/` (port 12000).
- Smoke test: `.venv/bin/python scripts/smoke_test.py` (expects "ALL TESTS PASSED").
- `models/*.gguf` and `llama.cpp/` are gitignored; don't commit them.

## Gotchas
- CORS is enabled (`PHI3_ALLOW_ORIGIN`, default `*`) so browser clients such as
  the Marbel AI GitHub Pages UI can call `/v1/chat/completions` directly. Set a
  specific origin in production.
- The `marbel` preset is the Phi-3 Mini LoRA adapter fine-tuned for Marbel AI
  (`data/marbel_instructions.jsonl`, adapter in `models/adapters/marbel-lora`),
  exported to `models/marbel-phi3-q4.gguf`. It is consumed by the Marbel AI repo
  (`MARBEL_API_BASE` in its `.env`).
- The sandbox `/workspace` volume is small (~25 GB). Stage large model
  intermediates (HF base, f16 GGUF, merged HF) under `/tmp` (larger overlay),
  then copy only the final quantized GGUF into `models/`.
- `save_pretrained` does not write the SentencePiece `tokenizer.model`, which
  `convert_hf_to_gguf.py` needs for Phi-3. `merge_and_convert.py` copies the
  tokenizer files from the base model to fix this.
- Quantizing needs a built `llama-quantize`:
  `cmake -B llama.cpp/build && cmake --build llama.cpp/build --target llama-quantize`.
