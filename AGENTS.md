# AGENTS.md

## Project
Local Phi-3 Mini chatbot (Microsoft Phi-3 Mini 4k Instruct) running on CPU via
llama.cpp. Provides a CLI, a FastAPI REST API (custom + OpenAI-compatible), a
small web chat UI, and an optional Foto & Video mode (image generation + short
MP4 clips).

## Layout
- `phi3/config.py` — central settings + model presets (`q4`, `q8`, `lora`, `marbel`).
- `phi3/engine.py` — thread-safe llama.cpp wrapper (chat + streaming).
- `phi3/model_slot.py` — one-entry cache shared by the text and image engines,
  so the two heavyweight models are never resident at the same time.
- `phi3/media.py` — image generation (`sd-turbo`) and MP4 assembly.
- `phi3/media_api.py` — media router (`/v1/media/status|image|video`).
- `phi3/server.py` — FastAPI app (`/health`, `/v1/chat`, `/v1/chat/stream`, `/`).
- `phi3/openai_api.py` — OpenAI-compatible router (`/v1/chat/completions`,
  `/v1/completions`, `/v1/models`).
- `phi3/cli.py` — terminal chat (`python -m phi3.cli`).
- `scripts/` — `download_model.py`, `download_media_model.py`, `smoke_test.py`,
  `api_test.py`, LoRA pipeline (`train_lora.py`, `merge_and_convert.py`,
  `test_adapter.py`).
- `static/index.html` — web UI (single file: design tokens, chat, settings,
  Foto & Video tab).

## Environment / build
- Use the project venv: `.venv/bin/python` (Python 3.13).
- Runtime deps: `pip install -r requirements.txt`. Media (Foto & Video) deps:
  `pip install -r requirements-media.txt` (diffusers, pillow, imageio-ffmpeg).
  LoRA deps: `pip install -r requirements-lora.txt`.
- `llama-cpp-python` is installed from the CPU wheel index
  (`--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu`).
- Keep `numpy>=2` — the GGUF converter's deps (`datasets`/`pyarrow`) require it,
  and `llama-cpp-python` 0.3.36 works with it.
- Models live in `models/` and are gitignored (`.gguf`). Download with
  `python scripts/download_model.py` (Q4) — Q8 comes from the bartowski repo
  (microsoft's repo has no Q8).
- Image model: `python scripts/download_media_model.py` fetches `sd-turbo` into
  `models/media/sd-turbo` (~2.5 GB, also gitignored). Override with
  `PHI3_MEDIA_MODEL`.

## Running / testing
- Start the server: `./run.sh` (port 12000; respects `PHI3_PORT`,
  `PHI3_MODEL_PATH`). Web UI: `https://antono4.github.io/phi3-mini-ai/static/`
  (static hosting; it auto-targets the public backend on port 12000). Local
  preview: `http://localhost:12000/`.
- Smoke test: `.venv/bin/python scripts/smoke_test.py` (expects "ALL TESTS PASSED").
- Full API test: `.venv/bin/python scripts/api_test.py` while the server runs
  (77 checks: presets, validation, SSE, OpenAI routes, CORS, error mapping, and
  media PNG/MP4 + seed reproduction). The media block self-skips when the image
  model is absent.
- `models/*.gguf` and `llama.cpp/` are gitignored; don't commit them.
  `models/media/` (the image weights) and `node_modules/` are gitignored too.

## Gotchas
- The web UI is one self-contained `static/index.html` (no build step, no CDN):
  design tokens in `:root`, dark/light via `data-theme` on `<html>`, and a small
  markdown renderer. That renderer detects block structure on the raw text and
  escapes each fragment as it emits it, so model output cannot inject HTML —
  keep the escape-on-emit order if you extend it. `body` uses
  `grid-template-columns: minmax(0, 1fr)`; with an `auto` track the topbar's
  content width widens the column and the page overflows at ~768px.
- Only one GGUF is kept in memory at a time. `get_engine()` closes the previous
  engine before loading another model; caching one engine per model path leaks
  ~2.5 GB per switch and will exhaust the sandbox (seen: 10.8 GB RSS). If you
  change the cache, keep the single-model behavior and the `close()` lock.
- The same slot now also holds the image model, so text and image never coexist.
  `ModelSlot` frees a model by calling `close()` on it, and a bare diffusers
  pipeline has no `close()` — that is why `media.py` wraps it in `MediaEngine`.
  Without that wrapper eviction is a no-op: the weights stay resident and the
  next text load piles on top (seen: 8.2 GB climbing past 10.7 GB). The wrapper
  also holds a lock, because a diffusers pipeline is not safe to call from two
  threads at once (shared buffers, per-call step callback).
- Media generation is synchronous and CPU-bound, so a single image is seconds and
  a video is minutes (one frame per image). The UI shows a busy state and an ETA;
  keep video short (`MAX_VIDEO_SECONDS = 4`) or the request outlives the client.
- Generated frames are written to a private `tempfile.mkdtemp()` dir and never
  served over HTTP. Downloads happen client-side from a blob. Don't add a static
  mount for generated media — that would expose every user's output by URL.
- `static/index.html` hides panels with the `hidden` attribute. Because
  `[hidden]` sets `display: none` at low specificity, any `display: flex/grid`
  rule on the same element silently overrides it; gate such rules on
  `:not([hidden])` instead of relying on the attribute alone.
- `scripts/api_test.py` returns decoded text, so binary endpoints must be called
  with `binary=True`. Header names come back lowercased (HTTP headers are
  case-insensitive; a plain dict is not) — look them up as `content-type`.
- Inference errors are mapped to 4xx, not 500: a missing model file → 404, a
  prompt over `n_ctx` (llama.cpp `ValueError`) → 400. The streaming route emits
  `event: error` before `event: done`; the web UI unwraps that event.
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
