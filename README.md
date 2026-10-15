# Phi-3 Mini AI

Chatbot AI lokal berbasis **Phi-3 Mini 4k Instruct** (Microsoft), dijalankan
dengan `llama.cpp` pada CPU. Menyediakan tiga cara pakai: **CLI**, **REST API**,
dan **Web UI**.

Model yang dipakai: `Phi-3-mini-4k-instruct-q4.gguf` (kuantisasi Q4, ~2,3 GB)
sehingga cukup untuk dijalankan tanpa GPU.

## Fitur

- Inferensi lokal penuh â€” tanpa panggilan API eksternal saat runtime.
- Chat CLI dengan streaming token dan riwayat percakapan.
- REST API (non-streaming & SSE streaming).
- **Endpoint kompatibel OpenAI** (`/v1/chat/completions`, `/v1/completions`,
  `/v1/models`) sehingga SDK/klien OpenAI (mis. `openai-python`, LangChain)
  bisa langsung dipakai.
- Pilihan model lewat preset `q4` (default, ~2,3 GB) dan `q8` (~4,7 GB).
- Web UI chat satu berkas (disajikan langsung oleh server): tema terang/gelap,
  render markdown + blok kode, tombol salin dan ulangi jawaban, serta pemilih
  preset model — tanpa build step dan tanpa CDN.
- **Pipeline fine-tuning LoRA**: latih adapter, merge ke base, ekspor ke GGUF.
- Konfigurasi lewat variabel environment.

## Instalasi

```bash
pip install -r requirements.txt
# unduh model (~2,3 GB) ke folder models/
python scripts/download_model.py
```

## Menjalankan

### 1. Web UI + API (disarankan)

```bash
./run.sh
# atau
uvicorn phi3.server:app --host 0.0.0.0 --port 12000
```

Buka `http://localhost:12000` di browser.

### 2. Chat di terminal

```bash
python -m phi3.cli
```

Perintah dalam chat: `/reset`, `/help`, `/exit`.

One-shot prompt:

```bash
python -m phi3.cli -p "Jelaskan fotosintesis secara singkat."
```

## API

| Method | Path | Deskripsi |
| ------ | ---- | --------- |
| GET | `/health` | Status server & info model |
| POST | `/v1/chat` | Completion JSON (non-streaming) |
| POST | `/v1/chat/stream` | Streaming token via SSE |
| POST | `/v1/chat/completions` | Chat completion kompatibel OpenAI (JSON/SSE) |
| POST | `/v1/completions` | Text completion kompatibel OpenAI (legacy) |
| GET | `/v1/models` | Daftar model (preset) |
| GET | `/` | Web UI chat |

Contoh:

```bash
curl -X POST http://localhost:12000/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Halo!"}],"max_tokens":128}'
```

Streaming:

```bash
curl -N -X POST http://localhost:12000/v1/chat/stream \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hitung 1 sampai 5."}]}'
```

Format body `POST /v1/chat`:

```json
{
  "messages": [{"role": "user", "content": "..."}],
  "max_tokens": 512,
  "temperature": 0.7,
  "top_p": 0.9
}
```

## Konfigurasi

Semua pengaturan dapat dioverride lewat environment variable:

| Variabel | Default | Keterangan |
| -------- | ------- | ---------- |
| `PHI3_MODEL_PATH` | `models/Phi-3-mini-4k-instruct-q4.gguf` | Lokasi file model |
| `PHI3_N_CTX` | `4096` | Ukuran context window |
| `PHI3_N_THREADS` | jumlah CPU | Thread inferensi |
| `PHI3_N_GPU_LAYERS` | `0` | Layer ke GPU (0 = CPU saja) |
| `PHI3_MAX_TOKENS` | `512` | Panjang balasan maksimum |
| `PHI3_TEMPERATURE` | `0.7` | Kreativitas sampling |
| `PHI3_TOP_P` | `0.9` | Nucleus sampling |
| `PHI3_SYSTEM_PROMPT` | (bawaan) | Peran/persona asisten |
| `PHI3_PORT` | `12000` | Port server |
| `PHI3_ALLOW_ORIGIN` | `*` | Origin yang diizinkan CORS |

## Pengujian

```bash
python scripts/smoke_test.py        # engine + skema (butuh model)
python scripts/api_test.py          # uji menyeluruh REST + OpenAI API (server harus jalan)
```

`api_test.py` memeriksa health/preset, validasi input, keempat preset, streaming
SSE, endpoint kompatibel OpenAI, CORS, dan penanganan error (model hilang → 404,
prompt melebihi context window → 400).

## Struktur proyek

```
phi3/
  config.py    # konfigurasi terpusat (+ preset model)
  engine.py    # wrapper llama.cpp (chat + streaming)
  openai_api.py# endpoint kompatibel OpenAI
  cli.py       # chat terminal
  server.py    # FastAPI: UI + REST API
static/
  index.html   # web UI chat
scripts/
  download_model.py
  train_lora.py        # latih adapter LoRA
  merge_and_convert.py # merge adapter + ekspor GGUF
  test_adapter.py      # uji adapter
  smoke_test.py
data/
  sample_instructions.jsonl
requirements.txt       # dependensi runtime
requirements-lora.txt  # dependensi pipeline LoRA
run.sh         # start server
```

## Endpoint kompatibel OpenAI

Gunakan SDK `openai` biasa dengan `base_url` ke server lokal:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:12000/v1", api_key="local")
resp = client.chat.completions.create(
    model="phi-3-mini",
    messages=[{"role": "user", "content": "Halo!"}],
)
print(resp.choices[0].message.content)
```

Nilai `model` boleh `phi-3-mini` (default), atau preset `q4`/`q8` untuk memilih
bobot tertentu.

## Fine-tuning LoRA

Pipeline opsional untuk melatih adapter LoRA, menggabungkannya ke base model,
lalu mengekspor GGUF yang bisa dimuat server. Dependensi tambahan:

```bash
pip install -r requirements-lora.txt
```

1. Unduh bobot HuggingFace base (Phi-3 Mini 4k Instruct):

   ```bash
   python -c "from huggingface_hub import snapshot_download as s; \
   s('microsoft/Phi-3-mini-4k-instruct', local_dir='models/phi3-hf-base', \
   allow_patterns=['*.json','*.model','*.safetensors','*.py','*.txt','tokenizer*','*.jinja','*.md'])"
   ```

2. Latih adapter pada dataset instruksi (JSONL `instruction`/`response`):

   ```bash
   python scripts/train_lora.py --data data/sample_instructions.jsonl \
       --epochs 1 --rank 8
   ```

3. Uji adapter:

   ```bash
   python scripts/test_adapter.py --prompt "What is the capital of Indonesia?"
   ```

4. Merge + ekspor GGUF terkuantisasi (perlu checkout `llama.cpp`):

   ```bash
   git clone https://github.com/ggerganov/llama.cpp
   cmake -B llama.cpp/build && cmake --build llama.cpp/build --target llama-quantize
   python scripts/merge_and_convert.py --quant Q4_K_M
   ```

   Hasilnya: `models/phi3-lora-q4.gguf`. Jalankan server dengan
   `PHI3_MODEL_PATH=models/phi3-lora-q4.gguf ./run.sh`.

### Model `marbel` (untuk Marbel AI)

Preset `marbel` adalah adapter LoRA yang dilatih khusus untuk
[Marbel AI](https://antono4.github.io/MarbelAI/) pada dataset
`data/marbel_instructions.jsonl` (asisten gratis berbahasa Indonesia), lalu
di-merge dan dikuantisasi menjadi `models/marbel-phi3-q4.gguf`:

```bash
python scripts/train_lora.py --data data/marbel_instructions.jsonl \
    --output models/adapters/marbel-lora --epochs 1 --rank 8 --max-length 128
python scripts/merge_and_convert.py --adapter models/adapters/marbel-lora \
    --merged /tmp/marbel-merged --gguf models/marbel-phi3-q4.gguf --quant Q4_K_M
```

Jalankan server dengan preset itu:

```bash
PHI3_MODEL=marbel ./run.sh
```

Server mengaktifkan **CORS** (`PHI3_ALLOW_ORIGIN`, default `*`) sehingga halaman
Marbel AI di GitHub Pages dapat memanggil `/v1/chat/completions` langsung dari
browser. Atur ke origin tertentu untuk produksi.

## Catatan performa

Berjalan di CPU (4 vCPU, ~15 GB RAM). Kecepatan generasi kira-kira 10â€“15 token/detik.
Naikkan `PHI3_N_GPU_LAYERS` bila ada GPU dengan build `llama-cpp-python` yang mendukung CUDA.
