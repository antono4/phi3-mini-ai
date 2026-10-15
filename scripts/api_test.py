#!/usr/bin/env python3
"""Comprehensive end-to-end API test harness (no external deps beyond stdlib)."""
import json
import sys
import urllib.error
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:12000"
fails = []
passes = 0


def check(name, cond, detail=""):
    global passes
    if cond:
        passes += 1
        print(f"PASS | {name}" + (f" :: {detail}" if detail else ""))
    else:
        fails.append(name)
        print(f"FAIL | {name} :: {detail}")


def req(method, path, body=None, headers=None, timeout=180):
    url = BASE + path
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(url, data=data, method=method)
    r.add_header("Content-Type", "application/json")
    for k, v in (headers or {}).items():
        r.add_header(k, v)
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            raw = resp.read().decode()
            return resp.status, dict(resp.headers), raw
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read().decode()


# ---------------------------------------------------------------- health
st, h, raw = req("GET", "/health")
health = json.loads(raw)
check("health 200", st == 200, f"status={st}")
check("health status ok", health.get("status") == "ok")
check("health has model", bool(health.get("model")), health.get("model"))
check("health preset matches loaded model", health.get("preset") == "q4", str(health.get("preset")))
check("health presets sorted", health.get("presets") == sorted(health.get("presets", [])))
names = [p["name"] for p in health.get("presets_info", [])]
check("presets_info covers all presets", names == sorted(health.get("presets", [])), str(names))
check("presets_info availability flags", all(isinstance(p["available"], bool) for p in health["presets_info"]))

# ------------------------------------------------------------ CORS preflight
st, hdr, _ = req("OPTIONS", "/v1/chat", headers={"Origin": "https://antono4.github.io",
                                                 "Access-Control-Request-Method": "POST",
                                                 "Access-Control-Request-Headers": "content-type"})
check("CORS preflight 200/204", st in (200, 204), f"status={st}")
check("CORS allows Pages origin", hdr.get("access-control-allow-origin") in ("*", "https://antono4.github.io"),
      str(hdr.get("access-control-allow-origin")))

# ------------------------------------------------------- custom API presets
for preset in ["q4", "q8", "lora", "marbel"]:
    st, _, raw = req("POST", "/v1/chat", {
        "messages": [{"role": "user", "content": "Sebutkan satu warna."}],
        "model": preset, "max_tokens": 24,
    })
    try:
        d = json.loads(raw)
        reply = d.get("content", "")
    except Exception:
        reply = ""
    check(f"/v1/chat preset {preset}", st == 200 and bool(reply.strip()), f"status={st} reply={reply!r}")

# -------------------------------------------------------------- validation
st, _, raw = req("POST", "/v1/chat", {"messages": []})
check("/v1/chat empty messages -> 422", st == 422, f"status={st}")

st, _, raw = req("POST", "/v1/chat", {"messages": [{"role": "bogus", "content": "hi"}]})
check("/v1/chat bad role -> 422", st == 422, f"status={st}")

st, _, raw = req("POST", "/v1/chat", {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 99999})
check("/v1/chat max_tokens out of range -> 422", st == 422, f"status={st}")

st, _, raw = req("POST", "/v1/chat", {"messages": [{"role": "user", "content": "hi"}], "temperature": 5.0})
check("/v1/chat temperature out of range -> 422", st == 422, f"status={st}")

st, _, raw = req("POST", "/v1/chat", {"messages": [{"role": "user", "content": "hi"}], "top_p": 0})
check("/v1/chat top_p=0 -> 422", st == 422, f"status={st}")

# A requested model with no file on disk is a client error, not a 500.
st, _, raw = req("POST", "/v1/chat", {"messages": [{"role": "user", "content": "hi"}], "model": "does-not-exist", "max_tokens": 8})
check("/v1/chat unknown model -> 404", st == 404, f"status={st}")
check("unknown model error is informative", "Model not found" in raw and "download_model.py" in raw, raw[:120])

st, _, raw = req("POST", "/v1/chat", {"messages": [{"role": "user", "content": "hi"}], "model": "/tmp/nope.gguf", "max_tokens": 8})
check("/v1/chat missing path -> 404", st == 404, f"status={st}")

# OpenAI route stays lenient on unknown names (SDKs send gpt-3.5-turbo etc.)
st, _, raw = req("POST", "/v1/chat/completions", {"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8})
check("openai unknown name falls back -> 200", st == 200, f"status={st}")

# ------------------------------------------------------- custom SSE stream
st, hdr, raw = req("POST", "/v1/chat/stream", {"messages": [{"role": "user", "content": "Hitung 1 sampai 3."}], "max_tokens": 32})
check("/v1/chat/stream 200", st == 200, f"status={st}")
check("/v1/chat/stream content-type", "text/event-stream" in hdr.get("content-type", ""), hdr.get("content-type"))
check("/v1/chat/stream has data events", raw.count("data:") >= 1, f"{raw.count('data:')} events")
check("/v1/chat/stream ends with done", "event: done" in raw and "[DONE]" in raw)

# ------------------------------------------------- OpenAI-compatible routes
st, _, raw = req("GET", "/v1/models")
d = json.loads(raw)
ids = [m["id"] for m in d.get("data", [])]
check("/v1/models 200", st == 200, f"status={st}")
check("/v1/models lists presets", set(ids) == set(health["presets"]), str(ids))
check("/v1/models object=list", d.get("object") == "list")

st, _, raw = req("POST", "/v1/chat/completions", {
    "model": "q4", "messages": [{"role": "user", "content": "Sebutkan satu warna."}], "max_tokens": 24})
d = json.loads(raw)
check("/v1/chat/completions 200", st == 200, f"status={st}")
check("openai response shape", d.get("object") == "chat.completion" and d["choices"][0]["message"]["role"] == "assistant",
      str(d.get("object")))
check("openai usage tokens", d.get("usage", {}).get("total_tokens", 0) > 0, str(d.get("usage")))

st, hdr, raw = req("POST", "/v1/chat/completions", {
    "model": "q4", "messages": [{"role": "user", "content": "Hitung 1 sampai 3."}],
    "max_tokens": 32, "stream": True})
check("openai streaming 200", st == 200, f"status={st}")
check("openai streaming SSE chunks", "chat.completion.chunk" in raw and raw.strip().endswith(""))
check("openai streaming finish_reason stop", '"finish_reason": "stop"' in raw)

st, _, raw = req("POST", "/v1/completions", {"model": "q4", "prompt": "Ibu kota Indonesia adalah", "max_tokens": 16})
d = json.loads(raw)
check("/v1/completions 200", st == 200, f"status={st}")
check("/v1/completions object", d.get("object") == "text_completion", str(d.get("object")))
check("/v1/completions has text", bool(d["choices"][0].get("text", "").strip()), repr(d["choices"][0].get("text")))

st, _, raw = req("POST", "/v1/completions", {"model": "q4", "prompt": "Hitung 1 sampai 3.", "max_tokens": 32, "stream": True})
check("/v1/completions streaming 200", st == 200, f"status={st}")
check("/v1/completions streaming SSE", "text_completion" in raw and raw.strip().endswith("data: [DONE]"))

st, _, raw = req("POST", "/v1/chat/completions", {"messages": []})
check("openai empty messages -> 422", st == 422, f"status={st}")

# ---------------------------------------------------------------- static UI
st, hdr, raw = req("GET", "/")
check("GET / serves UI 200", st == 200, f"status={st}")
check("GET / is html", "text/html" in hdr.get("content-type", ""), hdr.get("content-type"))
check("UI references BACKEND_DEFAULT", "BACKEND_DEFAULT" in raw)

st, _, _ = req("GET", "/nope")
check("unknown path -> 404", st == 404, f"status={st}")

st, _, _ = req("GET", "/v1/chat")
check("GET on POST route -> 405", st == 405, f"status={st}")

# ------------------------------------------- over-long prompt (context window)
long_prompt = "kata " * 6000
st, _, raw = req("POST", "/v1/chat", {"messages": [{"role": "user", "content": long_prompt}], "max_tokens": 8})
check("/v1/chat overflow -> 400 (not 500)", st == 400, f"status={st}")
check("overflow error mentions context", "context" in raw.lower(), raw[:120])

st, _, raw = req("POST", "/v1/chat/completions", {"model": "q4", "messages": [{"role": "user", "content": long_prompt}], "max_tokens": 8})
check("openai overflow -> 400 (not 500)", st == 400, f"status={st}")

st, hdr, raw = req("POST", "/v1/chat/stream", {"messages": [{"role": "user", "content": long_prompt}], "max_tokens": 8})
check("stream overflow -> SSE error event", "event: error" in raw and "context" in raw.lower(), raw[:120])

print(f"\n{'='*40}\n{passes} passed, {len(fails)} failed")
if fails:
    print("FAILURES:")
    for f in fails:
        print("  -", f)
sys.exit(1 if fails else 0)