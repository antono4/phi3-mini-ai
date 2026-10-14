"""FastAPI server exposing Phi-3 Mini over HTTP.

Endpoints:
    GET  /health            liveness + model info
    POST /v1/chat           non-streaming JSON completion
    POST /v1/chat/stream    Server-Sent Events token stream
    GET  /                  small chat UI served from static/index.html
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from . import __version__
from .config import MODEL_PRESETS, settings
from .engine import get_engine
from .openai_api import router as openai_router

app = FastAPI(title="Phi-3 Mini API", version=__version__)
# Allow browser clients (e.g. the Marbel AI GitHub Pages UI) to call this API
# directly from another origin. Restrict with PHI3_ALLOW_ORIGIN for production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.allow_origin],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)
app.include_router(openai_router)
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


class Message(BaseModel):
    role: str = Field(pattern="^(system|user|assistant)$")
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]
    model: str | None = Field(default=None, description="Model preset (q4/q8) or path")
    max_tokens: int | None = Field(default=None, ge=1, le=4096)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    stream: bool = False


def _as_dicts(req: ChatRequest) -> list[dict[str, str]]:
    if not req.messages:
        raise HTTPException(status_code=422, detail="messages must not be empty")
    return [m.model_dump() for m in req.messages]


def _engine_for(req: ChatRequest):
    return get_engine(settings.with_model(req.model))


@app.get("/health")
def health() -> dict:
    model_path = settings.resolved_model_path()
    return {
        "status": "ok",
        "model": model_path.name,
        "model_present": model_path.exists(),
        "presets": sorted(MODEL_PRESETS),
        "n_ctx": settings.n_ctx,
        "n_threads": settings.n_threads,
    }


@app.get("/")
def index() -> FileResponse:
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="UI not installed")
    return FileResponse(index_file)


@app.post("/v1/chat")
def chat(req: ChatRequest) -> dict:
    engine = _engine_for(req)
    result = engine.chat(
        _as_dicts(req),
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
    )
    return result


@app.post("/v1/chat/stream")
def chat_stream(req: ChatRequest) -> StreamingResponse:
    engine = _engine_for(req)
    payload = _as_dicts(req)

    def event_source():
        try:
            for piece in engine.stream(
                payload,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
            ):
                yield f"data: {json.dumps({'delta': piece})}\n\n"
        except Exception as exc:  # surface errors to the client as an SSE event
            yield f"event: error\ndata: {json.dumps({'error': str(exc)})}\n\n"
        yield "event: done\ndata: [DONE]\n\n"

    return StreamingResponse(event_source(), media_type="text/event-stream")
