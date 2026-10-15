"""FastAPI server exposing Phi-3 Mini over HTTP.

Endpoints:
    GET  /health            liveness + model info
    POST /v1/chat           non-streaming JSON completion
    POST /v1/chat/stream    Server-Sent Events token stream
    /v1/media/*             image generation and video assembly (see media_api)
    GET  /                  small chat UI served from static/index.html
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from . import __version__
from .config import MODEL_PRESETS, PROJECT_ROOT, settings
from .engine import get_engine
from .media_api import router as media_router
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
app.include_router(media_router)
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


@app.exception_handler(ValueError)
async def _value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    """Map inference errors to 4xx instead of leaking 500 tracebacks.

    llama.cpp raises ValueError for prompts that exceed the context window, and
    the engine raises it for an empty message list. Both are caused by the
    request, so a 400 with the reason is a better answer than a server error.
    """
    return JSONResponse(status_code=400, content={"detail": str(exc)})


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
    config = settings.with_model(req.model)
    try:
        config.require_model()
    except FileNotFoundError as exc:
        # A requested preset/path with no file on disk is a client error, not a
        # server fault; surface it as 404 instead of a 500 traceback.
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return get_engine(config)


@app.get("/health")
def health() -> dict:
    model_path = settings.resolved_model_path()
    # Which preset (if any) points at the model currently loaded, so a UI can
    # default its model selector to what the server actually serves.
    active = next(
        (
            name
            for name, rel in MODEL_PRESETS.items()
            if (PROJECT_ROOT / rel).resolve() == model_path
        ),
        None,
    )
    return {
        "status": "ok",
        "model": model_path.name,
        "model_present": model_path.exists(),
        "preset": active,
        "presets": sorted(MODEL_PRESETS),
        "presets_info": [
            {
                "name": name,
                "available": (PROJECT_ROOT / rel).is_file(),
                "path": rel,
            }
            for name, rel in sorted(MODEL_PRESETS.items())
        ],
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
