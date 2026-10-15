"""OpenAI-compatible endpoints so existing clients work unchanged.

Mount as a router on the main app. Supports both streaming and
non-streaming chat completions plus the legacy text completion route.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .config import MODEL_PRESETS, settings
from .engine import get_engine

router = APIRouter()


def _engine_for(model: str | None):
    """Pick an engine when the client asks for a known preset, else default.

    Unknown model names stay lenient (OpenAI clients send arbitrary names like
    ``gpt-3.5-turbo``), but a known preset/path whose file is missing is a
    client error and is reported as 404 rather than a 500 traceback.
    """
    config = settings.with_model(model) if model in MODEL_PRESETS else settings
    try:
        config.require_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return get_engine(config)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "phi-3-mini"
    messages: list[ChatMessage]
    max_tokens: int | None = Field(default=None, ge=1, le=4096)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    stream: bool = False


class CompletionRequest(BaseModel):
    model: str = "phi-3-mini"
    prompt: str
    max_tokens: int | None = Field(default=None, ge=1, le=4096)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    stream: bool = False


def _chat_chunks(completion_id: str, created: int, model: str, pieces):
    for piece in pieces:
        delta = {"content": piece}
        chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {"index": 0, "delta": delta, "finish_reason": None}
            ],
        }
        yield f"data: {json.dumps(chunk)}\n\n"
    final = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(final)}\n\n"
    yield "data: [DONE]\n\n"


@router.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    if not req.messages:
        raise HTTPException(status_code=422, detail="messages must not be empty")
    engine = _engine_for(req.model)
    messages = [m.model_dump() for m in req.messages]
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    model_name = req.model or settings.resolved_model_path().stem

    if req.stream:
        pieces = engine.stream(
            messages,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
        )
        return StreamingResponse(
            _chat_chunks(completion_id, created, model_name, pieces),
            media_type="text/event-stream",
        )

    result = engine.chat(
        messages,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
    )
    usage = result.get("usage", {})
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result["content"]},
                "finish_reason": result.get("finish_reason") or "stop",
            }
        ],
        "usage": {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        },
    }


@router.post("/v1/completions")
def completions(req: CompletionRequest):
    engine = _engine_for(req.model)
    messages = [{"role": "user", "content": req.prompt}]
    completion_id = f"cmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    model_name = req.model or settings.resolved_model_path().stem

    if req.stream:
        def chunks():
            for piece in engine.stream(
                messages,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
            ):
                payload: dict[str, Any] = {
                    "id": completion_id,
                    "object": "text_completion",
                    "created": created,
                    "model": model_name,
                    "choices": [
                        {"index": 0, "text": piece, "finish_reason": None}
                    ],
                }
                yield f"data: {json.dumps(payload)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(chunks(), media_type="text/event-stream")

    result = engine.chat(
        messages,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
    )
    usage = result.get("usage", {})
    return {
        "id": completion_id,
        "object": "text_completion",
        "created": created,
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "text": result["content"],
                "finish_reason": result.get("finish_reason") or "stop",
            }
        ],
        "usage": {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        },
    }


@router.get("/v1/models")
def list_models() -> dict:
    now = int(time.time())
    ids = list(MODEL_PRESETS)
    return {
        "object": "list",
        "data": [
            {"id": name, "object": "model", "created": now, "owned_by": "local"}
            for name in ids
        ],
    }
