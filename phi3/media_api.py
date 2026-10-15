"""HTTP routes for image generation and video assembly.

Endpoints:
    GET  /v1/media/status    what the media model can do right now
    POST /v1/media/image     text -> PNG (base64 JSON, or raw bytes)
    POST /v1/media/video     text -> MP4 (base64 JSON, or raw bytes)

Generation takes seconds on CPU, so these are plain synchronous endpoints
rather than streams; the UI shows a busy state while waiting.
"""

from __future__ import annotations

import base64

from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel, Field

from . import media

router = APIRouter(prefix="/v1/media", tags=["media"])


class ImageRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=1000)
    negative_prompt: str | None = Field(default=None, max_length=500)
    steps: int | None = Field(default=None, ge=1, le=8)
    size: int | None = Field(default=None, ge=128, le=768)
    seed: int | None = Field(default=None, ge=-1)
    # "json" returns base64 for the UI to render inline; "raw" returns the file
    # itself, which is what a download link or curl wants.
    format: str = Field(default="json", pattern="^(json|raw)$")


class VideoRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=1000)
    seconds: float = Field(default=2.0, ge=1.0, le=media.MAX_VIDEO_SECONDS)
    fps: int = Field(default=8, ge=4, le=16)
    size: str = Field(default="512x288", pattern="^(512p|512x288|384p)$")
    steps: int | None = Field(default=None, ge=1, le=4)
    seed: int | None = Field(default=None, ge=-1)
    format: str = Field(default="json", pattern="^(json|raw)$")


def _media_guard() -> None:
    if not media.model_available():
        raise HTTPException(
            status_code=503,
            detail=(
                "Media model is not installed. Run "
                "`python scripts/download_media_model.py` or set PHI3_MEDIA_MODEL."
            ),
        )


def _failure(exc: Exception, what: str) -> HTTPException:
    """Map generation errors to a 4xx the client can act on.

    Bad input is already a ValueError (handled app-wide as 400); what is left is
    a missing dependency or a broken install, which is a server-side problem.
    """
    if isinstance(exc, RuntimeError):
        return HTTPException(status_code=503, detail=str(exc))
    return HTTPException(status_code=500, detail=f"{what} failed: {exc}")


@router.get("/status")
def media_status() -> dict:
    path = media._model_path()
    return {
        "available": media.model_available(),
        "model": path.name,
        "model_path": str(path),
        "sizes": sorted(media.VIDEO_SIZES),
        "max_video_seconds": media.MAX_VIDEO_SECONDS,
        "default_steps": media.DEFAULT_STEPS,
        "device": "cpu",
    }


@router.post("/image")
def create_image(req: ImageRequest) -> Response:
    _media_guard()
    try:
        data, used_seed = media.generate_image(
            req.prompt,
            negative_prompt=req.negative_prompt,
            steps=req.steps,
            size=req.size,
            seed=req.seed,
        )
    except ValueError:
        raise
    except Exception as exc:
        raise _failure(exc, "image generation") from exc

    if req.format == "raw":
        return Response(content=data, media_type="image/png",
                        headers={"X-Seed": str(used_seed)})
    return Response(
        content=_json({"image": base64.b64encode(data).decode("ascii"),
                       "mime": "image/png", "seed": used_seed}),
        media_type="application/json",
    )


@router.post("/video")
def create_video(req: VideoRequest) -> Response:
    _media_guard()
    try:
        data, used_seed, frames = media.generate_video(
            req.prompt,
            seconds=req.seconds,
            fps=req.fps,
            size=req.size,
            steps=req.steps,
            seed=req.seed,
        )
    except ValueError:
        raise
    except Exception as exc:
        raise _failure(exc, "video generation") from exc

    if req.format == "raw":
        return Response(content=data, media_type="video/mp4",
                        headers={"X-Seed": str(used_seed), "X-Frames": str(frames)})
    return Response(
        content=_json({"video": base64.b64encode(data).decode("ascii"),
                       "mime": "video/mp4", "seed": used_seed, "frames": frames,
                       "fps": req.fps}),
        media_type="application/json",
    )


def _json(payload: dict) -> bytes:
    import json

    return json.dumps(payload).encode("utf-8")
