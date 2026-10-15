"""Image generation and short video assembly.

The text model cannot produce pixels, so Foto/Video mode uses a separate
Stable Diffusion model. SD-Turbo is used because it is step-distilled: one
denoising step gives a usable image, which is the difference between ~6s and
several minutes per frame on a CPU-only machine.

There is no local text-to-video model here. A 33B omni-modal model needs a GPU
and tens of GB of VRAM, so "video" means generating keyframes with the image
model and encoding them into a clip. Frames are produced one at a time and
released immediately, which keeps peak memory at roughly a single image.
"""

from __future__ import annotations

import io
import math
import os
import shutil
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path

from .config import PROJECT_ROOT, settings
from .model_slot import slot

# SD-Turbo is distilled for guidance-free sampling; a non-zero guidance scale
# with one step produces washed-out output, so callers pass 0.
DEFAULT_STEPS = 1
DEFAULT_SIZE = 512
MIN_SIZE, MAX_SIZE = 128, 768
# (width, height) pairs divisible by 8, which the VAE requires.
VIDEO_SIZES = {"512p": (512, 512), "512x288": (512, 288), "384p": (384, 384)}
MAX_VIDEO_SECONDS = 4.0

def _model_path() -> Path:
    raw = os.environ.get("PHI3_MEDIA_MODEL")
    if raw:
        return Path(raw).expanduser()
    return PROJECT_ROOT / "models" / "media" / "sd-turbo"


def model_available() -> bool:
    return (_model_path() / "model_index.json").is_file()


def _load_pipeline():
    """Build the diffusers pipeline. Imported lazily so the text-only server
    starts without torch installed."""
    try:
        import torch
        from diffusers import AutoPipelineForText2Image
    except ImportError as exc:  # pragma: no cover - dependency hint
        raise RuntimeError(
            "Image generation needs the media extras. Install them with:\n"
            "  .venv/bin/pip install -r requirements-media.txt\n"
            "and download the model with:\n"
            "  .venv/bin/python scripts/download_media_model.py"
        ) from exc

    path = _model_path()
    if not (path / "model_index.json").is_file():
        raise RuntimeError(
            f"Media model not found at {path}. Download it with "
            f"`python scripts/download_media_model.py` or set PHI3_MEDIA_MODEL."
        )

    torch.set_num_threads(max(1, settings.n_threads))
    pipe = AutoPipelineForText2Image.from_pretrained(
        str(path),
        torch_dtype=torch.float32,
        variant="fp16",
        safety_checker=None,
        # Loading straight to CPU avoids a second full copy of the weights.
        low_cpu_mem_usage=True,
    )
    pipe.set_progress_bar_config(disable=True)
    return MediaEngine(pipe)


class MediaEngine:
    """Wraps the pipeline with a lock and a ``close()`` the model slot can call.

    ``ModelSlot`` frees a model by calling ``close()`` on it. A bare diffusers
    pipeline has no such method, so without this wrapper evicting the image
    model would leave its weights resident and the text model's reload would
    pile on top of them.

    A pipeline is also not safe to call from two threads at once: requests share
    internal buffers, and the step callback is swapped per call. The lock makes
    concurrent media requests queue instead of corrupting each other.
    """

    def __init__(self, pipe) -> None:
        self._pipe = pipe
        self._lock = threading.Lock()
        self._closed = False

    def __call__(self, **kwargs):
        with self._lock:
            if self._closed:
                raise RuntimeError("media model was unloaded; retry")
            return self._pipe(**kwargs)

    @contextmanager
    def session(self):
        """Hold the lock across a whole multi-frame job.

        A video is many pipeline calls; taking the lock once per clip means an
        eviction cannot land between frames and kill the clip half-encoded.
        """
        with self._lock:
            if self._closed:
                raise RuntimeError("media model was unloaded; retry")
            yield self._pipe

    def close(self) -> None:
        # Taking the lock first means an in-flight frame finishes before the
        # weights are dropped, instead of being freed mid-denoise.
        with self._lock:
            self._closed = True
            self._pipe = None


def _coerce_size(value: int | None, default: int = DEFAULT_SIZE) -> int:
    """Round to a multiple of 8 and clamp to the supported range."""
    size = default if value is None else int(value)
    size = max(MIN_SIZE, min(MAX_SIZE, size))
    return max(8, (size // 8) * 8)


def generate_image(
    prompt: str,
    *,
    negative_prompt: str | None = None,
    steps: int | None = None,
    size: int | None = None,
    seed: int | None = None,
) -> tuple[bytes, int]:
    """Render one image and return ``(png_bytes, used_seed)``."""
    prompt = (prompt or "").strip()
    if not prompt:
        raise ValueError("prompt must not be empty")
    if len(prompt) > 1000:
        raise ValueError("prompt is too long (max 1000 characters)")

    steps = max(1, min(8, steps or DEFAULT_STEPS))
    size = _coerce_size(size)
    used_seed = _resolve_seed(seed)

    import torch

    engine = slot.acquire(f"media:{_model_path()}", _load_pipeline)
    generator = torch.Generator(device="cpu").manual_seed(used_seed)
    kwargs = {"prompt": prompt, "num_inference_steps": steps, "guidance_scale": 0.0,
              "height": size, "width": size, "generator": generator}
    if negative_prompt:
        kwargs["negative_prompt"] = negative_prompt

    image = engine(**kwargs).images[0]
    buf = io.BytesIO()
    image.save(buf, format="PNG", optimize=True)
    return buf.getvalue(), used_seed


def _resolve_seed(seed: int | None) -> int:
    """-1 or None means "pick one", matching the text model's PHI3_SEED."""
    if seed is None or seed < 0:
        return int.from_bytes(os.urandom(4), "big") % (2**31)
    return int(seed) % (2**31)


def generate_video(
    prompt: str,
    *,
    seconds: float = 2.0,
    fps: int = 8,
    size: str = "512x288",
    steps: int | None = None,
    seed: int | None = None,
) -> tuple[bytes, int, int]:
    """Render keyframes and encode them into an MP4.

    Returns ``(mp4_bytes, used_seed, frame_count)``. Each frame is written to a
    temporary file and dropped from memory right away, so peak memory stays near
    one image instead of the whole clip.
    """
    prompt = (prompt or "").strip()
    if not prompt:
        raise ValueError("prompt must not be empty")
    if len(prompt) > 1000:
        raise ValueError("prompt is too long (max 1000 characters)")
    if size not in VIDEO_SIZES:
        raise ValueError(f"size must be one of: {', '.join(sorted(VIDEO_SIZES))}")

    seconds = max(1.0, min(MAX_VIDEO_SECONDS, float(seconds)))
    fps = max(4, min(16, int(fps)))
    frames = max(4, min(int(round(seconds * fps)), 64))
    width, height = VIDEO_SIZES[size]
    steps = max(1, min(4, steps or DEFAULT_STEPS))
    used_seed = _resolve_seed(seed)

    import torch

    engine = slot.acquire(f"media:{_model_path()}", _load_pipeline)
    # Frames go to a private temp dir rather than anywhere served over HTTP:
    # nothing generated here should be reachable by URL.
    tmp = Path(tempfile.mkdtemp(prefix="phi3vid_"))
    stem = f"vid_{used_seed}"
    paths: list[Path] = []

    try:
        with engine.session() as pipe:
            for index in range(frames):
                # A slight per-frame seed change keeps the clip from being a
                # still image while still looking coherent.
                generator = torch.Generator(device="cpu").manual_seed(used_seed + index)
                image = pipe(prompt=prompt, num_inference_steps=steps, guidance_scale=0.0,
                             height=height, width=width, generator=generator).images[0]
                path = tmp / f"{stem}_{index:03d}.png"
                image.save(path)
                paths.append(path)
                del image

        return _encode(paths, fps), used_seed, frames
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _encode(paths: list[Path], fps: int) -> bytes:
    """Encode PNG frames into an H.264 MP4 using the ffmpeg imageio bundles."""
    try:
        import imageio_ffmpeg
    except ImportError as exc:  # pragma: no cover - dependency hint
        raise RuntimeError(
            "Video encoding needs imageio-ffmpeg. Install it with:\n"
            "  .venv/bin/pip install -r requirements-media.txt"
        ) from exc

    if not paths:
        raise ValueError("no frames to encode")

    import subprocess

    exe = imageio_ffmpeg.get_ffmpeg_exe()
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as handle:
        out = Path(handle.name)

    # A single palette-free pass is fine for these clips and avoids a second
    # read of every frame. yuv420p keeps the result playable in browsers.
    cmd = [
        exe, "-y", "-loglevel", "error",
        "-framerate", str(fps),
        "-i", str(paths[0].parent / (paths[0].stem.rsplit("_", 1)[0] + "_%03d.png")),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(out),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=300)
        if proc.returncode != 0:
            detail = proc.stderr.decode("utf-8", "replace").strip()[-500:]
            raise RuntimeError(f"ffmpeg failed: {detail or 'unknown error'}")
        data = out.read_bytes()
    finally:
        out.unlink(missing_ok=True)
    if not data:
        raise RuntimeError("ffmpeg produced an empty file")
    return data


def estimate_video_seconds(frames: int, size: str) -> float:
    """Rough per-frame cost, used only for the UI hint."""
    per_frame = {"512p": 22.0, "512x288": 13.0, "384p": 12.0}.get(size, 15.0)
    return math.ceil(frames * per_frame)
