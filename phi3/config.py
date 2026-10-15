"""Configuration for the Phi-3 Mini app.

All values can be overridden with environment variables so the same code runs
locally and inside the container without edits.
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Named GGUF presets so a model can be chosen by short name instead of a path.
MODEL_PRESETS: dict[str, str] = {
    "q4": "models/Phi-3-mini-4k-instruct-q4.gguf",
    "q8": "models/Phi-3-mini-4k-instruct-q8.gguf",
    "lora": "models/phi3-lora-q4.gguf",
    "marbel": "models/marbel-phi3-q4.gguf",
}


def resolve_model(value: str | None) -> Path:
    """Turn a preset name or path into a concrete model path."""
    if not value:
        value = "q4"
    if value in MODEL_PRESETS:
        return PROJECT_ROOT / MODEL_PRESETS[value]
    return Path(value).expanduser()


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


@dataclass(frozen=True)
class Settings:
    model_path: Path = field(
        default_factory=lambda: resolve_model(
            os.environ.get("PHI3_MODEL") or os.environ.get("PHI3_MODEL_PATH")
        )
    )
    n_ctx: int = field(default_factory=lambda: _env_int("PHI3_N_CTX", 4096))
    n_threads: int = field(
        default_factory=lambda: _env_int("PHI3_N_THREADS", max(1, (os.cpu_count() or 4)))
    )
    n_batch: int = field(default_factory=lambda: _env_int("PHI3_N_BATCH", 512))
    n_gpu_layers: int = field(default_factory=lambda: _env_int("PHI3_N_GPU_LAYERS", 0))
    max_tokens: int = field(default_factory=lambda: _env_int("PHI3_MAX_TOKENS", 512))
    temperature: float = field(default_factory=lambda: _env_float("PHI3_TEMPERATURE", 0.7))
    top_p: float = field(default_factory=lambda: _env_float("PHI3_TOP_P", 0.9))
    repeat_penalty: float = field(
        default_factory=lambda: _env_float("PHI3_REPEAT_PENALTY", 1.1)
    )
    seed: int = field(default_factory=lambda: _env_int("PHI3_SEED", -1))
    system_prompt: str = field(
        default_factory=lambda: os.environ.get(
            "PHI3_SYSTEM_PROMPT",
            "You are a helpful, concise AI assistant powered by Phi-3 Mini. "
            "Answer clearly and accurately.",
        )
    )
    host: str = field(default_factory=lambda: os.environ.get("PHI3_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: _env_int("PHI3_PORT", 12000))
    allow_origin: str = field(
        default_factory=lambda: os.environ.get("PHI3_ALLOW_ORIGIN", "*")
    )

    def resolved_model_path(self) -> Path:
        return Path(self.model_path).expanduser().resolve()

    def with_model(self, model: str | None) -> "Settings":
        """Return a copy of these settings pointed at another model."""
        return dataclasses.replace(self, model_path=resolve_model(model))

    def require_model(self) -> Path:
        path = self.resolved_model_path()
        if not path.exists():
            raise FileNotFoundError(
                f"Model not found at {path}. Download it with "
                f"`python scripts/download_model.py <preset>`, where <preset> is "
                f"one of: {', '.join(sorted(MODEL_PRESETS))}."
            )
        return path


settings = Settings()
