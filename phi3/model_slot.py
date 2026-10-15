"""Keeps at most one heavyweight model resident at a time.

The text model (Phi-3, a couple of GB) and the image model (several GB at peak)
do not both fit in the memory this project targets, so loading either one evicts
the other. Both caches share a single slot, which makes that exclusion
automatic: whichever model is asked for last is the one left in memory.

The slot holds a lock only while swapping. Eviction calls ``close()`` on the
outgoing model, and each engine's ``close()`` waits for its own generation lock,
so a switch cannot pull a model out from under an in-flight request.
"""

from __future__ import annotations

import threading
from typing import Any, Callable


class ModelSlot:
    """A one-entry cache keyed by a string; loading a new key evicts the old."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._key: str | None = None
        self._model: Any = None

    def acquire(self, key: str, factory: Callable[[], Any]) -> Any:
        """Return the cached model for ``key``, building it if needed.

        ``factory`` runs while the slot lock is held, so two requests for a cold
        model load it once instead of racing.
        """
        with self._lock:
            if self._key == key and self._model is not None:
                return self._model
            self._evict_locked()
            model = factory()
            self._key, self._model = key, model
            return model

    def evict(self) -> None:
        """Drop the resident model and release its memory."""
        with self._lock:
            self._evict_locked()

    def _evict_locked(self) -> None:
        model, self._model, self._key = self._model, None, None
        if model is None:
            return
        close = getattr(model, "close", None)
        if callable(close):
            close()

    def current(self) -> str | None:
        with self._lock:
            return self._key


# Shared by the text engine and the media engine so they never coexist.
slot = ModelSlot()
