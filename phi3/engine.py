"""Thin, thread-safe wrapper around llama.cpp for Phi-3 Mini inference."""

from __future__ import annotations

import queue
import threading
from collections.abc import Iterator
from typing import Any

from llama_cpp import Llama

from .config import Settings, settings as default_settings


class Phi3Engine:
    """Loads the GGUF model once and serves chat completions.

    llama.cpp is not thread-safe for concurrent generation, so a lock
    serialises requests when the shared engine is used by the HTTP server.
    """

    def __init__(self, config: Settings | None = None) -> None:
        self.config = config or default_settings
        self.key = str(self.config.resolved_model_path())
        self._lock = threading.Lock()
        self._closed = False
        self._llm: Llama | None = None

    @property
    def llm(self) -> Llama:
        if self._closed:
            raise RuntimeError("engine was switched to another model; retry")
        if self._llm is None:
            model_path = self.config.require_model()
            self._llm = Llama(
                model_path=str(model_path),
                n_ctx=self.config.n_ctx,
                n_threads=self.config.n_threads,
                n_batch=self.config.n_batch,
                n_gpu_layers=self.config.n_gpu_layers,
                verbose=False,
            )
        return self._llm

    def close(self) -> None:
        """Release the loaded model and free its memory.

        Takes the generation lock first so a caller that is mid-generation is
        not pulled out from under the native context.
        """
        with self._lock:
            self._closed = True
            if self._llm is not None:
                close = getattr(self._llm, "close", None)
                if callable(close):
                    close()
                self._llm = None

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> dict[str, Any]:
        """Return the full assistant reply for a conversation."""
        prepared = self._prepare(messages)
        with self._lock:
            result = self.llm.create_chat_completion(
                messages=prepared,
                max_tokens=max_tokens or self.config.max_tokens,
                temperature=(
                    temperature if temperature is not None else self.config.temperature
                ),
                top_p=top_p if top_p is not None else self.config.top_p,
                repeat_penalty=self.config.repeat_penalty,
                seed=self.config.seed,
            )
        return self._normalise(result)

    def stream(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> Iterator[str]:
        """Yield assistant text incrementally.

        Generation runs in a worker thread that owns the generation lock and
        releases it when it finishes. Tokens reach the caller through a queue.
        This decouples the lock lifetime from the HTTP consumer: a client that
        disconnects early cannot leave the lock held, which would otherwise
        stall every later request.
        """
        prepared = self._prepare(messages)
        stop = threading.Event()
        queue_: queue.Queue[Any] = queue.Queue()
        sentinel = object()

        def worker() -> None:
            with self._lock:
                try:
                    stream = self.llm.create_chat_completion(
                        messages=prepared,
                        max_tokens=max_tokens or self.config.max_tokens,
                        temperature=(
                            temperature
                            if temperature is not None
                            else self.config.temperature
                        ),
                        top_p=top_p if top_p is not None else self.config.top_p,
                        repeat_penalty=self.config.repeat_penalty,
                        seed=self.config.seed,
                        stream=True,
                    )
                    try:
                        for chunk in stream:
                            if stop.is_set():
                                break
                            text = chunk["choices"][0].get("delta", {}).get("content")
                            if text:
                                queue_.put(text)
                    finally:
                        close = getattr(stream, "close", None)
                        if callable(close):
                            close()
                except Exception as exc:  # forward to the consumer
                    queue_.put(exc)
                finally:
                    queue_.put(sentinel)

        threading.Thread(target=worker, daemon=True).start()
        try:
            while True:
                item = queue_.get()
                if item is sentinel:
                    break
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            stop.set()

    def _prepare(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        if not messages:
            raise ValueError("messages must not be empty")
        if messages[0].get("role") != "system":
            messages = [{"role": "system", "content": self.config.system_prompt}, *messages]
        return messages

    @staticmethod
    def _normalise(result: dict[str, Any]) -> dict[str, Any]:
        choice = result["choices"][0]
        return {
            "content": choice["message"]["content"],
            "role": choice["message"].get("role", "assistant"),
            "finish_reason": choice.get("finish_reason"),
            "usage": result.get("usage", {}),
        }


_current: tuple[str, Phi3Engine] | None = None
_switch_lock = threading.Lock()


def get_engine(config: Settings | None = None) -> Phi3Engine:
    """Return the engine for the requested model.

    Only one model is kept in memory at a time: loading a second GGUF while the
    first is still resident doubles the RAM footprint and can exhaust the
    machine, so asking for a different model closes the previous engine first.
    The returned engine is safe to use after the switch because the caller's
    generation holds the engine's own lock, which ``close`` waits on.
    """
    global _current
    config = config or default_settings
    key = str(config.resolved_model_path())
    with _switch_lock:
        if _current is not None and _current[0] == key:
            return _current[1]
        if _current is not None:
            _current[1].close()
        engine = Phi3Engine(config)
        _current = (key, engine)
        return engine
