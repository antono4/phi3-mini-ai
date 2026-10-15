"""Interactive terminal chat with Phi-3 Mini.

Usage:
    python -m phi3.cli            # streaming REPL
    python -m phi3.cli --no-stream
    python -m phi3.cli -p "Hello" # one-shot prompt
"""

from __future__ import annotations

import argparse
import sys

from .config import MODEL_PRESETS, settings
from .engine import get_engine

BANNER = """\
Phi-3 Mini chat  (model: {model})
Commands: /reset  /exit  /help
"""


def _run_once(engine, prompt: str, stream: bool) -> None:
    messages = [{"role": "user", "content": prompt}]
    if stream:
        for piece in engine.stream(messages):
            sys.stdout.write(piece)
            sys.stdout.flush()
        print()
    else:
        reply = engine.chat(messages)
        print(reply["content"])


def _repl(engine, stream: bool) -> None:
    print(BANNER.format(model=settings.resolved_model_path().name))
    history: list[dict[str, str]] = []
    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user in {"/exit", "/quit"}:
            break
        if user == "/help":
            print("Commands: /reset (clear history), /exit, /help")
            continue
        if user == "/reset":
            history.clear()
            print("(history cleared)")
            continue

        history.append({"role": "user", "content": user})
        print("phi> ", end="", flush=True)
        if stream:
            collected = []
            for piece in engine.stream(history):
                sys.stdout.write(piece)
                sys.stdout.flush()
                collected.append(piece)
            print()
            history.append({"role": "assistant", "content": "".join(collected)})
        else:
            reply = engine.chat(history)
            print(reply["content"])
            history.append({"role": "assistant", "content": reply["content"]})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Chat with Phi-3 Mini locally.")
    parser.add_argument("-p", "--prompt", help="send a single prompt and exit")
    parser.add_argument(
        "--no-stream", action="store_true", help="disable token streaming"
    )
    parser.add_argument(
        "--model",
        default=None,
        help=f"model preset ({'/'.join(sorted(MODEL_PRESETS))}) or a path to a GGUF file",
    )
    args = parser.parse_args(argv)

    engine = get_engine(settings.with_model(args.model))
    try:
        engine.config.require_model()
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1
    stream = not args.no_stream
    if args.prompt:
        _run_once(engine, args.prompt, stream)
    else:
        _repl(engine, stream)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
