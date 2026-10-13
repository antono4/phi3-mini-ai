"""Quick smoke test for the Phi-3 engine and API schemas.

Run: python scripts/smoke_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phi3.config import settings
from phi3.engine import get_engine
from phi3.server import ChatRequest, Message


def main() -> int:
    settings.require_model()
    engine = get_engine()

    reply = engine.chat([{"role": "user", "content": "Reply with the word: ready"}])
    assert reply["content"].strip(), "empty reply"
    print("chat ok:", reply["content"].strip()[:60])

    pieces = list(engine.stream([{"role": "user", "content": "Count to three."}]))
    assert pieces, "no streamed tokens"
    print("stream ok:", "".join(pieces).strip()[:60])

    req = ChatRequest(messages=[Message(role="user", content="hi")])
    assert req.messages[0].role == "user"
    print("schema ok")
    print("ALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
