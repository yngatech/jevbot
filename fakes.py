"""What the offline tests share: jev_bot imported without real keys, and fakes for OpenRouter, Rocky's answers and
Rocky's state. Not a test module itself: unittest only collects test_*.py.
"""

import contextlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

os.environ.setdefault("DISCORD_TOKEN_JEV", "test")
os.environ.setdefault("OPENROUTER_API_KEY", "test")

import jev_bot as j  # noqa: E402
import rocky_harness as rh  # noqa: E402


def patch_all(case, patches):
    # [(obj, attr, value), ...], each undone when the test ends
    for obj, attr, value in patches:
        p = patch.object(obj, attr, value)
        p.start()
        case.addCleanup(p.stop)


def traced(case):
    # A fresh trace for the test, as each reply and status gets
    t = {"cost": 0.0, "requests": 0}
    case.addCleanup(j.trace.reset, j.trace.set(t))
    return t


def isolate_rocky(case, here):
    # Rocky's logs and memories, empty and kept in `here`; and none of the checkout's own commits
    patch_all(case, [(rh, "LOG_DIR", Path(here) / "rocky_logs"), (rh, "MEMORY_PATH", Path(here) / "rocky_memories.json"),
                     (rh, "logs", {}), (rh, "log_saves", {}), (rh, "memories", {}), (rh, "memories_loaded", True),
                     (rh, "memories_save", None), (rh, "head", lambda: None)])


def answer(text="", calls=(), reasoning=None):
    # Rocky's response to a request: what it says, and the tools it calls as [(name, arguments)]
    return {"role": "assistant", "content": text or None, "reasoning_details": reasoning,
            "tool_calls": [{"id": f"call_{i}", "type": "function",
                            "function": {"name": name, "arguments": json.dumps(args)}}
                           for i, (name, args) in enumerate(calls)] or None}


def session():
    # An aiohttp session, its own `async with`
    s = MagicMock()
    s.__aenter__.return_value = s
    return s


@contextlib.contextmanager
def openrouter(*replies):
    # OpenRouter answering each request with the next (status, body), the last one again once they run out. Yields the
    # session, to see what was asked, with set_credit on it; retries don't wait.
    replies = list(replies)

    def reply():
        status, body = replies.pop(0) if len(replies) > 1 else replies[0]
        return SimpleNamespace(status=status, json=AsyncMock(return_value=body),
                               text=AsyncMock(return_value=json.dumps(body)))

    s = session()
    s.post.return_value.__aenter__.side_effect = reply
    with patch.object(j.aiohttp, "ClientSession", return_value=s), \
            patch.object(j, "set_credit", AsyncMock()) as s.set_credit, patch.object(j.asyncio, "sleep", AsyncMock()):
        yield s
