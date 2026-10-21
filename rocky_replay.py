"""
Replays messages to `!model rocky` on a copy of the live channel log — to see how a change to Rocky answers before it
ships. Each run starts from the log and memories as they are, adds the message, and prints what Rocky did: its tool
calls and their results, its reactions, and what it said. Nothing is written back, and its thinking isn't printed.

    python rocky_replay.py "@rocky how are you doing?" "lol rocky"
    python rocky_replay.py -n 3 --file scenarios.txt     # one message a line, # for comments
    python rocky_replay.py --at "2026-10-16 19:00" "rocky get us a beer"
    python rocky_replay.py --fresh "@rocky get me a stout"   # a short synthetic chat instead of the live log
    python rocky_replay.py --image cat.png "@rocky what's this"  # sent with a picture, which look sees
    python rocky_replay.py "@rocky soup?" --with "fern: @rocky tea?"  # another to it, there before its turn

Run on master and on your branch to compare. It reads the live checkout's files (where this checkout's .env links to,
or this one's): its Rocky log, memories, rocky_lines.txt and rocky_body.txt. A message comes 2 minutes after the log's last event,
unless --at says when. On OPENROUTER_API_KEY_EVAL, one request at a time: a turn on the live log is ~70K tokens, and
the first of a run writes it to the cache, which counts towards Anthropic's input tokens a minute.
"""

import argparse
import asyncio
import copy
import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import jev_bot as j
import rocky_harness as rh

HERE = Path(__file__).parent
AUTHOR = "pip"
FRESH = [("moss", "anyone watching the match tonight"), ("fern", "nah too tired"), ("moss", "fair, I'll get pizza then")]
RESULT_CHARS = 80


def live_dir():
    env = HERE / ".env"
    return env.resolve().parent if env.exists() else HERE


def load(live, channel_id, fresh, at):
    rh.bind(j)
    # Read from the live checkout, and nothing written anywhere
    rh.LOG_DIR, rh.MEMORY_PATH = live / "rocky_logs", live / "rocky_memories.json"
    rh.save_log = rh.save_memories = lambda *a: None
    if (live / "rocky_lines.txt").exists():
        j.ROCKY_LINES = j.read_lines(live / "rocky_lines.txt")
    if (live / "rocky_body.txt").exists():
        j.ROCKY_BODY = j.read_lines(live / "rocky_body.txt", "\n\n")
    try:
        memories = json.loads(rh.MEMORY_PATH.read_text())
    except FileNotFoundError:
        memories = {}
    if fresh:
        log = rh.Log("bot-spam", "1", 1)
        t0 = at or datetime.now(j.TIMEZONE)
        for k, (who, text) in enumerate(FRESH):
            log.message(who, text, t0 - timedelta(minutes=len(FRESH) - k))
    else:
        if not (rh.LOG_DIR / f"{channel_id}.json").exists():
            raise SystemExit(f"no Rocky log for channel {channel_id} in {rh.LOG_DIR}")
        log = rh.log_for(SimpleNamespace(id=channel_id, name="replay", guild=SimpleNamespace(id=1)))
    last = max((e["at"] for e in log.events if e.get("at")), default=datetime.now(j.TIMEZONE))
    return log, memories, at or last + timedelta(minutes=2)


async def replay(base, memories, text, when, author, picture=None, also=()):
    log = copy.deepcopy(base)
    rh.memories.clear()
    rh.memories.update(copy.deepcopy(memories))
    rh.memories_loaded = True
    rh.clock = lambda: when
    sent = log.message(author, f"{text} [photo]" if picture else text, when, to_bot=True)
    for who, said in also:  # came while it waited, so they're in its turn
        log.message(who, said, when + timedelta(seconds=2), to_bot=True)
    # Only the replayed message has a picture to see — the log's have no Discord to fetch theirs from
    rh.pictures = AsyncMock(side_effect=lambda _, n: [picture] if picture and n == sent["n"] else None)
    turn = await rh.take_turn(log, sent)
    steps = []
    for e in turn.events:
        if e["type"] == "rocky":
            steps += [f"{c['name']}{c['arguments']}" for c in e["calls"]]
        else:
            steps.append(f"  → {e['content'][:RESULT_CHARS]}")
    return turn, steps


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("messages", nargs="*", help="what's said to Rocky, one run each")
    ap.add_argument("--file", help="more messages, one a line")
    ap.add_argument("-n", "--runs", type=int, default=1, help="runs of each message (default 1)")
    ap.add_argument("--at", type=datetime.fromisoformat, help="when it's said, on TIMEZONE (default 2 minutes after the log ends)")
    ap.add_argument("--author", default=AUTHOR, help=f"who says it (default {AUTHOR})")
    ap.add_argument("--channel", type=int, help="the channel's log (default the first LONG_CONTEXT_CHANNEL_IDS)")
    ap.add_argument("--fresh", action="store_true", help="a short synthetic chat instead of the live log")
    ap.add_argument("--image", help="a picture sent with each message, for look — a synthetic one, not from the chat")
    ap.add_argument("--with", dest="also", action="append", default=[], metavar="AUTHOR: TEXT",
                    help="another message to Rocky, a moment after, before its turn (repeatable)")
    ap.add_argument("--model", default="rocky", help="an LLM with the harness (default rocky)")
    args = ap.parse_args()
    messages = list(args.messages)
    if args.file:
        messages += [l.strip() for l in Path(args.file).read_text().splitlines() if l.strip() and not l.startswith("#")]
    if not messages:
        ap.error("say something to Rocky")
    if any(":" not in a for a in args.also):
        ap.error("--with takes AUTHOR: TEXT")
    also = [tuple(x.strip() for x in a.split(":", 1)) for a in args.also]
    if not j.LLMS.get(args.model, {}).get("harness"):
        ap.error(f"{args.model} isn't an LLM with the harness")
    # Its own key, so replays have their own limits and usage on OpenRouter and Anthropic, apart from the bot's
    eval_key = os.environ.get("OPENROUTER_API_KEY_EVAL")
    if not eval_key:
        ap.error("set OPENROUTER_API_KEY_EVAL in .env — replays don't use the bot's OPENROUTER_API_KEY")
    j.HEADERS = {**j.HEADERS, "Authorization": f"Bearer {eval_key}"}
    channel = args.channel or next(iter(sorted(j.LONG_CHAT_CHANNELS)), None)
    if not args.fresh and not channel:
        ap.error("set --channel or LONG_CONTEXT_CHANNEL_IDS, or use --fresh")
    at = args.at and (args.at if args.at.tzinfo else args.at.replace(tzinfo=j.TIMEZONE))
    logging.getLogger("jev").setLevel(logging.WARNING)
    picture = args.image and rh.shrunk(Path(args.image).read_bytes())
    if args.image and not picture:
        ap.error(f"{args.image} isn't a picture")

    live = live_dir()
    base, memories, when = load(live, channel, args.fresh, at)
    print(f"{'fresh chat' if args.fresh else f'{live.name} log: {len(base.events)} events'}, "
          f"{len(j.ROCKY_LINES)} Rocky lines, {len(j.ROCKY_BODY)} body passages, at {when:%a %d %b %H:%M}")
    spent = {"cost": 0.0, "requests": 0}
    j.trace.set(spent)
    for text in messages:
        for i in range(args.runs):
            turn, steps = await replay(base, memories, text, when, args.author, picture, also)
            shown = "".join(f"\n    {l}" for l in j.action_lines(turn.shown))
            print(f"[{text}] #{i + 1} reqs={turn.requests} react={[e for _, e in turn.reactions]} said={turn.text!r}"
                  + "".join(f"\n    {s}" for s in steps) + shown, flush=True)
    print(f"{spent['requests']} requests, ${spent['cost']:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
