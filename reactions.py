"""
How people reacted to what jev posted: its replies by who wrote them (Jev, or an LLM picked with !model) and by
length, its statuses, and the most-reacted. Reads logs/ for what jev posted, and Discord for the reactions on it now,
since they aren't logged and keep coming after jev moves on. Reads only; costs nothing.

👎 is a Rocky meme as often as a complaint, so it's counted apart: "besides 👎" is the share with some other reaction.
"audience" is how many people other than jev spoke in the channel in the AUDIENCE_MINUTES after a post, so a quiet
evening can be told from a flop.

    uv run --with-requirements requirements.txt python reactions.py [--days 7] [--top 10] [--logs DIR]
"""

import argparse
import asyncio
import json
import logging
from collections import Counter, defaultdict
from datetime import timedelta
from pathlib import Path

import discord

# Only the report: not jev's startup lines, or discord.py's about voice and intents it doesn't need here
logging.getLogger("jev").setLevel(logging.WARNING)
logging.getLogger("discord").setLevel(logging.ERROR)

import jev_bot as j  # noqa: E402

THUMBS = "👎"
AUDIENCE_MINUTES = 30
LENGTHS = [(1, "1 word"), (3, "2-3 words"), (6, "4-6 words"), (None, "7+ words")]


# jev's replies and statuses that made it to Discord, from the logs
def posts(paths, status_channel):
    found = []
    for path in paths:
        with open(path) as f:
            for line in f:
                try:
                    t = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if t.get("reply_id") and t.get("channel_id"):
                    found.append({"kind": "reply", "channel_id": t["channel_id"], "id": t["reply_id"],
                                  "writer": t.get("llm") or "jev", "text": t.get("reply") or "", "at": t["at"]})
                elif t.get("status_id") and status_channel:
                    found.append({"kind": "status", "channel_id": status_channel, "id": t["status_id"],
                                  "writer": t.get("llm") or "jev", "text": t.get("status") or "", "at": t["at"]})
    return found


# People's reactions to a message, emoji to count — not jev's own
def reactions_of(message):
    return {j.emoji_text(r.emoji): n for r in message.reactions if (n := r.count - r.me)}


# How many people other than jev spoke in the channel in the AUDIENCE_MINUTES after `at`
def audience(at, said, bot_id):
    end = at + timedelta(minutes=AUDIENCE_MINUTES)
    return len({who for when, who in said if at < when <= end and who != bot_id})


# Each post's reactions and audience, from the channel's history over the stretch the posts cover — a page of 100
# messages per request instead of one per post. Posts whose message or channel is gone are dropped.
async def collect(client, found):
    by_channel = defaultdict(list)
    for p in found:
        by_channel[p["channel_id"]].append(p)
    kept = []
    for channel_id, ps in by_channel.items():
        try:
            channel = await client.fetch_channel(channel_id)
        except discord.HTTPException as e:
            print(f"Skipping {len(ps)} post(s) in channel {channel_id}: {e}")
            continue
        first, last = min(p["id"] for p in ps), max(p["id"] for p in ps)
        end = discord.utils.snowflake_time(last) + timedelta(minutes=AUDIENCE_MINUTES)
        messages, said = {}, []
        async for m in channel.history(after=discord.Object(first - 1), before=end, limit=None, oldest_first=True):
            messages[m.id] = m
            said.append((m.created_at, m.author.id))
        for p in ps:
            if m := messages.get(p["id"]):
                kept.append({**p, "reactions": reactions_of(m), "audience": audience(m.created_at, said, client.user.id)})
    return sorted(kept, key=lambda p: p["at"])


def summary(ps):
    total = Counter()
    for p in ps:
        total.update(p["reactions"])
    n = len(ps)
    return {
        "posts": n,
        "reacted": sum(bool(p["reactions"]) for p in ps) / n,
        "besides_thumbs": sum(any(e != THUMBS for e in p["reactions"]) for p in ps) / n,
        "thumbs": sum(THUMBS in p["reactions"] for p in ps) / n,
        "per_post": sum(total.values()) / n,
        "audience": sum(p["audience"] for p in ps) / n,
        "top": total.most_common(5),
    }


def length(p):
    words = len(p["text"].split())
    return next(name for most, name in LENGTHS if most is None or words <= most)


def table(title, groups):
    print(f"\n{title}")
    print(f"  {'':<12}{'posts':>6}{'reacted':>9}{'besides ' + THUMBS:>11}{'with ' + THUMBS:>8}{'reacts':>8}"
          f"{'audience':>10}  top")
    for name, ps in groups:
        if not ps:
            continue
        s = summary(ps)
        top = " ".join(f"{e}×{n}" for e, n in s["top"])
        print(f"  {name:<12}{s['posts']:>6}{s['reacted']:>9.0%}{s['besides_thumbs']:>11.0%}{s['thumbs']:>8.0%}"
              f"{s['per_post']:>8.2f}{s['audience']:>10.1f}  {top}")


def grouped(ps, key, order=None):
    groups = defaultdict(list)
    for p in ps:
        groups[key(p)].append(p)
    names = order or sorted(groups, key=lambda g: -len(groups[g]))
    return [(name, groups[name]) for name in names if name in groups]


def report(ps, top):
    replies = [p for p in ps if p["kind"] == "reply"]
    statuses = [p for p in ps if p["kind"] == "status"]
    print(f"{len(ps)} posts from {ps[0]['at'][:10]} to {ps[-1]['at'][:10]}" if ps else "Nothing posted in that time")
    if replies:
        table("Replies, by writer", grouped(replies, lambda p: p["writer"]))
        table("Jev's replies, by length", grouped([p for p in replies if p["writer"] == "jev"], length,
                                                  [name for _, name in LENGTHS]))
    if statuses:
        table("Statuses, by writer", grouped(statuses, lambda p: p["writer"]))
    best = sorted((p for p in ps if p["reactions"]), key=lambda p: -sum(p["reactions"].values()))[:top]
    if best:
        print("\nMost reacted")
        for p in best:
            said = " ".join(p["text"].split())
            said = said if len(said) <= 60 else said[:59] + "…"
            reacts = " ".join(f"{e}×{n}" for e, n in sorted(p["reactions"].items(), key=lambda r: -r[1]))
            print(f"  {p['at'][:10]} {p['kind']:<6} {p['writer']:<8} {said}  {reacts}")


async def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    parser.add_argument("--days", type=int, help="only the last DAYS days of logs (default: all)")
    parser.add_argument("--logs", type=Path, default=j.LOG_DIR, help="where jev's logs are (default: %(default)s)")
    parser.add_argument("--top", type=int, default=10, help="how many of the most-reacted posts to list")
    args = parser.parse_args()

    paths = sorted(args.logs.glob("*.jsonl"))
    if args.days:
        paths = paths[-args.days:]
    found = posts(paths, j.STATUS_CHANNEL)
    # Only Discord's HTTP API, no gateway connection, so it runs alongside the bot on the same token
    client = discord.Client(intents=discord.Intents.none())
    await client.login(j.TOKEN)
    try:
        ps = await collect(client, found)
    finally:
        await client.close()
    if gone := len(found) - len(ps):
        print(f"{gone} post(s) no longer on Discord, left out")
    report(ps, args.top)


if __name__ == "__main__":
    asyncio.run(main())
