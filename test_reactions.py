"""Offline checks for reactions.py: reading posts from the logs, matching them to Discord messages, and the counts.

    uv run --with-requirements requirements.txt python -m unittest test_reactions
"""

import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("DISCORD_TOKEN_JEV", "test")
os.environ.setdefault("OPENROUTER_API_KEY", "test")

import reactions as r  # noqa: E402

BOT, ALICE, BOB = 1, 2, 3
T0 = datetime(2026, 10, 1, 12, tzinfo=timezone.utc)


def reaction(emoji, count, me=False):
    return SimpleNamespace(emoji=emoji, count=count, me=me)


def message(id, author, minutes, reactions=()):
    return SimpleNamespace(id=id, author=SimpleNamespace(id=author), created_at=T0 + timedelta(minutes=minutes),
                           reactions=list(reactions))


def post(reacts, audience=0, writer="jev", text="Yes yes"):
    return {"kind": "reply", "writer": writer, "text": text, "reactions": reacts, "audience": audience}


class Posts(unittest.TestCase):
    def test_reads_replies_and_statuses_that_were_posted(self):
        lines = [
            {"at": "2026-10-01T12:00:00+00:00", "channel_id": 10, "reply": "No not wobbly", "reply_id": 100},
            {"at": "2026-10-01T12:01:00+00:00", "channel_id": 10, "reply": "Is good, good, good.", "reply_id": 101,
             "llm": "deepseek"},
            {"at": "2026-10-01T12:02:00+00:00", "channel_id": 10, "reaction": "😂"},  # a reaction, not a post
            {"at": "2026-10-01T12:03:00+00:00", "channel_id": 10, "reply": "..."},  # failed to send
            {"at": "2026-10-01T15:00:00+00:00", "status": "I feel wobbly", "status_id": 200},
        ]
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "2026-10-01.jsonl"
            path.write_text("\n".join(json.dumps(t) for t in lines) + "\nnot json\n")
            found = r.posts([path], status_channel=20)
        self.assertEqual([(p["kind"], p["channel_id"], p["id"], p["writer"]) for p in found],
                         [("reply", 10, 100, "jev"), ("reply", 10, 101, "deepseek"), ("status", 20, 200, "jev")])

    def test_statuses_are_skipped_without_a_status_channel(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "2026-10-01.jsonl"
            path.write_text(json.dumps({"at": "2026-10-01T15:00:00+00:00", "status": "I feel", "status_id": 200}))
            self.assertEqual(r.posts([path], status_channel=0), [])


class Counts(unittest.TestCase):
    def test_reactions_leave_out_jevs_own(self):
        m = message(1, BOT, 0, [reaction("😂", 3, me=True), reaction("👎", 1), reaction("🤷", 1, me=True)])
        self.assertEqual(r.reactions_of(m), {"😂": 2, "👎": 1})

    def test_audience_counts_people_other_than_jev_in_the_window(self):
        said = [(T0 - timedelta(minutes=1), ALICE),  # before
                (T0 + timedelta(minutes=1), BOT), (T0 + timedelta(minutes=2), BOB), (T0 + timedelta(minutes=3), BOB),
                (T0 + timedelta(minutes=r.AUDIENCE_MINUTES + 1), ALICE)]  # after the window
        self.assertEqual(r.audience(T0, said, BOT), 1)

    def test_thumbs_down_is_counted_apart(self):
        s = r.summary([post({"👎": 2}), post({"😂": 1, "👎": 1}), post({}), post({"❤️": 1})])
        self.assertEqual((s["reacted"], s["besides_thumbs"], s["thumbs"]), (.75, .5, .5))
        self.assertEqual(s["per_post"], 1.25)
        self.assertEqual(s["top"][0], ("👎", 3))

    def test_length_buckets(self):
        self.assertEqual([r.length(post({}, text=t)) for t in ["Dunno", "Dunno google", "a b c d", "a b c d e f g"]],
                         ["1 word", "2-3 words", "4-6 words", "7+ words"])


class Collect(unittest.IsolatedAsyncioTestCase):
    async def test_matches_posts_to_messages_and_drops_the_gone(self):
        history = [message(100, BOT, 0, [reaction("😂", 2)]), message(150, ALICE, 1), message(160, BOB, 5),
                   message(102, BOT, 40)]
        calls = []

        class Channel:
            async def history(self, **kwargs):
                calls.append(kwargs)
                for m in history:
                    yield m

        class Client:
            user = SimpleNamespace(id=BOT)

            async def fetch_channel(self, id):
                if id == 99:
                    raise r.discord.NotFound(SimpleNamespace(status=404, reason="Not Found"), "Unknown Channel")
                return Channel()

        found = [{"kind": "reply", "channel_id": 10, "id": 100, "at": "2026-10-01T12:00:00+00:00"},
                 {"kind": "reply", "channel_id": 10, "id": 101, "at": "2026-10-01T12:20:00+00:00"},  # deleted
                 {"kind": "reply", "channel_id": 10, "id": 102, "at": "2026-10-01T12:40:00+00:00"},
                 {"kind": "reply", "channel_id": 99, "id": 300, "at": "2026-10-01T13:00:00+00:00"}]  # channel gone
        kept = await r.collect(Client(), found)
        self.assertEqual([(p["id"], p["reactions"], p["audience"]) for p in kept], [(100, {"😂": 2}, 2), (102, {}, 0)])
        self.assertEqual(len(calls), 1)  # one history walk for the channel, not a request per post
        self.assertEqual(calls[0]["after"].id, 99)


if __name__ == "__main__":
    unittest.main()
