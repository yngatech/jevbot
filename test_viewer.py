"""Offline checks for rocky_viewer: what its API answers from synthetic logs, and that it only sends what changed.

    uv run --with-requirements requirements.txt python -m unittest test_viewer
"""

import json
import tempfile
import unittest
import unittest.mock
from pathlib import Path

from aiohttp.test_utils import TestClient, TestServer
from yarl import URL

import rocky_viewer

LOG = {"channel": "bot-spam", "space": "400", "next_n": 3, "ids": {"900": 1, "901": "you", "902": 2},
       "shown": [{"n": 1, "about": "pip", "fact": "has a cat called Biscuit"}], "seeded": True, "asked": None,
       "events": [
           {"type": "message", "n": 1, "author": "pip", "at": "2026-10-12T18:00:00+00:00", "text": "rocky <3 soup?",
            "to_bot": True, "reply_to": None, "woke": True},
           {"type": "rocky", "at": "2026-10-12T18:00:02.500000+00:00", "text": "Soup good.", "id": 901,
            "calls": [{"id": "call_0", "name": "react", "arguments": "{\"emoji\": \"🍲\"}"}],
            "reasoning": [{"type": "reasoning.text", "text": "soup"}]},
           {"type": "result", "at": "2026-10-12T18:00:02.500000+00:00", "id": "call_0", "content": "Reacted."},
           {"type": "message", "n": 2, "author": "kettle", "at": "2026-10-12T18:01:00+00:00", "text": "<b>hi</b>",
            "to_bot": False, "reply_to": None}]}

# A turn's line as jev_bot writes it: the events' times come out as str() does them
def trace(at, message_id, channel_id=300, **fields):
    return {"at": at, "channel": "bot-spam", "channel_id": channel_id, "message_id": message_id, "author": "pip",
            "message": "rocky <3 soup?", "cost": 0.0004, "llm": "rocky", "history": ["big"], "transcript": "big",
            "rocky_events": [{"type": "rocky", "at": "2026-10-12 18:00:02.500000+00:00", "text": "Soup good."}], **fields}


class ViewerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        self.root = Path(d.name)
        (self.root / "rocky_logs").mkdir()
        (self.root / "logs").mkdir()
        (self.root / "rocky_logs" / "300.json").write_text(json.dumps(LOG))
        (self.root / "rocky_memories.json").write_text(json.dumps(
            {"400": {"next": 3, "facts": [{"n": 2, "about": "pip", "fact": "likes soup"}]}}))
        self.day = self.root / "logs" / "2026-10-12.jsonl"
        self.day.write_text(json.dumps(trace("2026-10-12T18:00:00+00:00", 900)) + "\n" +
                            json.dumps({"at": "2026-10-12T18:00:30+00:00", "channel_id": 300, "llm": "haiku"}) + "\n" +
                            json.dumps(trace("2026-10-12T18:00:40+00:00", 950, channel_id=301)) + "\n")
        self.client = TestClient(TestServer(rocky_viewer.app(self.root)))
        await self.client.start_server()
        self.addAsyncCleanup(self.client.close)

    async def get(self, url, **kwargs):
        r = await self.client.get(url, **kwargs)
        return r, (await r.json() if r.status == 200 else None)

    async def test_lists_the_channels_with_logs(self):
        r, body = await self.get("/api/channels")
        self.assertEqual(body["channels"], [{"id": "300", "channel": "bot-spam", "space": "400", "events": 4,
                                             "last": "2026-10-12T18:01:00+00:00"}])

    async def test_a_channel_has_its_log_memories_and_its_turns_by_when_they_were_logged(self):
        r, body = await self.get("/api/channels/300")
        self.assertEqual(body["events"], LOG["events"])
        self.assertEqual(body["facts"], [{"n": 2, "about": "pip", "fact": "likes soup"}])
        # Only Rocky's turns in this channel, keyed as the log writes its first response's time
        self.assertEqual(list(body["turns"]), ["2026-10-12T18:00:02.500000+00:00"])
        turn = body["turns"]["2026-10-12T18:00:02.500000+00:00"]
        self.assertEqual(turn["cost"], 0.0004)
        self.assertNotIn("history", turn)  # most of a line, and the log already shows what it saw
        self.assertNotIn("transcript", turn)

    async def test_it_answers_only_what_changed(self):
        r, _ = await self.get("/api/channels/300")
        tag = r.headers["ETag"]
        r, _ = await self.get("/api/channels/300", headers={"If-None-Match": tag})
        self.assertEqual(r.status, 304)
        with open(self.day, "a") as f:  # a new turn, and a line still being written
            f.write(json.dumps(trace("2026-10-12T18:02:00+00:00", 902)) + "\n" + '{"at": "2026-10-12T18:0')
        r, body = await self.get("/api/channels/300/turns", headers={"If-None-Match": tag})
        self.assertEqual(r.status, 200)
        self.assertEqual([t["message_id"] for t in body["turns"]], [902, 900])  # newest first, half a line left out
        with open(self.day, "a") as f:
            f.write('3:00+00:00", "channel_id": 300, "message_id": 903, "llm": "rocky"}\n')
        r, body = await self.get("/api/channels/300/turns")
        self.assertEqual([t["message_id"] for t in body["turns"]], [903, 902, 900])

    async def test_turns_come_a_page_at_a_time(self):
        with patch_page(2):
            with open(self.day, "a") as f:
                for i in range(3):
                    f.write(json.dumps(trace(f"2026-10-12T19:0{i}:00+00:00", 960 + i)) + "\n")
            r, first = await self.get("/api/channels/300/turns")
            r, second = await self.get("/api/channels/300/turns?page=1")
        self.assertEqual((first["total"], [t["message_id"] for t in first["turns"]]), (4, [962, 961]))
        self.assertEqual([t["message_id"] for t in second["turns"]], [960, 900])

    async def test_nothing_else_is_served(self):
        for url in ["/api/channels/301", "/api/channels/abc", "/static/../rocky_viewer.py",
                    "/static/%2e%2e/rocky_viewer.py", "/static/..%2f..%2f.env", "/rocky_logs/300.json"]:
            # sent as written — the client would otherwise take the ".." out itself
            r = await self.client.session.get(URL(str(self.client.make_url("")).rstrip("/") + url, encoded=True))
            self.assertEqual(r.status, 404, url)
        r, _ = await self.get("/api/channels/300/turns?page=x")
        self.assertEqual(r.status, 400)

    async def test_the_page_runs_only_its_own_script(self):
        r = await self.client.get("/static/app.js")
        self.assertEqual(r.status, 200)
        r = await self.client.get("/")
        self.assertEqual(r.status, 200)
        self.assertIn("script-src 'self'", r.headers["Content-Security-Policy"])
        self.assertNotIn("unsafe-inline", r.headers["Content-Security-Policy"])
        self.assertIn('src="/static/app.js"', await r.text())


def patch_page(size):
    return unittest.mock.patch.object(rocky_viewer, "TURN_PAGE", size)


if __name__ == "__main__":
    unittest.main()
