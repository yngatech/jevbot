"""Offline checks for rocky_replay: it reads a live checkout's log and memories, replays on a copy, and writes nothing.
The model's answers are synthetic.

    uv run --with-requirements requirements.txt python -m unittest test_replay
"""

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fakes import answer, isolate_rocky, patch_all
import jev_bot as j
import rocky_harness as rh
import rocky_replay

CHANNEL = 300
FACT = {"n": 1, "about": "kettle", "fact": "likes soup"}
LOG = {"channel": "bot-spam", "space": "1", "next_n": 2, "ids": {}, "shown": [FACT], "seeded": True, "asked": None,
       "code": None, "events": [{"type": "message", "n": 1, "author": "kettle", "uid": None, "text": "soup tonight?",
                                 "at": "2026-10-12T18:00:00+00:00", "to_bot": False, "reply_to": None}]}
MEMORIES = {"1": {"next": 2, "facts": [FACT]}}


class ReplayTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        self.live = Path(d.name)
        (self.live / "rocky_logs").mkdir()
        (self.live / "rocky_logs" / f"{CHANNEL}.json").write_text(json.dumps(LOG))
        (self.live / "rocky_memories.json").write_text(json.dumps(MEMORIES))
        (self.live / "rocky_lines.txt").write_text("# a comment\nAmaze, amaze, amaze!\n")
        (self.live / "rocky_body.txt").write_text("# a comment\n\nHe has five legs\nand five hands.\n\nNo face.\n")
        isolate_rocky(self, d.name)  # put back after, as the replay points them at its copy
        patch_all(self, [(rh, "save_log", rh.save_log), (rh, "save_memories", rh.save_memories), (rh, "clock", rh.clock),
                         (rh, "pictures", rh.pictures), (j, "ROCKY_LINES", []), (j, "ROCKY_BODY", []), (j, "HEADERS", dict(j.HEADERS)),
                         (j, "TIMEZONE", j.ZoneInfo("UTC")), (rocky_replay, "live_dir", lambda: self.live)])
        self.before = {p: p.read_text() for p in self.live.rglob("*") if p.is_file()}

    async def run_main(self, *argv, env=None):
        llm = AsyncMock(side_effect=lambda *a, **k: answer("Soup good.", [("remember", {"about": "pip", "fact": "is hungry"})]))
        out = io.StringIO()
        with patch.object(j, "llm_message", llm), patch.object(sys, "argv", ["rocky_replay.py", *argv]), \
                patch.dict(os.environ, env or {"OPENROUTER_API_KEY_EVAL": "eval-key"}), contextlib.redirect_stdout(out):
            await rocky_replay.main()
        return llm, out.getvalue()

    async def test_each_run_starts_from_the_live_log_and_writes_nothing(self):
        llm, out = await self.run_main("--channel", str(CHANNEL), "-n", "2", "rocky soup?")
        self.assertEqual(llm.await_count, 2)
        for call in llm.await_args_list:
            messages = call.args[1]
            self.assertIn("Amaze, amaze, amaze!", j.text_of(messages[0]["content"]))  # the live rocky_lines
            self.assertIn("He has five legs\nand five hands.\n\nNo face.", j.text_of(messages[0]["content"]))  # and rocky_body
            self.assertIn("kettle: likes soup", j.text_of(messages[1]["content"]))  # the live <memories>
            text = j.text_of(messages[-1]["content"])
            self.assertIn("soup tonight?", text)
            self.assertEqual(text.count("rocky soup?"), 1)  # not the run before's message too
            self.assertIn('at="18:02"', text)  # 2 minutes after the log ends
        # run 2 remembered on the live memories, not on run 1's as well
        self.assertEqual([f["fact"] for f in rh.memories["1"]["facts"]], ["likes soup", "is hungry"])
        self.assertEqual(j.HEADERS["Authorization"], "Bearer eval-key")
        self.assertIn("[rocky soup?] #2", out)
        self.assertIn("🧠 remembered: pip is hungry", out)
        self.assertEqual({p: p.read_text() for p in self.live.rglob("*") if p.is_file()}, self.before)

    async def test_with_another_message_both_are_its_to_answer(self):
        llm, _ = await self.run_main("--channel", str(CHANNEL), "rocky soup?", "--with", "fern: rocky tea?")
        text = j.text_of(llm.await_args.args[1][-1]["content"])
        self.assertIn('from="fern"', text)
        self.assertEqual(text.count('woke_you="yes"'), 2)

    async def test_it_needs_the_eval_key(self):
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            await self.run_main("--channel", str(CHANNEL), "hi", env={"OPENROUTER_API_KEY_EVAL": ""})

    async def test_a_fresh_chat_ends_when_its_said(self):
        llm, _ = await self.run_main("--fresh", "--at", "2026-10-16 19:00", "rocky get us a beer")
        text = j.text_of(llm.await_args.args[1][-1]["content"])
        self.assertNotIn("soup tonight?", text)
        self.assertIn("anyone watching the match tonight", text)
        self.assertIn('at="19:00"', text)

    async def test_a_picture_gets_one_photo_tag(self):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        image = Path(d.name) / "dot.png"
        rh.Image.new("RGB", (8, 8), "red").save(image)
        for said in ("rocky look", "rocky [photo]"):
            llm, _ = await self.run_main("--fresh", "--image", str(image), said)
            text = j.text_of(llm.await_args.args[1][-1]["content"])
            self.assertEqual(text.count("[photo]"), 1, said)


if __name__ == "__main__":
    unittest.main()
