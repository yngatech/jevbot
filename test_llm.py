"""Offline behavioral checks for the LLM writer: !model, replies, statuses and !why, with the LLM's answers synthetic.

    uv run --with-requirements requirements.txt python -m unittest test_llm
"""

import os
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

os.environ.setdefault("DISCORD_TOKEN_JEV", "test")
os.environ.setdefault("OPENROUTER_API_KEY", "test")

import jev_bot as j  # noqa: E402

TOKENS = [{"token": "Is", "p": 0.6, "top": [["Is", 0.6], ["Biscuit", 0.3]]},
          {"token": " called", "p": 0.02, "top": [[" Biscuit", 0.9], [" cat", 0.05]]}]


class _Case(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.trace = {"cost": 0.0, "requests": 0}
        token = j.trace.set(self.trace)
        self.addCleanup(j.trace.reset, token)
        for obj, attr, value in [(j, "MODEL_PATH", Path(self.dir.name) / "model.json"), (j, "model_name", "jev"),
                                 (j, "has_credit", True), (j, "channel_history", defaultdict(list)),
                                 (j, "TIMEZONE", j.ZoneInfo("UTC"))]:  # not whatever .env sets
            p = patch.object(obj, attr, value)
            p.start()
            self.addCleanup(p.stop)


class ModelCommandTests(_Case):
    def setUp(self):
        super().setUp()
        self.user = SimpleNamespace(id=100, bot=True, display_name="rocky")
        p = patch.object(j.bot._connection, "user", self.user)
        p.start()
        self.addCleanup(p.stop)

    def message(self, content):
        return SimpleNamespace(id=600, content=content, guild=None, channel=SimpleNamespace(id=300),
                               author=SimpleNamespace(id=500, bot=False, display_name="pip"),
                               mentions=[], role_mentions=[], reference=None, reply=AsyncMock())

    def test_command_takes_model_with_or_without_a_name(self):
        for content, want in [("!model", "!model"), ("!model deepseek", "!model"), ("!MODEL Haiku", "!model"),
                              ("!model deepseek please", None), ("!why", "!why"), ("!why now", None),
                              ("what model are you", None)]:
            with self.subTest(content=content):
                self.assertEqual(j.command(self.message(content)), want)

    async def test_on_message_routes_model_without_answering(self):
        m = self.message("!model kimi")
        with patch.object(j, "DM_USERS", {500}), patch.object(j, "model_command", new_callable=AsyncMock) as cmd, \
                patch.object(j, "respond", new_callable=AsyncMock) as respond:
            await j.on_message(m)
        cmd.assert_awaited_once_with(m)
        respond.assert_not_awaited()

    async def test_switching_is_kept_and_unknown_names_change_nothing(self):
        m = self.message("!model deepseek")
        await j.model_command(m)
        self.assertEqual(j.model_name, "deepseek")
        self.assertEqual(j.load_model(), "deepseek")  # what a restart would pick up
        self.assertIn("Now writing with **deepseek** (deepseek/deepseek-v4-pro)", m.reply.call_args.args[0])

        m = self.message("!model gpt9")
        await j.model_command(m)
        self.assertEqual(j.model_name, "deepseek")
        self.assertIn("No model called 'gpt9'", m.reply.call_args.args[0])

        m = self.message("!model")
        await j.model_command(m)
        self.assertIn("Writing with **deepseek**", m.reply.call_args.args[0])

    def test_unknown_or_unreadable_saved_model_falls_back_to_jev(self):
        for saved in ['{"model": "gpt9"}', "not json"]:
            with self.subTest(saved=saved):
                j.MODEL_PATH.write_text(saved)
                self.assertEqual(j.load_model(), "jev")


class ReplyTests(_Case):
    async def test_reply_goes_to_the_chosen_model_and_is_cleaned(self):
        j.model_name = "deepseek"
        llm = AsyncMock(return_value=('rocky: "Is called Biscuit. Small predator."\nrocky: more', TOKENS))
        with patch.object(j, "llm", llm), patch.object(j, "loom", new_callable=AsyncMock) as loom:
            r = await j.generate_reply("what's my cat called?", "kettle", "rocky", history=[])
        self.assertEqual(r, "Is called Biscuit. Small predator.")
        loom.assert_not_awaited()
        name, messages = llm.call_args.args
        self.assertEqual(name, "deepseek")
        self.assertIn("talk like Rocky", messages[0]["content"])
        self.assertRegex(messages[1]["content"],
                         r"^The chat so far, times in UTC:\n\n--- \w+ \d+ \w+\n\d\d:\d\d kettle: what's my cat called\?\n\n")
        self.assertNotIn("rocky: \n", messages[1]["content"])  # its empty turn is asked for, not shown
        self.assertIn("Write rocky's reply to kettle's last message.", messages[1]["content"])  # who it answers
        self.assertEqual((self.trace["llm"], self.trace["llm_tokens"]), ("deepseek", TOKENS))
        self.assertEqual(self.trace["llm_model"], "deepseek/deepseek-v4-pro")
        self.assertIn("kettle: what's my cat called?", self.trace["transcript"])  # for Context

    async def test_reply_logs_the_prompt_size_for_context(self):
        j.model_name = "kimi"
        session = MagicMock()
        session.__aenter__.return_value = session
        session.post.return_value.__aenter__.return_value = SimpleNamespace(status=200, json=AsyncMock(return_value={
            "choices": [{"message": {"content": "Is Biscuit."}}], "usage": {"prompt_tokens": 3_100, "cost": 0.001}}))
        with patch.object(j.aiohttp, "ClientSession", return_value=session), patch.object(j, "set_credit", AsyncMock()):
            await j.generate_reply("what's my cat called?", "kettle", "rocky", history=[])
        messages = session.post.call_args.kwargs["json"]["messages"]
        self.assertEqual(self.trace["llm_window"], 262_144)
        self.assertEqual(self.trace["llm_prompt_tokens"], 3_100)
        self.assertEqual(self.trace["llm_prompt_chars"], sum(len(m["content"]) for m in messages))
        chat = self.trace["transcript"].rsplit("\n", 1)[0]
        self.assertIn(chat, messages[1]["content"])
        self.assertEqual(self.trace["llm_chat_chars"], len(chat))

    async def test_transcript_shows_the_local_time_when_the_minute_changes_and_a_line_per_day(self):
        j.model_name = "deepseek"
        utc = j.timezone.utc
        history = [{"role": "user", "name": "pip", "content": "night all", "to_bot": False,
                    "at": j.datetime(2026, 10, 9, 22, 50, tzinfo=utc)},
                   {"role": "user", "name": "kettle", "content": "rocky you up?", "at": j.datetime(2026, 10, 9, 22, 58, tzinfo=utc),
                    "reply": "Always up.", "reply_at": j.datetime(2026, 10, 9, 22, 58, 40, tzinfo=utc)},
                   {"role": "user", "name": "kettle", "content": "nice", "at": j.datetime(2026, 10, 9, 22, 59, tzinfo=utc)},
                   {"role": "user", "name": "pip", "content": "old reply", "at": j.datetime(2026, 10, 9, 23, 30, tzinfo=utc),
                    "reply": "No time on this one."}]
        llm = AsyncMock(return_value=("Is morning. Go.", []))
        with patch.object(j, "llm", llm), patch.object(j, "TIMEZONE", j.ZoneInfo("Europe/London")):
            await j.generate_reply("morning", "kettle", "rocky", history=history,
                                   at=j.datetime(2026, 10, 10, 7, 5, tzinfo=utc))
        self.assertIn("The chat so far, times in Europe/London:\n\n"
                      "--- Fri 9 Oct\n23:50 pip: night all\n23:58 kettle: rocky you up?\nrocky: Always up.\n"
                      "23:59 kettle: nice\n--- Sat 10 Oct\n00:30 pip: old reply\nrocky: No time on this one.\n"
                      "08:05 kettle: morning\n\n",
                      llm.call_args.args[1][1]["content"])
        self.assertIn("08:05 kettle: morning\nrocky: ", self.trace["transcript"])  # for Context

    async def test_jev_transcript_has_no_times(self):
        history = [{"role": "user", "name": "pip", "content": "hi", "at": j.datetime.now(j.timezone.utc)}]
        with patch.object(j, "loom", new_callable=AsyncMock, return_value=["Hi"]):
            await j.generate_reply("cat?", "kettle", "rocky", history=history, at=j.datetime.now(j.timezone.utc))
        self.assertEqual(self.trace["transcript"], "pip: hi\nkettle: cat?\nrocky: ")

    async def test_jev_still_looms(self):
        with patch.object(j, "llm", new_callable=AsyncMock) as llm, \
                patch.object(j, "loom", new_callable=AsyncMock, return_value=["Biscuit"]):
            self.assertEqual(await j.generate_reply("cat?", "kettle", "rocky", history=[]), "Biscuit")
        llm.assert_not_awaited()

    async def test_switching_mid_reply_finishes_with_the_model_it_started_with(self):
        j.model_name = "kimi"

        async def answer(name, messages):
            j.model_name = "jev"  # !model jev arrives while the request is out
            return "First empty", []

        with patch.object(j, "llm", side_effect=answer) as llm:
            await j.generate_reply("hi", "pip", "rocky", history=[])
        self.assertEqual(llm.call_args.args[0], "kimi")

    async def test_empty_answer_is_retried_then_gives_up_with_dots(self):
        j.model_name = "kimi"
        llm = AsyncMock(side_effect=[("", []), ("Hello friend.", [])])
        with patch.object(j, "llm", llm):
            self.assertEqual(await j.generate_reply("hi", "pip", "rocky", history=[]), "Hello friend.")
        llm = AsyncMock(return_value=("  ", []))
        with patch.object(j, "llm", llm):
            self.assertEqual(await j.generate_reply("hi", "pip", "rocky", history=[]), "...")
        self.assertEqual(llm.await_count, j.LLM_ATTEMPTS)

    async def test_haiku_gets_shuffled_lines_the_tail_and_the_question_dice(self):
        j.model_name = "haiku"
        for roll, tagless in [(0.1, False), (0.9, True)]:
            with self.subTest(roll=roll), patch.object(j.random, "random", return_value=roll), \
                    patch.object(j, "rocky_prompt", return_value="voice") as prompt, \
                    patch.object(j, "llm", AsyncMock(return_value=("Biscuit.", []))) as llm:
                await j.generate_reply("cat?", "kettle", "rocky", history=[])
            ask = llm.call_args.args[1][1]["content"]
            self.assertIn("one short sentence at most", ask)
            self.assertEqual('no ", question?" tag' in ask, tagless)
            prompt.assert_called_with("rocky", True)


class StatusTests(_Case):
    async def test_status_keeps_its_opener_on_one_line(self):
        j.model_name = "deepseek"
        for answer, want in [("I wonder if sleep is same\nwhen no one watches.", "I wonder if sleep is same when no one watches."),
                             ("Why cat knock tea, question?", "I wonder why cat knock tea, question?"),
                             ("I wonder " + "very " * 40, None)]:
            with self.subTest(answer=answer), patch.object(j, "llm", AsyncMock(return_value=(answer, TOKENS))):
                mood = await j.generate_status("rocky", ["i", "wonder"], "kettle: my cat knocked my tea over")
            self.assertLessEqual(len(mood), 128)
            if want:
                self.assertEqual(mood, want)
        self.assertIn("kettle: my cat knocked my tea over", self.trace["transcript"])
        self.assertEqual(self.trace["llm"], "deepseek")

    async def test_accepted_status_keeps_its_tokens_for_why_and_its_prompt_size_for_context(self):
        j.model_name = "deepseek"
        sizes = iter([2_000, 3_000])
        async def llm(name, messages):
            j.note(llm_window=1, llm_prompt_tokens=next(sizes), llm_prompt_chars=1)
            return "I feel cat is tiny disaster.", TOKENS
        with patch.object(j, "llm", llm), patch.object(j, "status_score", AsyncMock(side_effect=[0.1, 0.9])):
            mood = await j.generate_filtered_status("rocky", ["i", "feel"], "kettle: cat")
        self.assertEqual(mood, "I feel cat is tiny disaster.")
        self.assertEqual((self.trace["llm"], self.trace["llm_tokens"]), ("deepseek", TOKENS))
        self.assertEqual((self.trace["llm_prompt_tokens"], self.trace["llm_chat_chars"]), (3_000, len("kettle: cat")))


class WhyTests(_Case):
    def test_panels_add_a_token_sampled_from_outside_its_top_few(self):
        first, second = j.llm_why_panels(TOKENS)
        self.assertEqual((first["picked"], first["so_far"], first["rows"][0]), ("Is", "", ["Is", 0.6, 0.6]))
        self.assertEqual((second["so_far"], second["picked"]), ("Is", " called"))
        self.assertEqual([r[0] for r in second["rows"]], [" Biscuit", " cat", " called"])

    async def test_why_charts_tokens_or_says_there_are_none(self):
        m = SimpleNamespace(channel=SimpleNamespace(id=300), guild=None,
                            reply=AsyncMock(return_value=SimpleNamespace(attachments=[])), reference=None)
        logged = {"llm": "kimi", "reply": "Biscuit", "bot_name": "rocky"}
        with patch.object(j, "find_trace", return_value=logged), patch.object(j.why_chart, "render") as render:
            await j.why(m)
        render.assert_not_called()
        self.assertIn("kimi doesn't say how likely its words were", m.reply.call_args.args[0])

        logged = {"llm": "deepseek", "reply": "Is called", "bot_name": "rocky", "llm_tokens": TOKENS}
        with patch.object(j, "find_trace", return_value=logged), \
                patch.object(j.why_chart, "render", return_value=b"png") as render:
            await j.why(m)
        self.assertTrue(render.call_args.kwargs["tokens"])
        self.assertEqual(len(render.call_args.args[2]), 2)
        self.assertEqual(render.call_args.kwargs["note"], "Model: deepseek/deepseek-v4-pro")  # logged before llm_model
        self.assertEqual(m.reply.call_args.kwargs["file"].filename, "why-is-called.png")

        logged["llm_model"] = "deepseek/deepseek-v4-flash"
        with patch.object(j, "find_trace", return_value=logged), \
                patch.object(j.why_chart, "render", return_value=b"png") as render:
            await j.why(m)
        self.assertEqual(render.call_args.kwargs["note"], "Model: deepseek/deepseek-v4-flash")

    def test_repeated_alternatives_and_the_end_token_are_dropped(self):
        lp = lambda p: __import__("math").log(p)
        raw = [{"token": "Wh", "logprob": lp(0.94), "top_logprobs": [{"token": "Wh", "logprob": lp(0.94)},
                                                                   {"token": "M", "logprob": lp(0.02)}]},
               {"token": "isk", "logprob": lp(0.9), "top_logprobs": [{"token": "isk", "logprob": lp(0.9)},
                                                                    {"token": "is", "logprob": lp(0.06)}]},
               {"token": "ers", "logprob": lp(0.99), "top_logprobs": [{"token": "isk", "logprob": lp(0.9)},
                                                                     {"token": "is", "logprob": lp(0.06)}]},
               {"token": "<｜end▁of▁sentence｜>", "logprob": lp(0.5), "top_logprobs": []}]
        tokens = j.llm_tokens(raw)
        self.assertEqual([t["token"] for t in tokens], ["Wh", "isk", "ers"])
        self.assertEqual(tokens[1]["top"], [["isk", 0.9], ["is", 0.06]])
        self.assertEqual(tokens[2]["top"], [])
        self.assertEqual(j.llm_why_panels(tokens)[2]["rows"], [["ers", 0.99, 0.99]])

    def test_chart_files_are_named_after_what_they_show(self):
        self.assertEqual(j.why_filename("Day good. Rain make race more interesting."), "why-day-good-rain-make-race-more.png")
        self.assertEqual(j.why_filename("lol that cat 😂", reacted=True), "why-reacted-to-lol-that-cat.png")
        self.assertEqual(j.why_filename("🤷"), "why.png")
        self.assertLessEqual(len(j.why_filename("supercalifragilistic " * 6)), 64)

    def test_clean_takes_the_first_paragraph_without_a_name(self):
        self.assertEqual(j.clean_llm('Rocky: "No. Just no."\n\nkettle: ok', "rocky"), "No. Just no.")
        self.assertEqual(j.clean_llm(None, "rocky"), "")


if __name__ == "__main__":
    unittest.main()
