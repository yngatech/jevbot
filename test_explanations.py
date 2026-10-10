"""Offline checks for cached Why and Context answers, using synthetic logs and Discord sends.

    uv run --with-requirements requirements.txt python -m unittest test_explanations
"""

import asyncio
import json
import os
import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

os.environ.setdefault("DISCORD_TOKEN_JEV", "test")
os.environ.setdefault("OPENROUTER_API_KEY", "test")

import discord
import jev_bot as j


class ExplanationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.user = SimpleNamespace(id=100, display_name="Testbot")
        self.guild = SimpleNamespace(me=self.user, emojis=[])
        self.channel = SimpleNamespace(id=300)
        self.logged = {"at": "2026-10-01T12:00:00+00:00", "channel_id": 300, "message_id": 10, "reply_id": 20,
                       "author": "Speaker", "bot_name": "Testbot", "message": "tea?", "history": [],
                       "reply": "Tea", "transcript": "Speaker: tea?\n\nTestbot: ", "asked": .9,
                       "steps": [{"word": "tea", "top": [["tea", .8, .8], ["coffee", .2, .2]]}]}
        self.sent = []
        for obj, attr, value in [(j, "LOG_DIR", Path(self.dir.name)), (j.bot._connection, "user", self.user),
                                 (j, "explanation_cache", OrderedDict()), (j, "explanation_pending", {})]:
            p = patch.object(obj, attr, value)
            p.start()
            self.addCleanup(p.stop)

    def target(self, t=None, *, original=False):
        t = self.logged if t is None else t
        return Mock(spec=discord.Message, id=t["message_id"] if original else t["reply_id"],
                    author=SimpleNamespace(id=500) if original else self.user,
                    content=t["message"] if original else t["reply"])

    async def capture(self, content=None, **kwargs):
        file = kwargs.get("file")
        data = None
        if file is not None and file is not discord.utils.MISSING:
            data = file.fp.read()
            file.close()
            file.fp.close()  # closing a sent stream mustn't break the next request
        self.sent.append((content, file, data, kwargs))

    def command(self, target=None):
        return SimpleNamespace(channel=self.channel, guild=self.guild, reply=AsyncMock(side_effect=self.capture),
                               reference=SimpleNamespace(message_id=target.id, resolved=target) if target else None)

    def interaction(self):
        return SimpleNamespace(channel_id=self.channel.id, guild=self.guild,
                               response=SimpleNamespace(defer=AsyncMock()),
                               followup=SimpleNamespace(send=AsyncMock(side_effect=self.capture)))

    async def explain(self, t=None, kind="why"):
        await {"why": j.explain_why, "context": j.explain_context}[kind](
            self.channel.id, self.guild, self.target(t), self.capture)

    async def test_commands_and_actions_reuse_a_real_chart_with_fresh_files(self):
        j.write_trace(self.logged)
        with patch.object(j.why_chart, "render", wraps=j.why_chart.render) as render:
            await j.why(self.command())  # latest
            interaction = self.interaction()
            await j.why_action.callback(interaction, self.target())  # jev's reply
            await j.why(self.command(self.target(original=True)))  # the message jev answered
        render.assert_called_once()
        files = [s[1] for s in self.sent]
        self.assertEqual(len({id(f) for f in files}), 3)
        self.assertTrue(all(f.fp.closed for f in files))
        self.assertTrue(all(f.filename == "why-tea.png" for f in files))
        png = self.sent[0][2]
        self.assertTrue(png.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertTrue(all(s[2] == png for s in self.sent))
        self.assertTrue(self.sent[1][3]["ephemeral"])
        self.assertEqual(self.sent[1][3]["allowed_mentions"].to_dict()["parse"], [])
        interaction.response.defer.assert_awaited_once_with(ephemeral=True, thinking=True)

    async def test_context_commands_and_actions_reuse_inline_and_attached_transcripts(self):
        for text in ("Speaker: tea? @everyone ```\n\nTestbot: ", "Speaker: " + "tea " * 600):
            with self.subTest(attached=len(text) > j.CONTEXT_LIMIT):
                self.logged["transcript"] = text
                j.write_trace(self.logged)
                self.sent.clear()
                with patch.object(j, "logged_context", wraps=j.logged_context) as context:
                    await j.context(self.command())
                    await j.context_action.callback(self.interaction(), self.target())
                context.assert_called_once()
                self.assertEqual(self.sent[0][0], self.sent[1][0])
                if len(text) > j.CONTEXT_LIMIT:
                    self.assertEqual([s[2] for s in self.sent], [text.encode(), text.encode()])
                    self.assertIsNot(self.sent[0][1], self.sent[1][1])
                    self.assertEqual(self.sent[1][1].filename, "context.txt")
                else:
                    self.assertIn(j.unfence(text), self.sent[0][0])
                    self.assertIsNone(self.sent[0][1])
                self.assertTrue(self.sent[1][3]["ephemeral"])
                self.assertEqual(self.sent[1][3]["allowed_mentions"].to_dict()["parse"], [])

    async def test_latest_updates_and_an_older_target_keeps_its_own_chart(self):
        j.write_trace(self.logged)
        newer = dict(self.logged, message_id=11, reply_id=21, reply="Coffee",
                     transcript="Speaker: coffee?\n\nTestbot: ",
                     steps=[{"word": "coffee", "top": [["coffee", .9, .9]]}])
        with patch.object(j.why_chart, "render", side_effect=[b"tea chart", b"coffee chart"]) as render:
            await j.why(self.command())
            await j.context(self.command())
            j.write_trace(newer)
            await j.why(self.command())
            await j.context(self.command())
            await self.explain()
        self.assertEqual(render.call_count, 2)
        self.assertEqual([self.sent[i][2] for i in (0, 2, 4)], [b"tea chart", b"coffee chart", b"tea chart"])
        self.assertIn("tea?", self.sent[1][0])
        self.assertIn("coffee?", self.sent[3][0])

    async def test_changed_logs_rerender_even_when_the_reply_text_is_the_same(self):
        j.write_trace(self.logged)
        with patch.object(j.why_chart, "render", side_effect=[b"before", b"after"]) as render:
            await self.explain()
            changed = dict(self.logged, steps=[{"word": "tea", "top": [["tea", .5, .5]]}])
            next(j.LOG_DIR.glob("*.jsonl")).write_text(json.dumps(changed) + "\n")
            await self.explain()
        self.assertEqual(render.call_count, 2)
        self.assertEqual([s[2] for s in self.sent], [b"before", b"after"])

    async def test_missing_and_deleted_logs_do_not_return_a_cached_answer(self):
        with patch.object(j.why_chart, "render", return_value=b"chart") as render:
            await self.explain()
            self.assertEqual(self.sent[-1][0], "Nothing logged for that")
            j.write_trace(self.logged)
            await self.explain()
            await self.explain(kind="context")
            next(j.LOG_DIR.glob("*.jsonl")).unlink()
            await self.explain()
            await self.explain(kind="context")
        render.assert_called_once()
        self.assertEqual([s[0] for s in self.sent[-2:]], ["Nothing logged for that"] * 2)

    async def test_edited_target_still_has_to_match_its_log(self):
        j.write_trace(self.logged)
        target = self.target()
        with patch.object(j.why_chart, "render", return_value=b"chart") as render:
            await j.why(self.command(target))
            target.content = "Edited reply"
            await j.why(self.command(target))
        render.assert_called_once()
        self.assertEqual(self.sent[-1][0], "Nothing logged for that")

    async def test_llm_and_status_charts_are_cached(self):
        llm = dict(self.logged, llm="deepseek", llm_tokens=[{"token": "Tea", "p": .8, "top": [["Tea", .8]]}])
        status = {"at": self.logged["at"], "status": "I feel tea is good", "status_id": 40,
                  "bot_name": "Testbot", "transcript": "Testbot's diary, today: I feel",
                  "steps": [{"word": "tea", "top": [["tea", .8, .8]]}]}
        with patch.object(j, "STATUS_CHANNEL", self.channel.id), \
                patch.object(j.why_chart, "render", wraps=j.why_chart.render) as render:
            j.write_trace(llm)
            await j.why(self.command())
            await self.explain(llm)
            self.assertTrue(render.call_args.kwargs["tokens"])
            j.write_trace(status)
            await j.why(self.command())
            await j.why(self.command())
            await j.context(self.command())
            await j.context(self.command())
        self.assertEqual(render.call_count, 2)
        self.assertTrue(all(s[2].startswith(b"\x89PNG") for s in self.sent[:4]))
        self.assertEqual(self.sent[4][0], self.sent[5][0])
        self.assertIn("before the status", self.sent[4][0])

    async def test_reaction_chart_reuse_skips_emoji_fetches_and_follows_custom_emoji_changes(self):
        reaction = dict(self.logged, reaction=":spark:", reaction_candidates=[[":spark:", .8, .8]], steps=[])
        j.write_trace(reaction)
        self.guild.emojis = [SimpleNamespace(name="spark", id=900)]
        with patch.object(j, "emoji_image", new_callable=AsyncMock, return_value=None) as fetch, \
                patch.object(j.why_chart, "render", return_value=b"reaction chart") as render:
            await self.explain()
            await self.explain()
            fetch.assert_awaited_once()
            self.guild.emojis = [SimpleNamespace(name="spark", id=901)]
            await self.explain()
        self.assertEqual(render.call_count, 2)
        self.assertEqual(fetch.await_count, 2)
        self.assertEqual(self.sent[0][1].filename, "why-reacted-to-tea.png")

    async def test_legacy_reaction_and_llm_without_tokens_reuse_their_text(self):
        for fields, expected in [({"reaction": "👍", "reaction_candidates": ["👍", "👎"], "steps": []}, "Reacted 👍"),
                                 ({"llm": "kimi"}, "kimi doesn't say how likely its words were")]:
            with self.subTest(fields=fields), patch.object(j.why_chart, "render") as render, \
                    patch.object(j, "why_response", wraps=j.why_response) as prepare:
                j.write_trace(dict(self.logged, **fields))
                await self.explain()
                await self.explain()
                prepare.assert_awaited_once()
                render.assert_not_called()
                self.assertIn(expected, self.sent[-1][0])

    async def test_channel_and_legacy_bot_name_changes_do_not_share_cached_charts(self):
        del self.logged["bot_name"]
        j.write_trace(self.logged)
        with patch.object(j.why_chart, "render", return_value=b"chart") as render:
            await self.explain()
            self.user.display_name = "Newbot"
            await self.explain()
            self.assertEqual(render.call_args.args[0], "Newbot")
            other = dict(self.logged, channel_id=301)
            j.write_trace(other)
            await j.explain_why(301, self.guild, self.target(other), self.capture)
        self.assertEqual(render.call_count, 3)

    async def test_entry_limit_evicts_the_least_recently_used_chart(self):
        traces = [dict(self.logged, message_id=10 + n, reply_id=20 + n) for n in range(3)]
        for t in traces:
            j.write_trace(t)
        with patch.object(j, "EXPLANATION_CACHE_ENTRIES", 2), \
                patch.object(j.why_chart, "render", return_value=b"chart") as render:
            for n in (0, 1, 0, 2, 0, 1):
                await self.explain(traces[n])
        self.assertEqual(render.call_count, 4)

    async def test_byte_limit_evicts_old_charts_and_oversized_charts_are_still_sent(self):
        newer = dict(self.logged, message_id=11, reply_id=21)
        j.write_trace(self.logged)
        j.write_trace(newer)
        with patch.object(j, "EXPLANATION_CACHE_BYTES", 80), \
                patch.object(j.why_chart, "render", return_value=b"a" * 40) as render:
            await self.explain()
            await self.explain(newer)
            await self.explain()
        self.assertEqual(render.call_count, 3)
        j.explanation_cache.clear()
        with patch.object(j, "EXPLANATION_CACHE_BYTES", 80), \
                patch.object(j.why_chart, "render", return_value=b"a" * 100) as render:
            await self.explain()
            await self.explain()
        self.assertEqual(render.call_count, 2)
        self.assertEqual([s[2] for s in self.sent[-2:]], [b"a" * 100] * 2)

    async def test_failed_render_is_retried(self):
        j.write_trace(self.logged)
        with patch.object(j.why_chart, "render", side_effect=[RuntimeError("render failed"), b"chart"]) as render:
            with self.assertRaisesRegex(RuntimeError, "render failed"):
                await self.explain()
            await self.explain()
            await self.explain()
        self.assertEqual(render.call_count, 2)
        self.assertEqual([s[2] for s in self.sent], [b"chart", b"chart"])

    async def test_concurrent_requests_share_preparation_even_if_one_is_cancelled(self):
        for cancel in (False, True):
            with self.subTest(cancel=cancel):
                j.explanation_cache.clear()
                started, finish = asyncio.Event(), asyncio.Event()

                async def prepare(*args):
                    started.set()
                    await finish.wait()
                    return None, b"chart", "why-tea.png"

                with patch.object(j, "why_response", side_effect=prepare) as render:
                    first = asyncio.create_task(j.cached_explanation("why", 300, self.guild, self.logged))
                    await asyncio.wait_for(started.wait(), 1)
                    second = asyncio.create_task(j.cached_explanation("why", 300, self.guild, self.logged))
                    await asyncio.sleep(0)  # the second caller joins the work before it finishes
                    if cancel:
                        first.cancel()
                        with self.assertRaises(asyncio.CancelledError):
                            await first
                    finish.set()
                    self.assertEqual(await second, (None, b"chart", "why-tea.png"))
                    if not cancel:
                        self.assertEqual(await first, await second)
                    self.assertEqual(await j.cached_explanation("why", 300, self.guild, self.logged), await second)
                render.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
