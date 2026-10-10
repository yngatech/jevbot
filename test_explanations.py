"""Offline checks for cached Why and Context answers, using synthetic logs and Discord sends.

    uv run --with-requirements requirements.txt python -m unittest test_explanations
"""

import asyncio
import json
import os
import tempfile
import time
import unittest
from collections import OrderedDict
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock, patch

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
        self.uploads = {}
        self.check_available = j.explanation_attachment_available
        self.available = AsyncMock(return_value=True)
        for obj, attr, value in [(j, "LOG_DIR", Path(self.dir.name)), (j.bot._connection, "user", self.user),
                                 (j, "explanation_cache", OrderedDict()), (j, "explanation_pending", {}),
                                 (j, "explanation_attachment_available", self.available)]:
            p = patch.object(obj, attr, value)
            p.start()
            self.addCleanup(p.stop)

    def target(self, t=None, *, original=False):
        t = self.logged if t is None else t
        return Mock(spec=discord.Message, id=t["message_id"] if original else t.get("status_id", t.get("reply_id")),
                    author=SimpleNamespace(id=500) if original else self.user,
                    content=t["message"] if original else t.get("status", t.get("reply")))

    async def capture(self, content=None, **kwargs):
        file = kwargs.get("file")
        data = None
        attachments = []
        if file is not None and file is not discord.utils.MISSING:
            data = file.fp.read()
            route = "ephemeral-attachments" if kwargs.get("ephemeral") else "attachments"
            expires = int(time.time()) + 3600
            url = (f"https://cdn.discordapp.com/{route}/300/{900 + len(self.uploads)}/{file.filename}"
                   f"?ex={expires:x}&is={expires - 3600:x}&hm=synthetic-signature")
            self.uploads[url] = data  # a stand-in for Discord's hosted files, outside the bot's cache
            attachments = [SimpleNamespace(url=url)]
            file.close()
            file.fp.close()
        self.sent.append((content, file, data, kwargs))
        return SimpleNamespace(attachments=attachments)

    def chart_bytes(self, sent):
        return sent[2] if sent[2] is not None else self.uploads[sent[3]["embed"].image.url]

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

    async def concurrent(self, phase, other=None, *, send=None, cancel=False):
        started, finish = asyncio.Event(), asyncio.Event()
        attr = {"render": "why_response", "validation": "explanation_attachment_available"}.get(phase)
        original = getattr(j, attr) if attr else send or self.capture

        async def gated(*args, **kwargs):
            if not started.is_set():
                started.set()
                await finish.wait()
            return await original(*args, **kwargs)

        def request(sender):
            return asyncio.create_task(j.send_explanation("why", 300, self.guild, self.logged, sender))

        with patch.object(j, attr, gated) if attr else nullcontext():
            tasks = [request((send or self.capture) if attr else gated)]
            try:
                await asyncio.wait_for(started.wait(), 1)
                tasks.append(request(other or self.capture))
                await asyncio.sleep(0)  # join the work before releasing the first caller
                if cancel:
                    tasks[0].cancel()
            finally:
                finish.set()
            return await asyncio.gather(*tasks, return_exceptions=True)

    async def test_commands_and_actions_reuse_a_real_chart_without_reuploading(self):
        j.write_trace(self.logged)
        with patch.object(j.why_chart, "render", wraps=j.why_chart.render) as render:
            await j.why(self.command())  # latest
            interaction = self.interaction()
            await j.why_action.callback(interaction, self.target())  # jev's reply
            await j.why(self.command(self.target(original=True)))  # the message jev answered
        render.assert_called_once()
        self.assertEqual(len(self.uploads), 1)
        self.assertTrue(self.sent[0][1].fp.closed)
        self.assertEqual(self.sent[0][1].filename, "why-tea.png")
        png = self.sent[0][2]
        self.assertTrue(png.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertTrue(all(self.chart_bytes(s) == png for s in self.sent))
        url = next(iter(self.uploads))
        self.assertEqual([s[3]["embed"].image.url for s in self.sent[1:]], [url, url])
        self.assertIs(self.sent[1][1], discord.utils.MISSING)
        cached, = j.explanation_cache.values()
        self.assertEqual(cached.url, url)
        self.assertTrue(all(value is None or isinstance(value, (str, int, float)) for value in vars(cached).values()))
        self.assertTrue(self.sent[1][3]["ephemeral"])
        self.assertEqual(self.sent[1][3]["allowed_mentions"].to_dict()["parse"], [])
        interaction.response.defer.assert_awaited_once_with(ephemeral=True, thinking=True)

    async def test_context_actions_reuse_inline_and_attached_transcripts(self):
        for text in ("Speaker: tea? @everyone ```\n\nTestbot: ", "Speaker: " + "tea " * 600):
            with self.subTest(attached=len(text) > j.CONTEXT_LIMIT):
                self.logged["transcript"] = text
                j.write_trace(self.logged)
                self.sent.clear()
                with patch.object(j, "logged_context", wraps=j.logged_context) as context:
                    await j.context_action.callback(self.interaction(), self.target())
                    await j.context_action.callback(self.interaction(), self.target())
                context.assert_called_once()
                if len(text) > j.CONTEXT_LIMIT:
                    self.assertEqual(self.sent[0][2], text.encode())
                    self.assertEqual(self.sent[0][1].filename, "context.txt")
                    self.assertIs(self.sent[1][1], discord.utils.MISSING)
                    url = next(reversed(self.uploads))
                    self.assertEqual(self.sent[1][0], f"{self.sent[0][0]}\n[context.txt]({url})")
                else:
                    self.assertEqual(self.sent[0][0], self.sent[1][0])
                    self.assertIn(j.unfence(text), self.sent[0][0])
                    self.assertIs(self.sent[0][1], discord.utils.MISSING)
                self.assertTrue(self.sent[1][3]["ephemeral"])
                self.assertEqual(self.sent[1][3]["allowed_mentions"].to_dict()["parse"], [])

    async def test_context_shows_how_much_of_an_llms_window_it_used(self):
        usage = {"llm": "haiku", "llm_window": 1_000_000, "llm_prompt_tokens": 9_800, "llm_prompt_chars": 30_000,
                 "llm_chat_chars": 20_000, "history": [{"author": "Speaker", "content": "hi"}] * 199}
        with patch.object(j, "STATUS_CHANNEL", self.channel.id):
            for name, t, chat in [
                    ("reply", dict(self.logged, **usage), "~7,000 tokens (200 messages)"),
                    ("long reply", dict(self.logged, **usage, transcript="Speaker: " + "tea " * 600),
                     "~7,000 tokens (200 messages)"),
                    ("status", {"at": self.logged["at"], "status": "I feel tea is good", "status_id": 40,
                                "bot_name": "Testbot", "transcript": "Recent chat", **usage}, "~7,000 tokens\n")]:
                with self.subTest(name):
                    self.sent.clear()
                    j.write_trace(t)
                    await j.context_action.callback(self.interaction(), self.target(t))
                    content = self.sent[0][0]
                    # chat's 20,000 characters count 1.25 times against the prompt's other 10,000: 7,000 of 9,800
                    self.assertIn("**haiku**: 9,800 of 1,000,000 tokens in its window — 1.0% used, 99.0% free\n"
                                  "- prompt: ~2,800 tokens\n- chat: " + chat, content + "\n")
                    self.assertLess(content.index("free"), content.index("```") if "```" in content else len(content))

    async def test_context_for_jev_or_an_older_log_has_no_window_usage(self):
        for t in (self.logged, dict(self.logged, llm="haiku")):
            with self.subTest(llm=t.get("llm")):
                self.sent.clear()
                j.write_trace(t)
                await j.context_action.callback(self.interaction(), self.target(t))
                self.assertTrue(self.sent[0][0].startswith('What Testbot saw before replying to "tea?":\n```'))
                self.assertNotIn("window", self.sent[0][0])

    async def test_a_private_upload_is_reused_with_all_signing_parameters(self):
        j.write_trace(self.logged)
        with patch.object(j.why_chart, "render", return_value=b"chart") as render:
            await j.why_action.callback(self.interaction(), self.target())
            await j.why(self.command())
            await j.why_action.callback(self.interaction(), self.target())
        render.assert_called_once()
        self.assertEqual(len(self.uploads), 1)
        url = next(iter(self.uploads))
        self.assertIn("/ephemeral-attachments/", url)
        self.assertIn("&hm=synthetic-signature", url)
        self.assertEqual([s[3]["embed"].image.url for s in self.sent[1:]], [url, url])
        self.assertTrue(self.sent[2][3]["ephemeral"])
        self.assertTrue(self.sent[0][3]["wait"])

    async def test_latest_updates_and_an_older_target_keeps_its_own_chart(self):
        j.write_trace(self.logged)
        newer = dict(self.logged, message_id=11, reply_id=21, reply="Coffee",
                     transcript="Speaker: coffee?\n\nTestbot: ",
                     steps=[{"word": "coffee", "top": [["coffee", .9, .9]]}])
        with patch.object(j.why_chart, "render", side_effect=[b"tea chart", b"coffee chart"]) as render:
            await j.why(self.command())
            await j.context_action.callback(self.interaction(), self.target())
            j.write_trace(newer)
            await j.why(self.command())
            await j.context_action.callback(self.interaction(), self.target(newer))
            await self.explain()
        self.assertEqual(render.call_count, 2)
        self.assertEqual([self.chart_bytes(self.sent[i]) for i in (0, 2, 4)], [b"tea chart", b"coffee chart", b"tea chart"])
        self.assertEqual(len(self.uploads), 2)
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
            await j.context_action.callback(self.interaction(), self.target(status))
            await j.context_action.callback(self.interaction(), self.target(status))
        self.assertEqual(render.call_count, 2)
        self.assertTrue(all(self.chart_bytes(s).startswith(b"\x89PNG") for s in self.sent[:4]))
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

    async def test_expiring_urls_are_regenerated_before_the_client_can_receive_an_expired_image(self):
        j.write_trace(self.logged)
        with patch.object(j.why_chart, "render", return_value=b"chart") as render:
            await self.explain()
            cached, = j.explanation_cache.values()
            with patch.object(j.time, "time", return_value=cached.expires - j.EXPLANATION_EXPIRY_MARGIN):
                await self.explain()
                await self.explain()
        self.assertEqual(render.call_count, 2)
        self.assertEqual(len(self.uploads), 2)
        self.assertEqual(self.sent[2][3]["embed"].image.url, next(reversed(self.uploads)))
        self.available.assert_awaited_once()  # the expiring URL was rejected before a network check

    async def test_a_deleted_hosted_attachment_is_regenerated_and_reuploaded(self):
        j.write_trace(self.logged)
        with patch.object(j.why_chart, "render", return_value=b"chart") as render:
            await self.explain()
            self.available.return_value = False
            await self.explain()
        self.assertEqual(render.call_count, 2)
        self.assertEqual(len(self.uploads), 2)
        self.assertEqual([s[2] for s in self.sent], [b"chart", b"chart"])

    async def test_unknown_expiry_or_missing_attachments_do_not_cache_an_unsafe_url(self):
        for url in (None, "https://cdn.discordapp.com/attachments/300/900/why-tea.png",
                    "https://cdn.discordapp.com/attachments/300/900/why-tea.png?ex=invalid&hm=synthetic"):
            with self.subTest(url=url), patch.object(j.why_chart, "render", return_value=b"chart") as render:
                j.explanation_cache.clear()

                async def send(*args, **kwargs):
                    await self.capture(*args, **kwargs)
                    return SimpleNamespace(attachments=[SimpleNamespace(url=url)] if url else [])

                await j.send_explanation("why", 300, self.guild, self.logged, send)
                await j.send_explanation("why", 300, self.guild, self.logged, send)
                self.assertEqual(render.call_count, 2)
                self.assertEqual(len(j.explanation_cache), 0)

    async def test_attachment_availability_checks_the_full_url_and_handles_errors(self):
        url = "https://cdn.discordapp.com/attachments/300/900/why-tea.png?ex=ffffffff&hm=synthetic"
        cases = [(status, None) for status in (200, 403, 404, 503)]
        cases += [(None, error) for error in (j.aiohttp.ClientError("synthetic network failure"), asyncio.TimeoutError())]
        for status, error in cases:
            with self.subTest(status=status, error=error):
                session = MagicMock()
                session.__aenter__.return_value = session
                session.head.return_value.__aenter__.return_value = SimpleNamespace(status=status)
                session.head.return_value.__aenter__.side_effect = error
                with patch.object(j.aiohttp, "ClientSession", return_value=session):
                    self.assertEqual(await self.check_available(url), status == 200)
                self.assertEqual(session.head.call_args.args[0], url)

    async def test_failed_render_is_retried(self):
        j.write_trace(self.logged)
        with patch.object(j.why_chart, "render", side_effect=[RuntimeError("render failed"), b"chart"]) as render:
            with self.assertRaisesRegex(RuntimeError, "render failed"):
                await self.explain()
            await self.explain()
            await self.explain()
        self.assertEqual(render.call_count, 2)
        self.assertEqual([self.chart_bytes(s) for s in self.sent], [b"chart", b"chart"])

    async def test_failed_upload_is_retried_without_caching_bytes_or_a_url(self):
        send = AsyncMock(side_effect=[RuntimeError("upload failed"), SimpleNamespace(attachments=[])])
        with patch.object(j.why_chart, "render", return_value=b"chart") as render:
            with self.assertRaisesRegex(RuntimeError, "upload failed"):
                await j.send_explanation("why", 300, self.guild, self.logged, send)
            self.assertEqual(len(j.explanation_cache), 0)
            self.assertTrue(send.call_args.kwargs["file"].fp.closed)
            await j.send_explanation("why", 300, self.guild, self.logged, self.capture)
            await j.send_explanation("why", 300, self.guild, self.logged, self.capture)
        self.assertEqual(render.call_count, 2)
        self.assertEqual(len(self.uploads), 1)

    async def test_rejected_embed_falls_back_to_a_fresh_attachment(self):
        async def send(*args, **kwargs):
            if kwargs.get("embed"):
                raise discord.Forbidden(Mock(status=403, reason="Forbidden"), "synthetic embed rejection")
            return await self.capture(*args, **kwargs)

        with patch.object(j.why_chart, "render", return_value=b"chart") as render:
            await j.send_explanation("why", 300, self.guild, self.logged, send)
            await j.send_explanation("why", 300, self.guild, self.logged, send)
        self.assertEqual(render.call_count, 2)
        self.assertEqual(len(self.uploads), 2)

    async def test_concurrent_requests_share_work_and_isolate_failures(self):
        for phase in ("render", "upload", "validation", "inline"):
            for outcome in ("success", "cancel", "failure"):
                with self.subTest(phase=phase, outcome=outcome):
                    j.explanation_cache.clear()
                    self.uploads.clear()
                    self.sent.clear()
                    self.available.reset_mock()
                    error = (RuntimeError("synthetic render failure") if phase == "render" else
                             discord.NotFound(Mock(status=404, reason="Not Found"), "synthetic expired interaction"))
                    response = ("Explanation", None, None) if phase == "inline" else (None, b"chart", "why-tea.png")
                    fails = outcome == "failure"
                    sender = AsyncMock(side_effect=error) if fails and phase != "render" else None
                    with patch.object(j, "why_response", return_value=response) as prepare:
                        if phase == "validation":
                            await j.send_explanation("why", 300, self.guild, self.logged, self.capture)
                            self.sent.clear()
                        if fails and phase == "render":
                            prepare.side_effect = [error, response]
                        first, second = await self.concurrent("upload" if phase == "inline" else phase,
                                                              send=sender, cancel=outcome == "cancel")
                        if fails:
                            self.assertIs(first, error)
                        elif outcome == "cancel":
                            self.assertIsInstance(first, asyncio.CancelledError)
                        else:
                            self.assertIsNone(first)
                        self.assertIsNone(second)
                        self.assertFalse(j.explanation_pending)
                        await j.send_explanation("why", 300, self.guild, self.logged, self.capture)
                    self.assertEqual(prepare.await_count, 1 + fails * (2 if phase == "validation" else 1))
                    self.assertEqual(len(self.uploads), 0 if phase == "inline" else 1 + (fails and phase == "validation"))
                    self.assertEqual(len(self.sent), 2 if fails else 3)
                    if phase == "inline":
                        self.assertTrue(all(s[0] == "Explanation" for s in self.sent))
                    else:
                        self.assertTrue(all(self.chart_bytes(s) == b"chart" for s in self.sent))
                        self.assertEqual(self.available.await_count, 2 if phase == "validation" else 1)

    async def test_a_concurrent_caller_with_rejected_embeds_gets_a_fresh_attachment(self):
        async def reject_embed(*args, **kwargs):
            if kwargs.get("embed"):
                raise discord.Forbidden(Mock(status=403, reason="Forbidden"), "synthetic embed rejection")
            return await self.capture(*args, **kwargs)

        for phase in ("upload", "validation"):
            with self.subTest(phase=phase):
                j.explanation_cache.clear()
                self.uploads.clear()
                self.sent.clear()
                with patch.object(j.why_chart, "render", return_value=b"chart") as render:
                    if phase == "validation":
                        await j.send_explanation("why", 300, self.guild, self.logged, self.capture)
                    self.assertEqual(await self.concurrent(phase, reject_embed), [None, None])
                    await j.send_explanation("why", 300, self.guild, self.logged, self.capture)
                self.assertEqual(render.call_count, 2)
                self.assertEqual(len(self.uploads), 2)
                self.assertEqual(self.sent[-1][3]["embed"].image.url, next(reversed(self.uploads)))


if __name__ == "__main__":
    unittest.main()
