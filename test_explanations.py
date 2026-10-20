"""Offline checks for diagnostic actions, using synthetic logs and Discord sends.

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
        files = kwargs.get("files") or ([file] if file is not None and file is not discord.utils.MISSING else [])
        data = None
        attachments = []
        for uploaded in files:
            uploaded_data = uploaded.fp.read()
            if data is None:
                file, data = uploaded, uploaded_data
            route = "ephemeral-attachments" if kwargs.get("ephemeral") else "attachments"
            expires = int(time.time()) + 3600
            url = (f"https://cdn.discordapp.com/{route}/300/{900 + len(self.uploads)}/{uploaded.filename}"
                   f"?ex={expires:x}&is={expires - 3600:x}&hm=synthetic-signature")
            self.uploads[url] = uploaded_data  # Discord's hosted files, outside the bot's cache
            attachments.append(SimpleNamespace(url=url, filename=uploaded.filename))
            uploaded.close()
            uploaded.fp.close()
        self.sent.append((content, file, data, kwargs))
        return SimpleNamespace(attachments=attachments)

    def embeds(self, sent=None):
        options = (sent or self.sent[-1])[3]
        return options.get("embeds") or [options["embed"]]

    def fields(self, sent=None):
        return {field.name: field.value for field in self.embeds(sent)[0].fields}

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
        json.dumps(vars(cached))  # cached metadata can be serialized; attachment bytes cannot
        self.assertEqual(cached.embeds, ())
        self.assertTrue(self.sent[1][3]["ephemeral"])
        self.assertEqual(self.sent[1][3]["allowed_mentions"].to_dict()["parse"], [])
        interaction.response.defer.assert_awaited_once_with(ephemeral=True, thinking=True)

    async def test_context_actions_send_inline_and_freshly_attached_transcripts(self):
        for text in ("Speaker: tea? @everyone ```\n\nTestbot: ", "Speaker: " + "tea " * 600):
            with self.subTest(attached=len(text) > j.CONTEXT_LIMIT):
                self.logged["transcript"] = text
                j.write_trace(self.logged)
                self.sent.clear()
                await j.context_action.callback(self.interaction(), self.target())
                await j.context_action.callback(self.interaction(), self.target())
                first, second = self.embeds(self.sent[0])[0], self.embeds(self.sent[1])[0]
                self.assertEqual(first.to_dict(), second.to_dict())
                if len(text) > j.CONTEXT_LIMIT:
                    # Uploaded each time: an attachment opens in Discord's viewer, a link to an old one doesn't
                    self.assertEqual([(s[1].filename, s[2]) for s in self.sent], [("context.txt", text.encode())] * 2)
                    self.assertIsNone(first.description)
                else:
                    self.assertIn(j.unfence(text), first.description)
                    self.assertEqual(self.uploads, {})
                self.assertEqual(j.explanation_cache, {})
                self.assertTrue(self.sent[1][3]["ephemeral"])
                self.assertEqual(self.sent[1][3]["allowed_mentions"].to_dict()["parse"], [])

    async def test_context_uses_exact_window_and_cache_counts_and_preserves_the_transcript(self):
        for long in (False, True):
            with self.subTest(long=long):
                t = dict(self.logged, llm="haiku", llm_window=1_000_000, llm_prompt_tokens=18_673,
                         llm_cached_tokens=14_533, llm_messages=291, llm_completion_tokens=24,
                         transcript="Speaker: " + "tea " * (600 if long else 1))
                j.write_trace(t)
                self.sent.clear()
                self.uploads.clear()
                await j.context_action.callback(self.interaction(), self.target(t))
                embed, = self.embeds(self.sent[0])
                self.assertEqual({f.name: f.value for f in embed.fields},
                                 {"Model": "Haiku", "Chat messages": "291", "Output": "24 tokens",
                                  "Context window": f"`▰{'▱' * 19}` 1.9% · 18,673 / 1,000,000 tokens",
                                  "Input cache": f"`{'▰' * 16}{'▱' * 4}` 77.8% · 14,533 cached · 4,140 uncached"})
                self.assertFalse(embed.image.url)
                uploads = {url.split("/")[-1].split("?")[0]: data for url, data in self.uploads.items()}
                if long:
                    self.assertEqual(uploads, {"context.txt": t["transcript"].encode()})
                else:
                    self.assertEqual(uploads, {})
                    self.assertIn(t["transcript"], embed.description)
                self.assertNotIn("estimated", json.dumps(embed.to_dict()))

    def test_text_bars_show_any_share_and_fill_only_when_full(self):
        self.assertEqual([j.text_bar(part, 1000, cells=4) for part in (0, 1, 500, 999, 1000)],
                         ["`▱▱▱▱`", "`▰▱▱▱`", "`▰▰▱▱`", "`▰▰▰▱`", "`▰▰▰▰`"])

    async def test_context_for_jev_or_an_older_log_has_a_transcript_without_made_up_usage(self):
        for t in (self.logged, dict(self.logged, llm="haiku")):
            with self.subTest(llm=t.get("llm")):
                j.write_trace(t)
                await j.context_action.callback(self.interaction(), self.target(t))
                embed, = self.embeds()
                self.assertEqual(embed.title, "What Testbot saw")
                self.assertIn(t["transcript"], embed.description)
                self.assertEqual(embed.fields[0].value, "Haiku" if t.get("llm") else "Jev")
                self.assertNotIn("Context window", [f.name for f in embed.fields])
                self.assertFalse(embed.image.url)

    async def test_stats_privately_show_only_model_total_cost_and_context_without_a_model_call(self):
        t = dict(self.logged, llm="haiku", llm_model="synthetic/previous-haiku", cost=.0016, byok_cost=.0015,
                 requests=2, seconds=1.2, llm_window=1_000_000, llm_prompt_tokens=9_800,
                 llm_cached_tokens=8_000, llm_messages=420, llm_completion_tokens=18,
                 transcript="Synthetic chat that Stats should not quote")
        j.write_trace(t)
        interaction = self.interaction()
        with patch.object(j, "model_name", "jev"), patch.object(j, "stats_response", wraps=j.stats_response) as stats, \
                patch.object(j, "llm", new_callable=AsyncMock) as llm, \
                patch.object(j, "post", new_callable=AsyncMock) as post:
            await j.stats_action.callback(interaction, self.target(t))
            await j.stats_action.callback(self.interaction(), self.target(t, original=True))
        self.assertEqual(self.fields(), {"Model": "synthetic/previous-haiku", "Total cost": "$0.001600 USD",
                                        "Context": "9,800 / 1,000,000 tokens · 1.0% used\n420 chat messages"})
        self.assertEqual(self.embeds(self.sent[0])[0].to_dict(), self.embeds(self.sent[1])[0].to_dict())
        self.assertNotIn(t["transcript"], str(self.fields()))
        self.assertEqual(len(self.uploads), 0)
        stats.assert_called_once()
        llm.assert_not_awaited()
        post.assert_not_awaited()
        interaction.response.defer.assert_awaited_once_with(ephemeral=True, thinking=True)
        self.assertTrue(self.sent[0][3]["ephemeral"])
        self.assertEqual(self.sent[0][3]["allowed_mentions"].to_dict()["parse"], [])

    async def test_stats_find_jev_replies_reactions_and_posted_statuses(self):
        with patch.object(j, "STATUS_CHANNEL", self.channel.id), patch.object(j, "model_name", "haiku"):
            for name, t, original in [
                    ("reply", dict(self.logged, cost=.02, requests=8), False),
                    ("reaction", dict(self.logged, reaction="👍", reply=None, cost=.00001, requests=2), True),
                    ("status", {"at": self.logged["at"], "status": "I feel tea is good", "status_id": 40,
                                "bot_name": "Testbot", "cost": .04008, "requests": 10}, False)]:
                with self.subTest(name=name):
                    j.write_trace(t)
                    await j.stats_action.callback(self.interaction(), self.target(t, original=original))
                    self.assertEqual(self.fields()["Model"], "Jev")
                    self.assertEqual(self.fields()["Total cost"], f"${t['cost']:.6f} USD")
                    self.assertTrue(self.fields()["Context"].startswith("Not recorded"))

    async def test_rocky_reacting_as_well_as_replying_is_still_its_turn(self):
        # !model rocky picks its own emoji: a reaction on its turn isn't Jev's reply-or-react check
        t = dict({k: v for k, v in self.logged.items() if k != "steps"},  # Jev's words: none in a turn of rocky's
                 llm="rocky", llm_model="synthetic/haiku", reply="Teal. Good colour.", reaction="🎨",
                 cost=.00115, requests=2, llm_window=1_000_000, llm_prompt_tokens=49_027, llm_cached_tokens=48_813,
                 llm_messages=474, transcript="testperson: remember my colour\nTestbot: Teal. Good colour.")
        j.write_trace(t)
        await j.stats_action.callback(self.interaction(), self.target(t))
        self.assertEqual(self.fields()["Model"], "synthetic/haiku")
        self.assertTrue(self.fields()["Context"].startswith("49,027 / 1,000,000 tokens"))
        self.assertIn(t["transcript"], j.logged_context(t))
        self.assertFalse(j.jev_reacted(t))
        self.assertTrue(j.jev_reacted(dict(self.logged, reaction="👍")))
        content, png, _ = await j.why_response(t, self.guild, "Testbot")
        self.assertIn("doesn't say how likely its words were", content)  # about its message, not a Jev reaction
        only_reacted = {k: v for k, v in t.items() if k != "reply"}
        content, png, _ = await j.why_response(only_reacted, self.guild, "Testbot")
        self.assertIn("rocky picked it itself", content)

    async def test_stats_for_an_llm_status_show_accepted_usage_and_total_retry_cost(self):
        t = {"at": self.logged["at"], "status": "I feel tea is good", "status_id": 40, "bot_name": "Testbot",
             "llm": "haiku", "llm_model": "synthetic/haiku", "cost": .00108, "byok_cost": .001,
             "requests": 4, "llm_window": 1_000_000, "llm_prompt_tokens": 3_000,
             "status_attempts": [{"accepted": False, "llm_prompt_tokens": 2_000},
                                 {"accepted": True, "llm_prompt_tokens": 3_000}]}
        with patch.object(j, "STATUS_CHANNEL", self.channel.id):
            j.write_trace(t)
            await j.stats_action.callback(self.interaction(), self.target(t))
        self.assertEqual(self.fields()["Total cost"], "$0.001080 USD")
        self.assertEqual(self.fields()["Context"], "3,000 / 1,000,000 tokens · 0.3% used")

    async def test_stats_handle_missing_usage_zero_cost_and_out_of_credit(self):
        for fields, model, cost in [
                ({}, "Jev", "Not recorded"),
                ({"llm": "haiku"}, "Haiku", "Not recorded"),
                ({"llm": "kimi", "cost": 0, "llm_completion_tokens": 0}, "Kimi", "$0.000000 USD"),
                ({"reply": j.NO_MONEY[0], "cost": 0, "requests": 0}, "None (out of credit)", "$0.000000 USD"),
                ({"llm": "haiku", "reply": j.NO_MONEY[0], "cost": 0, "requests": 0},
                 "None (out of credit)", "$0.000000 USD")]:
            with self.subTest(model=model):
                t = dict(self.logged, **fields)
                j.write_trace(t)
                await j.stats_action.callback(self.interaction(), self.target(t))
                self.assertEqual(self.fields()["Model"], model)
                self.assertEqual(self.fields()["Total cost"], cost)
                self.assertTrue(self.fields()["Context"].startswith("No generation" if model.startswith("None")
                                                                    else "Not recorded"))

    async def test_stats_handle_zero_input_and_missing_window_in_partial_logs(self):
        for tokens, window, expected in [(100, 262_144, "100 / 262,144 tokens · <0.1% used"),
                                        (0, 262_144, "0 / 262,144 tokens · 0.0% used"),
                                        (0, None, "0 tokens")]:
            with self.subTest(tokens=tokens, window=window):
                t = dict(self.logged, llm="kimi", llm_window=window, llm_prompt_tokens=tokens)
                j.write_trace(t)
                await j.stats_action.callback(self.interaction(), self.target(t))
                self.assertTrue(self.fields()["Context"].startswith(expected))

    async def test_diagnostics_explain_request_purposes_routing_and_billing_privately(self):
        t = dict(self.logged, llm="haiku", cost=.000658, byok_cost=.000577, requests=2, seconds=2.5,
                 llm_cached_tokens=14_533, llm_completion_tokens=24, api_calls=[
                     {"model": j.MODEL, "purpose": "decision", "status": 200, "attempt": 0,
                      "router_cost": .000081, "is_byok": False, "key_cost": 0},
                     {"model": j.LLMS["haiku"]["id"], "purpose": "reply", "status": 200, "attempt": 0,
                      "router_cost": 0, "is_byok": True, "key_cost": .000577}])
        j.write_trace(t)
        with patch.object(j, "llm", new_callable=AsyncMock) as llm:
            await j.diagnostics_action.callback(self.interaction(), self.target(t))
            await j.diagnostics_action.callback(self.interaction(), self.target(t, original=True))
        fields = self.fields()
        self.assertEqual(fields["Requests"], "2")
        self.assertEqual(fields["Total time"], "2.5s")
        self.assertIn("Jev · 1 request · paid through OpenRouter", fields["Check whether to reply or react"])
        self.assertIn("$0.000081 USD", fields["Check whether to reply or react"])
        self.assertIn("Haiku · 1 request · paid through Anthropic key", fields["Write the reply"])
        self.assertIn("$0.000577 USD", fields["Write the reply"])
        self.assertEqual(fields["Final generation"], "14,533 cached input tokens · 24 output tokens")  # no input count
        self.assertIn("Routed through OpenRouter", self.embeds()[0].footer.text)
        self.assertEqual(self.embeds(self.sent[0])[0].to_dict(), self.embeds(self.sent[1])[0].to_dict())
        self.assertTrue(self.sent[-1][3]["ephemeral"])
        llm.assert_not_awaited()

    async def test_diagnostics_show_the_final_generations_input_with_its_cached_part(self):
        for cached, expected in [(0, "71,141 input tokens (0 cached) · 70 output tokens"),
                                 (70_998, "71,141 input tokens (70,998 cached) · 70 output tokens")]:
            with self.subTest(cached=cached):
                t = dict(self.logged, llm="rocky", cost=.007, llm_prompt_tokens=71_141, llm_cached_tokens=cached,
                         llm_completion_tokens=70)
                j.write_trace(t)
                await j.diagnostics_action.callback(self.interaction(), self.target(t))
                self.assertEqual(self.fields()["Final generation"], expected)

    async def test_diagnostics_handle_old_logs_without_inventing_request_purposes(self):
        t = dict(self.logged, llm="haiku", cost=.001, byok_cost=.0009, requests=2)
        j.write_trace(t)
        await j.diagnostics_action.callback(self.interaction(), self.target(t))
        self.assertEqual(self.fields()["Requests"], "2")
        self.assertEqual(self.fields()["Model ID"], "Not recorded")
        self.assertIn("OpenRouter: $0.000100 USD", self.fields()["Billing"])
        self.assertNotIn("Write the reply", self.fields())
        self.assertIn("not recorded", self.embeds()[0].footer.text)

    async def test_diagnostics_count_failed_attempts_and_do_not_price_unknown_failures_as_free(self):
        t = dict(self.logged, llm="haiku", cost=.001, requests=1, api_calls=[
            {"model": j.LLMS["haiku"]["id"], "purpose": "reply", "status": 429, "attempt": 0,
             "router_cost": None, "is_byok": False, "key_cost": 0},
            {"model": j.LLMS["haiku"]["id"], "purpose": "reply", "status": 200, "attempt": 1,
             "router_cost": .001, "is_byok": False, "key_cost": 0}])
        j.write_trace(t)
        await j.diagnostics_action.callback(self.interaction(), self.target(t))
        self.assertEqual(self.fields()["Requests"], "2")
        self.assertIn("1 failed attempt", self.fields()["Write the reply"])
        self.assertIn("$0.001000 USD reported", self.fields()["Write the reply"])
        self.assertIn("1 HTTP retries", self.fields()["Retries"])
        self.assertNotIn("Stopped early", self.fields()["Write the reply"])  # older logs have no endings

    async def test_diagnostics_show_refusals_and_other_early_stops(self):
        call = {"model": j.LLMS["haiku"]["id"], "purpose": "reply", "status": 200, "router_cost": 0,
                "is_byok": False, "key_cost": 0, "id": "gen-synthetic", "provider": "Anthropic"}
        t = dict(self.logged, llm="haiku", cost=0, requests=4, api_calls=[
            call | {"attempt": 0, "finish_reason": "content_filter", "native_finish_reason": "refusal"},
            call | {"attempt": 0, "finish_reason": "content_filter", "native_finish_reason": "refusal"},
            call | {"attempt": 0, "finish_reason": "length", "native_finish_reason": "max_tokens"},
            call | {"attempt": 0, "finish_reason": "stop", "native_finish_reason": "end_turn"}])
        j.write_trace(t)
        await j.diagnostics_action.callback(self.interaction(), self.target(t))
        self.assertIn("Stopped early: refusal ×2, max_tokens", self.fields()["Write the reply"])

    async def test_stats_require_a_matching_target_in_the_same_channel_and_existing_logs(self):
        j.write_trace(dict(self.logged, cost=.02))
        await j.stats_action.callback(self.interaction(), self.target())
        edited = self.target()
        edited.content = "Changed reply"
        await j.stats_action.callback(self.interaction(), edited)
        wrong_id = self.target()
        wrong_id.id = 99
        await j.stats_action.callback(self.interaction(), wrong_id)
        elsewhere = self.interaction()
        elsewhere.channel_id = 301
        await j.stats_action.callback(elsewhere, self.target())
        next(j.LOG_DIR.glob("*.jsonl")).unlink()
        await j.stats_action.callback(self.interaction(), self.target())
        self.assertTrue(all(s[0] == "Nothing logged for that" for s in self.sent[1:]))

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
        self.assertIn("tea?", self.embeds(self.sent[1])[0].description)
        self.assertIn("coffee?", self.embeds(self.sent[3])[0].description)

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

    async def test_why_command_and_actions_find_a_converted_user_mention(self):
        logged = dict(self.logged, reply="@Moss knows", sent="<@100000000000000002> knows")
        j.write_trace(logged)
        target = self.target(logged)
        target.content = logged["sent"]
        target.clean_content = logged["reply"]
        with patch.object(j.why_chart, "render", return_value=b"mention chart") as render:
            await j.why(self.command(target))
            await j.why_action.callback(self.interaction(), target)
            await j.context_action.callback(self.interaction(), target)
        render.assert_called_once()
        self.assertEqual(self.chart_bytes(self.sent[0]), b"mention chart")
        self.assertTrue(self.sent[1][3]["ephemeral"])
        self.assertIn("What Testbot saw", self.embeds(self.sent[2])[0].title)
        self.assertEqual(j.find_trace(self.channel.id, target), logged)
        target.content = "<@100000000000000002> edited"
        self.assertIsNone(j.find_trace(self.channel.id, target))  # the new sent field still detects edits

    async def test_legacy_converted_mentions_match_readable_text_but_require_the_same_id(self):
        logged = dict(self.logged, reply="@Moss knows")
        j.write_trace(logged)
        target = self.target(logged)
        target.content = "<@100000000000000002> knows"
        target.clean_content = "@Moss knows"
        self.assertEqual(j.find_trace(self.channel.id, target), logged)
        target.id += 1
        self.assertIsNone(j.find_trace(self.channel.id, target))

    async def test_reply_and_status_charts_are_cached(self):
        status = {"at": self.logged["at"], "status": "I feel tea is good", "status_id": 40,
                  "bot_name": "Testbot", "transcript": "Testbot's diary, today: I feel",
                  "steps": [{"word": "tea", "top": [["tea", .8, .8]]}]}
        with patch.object(j, "STATUS_CHANNEL", self.channel.id), \
                patch.object(j.why_chart, "render", wraps=j.why_chart.render) as render:
            j.write_trace(self.logged)
            await j.why(self.command())
            await self.explain(self.logged)
            j.write_trace(status)
            await j.why(self.command())
            await j.why(self.command())
            await j.context_action.callback(self.interaction(), self.target(status))
            await j.context_action.callback(self.interaction(), self.target(status))
        self.assertEqual(render.call_count, 2)
        self.assertTrue(all(self.chart_bytes(s).startswith(b"\x89PNG") for s in self.sent[:4]))
        self.assertEqual(self.embeds(self.sent[4])[0].to_dict(), self.embeds(self.sent[5])[0].to_dict())
        self.assertIn(status["transcript"], self.embeds(self.sent[4])[0].description)

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
                                 ({"llm": "haiku"}, "haiku doesn't say how likely its words were")]:
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
