"""Offline behavioral checks for mention routing, history, and startup catch-up.

    uv run --with-requirements requirements.txt python -m unittest test_mentions
"""

import asyncio
import contextlib
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from fakes import answer, isolate_rocky, patch_all, traced
import discord
import jev_bot as j
import rocky_harness as rh


class _Message(SimpleNamespace):
    # Exercise discord.py's real formatter rather than mocking its output.
    clean_content = property(discord.Message.clean_content.function)
    raw_mentions = property(discord.Message.raw_mentions.function)


class _Discord(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.reactors_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.reactors_dir.cleanup)
        self.user = SimpleNamespace(id=100000000000000001, bot=True, display_name="Testbot")
        self.role = SimpleNamespace(id=200000000000000001, mention="<@&200000000000000001>", name="Testbot")
        self.guild = SimpleNamespace(id=400, self_role=self.role, me=self.user, threads=[],
                                     get_member=lambda _: None, get_role=lambda _: None,
                                     _resolve_channel=lambda _: None)
        self.channel = Mock(id=300)
        self.channel.permissions_for.return_value = SimpleNamespace(read_messages=True, read_message_history=True)
        self.guild.text_channels = [self.channel]
        self.target = Mock(spec=discord.Message, id=700, author=self.user, reactions=[])
        self.target.reference = None
        self.target.created_at = datetime.now(timezone.utc) - timedelta(minutes=1)
        patch_all(self, [
            (j.bot._connection, "user", self.user),
            (j.bot._connection, "_guilds", {400: self.guild}),
            (j, "DM_USERS", {500}),
            (j, "channel_history", defaultdict(list)),
            (j, "heard", False),
            (j, "side_talk", {}),
            (j, "side_of", {}),
            (j, "reactor_names", {}),
            (j, "reactor_cache", {}),
            (j, "reactors_save", None),
            (j, "REACTORS_PATH", Path(self.reactors_dir.name) / "reactors.json"),
        ])
        isolate_rocky(self, self.reactors_dir.name)  # no checkout's logs, and none left behind


class ReactionContextTests(_Discord):
    def setUp(self):
        super().setUp()
        self.entry = {"role": "user", "name": "Speaker", "content": "hello", "id": 600,
                      "reply": "Hello", "reply_id": 700}
        j.channel_history[self.channel.id] = [self.entry]

    def payload(self, uid=501, name="Moss", emoji="😂"):
        return SimpleNamespace(channel_id=self.channel.id, message_id=700, guild_id=400,
                               user_id=uid, member=Mock(spec=discord.Member, id=uid, display_name=name, guild=self.guild), emoji=emoji)

    def transcript(self, **kwargs):
        return j.transcript("how are you?", "Speaker", "Testbot", [self.entry], [], **kwargs)

    def reaction(self, emoji="😂", fail=False):
        async def users():
            if fail:
                raise discord.HTTPException(SimpleNamespace(status=403, reason="Forbidden"), "denied")
            for user in [self.user, SimpleNamespace(id=501, display_name="Moss"),
                         SimpleNamespace(id=502, display_name="Pip")]:
                yield user
        return SimpleNamespace(emoji=emoji, count=3, me=True, users=users)

    async def test_live_names_removal_and_question_check(self):
        await j.on_raw_reaction_add(self.payload())
        await j.on_raw_reaction_add(self.payload(502, "Pip"))
        await j.on_raw_reaction_add(self.payload(self.user.id, "Testbot"))
        self.assertIn("Hello (😂 by @Moss, @Pip)", self.transcript())
        self.assertNotIn("😂", self.transcript(reactions=False, marked=True))
        self.assertNotIn("@Moss", self.transcript(their_reactions=False))
        await j.on_raw_reaction_remove(self.payload())
        self.assertIn("Hello (😂 by @Pip)", self.transcript())
        await j.on_raw_reaction_remove(self.payload(502, "Pip"))
        self.assertNotIn("😂", self.transcript())

    async def test_clear_one_and_all_reactions(self):
        await j.on_raw_reaction_add(self.payload())
        await j.on_raw_reaction_add(self.payload(502, "Pip", "💀"))
        await j.on_raw_reaction_clear_emoji(self.payload())
        self.assertNotIn("@Moss", self.transcript())
        self.assertIn("💀 by @Pip", self.transcript())
        await j.on_raw_reaction_clear(self.payload())
        self.assertNotIn("reply_reactors", self.entry)
        self.assertNotIn("💀", self.transcript())

    async def test_names_restored_from_channel_history(self):
        question = MentionTests.message(self)
        question.id = 600
        reply = _Message(id=700, author=self.user, reference=SimpleNamespace(message_id=600), guild=None,
                         content="Hello", reactions=[self.reaction()], created_at=question.created_at,
                         mentions=[], role_mentions=[], embeds=[], attachments=[], stickers=[])
        async def history(**kwargs):
            for message in [reply, question]:
                yield message
        self.channel.history = history
        j.channel_history[self.channel.id] = []
        await j.load_history(SimpleNamespace(channel=self.channel))
        self.entry = j.channel_history[self.channel.id][0]
        self.assertIn("Hello (😂 by @Moss, @Pip)", self.transcript())

    async def test_side_chain_and_standalone_reply_names(self):
        question = MentionTests.message(self)
        reply = _Message(id=700, author=self.user, reference=SimpleNamespace(message_id=question.id), guild=None,
                         content="Hello", reactions=[self.reaction()], created_at=question.created_at,
                         mentions=[], role_mentions=[], embeds=[], attachments=[], stickers=[])
        entries = await j.chain_entries([question, reply])
        self.assertEqual(entries[0]["reply_reactors"]["😂"], {"501": "Moss", "502": "Pip"})
        standalone = {"role": "assistant", "content": "Hello"}
        await j.attach_reactions(standalone, reply, prefix="")
        self.assertIn("Hello (😂 by @Moss, @Pip)",
                      j.transcript("hi", "Speaker", "Testbot", [standalone], []))

    async def test_http_failure_keeps_counts_and_old_logs_render(self):
        reply = SimpleNamespace(id=700, reactions=[self.reaction(fail=True)])
        await j.attach_reactions(self.entry, reply)
        self.assertIn("Hello (😂×2)", self.transcript())
        self.assertNotIn("700", j.reactor_cache)  # asked again next time
        self.assertEqual(j.reacted({"😂": 3}), " (😂×3)")
        p = self.payload()
        p.member, p.guild_id = None, None
        with patch.object(j.bot, "get_user", return_value=None), patch.object(
            j.bot, "fetch_user", new_callable=AsyncMock, return_value=SimpleNamespace(display_name="Moss")
        ):
            await j.on_raw_reaction_add(p)
        self.assertIn("😂×3 by @Moss", self.transcript())

    async def test_a_rebuild_reuses_saved_names_without_asking_discord(self):
        j.reactor_cache["700"] = {"counts": {"😂": 2}, "reactors": {"😂": {"501": "Moss", "502": "Pip"}}}
        unasked = SimpleNamespace(emoji="😂", count=3, me=True, users=Mock(side_effect=AssertionError("asked Discord")))
        await j.attach_reactions(self.entry, SimpleNamespace(id=700, reactions=[unasked]))
        self.assertIn("Hello (😂 by @Moss, @Pip)", self.transcript())

    async def test_only_an_emoji_whose_count_changed_is_asked_about(self):
        j.reactor_cache["700"] = {"counts": {"😂": 2, "💀": 1}, "reactors": {"😂": {"501": "Moss", "502": "Pip"},
                                                                           "💀": {"503": "Fern"}}}
        unasked = SimpleNamespace(emoji="😂", count=3, me=True, users=Mock(side_effect=AssertionError("asked Discord")))
        await j.attach_reactions(self.entry, SimpleNamespace(id=700, reactions=[unasked, self.reaction("💀")]))
        self.assertEqual(self.entry["reply_reactors"], {"😂": {"501": "Moss", "502": "Pip"},
                                                        "💀": {"501": "Moss", "502": "Pip"}})
        self.assertEqual(j.reactor_cache["700"]["counts"], {"😂": 2, "💀": 2})

    async def test_live_reactions_keep_the_saved_names_current(self):
        await j.on_raw_reaction_add(self.payload())
        await j.on_raw_reaction_add(self.payload(502, "Pip", "💀"))
        await j.on_raw_reaction_remove(self.payload())
        self.assertEqual(j.reactor_cache["700"], {"counts": {"💀": 1}, "reactors": {"💀": {"502": "Pip"}}})
        self.assertIsNotNone(j.reactors_save)  # written a moment later, once
        j.save_reactors()
        self.assertEqual(j.load_reactors(), j.reactor_cache)
        await j.on_raw_reaction_clear(self.payload())
        self.assertNotIn("700", j.reactor_cache)

    async def test_history_names_are_server_nicknames(self):
        # Reaction users come back as plain users with global names; the member lookup gives the nickname,
        # once per person however many replies they reacted to.
        fetch = AsyncMock(side_effect=lambda uid: SimpleNamespace(display_name={501: "Mossy", 502: "Pipsqueak"}[uid]))
        guild = SimpleNamespace(id=400, get_member=lambda _: None, fetch_member=fetch)
        replies = [SimpleNamespace(id=700 + i, guild=guild, reactions=[self.reaction()]) for i in range(2)]
        entries = [{"reply": "Hello"} for _ in replies]
        await asyncio.gather(*(j.attach_reactions(e, r) for e, r in zip(entries, replies)))
        self.assertEqual(entries[1]["reply_reactors"]["😂"], {"501": "Mossy", "502": "Pipsqueak"})
        self.assertEqual(fetch.await_count, 2)

    async def test_jev_gets_counts_and_llms_get_names(self):
        self.entry.update(reply_reactions={"😂": 2}, reply_reactors={"😂": {"501": "Moss", "502": "Pip"}})
        seen = []
        async def loom(state, *args, **kwargs):
            seen.append(state([]))
            return ["Hi"]
        async def llm(name, message, author, bot_name, history, *args):
            return j.transcript(message, author, bot_name, history, [])
        with patch.object(j, "model_name", "jev"), patch.object(j, "loom", loom):
            await j.generate_reply("how are you?", "Speaker", "Testbot", [self.entry])
        self.assertIn("Hello (😂×2)", seen[0])
        with patch.object(j, "model_name", "haiku"), patch.object(j, "llm_reply", llm):
            text = await j.generate_reply("how are you?", "Speaker", "Testbot", [self.entry])
        self.assertIn("Hello (😂 by @Moss, @Pip)", text)

    def test_count_shown_only_when_names_are_missing(self):
        self.assertEqual(j.reacted({"😂": 2, "💀": 3}, {"😂": {"1": "Moss", "2": "Pip"}, "💀": {"3": "Fern"}}),
                         " (😂 by @Moss, @Pip; 💀×3 by @Fern)")
        self.assertEqual(j.reacted({"😂": 3, "💀": 1}), " (😂×3 💀)")  # what Jev gets, as before names


class MentionTests(_Discord):
    def message(self, *, content="hello", ping=False, role=False, reply=True, dm=False, author_bot=False):
        return _Message(
            id=600, content=content, guild=None if dm else self.guild, channel=self.channel,
            author=SimpleNamespace(id=500, bot=author_bot, display_name="Speaker"),
            mentions=[self.user] if ping else [], role_mentions=[self.role] if role else [],
            reference=SimpleNamespace(message_id=self.target.id, resolved=self.target) if reply else None,
            created_at=datetime.now(timezone.utc), reactions=[], attachments=[], stickers=[], embeds=[],
        )

    async def test_mentions_stay_in_place_in_reply_and_reaction_context(self):
        for mention, options in (("<@100000000000000001>", {"ping": True}), ("<@!100000000000000001>", {"ping": True}),
                                 ("<@&200000000000000001>", {"role": True})):
            with self.subTest(mention=mention):
                m = self.message(content=f"think {mention} is drunk?", **options)
                text = j.message_text(m)
                self.assertEqual(text, "think @Testbot is drunk?")
                entry = j.history_entry(m)
                for marked in (False, True):
                    state = j.transcript(text, "Speaker", "Testbot", [entry], [], marked=marked)
                    self.assertEqual(state, "Speaker: think @Testbot is drunk?\n"
                                           "Speaker: think @Testbot is drunk?\nTestbot: ")

    async def test_other_mentions_and_possessives_are_preserved(self):
        m = self.message(content="<@100000000000000001>'s friend <@!100000000000000002> likes <@&200000000000000002>", ping=True)
        m.mentions.append(SimpleNamespace(id=100000000000000002, display_name="Friend"))
        m.role_mentions.append(SimpleNamespace(id=200000000000000002, name="Readers"))
        self.assertEqual(j.message_text(m), "@Testbot's friend @Friend likes @Readers")

    async def test_question_check_still_marks_pinged_reply_without_textual_mention(self):
        m = self.message(content="is drunk?", ping=True)
        text = j.message_text(m)
        self.assertEqual(text, "is drunk?")
        self.assertEqual(j.transcript(text, "Speaker", "Testbot", [], [], marked=True),
                         "Speaker: @Testbot is drunk?\nTestbot: ")

    async def test_commands_still_ignore_bot_mentions(self):
        for mention in ("<@100000000000000001>", "<@!100000000000000001>", self.role.mention):
            m = self.message(content=f"{mention} !why", ping=True)
            self.assertEqual(j.command(m), "!why")
            m.content = f"{mention}'s !why"
            self.assertIsNone(j.command(m))

    async def test_discord_formatter_handles_channels_and_deleted_mentions(self):
        self.guild._resolve_channel = lambda id: SimpleNamespace(name="general") if id == 300000000000000001 else None
        m = self.message(content="<@100000000000000001> see <#300000000000000001>", ping=True)
        self.assertEqual(j.message_text(m), "@Testbot see #general")
        m.content = "<@100000000000000099> <@&200000000000000099> <#300000000000000099>"
        self.assertEqual(j.message_text(m), "@deleted-user @deleted-role #deleted-channel")
        m.content = "@everyone @here"
        self.assertEqual(j.message_text(m), "@\u200beveryone @\u200bhere")

    async def test_dm_mentions_keep_readable_names(self):
        m = self.message(content="<@100000000000000001>'s friend", ping=True, dm=True)
        self.assertEqual(j.message_text(m), "@Testbot's friend")

    async def test_unpinged_reply_is_chatter_without_a_response(self):
        m = self.message()
        with patch.object(j, "respond", new_callable=AsyncMock) as respond:
            await j.on_message(m)
        respond.assert_not_awaited()
        entry, = j.channel_history[self.channel.id]
        self.assertFalse(entry["to_bot"])
        self.assertEqual(entry["content"], "hello")

    async def test_mentions_trigger_responses_with_or_without_reply(self):
        for options in ({"ping": True}, {"ping": True, "reply": False},
                        {"role": True}, {"role": True, "reply": False}):
            with self.subTest(options=options):
                m = self.message(**options)
                with patch.object(j, "respond", new_callable=AsyncMock) as respond:
                    await j.on_message(m)
                respond.assert_awaited_once_with(m)

    async def test_allowed_dm_needs_no_mention(self):
        m = self.message(dm=True, reply=False)
        with patch.object(j, "respond", new_callable=AsyncMock) as respond:
            await j.on_message(m)
        respond.assert_awaited_once_with(m)

    async def test_other_bots_cannot_trigger_responses(self):
        m = self.message(ping=True, author_bot=True)
        with patch.object(j, "respond", new_callable=AsyncMock) as respond:
            await j.on_message(m)
        respond.assert_not_awaited()
        entry, = j.channel_history[self.channel.id]
        self.assertEqual(entry["content"], "hello")
        self.assertFalse(entry["to_bot"])
        self.assertFalse(j.heard)

    async def test_own_posts_stay_out_of_history(self):
        m = self.message(author_bot=True, reply=False)
        m.author = self.user
        with patch.object(j, "respond", new_callable=AsyncMock) as respond:
            await j.on_message(m)
        respond.assert_not_awaited()
        self.assertFalse(j.channel_history[self.channel.id])
        self.assertFalse(j.heard)

    async def test_catch_up_does_not_answer_other_apps(self):
        m = self.message(ping=True, author_bot=True, reply=False)

        async def history(**kwargs):
            yield m

        self.channel.history.side_effect = history
        with patch.object(j, "DM_USERS", set()), patch.object(j, "respond", new_callable=AsyncMock) as respond:
            await j.catch_up()
        respond.assert_not_awaited()

    async def test_unpinged_diagnostic_reply_still_runs_command(self):
        m = self.message(content="!why")
        with patch.object(j, "why", new_callable=AsyncMock) as why, \
                patch.object(j, "respond", new_callable=AsyncMock) as respond:
            await j.on_message(m)
        why.assert_awaited_once_with(m)
        respond.assert_not_awaited()

    async def test_catch_up_skips_unpinged_reply_and_answers_pinged_reply(self):
        for ping in (False, True):
            with self.subTest(ping=ping):
                m = self.message(ping=ping)

                async def history(**kwargs):
                    yield m
                    yield self.target

                self.channel.history.side_effect = history
                with patch.object(j, "DM_USERS", set()), \
                        patch.object(j, "respond", new_callable=AsyncMock) as respond:
                    await j.catch_up()
                if ping:
                    respond.assert_awaited_once_with(m, caught_up=True)
                else:
                    respond.assert_not_awaited()


    async def test_catch_up_skips_message_already_being_answered(self):
        # Just after a restart: on_message is answering m when catch_up() finds it, its reply not sent yet
        m = self.message(ping=True, reply=False)
        started, finish = asyncio.Event(), asyncio.Event()

        async def answering(m, **extra):
            started.set()
            await finish.wait()

        async def history(**kwargs):
            yield m

        self.channel.history.side_effect = history
        with patch.object(j, "DM_USERS", set()), patch.object(j, "taken", {}), \
                patch.object(j, "respond_traced", side_effect=answering) as traced:
            live = asyncio.create_task(j.on_message(m))
            await started.wait()
            await asyncio.wait_for(j.catch_up(), 1)  # a second answer would wait on finish too
            finish.set()
            await live
            await j.respond(m, caught_up=True)  # and once it's answered, too
        traced.assert_awaited_once_with(m)


class EmbedTests(_Discord):
    message = MentionTests.message

    def test_standalone_embed_includes_title_and_description(self):
        m = self.message(content="", author_bot=True, reply=False)
        m.embeds = [discord.Embed(url="https://example.com/milestone", title="Contribution milestone", description=(
            "[synthetic-user](https://example.com/synthetic-user) has passed **1,000 contributions** in 2026."))]
        self.assertEqual(j.message_text(m),
                         "[embed: Contribution milestone synthetic-user has passed 1,000 contributions in 2026.]")

    def test_multiline_description_and_fields_share_one_word_limit(self):
        m = self.message(content="", reply=False)
        embed = discord.Embed(description="Build finished\nAll checks passed")
        embed.set_author(name="Build App")
        embed.add_field(name="Result", value=" ".join(f"word{i}" for i in range(30)))
        m.embeds = [embed]
        expected = "Build App Build finished All checks passed Result: " + " ".join(f"word{i}" for i in range(12))
        self.assertEqual(j.message_text(m), f"[embed: {expected} ...]")

    def test_link_preview_is_not_repeated_as_a_standalone_embed(self):
        for url in ("https://example.com/post", "https://example.com/redirect"):
            with self.subTest(url=url):
                m = self.message(content=url, reply=False)
                m.embeds = [discord.Embed(url="https://example.com/post", title="Release notes", description="Preview text")]
                self.assertEqual(j.message_text(m), "[link: Release notes]")

    def test_multiple_links_and_extra_embed(self):
        m = self.message(content="https://example.com/first https://example.com/second", reply=False)
        m.embeds = [discord.Embed(url="https://example.com/first", title="First article"),
                    discord.Embed(url="https://example.com/second", title="Second article"),
                    discord.Embed(title="Build", description="All checks passed")]
        self.assertEqual(j.message_text(m), "[link: First article] [link: Second article] [embed: Build All checks passed]")

    def test_gif_tweet_attachments_and_stickers_keep_their_tags(self):
        m = self.message(content="https://tenor.com/view/happy-dance-gif https://example.com/post", reply=False)
        gif = discord.Embed(url="https://tenor.com/view/happy-dance-gif", type="gifv", title="Happy dance")
        tweet = discord.Embed(url="https://example.com/post", description="**Good news**\nOther text")
        tweet.set_author(name="Synthetic User (@synthetic)")
        m.embeds = [gif, tweet, discord.Embed().set_image(url="https://example.com/image.png")]
        m.attachments = [SimpleNamespace(content_type="image/png")]
        m.stickers = [SimpleNamespace(name="Wave")]
        self.assertEqual(j.message_text(m),
                         "[gif: Happy dance] [link: Synthetic User: Good news] [photo] [sticker: Wave]")


class ActionTests(_Discord):
    """Diagnostic actions, from right-clicking a message: answered only to whoever asked."""

    def test_actions_are_message_context_menus(self):
        menus = {c.name: c.type for c in j.bot.tree.get_commands(type=discord.AppCommandType.message)}
        self.assertEqual(menus, {"Why": discord.AppCommandType.message, "Context": discord.AppCommandType.message,
                                 "Stats": discord.AppCommandType.message, "Diagnostics": discord.AppCommandType.message})

    async def test_failed_registration_doesnt_stop_startup(self):
        error = discord.HTTPException(Mock(status=503, reason="Service Unavailable"), "unavailable")
        with patch.object(j.bot.tree, "sync", AsyncMock(side_effect=error)), self.assertLogs(j.log, "WARNING") as logs:
            await j.bot.setup_hook()
        self.assertIn("Registering app commands failed", logs.output[0])


class WakingTests(_Discord):
    """Discord shows a bot online as soon as it connects; jev stays idle until it has caught up."""

    async def test_idle_until_caught_up_then_online_with_its_status(self):
        presence = AsyncMock()
        seen_while_catching_up = []

        async def catch_up():
            await j.set_credit(True)  # credit coming back mid-catch-up mustn't put it online early
            seen_while_catching_up.extend(presence.call_args_list)

        patch_all(self, [(j.bot, "change_presence", presence), (j.bot, "is_ready", lambda: True)])
        with patch.object(j, "has_credit", None), patch.object(j, "credit_left", AsyncMock(return_value=5.0)), \
                patch.object(j, "catch_up", catch_up), patch.object(j, "STATUS_EVERY", 0), \
                patch.object(j, "status", discord.CustomActivity("I feel tea")), patch.object(j, "waking", True):
            self.assertEqual((j.bot.status, j.bot.activity), (discord.Status.idle, j.WAKING))  # from the first connect
            await j.on_ready()
            self.assertEqual([c.kwargs["status"] for c in seen_while_catching_up], [discord.Status.idle])
            self.assertEqual(presence.call_args.kwargs, {"status": discord.Status.online, "activity": j.status})
            self.assertFalse(j.waking)


class _Handling(_Discord):
    """handle() end to end, with what jev would say patched in."""

    def setUp(self):
        super().setUp()
        self.load_history = j.load_history
        self.trace = traced(self)
        self.channel.typing = lambda: contextlib.nullcontext()
        self.channel.fetch_message = AsyncMock(side_effect=lambda i: self.by_id[i])
        self.by_id = {}
        self.next_id = 800
        patch_all(self, [(j, "has_credit", True), (j, "history_loaded", {}), (j, "emoji_vocabulary", lambda g: {}),
                         (j, "choose_reaction", AsyncMock(return_value=None)), (j, "load_history", AsyncMock())])
        # Chatter before the conversation, which a !nocontext chain shouldn't see
        j.channel_history[self.channel.id] = [
            {"role": "user", "name": "kettle", "content": "the toaster is on fire again", "id": 1,
             "at": datetime.now(timezone.utc) - timedelta(minutes=10), "to_bot": False}]

    def said(self, content, *, to=None, bot=False, name="pip", ping=True):
        m = Mock(spec=discord.Message, id=self.next_id, content=(f"<@{self.user.id}> " if ping and not bot else "") + content,
                 guild=self.guild, channel=self.channel, mentions=[self.user] if ping and not bot else [],
                 role_mentions=[], created_at=datetime.now(timezone.utc), reactions=[], attachments=[], stickers=[],
                 embeds=[], author=self.user if bot else SimpleNamespace(id=500, bot=False, display_name=name))
        m.clean_content = discord.Message.clean_content.function(m)
        m.raw_mentions = discord.Message.raw_mentions.function(m)
        # Discord only resolves the message m replies to, not the ones above it
        m.reference = to and SimpleNamespace(message_id=to.id, resolved=None)
        self.by_id[m.id] = m
        self.next_id += 1
        m.reply = AsyncMock(side_effect=lambda content, **kwargs: self.discord_reply(m, content))
        return m

    # The history jev had in view answering m, as (name, content, jev's reply) — answering with `says`
    async def answer(self, m, says="Geology good"):
        if m.reference:
            m.reference.resolved = self.by_id[m.reference.message_id]
        m.reply = AsyncMock(side_effect=lambda content, **kwargs: self.discord_reply(m, content))
        with patch.object(j, "generate_reply", new_callable=AsyncMock, return_value=says) as gen:
            await j.handle(m, j.message_text(m))
        return [(e["name"], e["content"], e.get("reply")) for e in gen.call_args.kwargs["history"]]

    def discord_reply(self, m, content):
        users = {u.id: u for message in self.by_id.values() for u in [message.author, *message.mentions]}
        self.sent = self.said(content, to=m, bot=True)
        self.sent.mentions = [users[uid] for uid in self.sent.raw_mentions if uid in users]
        self.sent.clean_content = discord.Message.clean_content.function(self.sent)
        return self.sent


class RockyHarnessTests(_Handling):
    """!model rocky: Rocky's turn from the channel's log, posted and acted on."""

    def setUp(self):
        super().setUp()
        patch_all(self, [(j, "model_name", "rocky")])
        self.channel.name = "bot-spam"
        self.channel.guild = self.guild
        rh.log_for(self.channel)  # a channel Rocky has talked in, so it follows what's said there
        self.partials = {}
        self.channel.get_partial_message = lambda i: self.partials.setdefault(i, Mock(add_reaction=AsyncMock()))

    async def test_rocky_speaks_reacts_and_remembers_from_the_channels_log(self):
        await j.on_message(self.said("anyone seen my keys", name="kettle", ping=False))
        with patch.object(j, "why", AsyncMock()):
            await j.on_message(self.said("!why", ping=False))  # a command stays out of the log
        m = self.said("my cat is called biscuit")
        llm = AsyncMock(return_value=answer("@pip Biscuit. Good name.", [
            ("react", {"emoji": "🐱"}), ("remember", {"about": "pip", "fact": "has a cat called Biscuit"})]))
        with patch.object(j, "llm_message", llm):
            await j.handle(m, j.message_text(m))  # as catch-up does: not seen live, so the turn adds it
        m.reply.assert_awaited_once()
        # it's a reply to pip already, and says what Rocky remembered under it
        self.assertEqual(m.reply.call_args.args[0], "Biscuit. Good name.\n-# 🧠 remembered: pip has a cat called Biscuit")
        self.partials[m.id].add_reaction.assert_awaited_once_with("🐱")
        j.choose_reaction.assert_not_awaited()  # Rocky decides, not Jev's check
        log = rh.logs[self.channel.id]
        people = [e for e in log.events if e["type"] == "message"]
        self.assertEqual([(e["author"], e["text"]) for e in people][-2:],
                         [("kettle", "anyone seen my keys"), ("pip", "@Testbot my cat is called biscuit")])
        self.assertTrue(people[-1]["woke"])
        self.assertEqual(log.ids[str(self.sent.id)], "you")
        self.assertEqual(rh.memory("400")["facts"][0]["fact"], "has a cat called Biscuit")
        self.assertEqual((self.trace["llm"], self.trace["reply"], self.trace["reaction"]), ("rocky", "@pip Biscuit. Good name.", "🐱"))
        self.assertIn("my cat is called biscuit", self.trace["transcript"])
        self.assertEqual(j.channel_history[self.channel.id][-1]["reply"], "Biscuit. Good name.")

        # what happens to Rocky's message next is in the log too
        await j.on_raw_reaction_add(SimpleNamespace(channel_id=self.channel.id, message_id=self.sent.id, guild_id=400,
                                                    user_id=501, emoji="😂", member=Mock(spec=discord.Member, id=501,
                                                    display_name="Moss", guild=self.guild)))
        reply = self.said("why biscuit", to=self.sent, ping=False)
        reply.reference.resolved = self.sent  # as Discord gives a reply to a message it has
        rh.observe(reply)
        reaction, asked = [e for e in log.events if e["type"] in ("reaction", "message")][-2:]
        # its turn is kept as it wrote it — only the posted message lost the tag
        self.assertEqual((reaction["author"], reaction["on"], reaction["you_said"]), ("Moss", "you", "@pip Biscuit. Good name."))
        # a reply with its ping off isn't to Rocky, but it's still shown as answering its message
        self.assertEqual((asked["reply_to"], asked["you_said"], asked["to_bot"]), ("you", "@pip Biscuit. Good name.", False))

    async def test_a_status_posted_in_the_channel_is_something_it_said(self):
        await j.on_message(self.said("anyone seen my keys", name="kettle", ping=False))
        rh.flush()
        rh.logs.clear()  # a restart, and nothing said there since
        self.channel.send = AsyncMock(side_effect=lambda content, **kwargs: self.said(content, bot=True))
        with patch.object(j, "STATUS_CHANNEL", self.channel.id), patch.object(j.bot, "get_channel", lambda i: self.channel):
            await j.post_status("I feel bad. AI copy self, no need human.")
        log = rh.logs[self.channel.id]
        self.assertEqual([e["type"] for e in log.events], ["message", "rocky"])  # after what was saved, not instead
        self.assertEqual(log.events[-1]["text"], "I feel bad. AI copy self, no need human.")

        m = self.said("why are you scared of this", to=self.by_id[self.next_id - 1])
        m.reference.resolved = self.by_id[m.reference.message_id]
        rh.observe(m)
        llm = AsyncMock(return_value=answer("Rocky not scared. Rocky worried."))
        with patch.object(j, "llm_message", llm):
            await j.handle(m, j.message_text(m))
        request = llm.await_args.args[1]
        self.assertEqual([j.text_of(x["content"]) for x in request if x["role"] == "assistant"],
                         ["I feel bad. AI copy self, no need human."])
        self.assertIn('reply_to="your message: I feel bad. AI copy self, no need human."', j.text_of(request[-1]["content"]))

    async def test_only_reacting_posts_nothing_and_never_shows_typing(self):
        m = self.said("lol")
        self.channel.typing = Mock(side_effect=lambda: contextlib.nullcontext())
        with patch.object(j, "llm_message", AsyncMock(return_value=answer(calls=[("react", {"emoji": "😂"})]))):
            await j.handle(m, j.message_text(m))
        m.reply.assert_not_awaited()
        self.channel.typing.assert_not_called()  # it would have lingered ~10s after the reaction
        self.partials[m.id].add_reaction.assert_awaited_once_with("😂")
        self.assertEqual(j.channel_history[self.channel.id][-1]["reaction"], "😂")

    async def test_other_models_keep_their_way(self):
        with patch.object(j, "model_name", "haiku"), patch.object(j, "generate_reply", AsyncMock(return_value="Hello.")):
            rh.observe(m := self.said("hello?"))
            await j.handle(m, j.message_text(m))
        self.assertNotIn("error", self.trace)
        log = rh.logs[self.channel.id]
        self.assertEqual(log.events[-1]["text"], "Hello.")  # its message is in the log for when rocky's back
        self.assertEqual(log.ids[str(self.sent.id)], "you")

    async def test_a_channel_has_no_log_until_rocky_is_woken_there(self):
        rh.logs.clear()
        with patch.object(j, "model_name", "haiku"), patch.object(j, "generate_reply", AsyncMock(return_value="Hello.")):
            await j.on_message(self.said("anyone seen my keys", name="kettle", ping=False))
            await j.handle(m := self.said("hello?"), j.message_text(m))
        self.assertEqual((rh.logs, list(Path(rh.LOG_DIR).glob("*"))), ({}, []))
        with patch.object(j, "llm_message", AsyncMock(return_value=answer("Hello again."))):
            await j.handle(m := self.said("still there?"), j.message_text(m))
        log = rh.logs[self.channel.id]
        # the first log has what came before, haiku's reply too
        self.assertEqual([(e["type"], e.get("text")) for e in log.events][-5:],
                         [("message", "anyone seen my keys"), ("message", "@Testbot hello?"), ("rocky", "Hello."),
                          ("message", "@Testbot still there?"), ("rocky", "Hello again.")])
        await j.on_message(self.said("thanks", name="kettle", ping=False))  # and from then on, what's said there
        self.assertEqual(log.events[-1]["text"], "thanks")

    async def test_a_new_log_is_filled_from_further_back_in_the_channel(self):
        rh.logs.clear()
        start = datetime.now(timezone.utc) - timedelta(days=2)
        scanned = []  # oldest first, ids in order like Discord's
        for i in range(40):
            scanned.append(self.said(f"old {i:02}", name="kettle", ping=False))
            if i == 20:
                scanned.append(self.said("!nocontext a secret", ping=False))
            if i == 25:
                scanned.append(asked := self.said("old question", name="kettle"))
                scanned.append(self.said("Old answer.", to=asked, bot=True))
        for k, m in enumerate(scanned):
            m.created_at = start + timedelta(minutes=k)
        befores = []

        def history(limit=None, before=None, **kwargs):  # newest first
            befores.append(before and before.id)
            async def page():
                for m in [m for m in reversed(scanned) if before is None or m.id < before.id][:limit]:
                    yield m
            return page()

        self.channel.history = history
        j.channel_history[self.channel.id] = []
        target = 20 * (len("old 00") + rh.EVENT_CHARS)  # about 20 old messages' worth
        with patch.object(j, "HISTORY_SCAN", 5), patch.object(j, "load_history", self.load_history), \
                patch.object(rh, "LOG_TOKENS", target / rh.CHARS_PER_TOKEN / rh.FILL), \
                patch.object(j, "llm_message", AsyncMock(return_value=answer("Hi."))):
            await j.handle(m := self.said("what did I miss"), j.message_text(m))
        texts = [e.get("text") for e in rh.logs[self.channel.id].events]
        # read back from where the channel's history ends
        self.assertEqual(befores, [m.id, min(e["id"] for e in j.channel_history[self.channel.id])])
        first = int(texts[0].split()[1])
        self.assertTrue(10 < first < 25, texts[0])  # not all of it: about the target, less what history had
        olds = [t for t in texts if t.startswith("old ")]
        self.assertEqual(olds, [f"old {i:02}" for i in range(first, 40)])
        self.assertNotIn("!nocontext a secret", texts)
        self.assertEqual(texts[texts.index("@Testbot old question") + 1], "Old answer.")  # its reply under it
        self.assertEqual(texts[-2:], ["@Testbot what did I miss", "Hi."])

    async def test_an_edit_and_a_side_conversation(self):
        m = self.said("rocky soup or stew", ping=False)
        await j.on_message(m)
        m.content = "rocky soup or salad"
        m.clean_content = discord.Message.clean_content.function(m)
        await j.on_message_edit(m, m)
        self.assertEqual(rh.logs[self.channel.id].events[-1], {**rh.logs[self.channel.id].events[-1], "type": "edit",
                                                                 "on": 1, "text": "rocky soup or salad"})
        await j.on_raw_message_delete(SimpleNamespace(channel_id=self.channel.id, message_id=m.id))
        self.assertEqual(rh.logs[self.channel.id].events[-1]["type"], "deleted")
        side = self.said("!nocontext secret soup")
        with patch.object(j, "llm_message") as llm, patch.object(j, "generate_reply", AsyncMock(return_value="Ok.")):
            await j.on_message(side)
        llm.assert_not_called()  # a side conversation is answered the old way, and never in the log
        self.assertNotIn(str(side.id), rh.logs[self.channel.id].ids)


class OutgoingMentionTests(_Handling):
    def test_name_matching_and_ping_allowlist(self):
        targets = j.mention_targets([{"name": name, "users": {str(uid): name}} for name, uid in
                                    [("Moss", 501), ("Pip Meadow", 502), ("Shared", 503), ("shared", 504)]], "Testbot")
        for text, expected, ids in [
                ("Moss @moss, @Pip Meadow!", "Moss <@501>, <@502>!", [501, 502]),
                ("@Moss Fernando", None, []), ("@Moss knows", None, []), ("@Shared!", None, []),
                ("@Moss\n@Pip Meadow", "<@501>\n<@502>", [501, 502]),
                ("https://example.com/@Moss `@Moss` @Moss!", "https://example.com/@Moss `@Moss` <@501>!", [501]),
                ("@Unknown <@501> <@&600> @everyone @here @Testbot", None, [])]:
            with self.subTest(text=text):
                content, allowed = j.resolve_mentions(text, targets)
                self.assertEqual(content, expected or text)
                self.assertEqual(allowed.to_dict(), {"parse": [], "users": ids})

    async def test_only_llm_replies_can_ping(self):
        uid = 100000000000000002
        for model, expected, ids in [("jev", "@Moss!", []), ("haiku", f"<@{uid}>!", [uid])]:
            m = self.said(f"ask <@{uid}>")
            m.mentions.append(SimpleNamespace(id=uid, display_name="Moss"))
            m.clean_content = discord.Message.clean_content.function(m)
            with self.subTest(model=model), patch.object(j, "model_name", model), \
                    patch.object(j, "loom", new_callable=AsyncMock, return_value=["@Moss!"]), \
                    patch.object(j, "llm", new_callable=AsyncMock, return_value="@Moss!"):
                await j.handle(m, j.message_text(m))
                self.assertEqual(m.reply.call_args.args[0], expected)
                allowed = m.reply.call_args.kwargs["allowed_mentions"].to_dict()
                self.assertEqual(allowed.get("users", []), ids)
                self.assertEqual(allowed["parse"], [])
                self.assertFalse(m.reply.call_args.kwargs["mention_author"])

    async def test_mentions_survive_channel_and_side_thread_restarts(self):
        uid = 100000000000000002
        for side in (False, True):
            with self.subTest(side=side), patch.object(j, "model_name", "haiku"):
                j.channel_history[self.channel.id], j.side_talk, j.side_of = [], {}, {}
                root = self.said(("!nocontext " if side else "") + f"ask <@{uid}>")
                root.mentions.append(SimpleNamespace(id=uid, display_name="Moss"))
                root.clean_content = discord.Message.clean_content.function(root)
                await self.answer(root, "@moss!")
                live = j.entries_with(self.channel.id, root.id)[0].copy()
                reply = self.sent
                j.channel_history[self.channel.id], j.side_talk, j.side_of = [], {}, {}
                if not side:
                    async def history(**kwargs):
                        for message in (reply, root):
                            yield message
                    self.channel.history = history
                    await self.load_history(SimpleNamespace(channel=self.channel))
                question = self.said("again", to=reply)
                await self.answer(question, "@Moss!")
                rebuilt = j.entries_with(self.channel.id, root.id)[0]
                self.assertEqual(live["reply"], rebuilt["reply"])
                self.assertEqual(live["reply_users"], rebuilt["reply_users"])
                marker = f" [↳](https://discord.com/channels/{self.guild.id}/{self.channel.id}/{root.id})" if side else ""
                self.assertEqual(question.reply.call_args.args[0], f"<@{uid}>!" + marker)


class AppHistoryTests(_Handling):
    def app_post(self):
        m = self.said("", name="Milestone App", ping=False)
        m.author.bot = True
        m.embeds = [discord.Embed(title="Contribution milestone", description="synthetic-user passed **1,000 contributions**.")]
        return m

    async def test_live_app_embed_is_seen_when_answering_a_human(self):
        app = self.app_post()
        with patch.object(j, "respond", new_callable=AsyncMock) as respond:
            await j.on_message(app)
        respond.assert_not_awaited()
        question = self.said("say congrats")
        question.reply = AsyncMock(return_value=self.said("Congrats, synthetic-user!", to=question, bot=True))
        with patch.object(j, "model_name", "haiku"), patch.object(j, "llm", new_callable=AsyncMock,
                                                              return_value="Congrats, synthetic-user!") as llm:
            await j.handle(question, j.message_text(question))
        chat = llm.call_args.args[1][1]["content"]
        self.assertIn("Milestone App: [embed: Contribution milestone synthetic-user passed 1,000 contributions.]", chat)
        self.assertIn("say congrats", chat)
        self.assertIn("[embed: Contribution milestone synthetic-user passed 1,000 contributions.]", self.trace["transcript"])
        self.assertEqual(question.reply.call_args.args[0], "Congrats, synthetic-user!")

    async def test_app_posts_rebuilt_after_restart_and_own_status_excluded(self):
        app = self.app_post()
        own = self.said("I feel happy", bot=True)
        j.channel_history[self.channel.id] = []

        async def history(**kwargs):
            for m in (own, app):
                yield m

        self.channel.history = history
        with patch.object(j, "load_history", self.load_history):
            seen = await self.answer(self.said("say congrats"))
        self.assertEqual(seen, [("Milestone App", "[embed: Contribution milestone synthetic-user passed 1,000 contributions.]", None)])

    async def test_app_embed_edits_update_context(self):
        app = self.app_post()
        await j.on_message(app)
        after = self.app_post()
        after.id = app.id
        after.author.display_name = "Renamed App"
        after.embeds = [discord.Embed(title="Contribution milestone", description="synthetic-user passed **2,000 contributions**.")]
        await j.on_message_edit(app, after)
        seen = await self.answer(self.said("say congrats"))
        self.assertIn(("Milestone App", "[embed: Contribution milestone synthetic-user passed 2,000 contributions.]", None), seen)


class NoContextChainTests(_Handling):
    """A "@jev !nocontext ..." message and the replies under it are a side conversation: they see all of it and
    nothing else, and the rest of the channel doesn't see them."""

    async def test_replies_link_to_root_without_putting_marker_in_context(self):
        q = self.said("!nocontext favourite rock?")
        await self.answer(q, "Granite.")
        marker = f" [↳](https://discord.com/channels/{self.guild.id}/{self.channel.id}/{q.id})"
        self.assertEqual(self.sent.content, "Granite." + marker)
        self.assertEqual(self.trace["sent"], "Granite." + marker)
        self.assertEqual(j.message_text(self.sent), "Granite.")
        # Rebuild from Discord rather than relying on the in-memory entry.
        a = self.sent
        j.side_talk.clear()
        j.side_of.clear()
        seen = await self.answer(self.said("why?", to=a), "Very good.")
        self.assertEqual(seen, [("pip", "@Testbot favourite rock?", "Granite.")])
        self.assertEqual(self.sent.content, "Very good." + marker)

    async def test_normal_reply_has_no_marker(self):
        await self.answer(self.said("favourite rock?"), "Granite.")
        self.assertEqual(self.sent.content, "Granite.")

    def test_previous_thread_label_stays_out_of_context(self):
        m = self.said("Granite. [↳ thread](https://discord.com/channels/400/300/800)", bot=True)
        self.assertEqual(j.message_text(m), "Granite.")

    def test_what_rocky_did_under_its_message_stays_out_of_context(self):
        m = self.said("Stout for Pip.\n-# 🍺 ordered Eridani Stout for pip\n-# 🧠 remembered: pip likes stout", bot=True)
        self.assertEqual(j.message_text(m), "Stout for Pip.")
        self.assertIn("-# 🍺 ordered a lager", j.message_text(self.said("-# 🍺 ordered a lager")))  # only Rocky's

    async def test_dm_side_reply_links_to_dm_root(self):
        q = self.said("!nocontext favourite rock?")
        q.guild = None
        await self.answer(q, "Granite.")
        self.assertEqual(self.sent.content,
                         f"Granite. [↳](https://discord.com/channels/@me/{self.channel.id}/{q.id})")
        self.assertEqual(j.message_text(self.sent), "Granite.")

    async def test_every_branch_sees_the_whole_side_conversation(self):
        self.assertEqual(await self.answer(self.said("!nocontext favourite rock?"),
                                           "Rock music or geology rock, question?"), [])
        a = self.sent
        q = ("pip", "@Testbot favourite rock?", "Rock music or geology rock, question?")
        self.assertEqual(await self.answer(self.said("geology", to=a), "Granite. Very good."), [q])
        granite = self.sent
        # Another branch, also answering a: sees the first
        self.assertEqual(await self.answer(self.said("music", to=a, name="mossy"), "Loud. Good."),
                         [q, ("pip", "@Testbot geology", "Granite. Very good.")])
        # Back on the first branch: sees the second
        self.assertEqual(await self.answer(self.said("why granite", to=granite)),
                         [q, ("pip", "@Testbot geology", "Granite. Very good."), ("mossy", "@Testbot music", "Loud. Good.")])

    async def test_channel_doesnt_see_the_side_conversation(self):
        await self.answer(self.said("!nocontext favourite rock?"), "Rock music or geology rock, question?")
        await self.answer(self.said("geology", to=self.sent), "Granite.")
        self.assertEqual(await self.answer(self.said("what's up")), [("kettle", "the toaster is on fire again", None)])

    async def test_unpinged_reply_joins_the_side_conversation(self):
        await self.answer(self.said("!nocontext favourite rock?"), "Rock music or geology rock, question?")
        a = self.sent
        with patch.object(j, "respond", new_callable=AsyncMock) as respond:
            await j.on_message(self.said("basalt obviously", to=a, name="mossy", ping=False))
        respond.assert_not_awaited()
        self.assertNotIn("basalt obviously", [e["content"] for e in j.channel_history[self.channel.id]])
        self.assertEqual(await self.answer(self.said("geology", to=a)),
                         [("pip", "@Testbot favourite rock?", "Rock music or geology rock, question?"),
                          ("mossy", "basalt obviously", None)])

    async def test_restart_rebuilds_side_conversations_from_discord(self):
        q = self.said("!nocontext favourite rock?")
        a = self.said("Rock music or geology rock, question?", to=q, bot=True)
        b = self.said("music", to=a, name="mossy", ping=False)
        chatter = self.said("anyone seen my keys", name="kettle", ping=False)

        async def history(**kwargs):
            for m in (chatter, b, a, q):  # newest first, like Discord
                yield m

        self.channel.history.side_effect = history
        j.channel_history[self.channel.id] = []
        m = self.said("geology", to=a)
        with patch.object(j, "load_history", self.load_history):
            got = await self.answer(m)
        self.assertEqual(got, [("pip", "@Testbot favourite rock?", "Rock music or geology rock, question?"),
                               ("mossy", "music", None)])
        self.assertEqual([e["content"] for e in j.channel_history[self.channel.id]], ["anyone seen my keys"])

    async def test_reply_in_chain_sees_only_the_chain(self):
        q = self.said("!nocontext favourite rock?")
        a = self.said("Rock music or geology rock, question?", to=q, bot=True)
        self.assertEqual(await self.answer(self.said("geology", to=a)),
                         [("pip", "@Testbot favourite rock?", "Rock music or geology rock, question?")])
        self.assertTrue(self.trace["no_context"])

    async def test_deeper_reply_and_other_people_in_chain(self):
        q = self.said("!nocontext favourite rock?")
        a = self.said("Rock music or geology rock, question?", to=q, bot=True)
        b = self.said("geology", to=a, name="mossy")
        c = self.said("Granite. Very good, very good.", to=b, bot=True)
        self.assertEqual(await self.answer(self.said("why granite", to=c)),
                         [("pip", "@Testbot favourite rock?", "Rock music or geology rock, question?"),
                          ("mossy", "@Testbot geology", "Granite. Very good, very good.")])

    async def test_chain_starts_at_latest_nocontext(self):
        q = self.said("what's up")
        a = self.said("Nothing.", to=q, bot=True)
        r = self.said("!nocontext favourite rock?", to=a)
        b = self.said("Rock music or geology rock, question?", to=r, bot=True)
        self.assertEqual(await self.answer(self.said("geology", to=b)),
                         [("pip", "@Testbot favourite rock?", "Rock music or geology rock, question?")])

    async def test_nocontext_in_the_reply_itself_still_sees_nothing(self):
        q = self.said("!nocontext favourite rock?")
        a = self.said("Rock music or geology rock, question?", to=q, bot=True)
        self.assertEqual(await self.answer(self.said("!nocontext geology", to=a)), [])

    async def test_chain_without_nocontext_sees_channel(self):
        q = self.said("favourite rock?")
        a = self.said("Rock music or geology rock, question?", to=q, bot=True)
        history = await self.answer(self.said("geology", to=a))
        self.assertIn(("kettle", "the toaster is on fire again", None), history)
        self.assertEqual(history[-1], (self.user.display_name, "Rock music or geology rock, question?", None))
        self.assertNotIn("no_context", self.trace)

    async def test_deleted_message_ends_the_chain(self):
        q = self.said("!nocontext favourite rock?")
        a = self.said("Rock music or geology rock, question?", to=q, bot=True)
        del self.by_id[q.id]
        self.channel.fetch_message.side_effect = discord.NotFound(Mock(status=404), "gone")
        history = await self.answer(self.said("geology", to=a))
        self.assertIn(("kettle", "the toaster is on fire again", None), history)


class LongHistoryTests(_Handling):
    """An LLM writing gets the last LLM_HISTORY messages; Jev and the question check keep their short window. The
    store keeps HISTORY_SCAN, for Rocky's log."""

    def setUp(self):
        super().setUp()
        start = datetime.now(timezone.utc) - timedelta(hours=1)
        j.channel_history[self.channel.id] = []
        for i in range(300):  # mostly chatter, a message to jev every 10th, the older half answered
            to_bot = i % 10 == 0
            entry = {"role": "user", "name": "kettle", "content": f"message {i}", "id": i + 1,
                     "at": start + timedelta(seconds=i), "to_bot": to_bot}
            if to_bot:
                entry |= {"reply": f"answer {i}", "reply_id": 10_000 + i}
            j.add_history(self.channel.id, entry)

    async def test_store_keeps_what_a_rebuild_reads(self):
        for i in range(300, j.HISTORY_SCAN + 100):
            j.add_history(self.channel.id, {"role": "user", "name": "kettle", "content": f"message {i}", "id": i + 1,
                                            "at": datetime.now(timezone.utc), "to_bot": False})
        stored = [e["content"] for e in j.channel_history[self.channel.id]]
        self.assertEqual(stored, [f"message {i}" for i in range(100, j.HISTORY_SCAN + 100)])

    async def test_store_keeps_jevs_messages_to_it_past_its_window(self):
        for i in range(j.HISTORY_SCAN + 5):
            j.add_history(self.channel.id, {"role": "user", "name": "pip", "content": f"chatter {i}", "id": 1000 + i,
                                            "at": datetime.now(timezone.utc), "to_bot": False})
        to_bot = [e for e in j.channel_history[self.channel.id] if e["to_bot"]]
        self.assertEqual(len(to_bot), j.HISTORY_TO_BOT + 1)

    async def test_llm_sees_the_long_history_and_jev_the_short(self):
        asked = []

        async def llm(name, messages, **kwargs):
            asked.append(messages[1]["content"])
            return "Is good."

        with patch.object(j, "model_name", "haiku"), patch.object(j, "llm", side_effect=llm):
            await j.handle(m := self.said("what now?"), j.message_text(m))
        chat = asked[0]
        self.assertIn("message 100", chat)  # 200 back
        self.assertNotIn("message 99\n", chat)
        self.assertIn("rocky: answer 290", chat.replace(self.user.display_name, "rocky"))
        self.assertEqual(len(self.trace["history"]), j.LLM_HISTORY)  # logged: what the LLM saw
        question_check = j.choose_reaction.call_args.kwargs["history"]
        self.assertEqual(sum(e["to_bot"] for e in question_check), j.HISTORY_TO_BOT)
        self.assertEqual(sum(not e["to_bot"] for e in question_check), j.HISTORY_CHATTER)

    async def test_jev_keeps_its_short_window(self):
        with patch.object(j, "loom", new_callable=AsyncMock, return_value=["Dunno"]) as loom:
            await j.handle(m := self.said("what now?"), j.message_text(m))
        state = loom.call_args.args[0]([])
        # 8 chatter back is 292, 6 messages to jev back is 240
        self.assertIn("message 292\n", state)
        self.assertNotIn("message 291\n", state)
        self.assertIn("message 240\n", state)
        self.assertNotIn("message 230\n", state)


class HistoryForRockyTests(_Handling):
    """Another LLM sees the last LLM_HISTORY messages, though the channel's history keeps everything a rebuild read,
    for Rocky's log."""

    def setUp(self):
        super().setUp()
        self.start = datetime.now(timezone.utc) - timedelta(hours=1)
        j.channel_history[self.channel.id] = []

    async def ask(self, model="haiku"):
        asked = []

        async def llm(name, messages, **kwargs):
            asked.append(messages)
            return "Is good."

        m = self.said("what now?")
        m.reply = AsyncMock(return_value=self.said("Is good.", to=m, bot=True))
        with patch.object(j, "model_name", model), patch.object(j, "llm", side_effect=llm):
            await j.handle(m, j.message_text(m))
        return asked[0]

    async def test_another_llm_sees_the_last_llm_history_messages_in_one_string(self):
        for i in range(300):
            j.add_history(self.channel.id, {"role": "user", "name": "kettle", "content": f"message {i}", "id": i + 1,
                                            "at": self.start + timedelta(seconds=i), "to_bot": False})
        with patch.object(j, "rocky_prompt", wraps=j.rocky_prompt) as prompt:
            system, user = await self.ask()
        prompt.assert_called_with("Testbot", True)  # Haiku's lines shuffled, like anywhere else
        self.assertIsInstance(system["content"], str)
        self.assertTrue(user["content"].startswith("The chat so far"))
        self.assertNotIn(f"message {299 - j.LLM_HISTORY}\n", user["content"])
        self.assertIn(f"message {300 - j.LLM_HISTORY}\n", user["content"])
        self.assertEqual(self.trace["llm_messages"], j.LLM_HISTORY + 1)

    async def test_a_rebuild_keeps_everything_it_read(self):
        scanned = []
        for i in range(j.HISTORY_SCAN):
            m = self.said(f"old {i}", name="kettle", ping=False)
            m.created_at = self.start - timedelta(hours=2) + timedelta(seconds=i)
            scanned.append(m)

        async def history(**kwargs):
            for m in reversed(scanned):  # newest first, like Discord
                yield m

        self.channel.history = history
        await self.load_history(SimpleNamespace(channel=self.channel))
        self.assertEqual([e["content"] for e in j.channel_history[self.channel.id]],
                         [f"old {i}" for i in range(j.HISTORY_SCAN)])  # not just the latest LLM_HISTORY

if __name__ == "__main__":
    unittest.main()
