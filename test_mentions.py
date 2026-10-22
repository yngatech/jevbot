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
        self.guild = SimpleNamespace(id=400, self_role=self.role, me=self.user, threads=[], get_thread=lambda _: None,
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
            (j, "taken", {}),
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

    # Who reacted to jev's reply, as people Rocky can ping (mention_targets)
    def names(self, entry=None):
        entry = {"name": "Speaker", **(entry or self.entry)}
        return {name for name, uid in j.mention_targets([entry], "Testbot", reactors=True).items()
                if uid and name != entry["name"]}  # not whoever asked

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
        self.assertEqual(self.names(), {"Moss", "Pip"})
        self.assertNotIn("😂", self.transcript(reactions=False, marked=True))
        self.assertIn("Hello (😂×2)", self.transcript())  # Jev gets the count
        self.assertNotIn("😂", self.transcript(their_reactions=False))
        await j.on_raw_reaction_remove(self.payload())
        self.assertEqual(self.names(), {"Pip"})
        await j.on_raw_reaction_remove(self.payload(502, "Pip"))
        self.assertNotIn("😂", self.transcript())

    async def test_clear_one_and_all_reactions(self):
        await j.on_raw_reaction_add(self.payload())
        await j.on_raw_reaction_add(self.payload(502, "Pip", "💀"))
        await j.on_raw_reaction_clear_emoji(self.payload())
        self.assertEqual(self.names(), {"Pip"})  # only on 💀 now
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
        self.assertEqual(self.names(), {"Moss", "Pip"})

    async def test_standalone_reply_names(self):
        question = MentionTests.message(self)
        reply = _Message(id=700, author=self.user, reference=SimpleNamespace(message_id=question.id), guild=None,
                         content="Hello", reactions=[self.reaction()], created_at=question.created_at,
                         mentions=[], role_mentions=[], embeds=[], attachments=[], stickers=[])
        standalone = {"role": "assistant", "content": "Hello"}
        await j.attach_reactions(standalone, reply, prefix="")
        self.assertEqual(self.names(standalone), {"Moss", "Pip"})

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
        self.assertEqual(self.names(), {"Moss"})

    async def test_a_rebuild_reuses_saved_names_without_asking_discord(self):
        j.reactor_cache["700"] = {"counts": {"😂": 2}, "reactors": {"😂": {"501": "Moss", "502": "Pip"}}}
        unasked = SimpleNamespace(emoji="😂", count=3, me=True, users=Mock(side_effect=AssertionError("asked Discord")))
        await j.attach_reactions(self.entry, SimpleNamespace(id=700, reactions=[unasked]))
        self.assertEqual(self.names(), {"Moss", "Pip"})

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

    async def test_jev_gets_counts(self):
        self.entry.update(reply_reactions={"😂": 2}, reply_reactors={"😂": {"501": "Moss", "502": "Pip"}})
        seen = []
        async def loom(state, *args, **kwargs):
            seen.append(state([]))
            return ["Hi"]
        with patch.object(j, "model_name", "jev"), patch.object(j, "loom", loom):
            await j.generate_reply("how are you?", "Speaker", "Testbot", [self.entry])
        self.assertIn("Hello (😂×2)", seen[0])
        self.assertNotIn("Moss", seen[0])

    def test_reactions_are_emoji_and_counts(self):
        self.assertEqual(j.reacted({"😂": 3, "💀": 1}), " (😂×3 💀)")
        self.assertEqual(j.reacted({}), "")


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
        # Chatter before the conversation, which a !nocontext thread shouldn't see
        j.channel_history[self.channel.id] = [
            {"role": "user", "name": "kettle", "content": "the toaster is on fire again", "id": 1,
             "at": datetime.now(timezone.utc) - timedelta(minutes=10), "to_bot": False}]

    def said(self, content, *, to=None, bot=False, name="pip", ping=True, channel=None):
        m = Mock(spec=discord.Message, id=self.next_id, content=(f"<@{self.user.id}> " if ping and not bot else "") + content,
                 guild=self.guild, channel=channel or self.channel, mentions=[self.user] if ping and not bot else [],
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
        self.sent = self.said(content, to=m, bot=True, channel=m.channel)
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

    async def test_jev_keeps_its_way(self):
        with patch.object(j, "model_name", "jev"), patch.object(j, "generate_reply", AsyncMock(return_value="Hello.")):
            rh.observe(m := self.said("hello?"))
            await j.handle(m, j.message_text(m))
        self.assertNotIn("error", self.trace)
        log = rh.logs[self.channel.id]
        self.assertEqual(log.events[-1]["text"], "Hello.")  # its message is in the log for when rocky's back
        self.assertEqual(log.ids[str(self.sent.id)], "you")

    async def test_a_channel_has_no_log_until_rocky_is_woken_there(self):
        rh.logs.clear()
        with patch.object(j, "model_name", "jev"), patch.object(j, "generate_reply", AsyncMock(return_value="Hello.")):
            await j.on_message(self.said("anyone seen my keys", name="kettle", ping=False))
            await j.handle(m := self.said("hello?"), j.message_text(m))
        self.assertEqual((rh.logs, list(Path(rh.LOG_DIR).glob("*"))), ({}, []))
        with patch.object(j, "llm_message", AsyncMock(return_value=answer("Hello again."))):
            await j.handle(m := self.said("still there?"), j.message_text(m))
        log = rh.logs[self.channel.id]
        # the first log has what came before, Jev's reply too
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

    async def test_an_edit_and_a_nocontext_message(self):
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
        side.create_thread = AsyncMock(side_effect=discord.Forbidden(Mock(status=403), "no threads"))
        with patch.object(j, "llm_message", AsyncMock(return_value=answer("Soup good."))):
            await j.on_message(side)
        side.reply.assert_awaited_once()  # answered alone, with no thread to answer in
        self.assertNotIn(str(side.id), rh.logs[self.channel.id].ids)
        rh.flush()
        self.assertEqual([p.name for p in Path(rh.LOG_DIR).glob("*")], [f"{self.channel.id}.json"])  # its log isn't kept


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

    async def test_only_rocky_can_ping(self):
        uid = 100000000000000002
        self.channel.name, self.channel.guild = "bot-spam", self.guild
        for model, expected, ids in [("jev", "@Moss!", []), ("rocky", f"<@{uid}>!", [uid])]:
            m = self.said(f"ask <@{uid}>")
            m.mentions.append(SimpleNamespace(id=uid, display_name="Moss"))
            m.clean_content = discord.Message.clean_content.function(m)
            with self.subTest(model=model), patch.object(j, "model_name", model), \
                    patch.object(j, "loom", new_callable=AsyncMock, return_value=["@Moss!"]), \
                    patch.object(j, "llm_message", AsyncMock(return_value=answer("@Moss!"))):
                await j.handle(m, j.message_text(m))
                self.assertEqual(m.reply.call_args.args[0], expected)
                allowed = m.reply.call_args.kwargs["allowed_mentions"].to_dict()
                self.assertEqual(allowed.get("users", []), ids)
                self.assertEqual(allowed["parse"], [])
                self.assertFalse(m.reply.call_args.kwargs["mention_author"])

    async def test_mentions_survive_a_restart(self):
        uid = 100000000000000002
        self.channel.name, self.channel.guild = "bot-spam", self.guild

        async def turn(m, says):
            if m.reference:
                m.reference.resolved = self.by_id[m.reference.message_id]
            with patch.object(j, "llm_message", AsyncMock(return_value=answer(says))):
                await j.handle(m, j.message_text(m))

        with patch.object(j, "model_name", "rocky"):
            j.channel_history[self.channel.id] = []
            root = self.said(f"ask <@{uid}>")
            root.mentions.append(SimpleNamespace(id=uid, display_name="Moss"))
            root.clean_content = discord.Message.clean_content.function(root)
            await turn(root, "@moss!")
            live = j.channel_history[self.channel.id][0].copy()
            reply = self.sent
            j.channel_history[self.channel.id] = []
            rh.logs.clear()

            async def history(**kwargs):
                for message in (reply, root):
                    yield message
            self.channel.history = history
            await self.load_history(SimpleNamespace(channel=self.channel))
            question = self.said("again", to=reply)
            await turn(question, "@Moss!")
            rebuilt = j.channel_history[self.channel.id][0]
            self.assertEqual(live["reply"], rebuilt["reply"])
            self.assertEqual(live["reply_users"], rebuilt["reply_users"])
            self.assertEqual(question.reply.call_args.args[0], f"<@{uid}>!")


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
        self.channel.name, self.channel.guild = "bot-spam", self.guild
        llm = AsyncMock(return_value=answer("Congrats, synthetic-user!"))
        with patch.object(j, "model_name", "rocky"), patch.object(j, "llm_message", llm):
            await j.handle(question, j.message_text(question))
        chat = j.text_of(llm.await_args.args[1][-1]["content"])
        self.assertIn('from="Milestone App"', chat)
        self.assertIn(">[embed: Contribution milestone synthetic-user passed 1,000 contributions.]</message>", chat)
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


class NoContextThreadTests(_Handling):
    """"@jev !nocontext ..." opens a thread on the message, a channel of its own: answered there with nothing from
    before, every message in it to jev, and for Rocky without memories."""

    def setUp(self):
        super().setUp()
        self.channel.name, self.channel.guild = "bot-spam", self.guild
        self.threads = {}
        self.partials = {}
        self.channel.get_partial_message = lambda i: self.partials.setdefault(i, Mock(add_reaction=AsyncMock()))

    # A !nocontext message to jev, which a thread can be opened on — with the message's id, as Discord's are
    def asked(self, content):
        m = self.said(content)

        async def create_thread(name, **kwargs):
            t = Mock(spec=discord.Thread, id=m.id, guild=self.guild)
            t.name = name
            # not the channel's Mock itself, which assigning would make the thread's child
            t.parent = SimpleNamespace(id=self.channel.id, fetch_message=lambda i: self.channel.fetch_message(i))
            t.typing = lambda: contextlib.nullcontext()
            t.send = AsyncMock(side_effect=lambda content, **kwargs: self.said(content, bot=True, channel=t))
            t.get_partial_message = lambda i: self.partials.setdefault(("thread", i), Mock(add_reaction=AsyncMock()))

            def history(limit=None, before=None, **kwargs):  # newest first: the thread's, not the message it's on
                async def page():
                    for x in sorted(self.by_id.values(), key=lambda x: -x.id):
                        if x.channel is t and (before is None or x.id < before.id):
                            yield x
                return page()
            t.history = history
            self.threads[m.id] = t
            return t
        m.create_thread = AsyncMock(side_effect=create_thread)
        return m

    async def test_jev_answers_in_the_thread_and_carries_on_there(self):
        q = self.asked("!nocontext favourite rock?")
        with patch.object(j, "model_name", "jev"):
            self.assertEqual(await self.answer(q, "Granite."), [])
            thread = self.threads[q.id]
            self.assertEqual(q.create_thread.call_args.kwargs["name"], "favourite rock?")
            q.reply.assert_not_awaited()
            thread.send.assert_awaited_once()
            self.assertEqual(thread.send.call_args.args[0], "Granite.")
            self.assertEqual((self.trace["no_context"], self.trace["channel_id"]), (True, thread.id))
            # !why finds it from the message or in the thread
            self.assertTrue(j.in_channel(self.trace, self.channel.id) and j.in_channel(self.trace, thread.id))
            # in the thread, nobody needs to tag jev, and it has the thread so far
            follow = self.said("why granite", ping=False, channel=thread)
            self.assertTrue(j.should_respond(follow))
            self.assertEqual(await self.answer(follow, "Very good."), [("pip", "@Testbot favourite rock?", "Granite.")])
            self.assertEqual(follow.reply.call_args.args[0], "Very good.")
        # and the channel it came from never sees any of it
        self.assertEqual([e["content"] for e in j.channel_history[self.channel.id]], ["the toaster is on fire again"])
        self.assertFalse(j.should_respond(self.said("anyone?", ping=False)))
        self.assertNotIn("rock", j.recent_chat())  # nor do the statuses, which are public

    async def test_catch_up_leaves_a_message_answered_in_its_thread(self):
        q = self.asked("!nocontext favourite rock?")
        with patch.object(j, "model_name", "jev"):
            await self.answer(q, "Granite.")

        def history(**kwargs):
            async def page():
                yield q
            return page()
        self.channel.history = history
        with patch.object(j, "DM_USERS", set()), patch.object(j, "respond", new_callable=AsyncMock) as respond:
            await j.catch_up()
        respond.assert_not_awaited()

    async def test_catch_up_answers_a_thread_opened_but_not_answered(self):
        q = self.asked("!nocontext favourite rock?")
        thread = await j.open_thread(q)  # and then a restart, before it answered
        rh.flush()
        rh.logs.clear()

        def history(**kwargs):
            async def page():
                yield q
            return page()
        self.channel.history = history
        with patch.object(j, "DM_USERS", set()), patch.object(j, "model_name", "jev"), \
                patch.object(j.bot, "fetch_channel", AsyncMock(return_value=thread)), \
                patch.object(j, "generate_reply", AsyncMock(return_value="Granite.")):
            await j.catch_up()
        q.create_thread.assert_awaited_once()  # the thread it opened, not another
        self.assertEqual(thread.send.call_args.args[0], "Granite.")

    async def test_editing_the_question_is_seen_in_its_thread(self):
        q = self.asked("!nocontext favourite rock?")
        with patch.object(j, "model_name", "jev"):
            await self.answer(q, "Granite.")
            q.content = f"<@{self.user.id}> !nocontext favourite colour?"
            q.clean_content = discord.Message.clean_content.function(q)
            await j.on_message_edit(q, q)
            follow = self.said("why", ping=False, channel=self.threads[q.id])
            self.assertEqual(await self.answer(follow), [("pip", "@Testbot favourite colour?", "Granite.")])

    async def test_replies_under_an_old_side_conversation_stay_out_of_history(self):
        q = self.said("!nocontext favourite rock?")
        a = self.said("Granite. [↳](https://discord.com/channels/400/300/800)", to=q, bot=True)
        b = self.said("why granite", to=a, name="mossy", ping=False)
        c = self.said("Hard.", to=b, bot=True)
        chatter = self.said("anyone seen my keys", name="kettle", ping=False)

        async def history(**kwargs):
            for m in (chatter, c, b, a, q):  # newest first, like Discord
                yield m
        self.channel.history = history
        j.channel_history[self.channel.id] = []
        await self.load_history(SimpleNamespace(channel=self.channel))
        self.assertEqual([e["content"] for e in j.channel_history[self.channel.id]], ["anyone seen my keys"])

    async def test_a_thread_it_cant_post_in_is_answered_in_the_channel(self):
        q = self.asked("!nocontext favourite rock?")
        opened = q.create_thread.side_effect

        async def create_thread(**kwargs):
            t = await opened(**kwargs)
            t.send = AsyncMock(side_effect=discord.Forbidden(Mock(status=403), "no sending in threads"))
            return t
        q.create_thread.side_effect = create_thread
        with patch.object(j, "model_name", "rocky"), patch.object(j, "llm_message", AsyncMock(return_value=answer("Granite."))):
            await j.handle(q, j.message_text(q))
        self.assertEqual(q.reply.call_args.args[0], "Granite.")

    async def test_nocontext_in_its_thread_carries_it_on(self):
        q = self.asked("!nocontext favourite rock?")
        with patch.object(j, "model_name", "jev"):
            await self.answer(q, "Granite.")
        thread = self.threads[q.id]
        again = self.said("!nocontext and basalt?", ping=False, channel=thread)
        self.assertFalse(j.no_context(again))
        with patch.object(j, "respond", new_callable=AsyncMock):
            await j.on_message(again)
        self.assertEqual(rh.logs[thread.id].events[-1]["text"], "and basalt?")

    async def test_a_nocontext_reply_shows_nothing_it_replies_to(self):
        earlier = self.said("the secret plan is soup", name="kettle", ping=False)
        q = self.asked("!nocontext what do you think")
        q.reference = SimpleNamespace(message_id=earlier.id, resolved=earlier)
        llm = AsyncMock(return_value=answer("Think good."))
        with patch.object(j, "model_name", "rocky"), patch.object(j, "llm_message", llm):
            await j.handle(q, j.message_text(q))
        self.assertNotIn("secret plan", str(llm.await_args.args[1]))

    async def test_rocky_in_the_thread_has_no_memories(self):
        rh.use(rh.Log("bot-spam", "400", self.channel.id),
               {"function": {"name": "remember", "arguments": '{"about": "pip", "fact": "likes soup"}'}}, {"n": 1})
        q = self.asked("!nocontext what do you know about me")
        llm = AsyncMock(return_value=answer("Nothing. New friend.", [("react", {"emoji": "🪨"})]))
        with patch.object(j, "model_name", "rocky"), patch.object(j, "llm_message", llm):
            await j.handle(q, j.message_text(q))
        thread = self.threads[q.id]
        self.assertEqual(thread.send.call_args.args[0], "Nothing. New friend.")
        self.partials[q.id].add_reaction.assert_awaited_once_with("🪨")  # on the message the thread is on
        request, tools = llm.await_args.args[1], llm.await_args.kwargs["tools"]
        self.assertNotIn("soup", str(request))
        self.assertNotIn("<memories>", str(request))
        self.assertNotIn("remember", [t["function"]["name"] for t in tools])
        log = rh.logs[thread.id]
        self.assertEqual([(e["type"], e.get("text")) for e in log.events],
                         [("message", "@Testbot what do you know about me"), ("rocky", "Nothing. New friend."),
                          ("result", None)])
        self.assertFalse(rh.has_log(self.channel.id))  # the channel's own log isn't started, or given it

        rh.flush()
        rh.logs.clear()  # a restart: the thread's log is still one without memories
        follow = self.said("ok remember I like rocks", ping=False, channel=thread)
        llm = AsyncMock(return_value=answer("Rocks good."))
        with patch.object(j, "model_name", "rocky"), patch.object(j, "llm_message", llm):
            await j.on_message(follow)
        self.assertEqual(follow.reply.call_args.args[0], "Rocks good.")
        self.assertNotIn("<memories>", str(llm.await_args.args[1]))
        self.assertIsNone(rh.logs[thread.id].space)
        self.assertEqual(rh.memory("400")["facts"][0]["fact"], "likes soup")

        # the message the thread is on is in the channel: a look or a reaction later goes there, not to the thread
        log = rh.logs[thread.id]
        self.assertEqual((log.channel_of(1), log.channel_of(3)), (self.channel.id, thread.id))
        again = self.said("react to my question", ping=False, channel=thread)
        llm = AsyncMock(return_value=answer(calls=[("react", {"emoji": "👍", "message": 1})]))
        with patch.object(j, "model_name", "rocky"), patch.object(j, "llm_message", llm), \
                patch.object(j.bot, "get_partial_messageable", lambda i: {self.channel.id: self.channel}[i]):
            await j.on_message(again)
        self.partials[q.id].add_reaction.assert_awaited_with("👍")

        # and when it's edited or deleted, the thread's log hears of it, loaded or not
        rh.flush()
        rh.logs.clear()
        q.content = f"<@{self.user.id}> !nocontext what do you know about pip"
        q.clean_content = discord.Message.clean_content.function(q)
        await j.on_message_edit(q, q)
        await j.on_raw_message_delete(SimpleNamespace(channel_id=self.channel.id, message_id=q.id))
        events = rh.logs[thread.id].events
        self.assertIn({"type": "edit", "on": 1, "text": "@Testbot what do you know about pip"},
                      [{k: e.get(k) for k in ("type", "on", "text")} for e in events])
        self.assertTrue(events[0]["deleted"])
        self.assertEqual(rh.logs[thread.id].channel, "what do you know about me")
        self.assertFalse(rh.has_log(self.channel.id))

    async def test_in_a_dm_its_answered_alone(self):
        q = self.said("!nocontext favourite rock?")
        q.guild = None
        rh.log_for(self.channel)  # a DM Rocky's talked in
        llm = AsyncMock(return_value=answer("Granite."))
        with patch.object(j, "model_name", "rocky"), patch.object(j, "llm_message", llm):
            await j.handle(q, j.message_text(q))
        self.assertEqual(q.reply.call_args.args[0], "Granite.")
        people = [m for m in llm.await_args.args[1] if m["role"] == "user"]
        self.assertEqual(len(people), 1)  # just the question: no memories, nothing from before
        self.assertIn("favourite rock?", j.text_of(people[0]["content"]))
        self.assertIn("This is #favourite rock? on Discord", j.text_of(llm.await_args.args[1][0]["content"]))  # as a thread is
        self.assertEqual(rh.logs[self.channel.id].events, [])  # nor is it in the DM's log
        rh.flush()
        self.assertEqual(list(Path(rh.LOG_DIR).glob("*")), [])  # and its own is never saved

    async def test_a_bare_nocontext_in_a_dm_is_nothing_to_answer(self):
        q = self.said("!nocontext", ping=False)
        q.guild = None
        llm = AsyncMock()
        with patch.object(j, "model_name", "rocky"), patch.object(j, "llm_message", llm):
            await j.handle(q, j.message_text(q))
        self.assertNotIn("error", self.trace)
        llm.assert_not_awaited()

    async def test_normal_reply_has_no_marker(self):
        await self.answer(self.said("favourite rock?"), "Granite.")
        self.assertEqual(self.sent.content, "Granite.")

    def test_old_side_conversation_links_stay_out_of_context(self):
        for text in ("Granite. [↳ thread](https://discord.com/channels/400/300/800)",
                     "Granite. [↳](https://discord.com/channels/@me/300/800)"):
            self.assertEqual(j.message_text(self.said(text, bot=True)), "Granite.")

    def test_what_rocky_did_under_its_message_stays_out_of_context(self):
        m = self.said("Stout for Pip.\n-# 🍺 ordered Eridani Stout for pip\n-# 🧠 remembered: pip likes stout", bot=True)
        self.assertEqual(j.message_text(m), "Stout for Pip.")
        self.assertIn("-# 🍺 ordered a lager", j.message_text(self.said("-# 🍺 ordered a lager")))  # only Rocky's


class LongHistoryTests(_Handling):
    """Jev and the question check keep their short window; the store keeps HISTORY_SCAN, for Rocky's log."""

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

    async def test_jev_keeps_its_short_window(self):
        with patch.object(j, "loom", new_callable=AsyncMock, return_value=["Dunno"]) as loom:
            await j.handle(m := self.said("what now?"), j.message_text(m))
        state = loom.call_args.args[0]([])
        # 8 chatter back is 292, 6 messages to jev back is 240
        self.assertIn("message 292\n", state)
        self.assertNotIn("message 291\n", state)
        self.assertIn("message 240\n", state)
        self.assertNotIn("message 230\n", state)
        question_check = j.choose_reaction.call_args.kwargs["history"]
        self.assertEqual(sum(e["to_bot"] for e in question_check), j.HISTORY_TO_BOT)
        self.assertEqual(sum(not e["to_bot"] for e in question_check), j.HISTORY_CHATTER)


class HistoryForRockyTests(_Handling):
    """The channel's history keeps everything a rebuild read, for Rocky's log."""

    def setUp(self):
        super().setUp()
        self.start = datetime.now(timezone.utc) - timedelta(hours=1)
        j.channel_history[self.channel.id] = []

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
                         [f"old {i}" for i in range(j.HISTORY_SCAN)])

if __name__ == "__main__":
    unittest.main()
