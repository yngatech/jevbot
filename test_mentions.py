"""Offline behavioral checks for mention routing, history, and startup catch-up.

    uv run --with-requirements requirements.txt python -m unittest test_mentions
"""

import contextlib
import os
import unittest
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

os.environ.setdefault("DISCORD_TOKEN_JEV", "test")
os.environ.setdefault("OPENROUTER_API_KEY", "test")

import discord
import jev_bot as j


class _Discord(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.user = SimpleNamespace(id=100, bot=True, display_name="Testbot")
        self.role = SimpleNamespace(id=200, mention="<@&200>")
        self.guild = SimpleNamespace(self_role=self.role, me=self.user, threads=[])
        self.channel = Mock(id=300)
        self.channel.permissions_for.return_value = SimpleNamespace(read_messages=True, read_message_history=True)
        self.guild.text_channels = [self.channel]
        self.target = Mock(spec=discord.Message, id=700, author=self.user, reactions=[])
        self.target.reference = None
        self.target.created_at = datetime.now(timezone.utc) - timedelta(minutes=1)
        for obj, attr, value in [
            (j.bot._connection, "user", self.user),
            (j.bot._connection, "_guilds", {400: self.guild}),
            (j, "DM_USERS", {500}),
            (j, "channel_history", defaultdict(list)),
            (j, "heard", False),
        ]:
            p = patch.object(obj, attr, value)
            p.start()
            self.addCleanup(p.stop)


class MentionTests(_Discord):
    def message(self, *, content="hello", ping=False, role=False, reply=True, dm=False, author_bot=False):
        return SimpleNamespace(
            id=600, content=content, guild=None if dm else self.guild, channel=self.channel,
            author=SimpleNamespace(id=500, bot=author_bot, display_name="Speaker"),
            mentions=[self.user] if ping else [], role_mentions=[self.role] if role else [],
            reference=SimpleNamespace(resolved=self.target) if reply else None,
            created_at=datetime.now(timezone.utc), reactions=[], attachments=[], stickers=[], embeds=[],
        )

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
        self.assertFalse(j.channel_history[self.channel.id])

    async def test_unpinged_diagnostic_reply_still_runs_command(self):
        for content, handler in (("!why", "why"), ("!context", "context")):
            with self.subTest(content=content):
                m = self.message(content=content)
                with patch.object(j, handler, new_callable=AsyncMock) as command, \
                        patch.object(j, "respond", new_callable=AsyncMock) as respond:
                    await j.on_message(m)
                command.assert_awaited_once_with(m)
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


class NoContextChainTests(_Discord):
    """Replies carrying on a "@jev !nocontext ..." conversation see only that chain, not the channel."""

    def setUp(self):
        super().setUp()
        self.trace = {"cost": 0.0, "requests": 0}
        token = j.trace.set(self.trace)
        self.addCleanup(j.trace.reset, token)
        self.channel.typing = lambda: contextlib.nullcontext()
        self.channel.fetch_message = AsyncMock(side_effect=lambda i: self.by_id[i])
        self.by_id = {}
        self.next_id = 800
        for attr, value in [("has_credit", True), ("history_loaded", {}), ("emoji_vocabulary", lambda g: {}),
                            ("choose_reaction", AsyncMock(return_value=None)), ("load_history", AsyncMock())]:
            p = patch.object(j, attr, value)
            p.start()
            self.addCleanup(p.stop)
        # Chatter before the conversation, which a !nocontext chain shouldn't see
        j.channel_history[self.channel.id] = [
            {"role": "user", "name": "kettle", "content": "the toaster is on fire again", "id": 1,
             "at": datetime.now(timezone.utc) - timedelta(minutes=10), "to_bot": False}]

    def said(self, content, *, to=None, bot=False, name="pip", ping=True):
        m = Mock(spec=discord.Message, id=self.next_id, content=(f"<@{self.user.id}> " if ping and not bot else "") + content,
                 guild=self.guild, channel=self.channel, mentions=[self.user] if ping and not bot else [],
                 role_mentions=[], created_at=datetime.now(timezone.utc), reactions=[], attachments=[], stickers=[],
                 embeds=[], author=self.user if bot else SimpleNamespace(id=500, bot=False, display_name=name))
        # Discord only resolves the message m replies to, not the ones above it
        m.reference = to and SimpleNamespace(message_id=to.id, resolved=None)
        self.by_id[m.id] = m
        self.next_id += 1
        return m

    async def answer(self, m):
        if m.reference:
            m.reference.resolved = self.by_id[m.reference.message_id]
        m.reply = AsyncMock(return_value=SimpleNamespace(id=999))
        with patch.object(j, "generate_reply", new_callable=AsyncMock, return_value="Geology good") as gen:
            await j.handle(m, j.message_text(m))
        return [(e["name"], e["content"], e.get("reply")) for e in gen.call_args.kwargs["history"]]

    async def test_reply_in_chain_sees_only_the_chain(self):
        q = self.said("!nocontext favourite rock?")
        a = self.said("Rock music or geology rock, question?", to=q, bot=True)
        self.assertEqual(await self.answer(self.said("geology", to=a)),
                         [("pip", "favourite rock?", "Rock music or geology rock, question?")])
        self.assertTrue(self.trace["no_context"])

    async def test_deeper_reply_and_other_people_in_chain(self):
        q = self.said("!nocontext favourite rock?")
        a = self.said("Rock music or geology rock, question?", to=q, bot=True)
        b = self.said("geology", to=a, name="mossy")
        c = self.said("Granite. Very good, very good.", to=b, bot=True)
        self.assertEqual(await self.answer(self.said("why granite", to=c)),
                         [("pip", "favourite rock?", "Rock music or geology rock, question?"),
                          ("mossy", "geology", "Granite. Very good, very good.")])

    async def test_chain_starts_at_latest_nocontext(self):
        q = self.said("what's up")
        a = self.said("Nothing.", to=q, bot=True)
        r = self.said("!nocontext favourite rock?", to=a)
        b = self.said("Rock music or geology rock, question?", to=r, bot=True)
        self.assertEqual(await self.answer(self.said("geology", to=b)),
                         [("pip", "favourite rock?", "Rock music or geology rock, question?")])

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


if __name__ == "__main__":
    unittest.main()
