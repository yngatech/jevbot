"""Offline checks for admins and /settings: who can change what, what each setting changes, and that it's kept.

    uv run --with-requirements requirements.txt python -m unittest test_settings
"""

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fakes import patch_all
import discord
import jev_bot as j
import rocky_harness as rh

ADMIN, OWNER, PERSON = 500, 501, 502


class _Case(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.path = Path(self.dir.name) / "settings.json"
        patch_all(self, [(j, "SETTINGS_PATH", self.path), (j, "saved_settings", {}), (j, "ADMINS", {ADMIN}),
                         (j, "STATUS_EVERY", 180), (rh, "LOG_TOKENS", 90_000),
                         (j.bot, "is_owner", AsyncMock(side_effect=lambda u: u.id == OWNER)),
                         (j.bot, "is_ready", lambda: False)])

    def interaction(self, user=ADMIN):
        return SimpleNamespace(user=SimpleNamespace(id=user), response=SimpleNamespace(
            send_message=AsyncMock(), edit_message=AsyncMock()))

    async def open_settings(self, user=ADMIN):
        i = self.interaction(user)
        await j.settings_command.callback(i)
        return i.response.send_message.call_args.kwargs

    def menus(self, view):
        return [item for item in view.walk_children() if isinstance(item, j.SettingMenu)]

    def menu(self, view, key):
        return next(item for item in self.menus(view) if item.key == key)

    def text(self, view):
        return [item.content for item in view.walk_children() if isinstance(item, discord.ui.TextDisplay)]

    async def pick(self, menu, value, user=ADMIN):
        menu._values = [str(value)]
        i = self.interaction(user)
        await menu.callback(i)
        return i


class SettingsCommandTests(_Case):
    def test_settings_is_a_slash_command(self):
        command = j.bot.tree.get_command("settings")
        self.assertIsInstance(command, discord.app_commands.Command)

    async def test_everyone_sees_the_settings_privately_but_only_admins_get_menus(self):
        sent = await self.open_settings(PERSON)
        self.assertTrue(sent["ephemeral"])
        self.assertEqual(self.text(sent["view"]), ["### Settings", "**New status**\nEvery 3 hours",
                                                   "**Context window**\n90K tokens, how far back Rocky remembers a channel",
                                                   "-# Only admins can change these."])
        self.assertEqual(self.menus(sent["view"]), [])

        for user in (ADMIN, OWNER):
            with self.subTest(user=user):
                view = (await self.open_settings(user))["view"]
                self.assertEqual(self.text(view), ["### Settings", "**New status**",
                                                   "**Context window** · how far back Rocky remembers a channel",
                                                   "-# Changes count straight away, and are kept across restarts."])
                self.assertEqual([m.key for m in self.menus(view)], ["STATUS_EVERY", "LOG_TOKENS"])
                status = self.menu(view, "STATUS_EVERY")
                self.assertEqual([o.label for o in status.options if o.default], ["Every 3 hours"])
                self.assertEqual([o.label for o in status.options][:2], ["Off", "Every hour"])
                self.assertEqual([o.label for o in self.menu(view, "LOG_TOKENS").options],
                                 ["25K tokens", "50K tokens", "90K tokens", "150K tokens", "200K tokens"])

    async def test_the_same_whichever_model_is_writing(self):
        for model in ("jev", "haiku", "rocky"):
            with self.subTest(model=model), patch.object(j, "model_name", model):
                view = (await self.open_settings())["view"]
                self.assertEqual([m.key for m in self.menus(view)], ["STATUS_EVERY", "LOG_TOKENS"])

    async def test_a_value_not_in_the_menu_still_shows_as_chosen(self):
        rh.LOG_TOKENS = 120_000
        view = (await self.open_settings())["view"]
        self.assertEqual([o.label for o in self.menu(view, "LOG_TOKENS").options if o.default], ["120K tokens"])

    async def test_picking_saves_it_and_redraws_the_answer(self):
        view = (await self.open_settings())["view"]
        i = await self.pick(self.menu(view, "LOG_TOKENS"), 150_000)
        self.assertEqual(rh.LOG_TOKENS, 150_000)
        redrawn = i.response.edit_message.call_args.kwargs
        self.assertEqual([o.value for o in self.menu(redrawn["view"], "LOG_TOKENS").options if o.default], ["150000"])

        await self.pick(self.menu(view, "STATUS_EVERY"), 0)
        self.assertEqual(json.loads(self.path.read_text()), {"LOG_TOKENS": 150_000, "STATUS_EVERY": 0})
        j.STATUS_EVERY, rh.LOG_TOKENS = 180, 90_000  # as a restart would start
        j.load_settings()
        self.assertEqual((j.STATUS_EVERY, rh.LOG_TOKENS), (0, 150_000))

    async def test_someone_no_longer_an_admin_changes_nothing(self):
        view = (await self.open_settings())["view"]
        j.ADMINS = set()
        i = await self.pick(self.menu(view, "STATUS_EVERY"), 60)
        i.response.send_message.assert_awaited_once_with("Only admins can change settings", ephemeral=True)
        i.response.edit_message.assert_not_awaited()
        self.assertEqual(j.STATUS_EVERY, 180)
        self.assertFalse(self.path.exists())

    def test_saved_values_out_of_range_are_ignored(self):
        self.path.write_text(json.dumps({"STATUS_EVERY": 1, "LOG_TOKENS": "lots", "MODEL": "x"}))
        with self.assertLogs(j.log, level="WARNING"):
            j.load_settings()
        self.assertEqual((j.STATUS_EVERY, rh.LOG_TOKENS), (180, 90_000))
        self.path.write_text(json.dumps({"LOG_TOKENS": 50_000}))
        j.load_settings()
        self.assertEqual(rh.LOG_TOKENS, 50_000)


class StatusIntervalTests(_Case):
    def setUp(self):
        super().setUp()
        patch_all(self, [(j, "heard", True), (j, "has_credit", True), (j, "status_turn", 0),
                         (j, "tried_at", datetime.now(timezone.utc) - timedelta(minutes=90)),
                         (j, "generate_filtered_status", AsyncMock(return_value=None)),
                         (j.bot._connection, "user", SimpleNamespace(display_name="rocky"))])

    async def test_a_new_interval_counts_from_the_last_status_straight_away(self):
        await j.update_status.coro()
        j.generate_filtered_status.assert_not_awaited()  # 90 minutes into 180

        j.change_setting("STATUS_EVERY", 60)
        with patch.object(j, "write_trace"):
            await j.update_status.coro()
        j.generate_filtered_status.assert_awaited_once()

        j.generate_filtered_status.reset_mock()
        j.heard = True  # it failed, so it tries again — but not for another hour
        await j.update_status.coro()
        j.generate_filtered_status.assert_not_awaited()

    async def test_off_makes_none(self):
        j.change_setting("STATUS_EVERY", 0)
        j.tried_at = None
        await j.update_status.coro()
        j.generate_filtered_status.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
