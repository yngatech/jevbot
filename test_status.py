"""Offline behavioral checks for status judging, retries, publishing, and trace history.

    uv run --with-requirements requirements.txt python -m unittest test_status
"""

import asyncio
import os
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

os.environ.setdefault("DISCORD_TOKEN_JEV", "test")
os.environ.setdefault("OPENROUTER_API_KEY", "test")

import jev_bot as j  # noqa: E402


class _StatusCase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.parent = {"cost": 0.0, "requests": 0}
        self.token = j.trace.set(self.parent)
        self.credit_patch = patch.object(j, "has_credit", True)
        self.credit_patch.start()
        self.addCleanup(self.credit_patch.stop)

    async def asyncTearDown(self):
        j.trace.reset(self.token)

    def candidates(self, moods, scores):
        moods, scores = iter(moods), iter(scores)

        async def generate(*args):
            mood = next(moods)
            j.note(transcript=f"Testbot's diary, today: {mood}", steps=[{"word": mood}], stop="<END>")
            j.trace.get()["cost"] += .02
            j.trace.get()["requests"] += 4
            return mood

        async def judge(mood):
            j.trace.get()["cost"] += .00004
            j.trace.get()["requests"] += 1
            return next(scores)

        return AsyncMock(side_effect=generate), AsyncMock(side_effect=judge)


class StatusTests(_StatusCase):
    async def test_accepts_first_candidate_without_rewriting(self):
        mood = "I feel good. Yup. Yup. Yup"
        generate, judge = self.candidates([mood], [.91])
        with patch.object(j, "generate_status", generate), patch.object(j, "status_score", judge):
            self.assertEqual(await j.generate_filtered_status("Testbot", j.STATUS_STARTS[0], "Speaker: hello"), mood)
        generate.assert_awaited_once_with("Testbot", j.STATUS_STARTS[0], "Speaker: hello")
        judge.assert_awaited_once_with(mood)
        self.assertEqual(len(self.parent["status_attempts"]), 1)
        self.assertEqual(self.parent["status_score"], .91)

    async def test_retries_with_same_opener_and_context_and_keeps_accepted_trace(self):
        bad, good = "I wonder what about is thing?", "I wonder why cats purr?"
        generate, judge = self.candidates([bad, good], [.2, j.STATUS_THRESHOLD])
        with patch.object(j, "generate_status", generate), patch.object(j, "status_score", judge):
            self.assertEqual(await j.generate_filtered_status("Testbot", j.STATUS_STARTS[2], "Speaker: cats purr"), good)
        self.assertEqual(generate.await_count, 2)
        self.assertTrue(all(call.args == ("Testbot", j.STATUS_STARTS[2], "Speaker: cats purr")
                            for call in generate.await_args_list))
        attempts = self.parent["status_attempts"]
        self.assertEqual([a["accepted"] for a in attempts], [False, True])
        self.assertEqual([a["status"] for a in attempts], [bad, good])
        self.assertEqual(self.parent["transcript"], attempts[1]["transcript"])
        self.assertEqual(self.parent["steps"], attempts[1]["steps"])
        self.assertAlmostEqual(self.parent["cost"], .04008)
        self.assertEqual(self.parent["requests"], 10)

    async def test_stops_after_three_rejections(self):
        generate, judge = self.candidates(["I wonder what is? Thing"] * 3, [.2, .3, .4])
        with patch.object(j, "generate_status", generate), patch.object(j, "status_score", judge):
            self.assertIsNone(await j.generate_filtered_status("Testbot", j.STATUS_STARTS[2]))
        self.assertEqual(generate.await_count, 3)
        self.assertEqual(judge.await_count, 3)
        self.assertNotIn("steps", self.parent)
        self.assertNotIn("transcript", self.parent)
        self.assertEqual(len(self.parent["status_attempts"]), 3)

    async def test_missing_decision_stops_regeneration(self):
        generate, judge = self.candidates(["I feel good"], [None])
        with patch.object(j, "generate_status", generate), patch.object(j, "status_score", judge):
            self.assertIsNone(await j.generate_filtered_status("Testbot", j.STATUS_STARTS[0]))
        self.assertEqual(generate.await_count, 1)
        self.assertFalse(self.parent["status_attempts"][0]["accepted"])

    async def test_empty_generation_skips_judge(self):
        generate, judge = self.candidates([None], [])
        with patch.object(j, "generate_status", generate), patch.object(j, "status_score", judge):
            self.assertIsNone(await j.generate_filtered_status("Testbot", j.STATUS_STARTS[0]))
        judge.assert_not_awaited()

    async def test_credit_exhaustion_stops_calls(self):
        async def generate(*args):
            j.has_credit = False
            j.note(error="out of credit")
            return "I feel good"

        judge = AsyncMock()
        with patch.object(j, "generate_status", AsyncMock(side_effect=generate)), patch.object(j, "status_score", judge):
            self.assertIsNone(await j.generate_filtered_status("Testbot", j.STATUS_STARTS[0]))
        judge.assert_not_awaited()
        self.assertEqual(self.parent["error"], "out of credit")

    async def test_credit_exhaustion_during_judging_does_not_accept(self):
        generate, _ = self.candidates(["I feel good"], [])

        async def judge(mood):
            j.has_credit = False
            return .99

        with patch.object(j, "generate_status", generate), patch.object(j, "status_score", AsyncMock(side_effect=judge)):
            self.assertIsNone(await j.generate_filtered_status("Testbot", j.STATUS_STARTS[0]))
        self.assertEqual(generate.await_count, 1)

    async def test_exception_restores_trace_and_accounts_for_cost(self):
        generate, _ = self.candidates(["I feel good"], [])
        with patch.object(j, "generate_status", generate), \
             patch.object(j, "status_score", AsyncMock(side_effect=RuntimeError("judge unavailable"))):
            with self.assertRaisesRegex(RuntimeError, "judge unavailable"):
                await j.generate_filtered_status("Testbot", j.STATUS_STARTS[0])
        self.assertIs(j.trace.get(), self.parent)
        self.assertEqual(self.parent["cost"], .02)
        self.assertEqual(self.parent["requests"], 4)

    async def test_judge_only_receives_finished_status(self):
        mood = "I'm thinking about pasta tonight."
        post = AsyncMock(return_value={"acceptable": {"noul": .83}})
        with patch.object(j, "post", post):
            self.assertEqual(await j.status_score(mood), .83)
        args = post.await_args.args
        self.assertTrue(args[1].endswith('Status to judge: "I\'m thinking about pasta tonight."'))
        self.assertEqual(args[2], {"acceptable": {"type": "noul", "instructions": j.STATUS_QUESTION}})
        self.assertIn("Its readers cannot see the conversation", args[1])

    async def test_missing_or_invalid_probabilities_are_unavailable(self):
        responses = [{}, {"acceptable": None}, {"acceptable": {}},
                     *[{"acceptable": {"noul": p}} for p in (None, True, "0.9", float("nan"), float("inf"), -1, 2)]]
        for response in responses:
            with self.subTest(response=response), patch.object(j, "post", AsyncMock(return_value=response)):
                self.assertIsNone(await j.status_score("I feel good"))
        with patch.object(j, "post", AsyncMock(return_value={"acceptable": {"noul": 0}})):
            self.assertEqual(await j.status_score("I wonder what is?"), 0)


class StatusUpdateTests(_StatusCase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.old = j.discord.CustomActivity("I feel rested today")
        for name, value in (("status", self.old), ("status_turn", 0), ("heard", True), ("gen_lock", asyncio.Lock())):
            patcher = patch.object(j, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch.object(j.bot._connection, "user", SimpleNamespace(display_name="Testbot"))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.saved, self.shown, self.posted, self.written = Mock(), AsyncMock(), AsyncMock(), Mock()
        for name, mock in (("save_status", self.saved), ("show_credit", self.shown),
                           ("post_status", self.posted), ("write_trace", self.written)):
            patcher = patch.object(j, name, mock)
            patcher.start()
            self.addCleanup(patcher.stop)

    async def test_generation_judging_and_publishing_flow(self):
        # Exercise the real loom/render path and the real judge, replacing only API results and Discord writes.
        words = iter(["what", "is", "thing", "about", "?",
                      "good", ".", "really", "seriously", "swear", j.END])
        scores = iter([.2, .95])

        async def next_word(*args):
            j.trace.get()["cost"] += .003
            j.trace.get()["requests"] += 4
            return {next(words): 1.0}, 0.0

        async def post(*args):
            j.trace.get()["cost"] += .00004
            j.trace.get()["requests"] += 1
            return {"acceptable": {"noul": next(scores)}}

        with patch.object(j, "next_word", AsyncMock(side_effect=next_word)), \
             patch.object(j, "post", AsyncMock(side_effect=post)):
            await j.update_status.coro()
        mood = "I feel good. Really seriously swear"
        self.assertEqual(j.status.name, mood)
        self.saved.assert_called_once()
        self.assertEqual(self.saved.call_args.args[:2], (mood, 0))
        self.shown.assert_awaited_once()
        self.posted.assert_awaited_once_with(mood)
        t = self.written.call_args.args[0]
        self.assertEqual(t["status"], mood)
        self.assertEqual(t["status_score"], .95)
        self.assertEqual(len(t["status_attempts"]), 2)
        self.assertEqual([a["accepted"] for a in t["status_attempts"]], [False, True])
        self.assertAlmostEqual(t["cost"], .03308)
        self.assertEqual(t["requests"], 46)
        self.assertEqual(j.why_panels(t)[0]["picked"], "good")
        self.assertEqual(t["transcript"], "Testbot's diary, today: I feel")
        self.assertFalse(j.heard)
        self.assertEqual(j.status_turn, 1)

    async def test_rejected_cycle_keeps_previous_status_and_does_not_post(self):
        generate, judge = self.candidates(["I wonder what is? Thing"] * 3, [.2, .3, .4])
        with patch.object(j, "generate_status", generate), patch.object(j, "status_score", judge):
            await j.update_status.coro()
        self.assertIs(j.status, self.old)
        self.saved.assert_not_called()
        self.shown.assert_not_awaited()
        self.posted.assert_not_awaited()
        self.assertTrue(j.heard)
        self.assertEqual(j.status_turn, 1)
        t = self.written.call_args.args[0]
        self.assertIsNone(t["status"])
        self.assertEqual(len(t["status_attempts"]), 3)

    async def test_quiet_or_out_of_credit_cycle_does_not_generate(self):
        for heard, credit in ((False, True), (True, False)):
            with self.subTest(heard=heard, credit=credit), patch.object(j, "heard", heard), \
                 patch.object(j, "has_credit", credit), patch.object(j, "generate_filtered_status", AsyncMock()) as generate:
                await j.update_status.coro()
                generate.assert_not_awaited()
        self.saved.assert_not_called()
        self.posted.assert_not_awaited()
        self.written.assert_not_called()

    async def test_unavailable_judge_keeps_previous_status_and_records_error(self):
        generate, _ = self.candidates(["I feel good"], [])
        with patch.object(j, "generate_status", generate), \
             patch.object(j, "status_score", AsyncMock(side_effect=RuntimeError("judge unavailable"))), \
             self.assertLogs(j.log, level="ERROR"):
            await j.update_status.coro()
        self.assertIs(j.status, self.old)
        self.saved.assert_not_called()
        self.posted.assert_not_awaited()
        self.assertTrue(j.heard)
        t = self.written.call_args.args[0]
        self.assertIn("judge unavailable", t["error"])
        self.assertEqual(t["cost"], .02)
        self.assertEqual(len(t["status_attempts"]), 1)
        self.assertIs(j.trace.get(), t)

    async def test_original_context_alternation_and_prefix_rotation(self):
        for turn in (0, 1, 2, 3):
            with self.subTest(turn=turn), patch.object(j, "status_turn", turn), patch.object(j, "heard", True), \
                 patch.object(j, "recent_chat", Mock(return_value="Speaker: dinner tonight")), \
                 patch.object(j, "generate_filtered_status", AsyncMock(return_value="I feel good")) as generate:
                await j.update_status.coro()
                generate.assert_awaited_once_with("Testbot", j.STATUS_STARTS[turn % 3],
                                                 "Speaker: dinner tonight" if turn % 2 else "")


if __name__ == "__main__":
    unittest.main()
