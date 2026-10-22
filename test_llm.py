"""Offline behavioral checks for the LLM: !model, its requests, statuses and !why, with the LLM's answers synthetic.

    uv run --with-requirements requirements.txt python -m unittest test_llm
"""

import tempfile
import unittest
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fakes import openrouter, patch_all, session, traced
import jev_bot as j


class _Case(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.trace = traced(self)
        patch_all(self, [(j, "MODEL_PATH", Path(self.dir.name) / "model.json"), (j, "model_name", "jev"),
                         (j, "has_credit", True), (j, "channel_history", defaultdict(list)),
                         (j, "TIMEZONE", j.ZoneInfo("UTC"))])  # not whatever .env sets


class ModelCommandTests(_Case):
    def setUp(self):
        super().setUp()
        self.user = SimpleNamespace(id=100, bot=True, display_name="rocky")
        patch_all(self, [(j.bot._connection, "user", self.user)])

    def message(self, content):
        return SimpleNamespace(id=600, content=content, guild=None, channel=SimpleNamespace(id=300),
                               author=SimpleNamespace(id=500, bot=False, display_name="pip"),
                               mentions=[], role_mentions=[], reference=None, reply=AsyncMock())

    def test_command_takes_model_with_or_without_a_name(self):
        for content, want in [("!model", "!model"), ("!model rocky", "!model"), ("!MODEL Rocky", "!model"),
                              ("!model rocky please", None), ("!why", "!why"), ("!why now", None),
                              ("what model are you", None)]:
            with self.subTest(content=content):
                self.assertEqual(j.command(self.message(content)), want)

    async def test_on_message_routes_model_without_answering(self):
        m = self.message("!model rocky")
        with patch.object(j, "DM_USERS", {500}), patch.object(j, "model_command", new_callable=AsyncMock) as cmd, \
                patch.object(j, "respond", new_callable=AsyncMock) as respond:
            await j.on_message(m)
        cmd.assert_awaited_once_with(m)
        respond.assert_not_awaited()

    async def test_switching_is_kept_and_unknown_names_change_nothing(self):
        m = self.message("!model rocky")
        await j.model_command(m)
        self.assertEqual(j.model_name, "rocky")
        self.assertEqual(j.load_model(), "rocky")  # what a restart would pick up
        self.assertIn("Now writing with **rocky** (anthropic/claude-haiku-5.5)", m.reply.call_args.args[0])

        m = self.message("!model gpt9")
        await j.model_command(m)
        self.assertEqual(j.model_name, "rocky")
        self.assertIn("No model called 'gpt9'", m.reply.call_args.args[0])

        m = self.message("!model haiku")  # gone: Haiku is Rocky now
        await j.model_command(m)
        self.assertEqual(j.model_name, "rocky")

        m = self.message("!model")
        await j.model_command(m)
        self.assertIn("Writing with **rocky**", m.reply.call_args.args[0])

    def test_unknown_or_unreadable_saved_model_falls_back_to_jev(self):
        for saved in ['{"model": "gpt9"}', '{"model": "haiku"}', "not json"]:
            with self.subTest(saved=saved):
                j.MODEL_PATH.write_text(saved)
                self.assertEqual(j.load_model(), "jev")


class RequestTests(_Case):
    async def test_a_jev_decision_and_an_llm_request_are_recorded_separately(self):
        with openrouter((200, {"answers": {"asked": {"noul": .9}}, "usage": {"cost": .000081}}),
                        (200, {"id": "gen-synthetic-2", "provider": "Anthropic",
                               "choices": [{"message": {"content": "Coffee first. Then walk."}, "finish_reason": "stop",
                                            "native_finish_reason": "stop"}],
                               "usage": {"cost": 0, "is_byok": True,
                                         "cost_details": {"upstream_inference_cost": .000577}}})), \
                patch.object(j.log, "warning") as warning:
            self.assertIsNone(await j.choose_reaction("coffee?", "Moss", "Rocky", ["☕"]))
            reply = await j.llm("rocky", [{"role": "user", "content": "coffee?"}])
        self.assertEqual(reply, "Coffee first. Then walk.")
        decision, call = self.trace["api_calls"]
        self.assertEqual([r["purpose"] for r in (decision, call)], ["decision", "reply"])
        self.assertEqual([r["is_byok"] for r in (decision, call)], [False, True])
        self.assertEqual(call["model"], j.LLMS["rocky"]["id"])
        # What each cost, and only the reply's on our own Anthropic key
        self.assertEqual([round(r["router_cost"] + r["key_cost"], 6) for r in (decision, call)], [.000081, .000577])
        self.assertAlmostEqual(self.trace["byok_cost"], .000577)
        self.assertNotIn("finish_reason", decision)  # Jev's decisions don't have one
        self.assertEqual((call["finish_reason"], call["provider"], call["id"]), ("stop", "Anthropic", "gen-synthetic-2"))
        self.assertIsNone(j.stopped_early(call))  # an ordinary ending, without a warning
        warning.assert_not_called()
        self.assertEqual(self.trace["requests"], 2)
        self.assertAlmostEqual(self.trace["cost"], .000658)

    async def test_a_request_logs_the_prompt_size_for_context(self):
        with openrouter((200, {"choices": [{"message": {"content": "Is Biscuit."}}],
                               "usage": {"prompt_tokens": 3_100, "completion_tokens": 18, "cost": 0.001,
                                         "prompt_tokens_details": {"cached_tokens": 2_900}}})) as s:
            await j.llm("rocky", [{"role": "system", "content": "voice"}, {"role": "user", "content": "cat?"}])
        messages = s.post.call_args.kwargs["json"]["messages"]
        self.assertEqual(self.trace["llm_window"], 1_000_000)
        self.assertEqual(self.trace["llm_prompt_tokens"], 3_100)
        self.assertEqual(self.trace["llm_completion_tokens"], 18)
        self.assertEqual(self.trace["llm_cached_tokens"], 2_900)
        self.assertEqual(self.trace["llm_prompt_chars"], sum(len(m["content"]) for m in messages))

    async def test_a_rate_limited_request_is_tried_again(self):
        limited = (429, {"error": {"code": 429}})
        with openrouter(limited, limited, (200, {"choices": [{"message": {"content": "Is Biscuit."}}],
                                                 "usage": {"prompt_tokens": 10}})):
            text = await j.llm("rocky", [{"role": "user", "content": "cat?"}])
        self.assertEqual(text, "Is Biscuit.")
        self.assertEqual([r["status"] for r in self.trace["api_calls"]], [429, 429, 200])
        self.assertEqual([r["attempt"] for r in self.trace["api_calls"]], [0, 1, 2])
        self.assertTrue(all(r["router_cost"] is None for r in self.trace["api_calls"]))

    async def test_a_refusal_is_recorded_and_warned_about_without_what_was_said(self):
        with openrouter((502, {"error": {"code": 502}}),
                        (200, {"id": "gen-synthetic-1", "provider": "Anthropic",
                               "choices": [{"message": {"content": "Secret refusal words."},
                                            "finish_reason": "content_filter", "native_finish_reason": "refusal"}],
                               "usage": {"cost": .0001}})), self.assertLogs(j.log, "WARNING") as logs:
            await j.llm("rocky", [{"role": "user", "content": "cat?"}])
        failed, refused = self.trace["api_calls"]  # nothing on why the failure stopped: it had no answer
        self.assertEqual({k: failed[k] for k in ["finish_reason", "native_finish_reason", "id", "provider"]},
                         dict.fromkeys(["finish_reason", "native_finish_reason", "id", "provider"]))
        self.assertEqual((refused["finish_reason"], refused["native_finish_reason"], refused["id"],
                          refused["provider"]), ("content_filter", "refusal", "gen-synthetic-1", "Anthropic"))
        self.assertEqual(j.stopped_early(refused), "refusal")
        warning, = [line for line in logs.output if "refused" in line]
        self.assertIn("content_filter, refusal", warning)
        self.assertIn("gen-synthetic-1", warning)
        self.assertNotIn("Secret", "".join(logs.output))

    async def test_an_odd_response_is_still_retried(self):
        with openrouter((502, {"choices": [None]}), (200, ["not", "a", "completion"]),
                        (200, {"choices": [{"message": {"content": "Is Biscuit."}, "finish_reason": "stop"}], "usage": {}})):
            reply = await j.llm("rocky", [{"role": "user", "content": "cat?"}])
        self.assertEqual(reply, "Is Biscuit.")
        self.assertEqual([r["finish_reason"] for r in self.trace["api_calls"]], [None, None, "stop"])

    async def test_jev_transcript_has_no_times(self):
        history = [{"role": "user", "name": "pip", "content": "hi", "at": j.datetime.now(j.timezone.utc)}]
        with patch.object(j, "loom", new_callable=AsyncMock, return_value=["Hi"]):
            await j.generate_reply("cat?", "kettle", "rocky", history=history)
        self.assertEqual(self.trace["transcript"], "pip: hi\nkettle: cat?\nrocky: ")

    async def test_jev_still_looms(self):
        with patch.object(j, "llm", new_callable=AsyncMock) as llm, \
                patch.object(j, "loom", new_callable=AsyncMock, return_value=["Biscuit"]):
            self.assertEqual(await j.generate_reply("cat?", "kettle", "rocky", history=[]), "Biscuit")
        llm.assert_not_awaited()

class CreditTests(_Case):
    """A request OpenRouter won't pay for: out of credit, or over the key's limit or its workspace's budget."""

    BUDGET = {"error": {"code": 403, "message": "Workspace budget exceeded",
                        "metadata": {"limit_source": "openrouter_workspace_budget", "limit_usd": 1, "spend_usd": 1}}}

    async def test_a_spent_budget_or_credit_stops_without_retrying(self):
        for status, body in [(403, self.BUDGET), (402, {"error": {"code": 402, "message": "Insufficient credits"}}),
                             (402, "not json")]:
            for call in ("llm", "post"):
                with self.subTest(status=status, body=body, call=call):
                    with openrouter((status, body)) as s:
                        if call == "llm":
                            self.assertEqual(await j.llm("rocky", [{"role": "user", "content": "cat?"}]), "")
                        else:
                            self.assertEqual(await j.post(s, "state", {"asked": {}}), {})
                    self.assertEqual(s.post.call_count, 1)
                    s.set_credit.assert_awaited_once_with(False)

    async def test_any_other_forbidden_is_retried(self):
        body = {"error": {"code": 403, "message": "Flagged", "metadata": {"reasons": ["synthetic"]}}}
        with openrouter((403, body)) as s:
            await j.llm("rocky", [{"role": "user", "content": "cat?"}])
        self.assertEqual(s.post.call_count, 3)
        s.set_credit.assert_not_awaited()

    async def test_credit_left_is_the_lowest_of_credit_key_limit_and_budget(self):
        for key, want in [({"limit_remaining": None}, 7.5), ({"limit_remaining": 4.0}, 4.0),
                          ({"limit_remaining": 4.0, "effective_budget": {"remaining_usd": 0.25}}, 0.25),
                          ({"limit_remaining": None, "effective_budget": None}, 7.5)]:
            with self.subTest(key=key):
                replies = iter([{"data": {"total_credits": 10, "total_usage": 2.5}}, {"data": key}])
                s = session()
                s.get.return_value.__aenter__.side_effect = lambda: SimpleNamespace(json=AsyncMock(return_value=next(replies)))
                with patch.object(j.aiohttp, "ClientSession", return_value=s):
                    self.assertEqual(await j.credit_left(), want)


class StatusTests(_Case):
    async def test_rejected_status_preserves_all_request_purposes_and_accepted_context(self):
        j.model_name = "rocky"
        replies = [(200, data) for size, score in [(2_000, .1), (3_000, .9)] for data in [
            {"choices": [{"message": {"content": "I feel coffee is good."}}],
             "usage": {"prompt_tokens": size, "completion_tokens": 18, "cost": 0, "is_byok": True,
                       "prompt_tokens_details": {"cached_tokens": size - 500},
                       "cost_details": {"upstream_inference_cost": .0004}}},
            {"answers": {"acceptable": {"noul": score}}, "usage": {"cost": .00004}}]]
        with openrouter(*replies):
            mood = await j.generate_filtered_status("Rocky", ["i", "feel"], "Moss: coffee")
        self.assertEqual(mood, "I feel coffee is good.")
        self.assertEqual([r["purpose"] for r in self.trace["api_calls"]],
                         ["status", "status_check", "status", "status_check"])
        # the accepted one's usage, and the cost of all of them
        self.assertEqual((self.trace["llm_prompt_tokens"], self.trace["llm_chat_chars"]), (3_000, len("Moss: coffee")))
        self.assertEqual((self.trace["llm_cached_tokens"], self.trace["llm_completion_tokens"]), (2_500, 18))
        self.assertEqual(self.trace["requests"], 4)
        self.assertAlmostEqual(self.trace["cost"], .00088)
        self.assertAlmostEqual(self.trace["byok_cost"], .0008)

    async def test_status_keeps_its_opener_on_one_line(self):
        j.model_name = "rocky"
        for answer, want in [("I wonder if sleep is same\nwhen no one watches.", "I wonder if sleep is same when no one watches."),
                             ("Why cat knock tea, question?", "I wonder why cat knock tea, question?"),
                             ("I wonder " + "very " * 40, None)]:
            with self.subTest(answer=answer), patch.object(j, "llm", AsyncMock(return_value=answer)):
                mood = await j.generate_status("rocky", ["i", "wonder"], "kettle: my cat knocked my tea over")
            self.assertLessEqual(len(mood), 128)
            if want:
                self.assertEqual(mood, want)
        self.assertIn("kettle: my cat knocked my tea over", self.trace["transcript"])
        self.assertEqual(self.trace["llm"], "rocky")

    async def test_a_status_gets_rockys_lines_in_a_fresh_order(self):
        j.model_name = "rocky"
        with patch.object(j, "rocky_prompt", return_value="voice") as prompt, \
                patch.object(j, "llm", AsyncMock(return_value="I feel good.")):
            await j.generate_status("rocky", ["i", "feel"])
        prompt.assert_called_with("rocky", True)


class WhyTests(_Case):
    async def test_why_says_an_llm_reply_has_nothing_to_chart(self):
        m = SimpleNamespace(channel=SimpleNamespace(id=300), guild=None,
                            reply=AsyncMock(return_value=SimpleNamespace(attachments=[])), reference=None)
        logged = {"llm": "haiku", "reply": "Biscuit", "bot_name": "rocky"}  # from before it was Rocky
        with patch.object(j, "find_trace", return_value=logged), patch.object(j.why_chart, "render") as render:
            await j.why(m)
        render.assert_not_called()
        self.assertIn("haiku doesn't say how likely its words were", m.reply.call_args.args[0])

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
