"""Offline checks for rocky_harness: the channel log, the requests it makes, Rocky's turns, and keeping all of it.
The model's answers are synthetic.

    uv run --with-requirements requirements.txt python -m unittest test_harness
"""

import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

os.environ.setdefault("DISCORD_TOKEN_JEV", "test")
os.environ.setdefault("OPENROUTER_API_KEY", "test")

import jev_bot as j  # noqa: E402
import rocky_harness as rh  # noqa: E402

T0 = datetime(2026, 10, 12, 18, 0, tzinfo=timezone.utc)


def answer(text="", calls=(), reasoning=None):
    return {"role": "assistant", "content": text or None, "reasoning_details": reasoning,
            "tool_calls": [{"id": f"call_{i}", "type": "function",
                            "function": {"name": name, "arguments": json.dumps(args)}}
                           for i, (name, args) in enumerate(calls)] or None}


class _Case(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.trace = {"cost": 0.0, "requests": 0}
        token = j.trace.set(self.trace)
        self.addCleanup(j.trace.reset, token)
        for obj, attr, value in [(rh, "LOG_DIR", Path(self.dir.name) / "logs"),
                                 (rh, "MEMORY_PATH", Path(self.dir.name) / "memories.json"),
                                 (rh, "logs", {}), (rh, "log_saves", {}), (rh, "memories", {}),
                                 (rh, "memories_loaded", True), (rh, "memories_save", None),
                                 (j, "TIMEZONE", j.ZoneInfo("UTC")), (j, "ROCKY_LINES", ["Amaze, amaze, amaze!"])]:
            p = patch.object(obj, attr, value)
            p.start()
            self.addCleanup(p.stop)
        self.log = rh.Log("bot-spam", "400", 300)

    def said(self, author, text, minutes, to_bot=False, **kwargs):
        return self.log.message(author, text, T0 + timedelta(minutes=minutes), to_bot=to_bot, **kwargs)


class RenderTests(_Case):
    def test_people_are_user_turns_and_rocky_the_assistant(self):
        self.said("kettle", "anyone up?", 0)
        q = self.said("pip", "rocky <3 soup?", 1, to_bot=True, discord_id=900)
        q["woke"] = True
        self.log.add("rocky", text="Soup good.", calls=[], id=901)
        self.log.ids["901"] = "you"
        self.said("pip", "why", 2, to_bot=True, reply_to="you", you_said="Soup good.")
        self.log.add("reaction", on="you", you_said="Soup good.", emoji="😂", author="moss")
        self.log.add("edit", on=1, text="anyone up??")
        system, memories, first, rocky, second = rh.render(self.log, "rocky")
        self.assertIn("woke_you", system["content"][0]["text"])
        self.assertEqual(system["content"][0]["cache_control"], rh.CACHE)
        self.assertEqual(memories["content"][0]["text"], "<memories>\nNothing yet.\n</memories>")
        self.assertEqual(first["content"], '<day>Mon 12 Oct</day>\n<message id="1" from="kettle" at="18:00">anyone up?</message>\n'
                                           '<message id="2" from="pip" at="18:01" to="you" woke_you="yes">rocky &lt;3 soup?</message>')
        self.assertEqual(rocky, {"role": "assistant", "content": "Soup good."})
        text = second["content"][0]["text"]
        self.assertIn('to="you" reply_to="your message: Soup good.">why</message>', text)
        self.assertIn('<reaction from="moss" emoji="😂" on="your message: Soup good."/>', text)
        self.assertIn('<edit id="1">anyone up??</edit>', text)
        self.assertEqual(second["content"][0]["cache_control"], rh.CACHE)  # the end of the request

    def test_a_new_turn_leaves_everything_before_it_as_it_was(self):
        q = self.said("pip", "soup?", 0, to_bot=True)
        q["woke"] = True
        before = [j.text_of(m.get("content")) for m in rh.render(self.log, "rocky")]
        self.log.add("rocky", text="", calls=[{"id": "c1", "name": "remember", "arguments": '{"about": "pip", "fact": "likes soup"}'}],
                     reasoning=[{"type": "reasoning.encrypted", "data": "x"}])
        rh.use(self.log, {"function": {"name": "remember", "arguments": '{"about": "pip", "fact": "likes soup"}'}}, q)
        self.log.add("result", id="c1", content="Remembered as [1].")
        self.said("kettle", "rocky hi", 1, to_bot=True)["woke"] = True
        after = rh.render(self.log, "rocky")
        self.assertEqual([j.text_of(m.get("content")) for m in after[:len(before)]], before)  # memories too: a snapshot
        self.assertEqual(after[3]["reasoning_details"], [{"type": "reasoning.encrypted", "data": "x"}])
        self.assertEqual(after[4], {"role": "tool", "tool_call_id": "c1", "content": "Remembered as [1]."})
        self.log.reorganise()  # at a jump
        self.assertIn("[1] pip: likes soup", j.text_of(rh.render(self.log, "rocky")[1]["content"]))

    def test_after_a_tool_call_the_cache_mark_stays_where_the_last_request_ended(self):
        q = self.said("pip", "rocky remember soup", 0, to_bot=True)
        q["woke"] = True
        first = rh.render(self.log, "rocky")
        self.log.add("rocky", text="", calls=[{"id": "c1", "name": "remember", "arguments": '{"about": "pip", "fact": "soup"}'}])
        self.log.add("result", id="c1", content="Remembered as [1].")
        follow_up = rh.render(self.log, "rocky")
        self.assertEqual(follow_up[2], first[2])  # people's turn, marked as the first request had it
        self.assertEqual(follow_up[2]["content"][0]["cache_control"], rh.CACHE)
        self.assertEqual(follow_up[-1]["role"], "tool")

    def test_memories_are_the_servers_not_the_channels(self):
        rh.use(self.log, {"function": {"name": "remember", "arguments": '{"about": "pip", "fact": "likes soup"}'}}, {"n": 1})
        other = rh.Log("general", "400", 301)
        other.reorganise()
        self.assertIn("pip: likes soup", rh.memories_text(other))
        self.assertNotIn("soup", rh.memories_text(rh.Log("dm", "dm-5", 5)))


class ToolTests(_Case):
    def call(self, name, **args):
        return rh.use(self.log, {"function": {"name": name, "arguments": json.dumps(args)}}, {"n": 7})

    def test_react_defaults_to_the_message_that_woke_it(self):
        self.assertEqual(self.call("react", emoji="😂"), (("react", 7, "😂"), "Reacted."))
        self.assertEqual(self.call("react", emoji="🎉", message=3)[0], ("react", 3, "🎉"))

    def test_remember_keeps_short_settled_new_facts_and_forget_drops_them(self):
        self.assertEqual(self.call("remember", about="@pip", fact="is vegetarian.")[1], "Remembered as [1].")
        self.assertEqual(self.call("remember", about="Pip", fact="Is vegetarian")[1], "Already remembered as [1].")
        self.assertIn("Not kept", self.call("remember", about="rocky", fact="likes Rocky — no, Project Hail Mary? Avoid")[1])
        self.assertIn("Not kept", self.call("remember", about="pip", fact="x" * 101)[1])
        self.assertEqual(rh.memory("400")["facts"][0]["fact"], "is vegetarian")
        self.assertEqual(self.call("forget", number=1), (("forget", 1, "pip: is vegetarian"), "Forgot it."))
        self.assertEqual(self.call("forget", number=1)[1], "No such fact.")
        self.assertIn("didn't work", rh.use(self.log, {"function": {"name": "remember", "arguments": '{"about"'}}, {"n": 1})[1])

    def test_a_wordless_answer_is_nothing_and_a_leading_tag_of_the_asker_goes(self):
        for text in ("(silent)", ".", "*nothing*", "[no reply]"):
            self.assertEqual(rh.said(text, "rocky"), "")
        self.assertEqual(rh.said("rocky: Soup good.", "rocky"), "Soup good.")
        self.assertEqual(rh.unaddressed("@Pip, soup good", "pip"), "soup good")
        self.assertEqual(rh.unaddressed("@Pipsqueak soup", "pip"), "@Pipsqueak soup")
        self.assertEqual(rh.unaddressed("soup @pip", "pip"), "soup @pip")


class TurnTests(_Case):
    async def turn(self, *answers):
        q = self.said("pip", "rocky my cat is called biscuit", 0, to_bot=True)
        llm = AsyncMock(side_effect=list(answers))
        with patch.object(j, "llm_message", llm):
            turn = await rh.take_turn(self.log, q)
        return turn, llm

    async def test_it_speaks_and_acts_in_one_response(self):
        turn, llm = await self.turn(answer("Biscuit. Good name.", [("react", {"emoji": "🐱"}),
                                                                    ("remember", {"about": "pip", "fact": "has a cat called Biscuit"})],
                                           reasoning=[{"type": "reasoning.text", "text": "a cat fact"}]))
        self.assertEqual((turn.text, turn.reactions, turn.kept, turn.requests),
                         ("Biscuit. Good name.", [(1, "🐱")], [("remember", 1, "pip: has a cat called Biscuit")], 1))
        (name, messages, max_tokens), kwargs = llm.call_args
        self.assertEqual((name, max_tokens, kwargs), ("rocky", rh.MAX_TOKENS, {"tools": rh.TOOLS, "reasoning": rh.REASONING}))
        self.assertIn('woke_you="yes"', j.text_of(messages[-1]["content"]))
        rocky = next(e for e in self.log.events if e["type"] == "rocky")
        self.assertIs(turn.spoke, rocky)
        self.assertEqual(rocky["reasoning"], [{"type": "reasoning.text", "text": "a cat fact"}])
        self.assertEqual([e["content"] for e in self.log.events if e["type"] == "result"], ["Reacted.", "Remembered as [1]."])

    async def test_after_only_remembering_it_gets_a_turn_to_speak(self):
        turn, llm = await self.turn(answer(calls=[("remember", {"about": "pip", "fact": "has a cat called Biscuit"})]),
                                    answer("Biscuit. Good name."))
        self.assertEqual((turn.text, turn.requests), ("Biscuit. Good name.", 2))
        self.assertEqual(llm.call_args.args[1][-1], {"role": "tool", "tool_call_id": "call_0", "content": "Remembered as [1]."})

    async def test_only_reacting_or_nothing_is_one_request(self):
        turn, llm = await self.turn(answer(calls=[("react", {"emoji": "😂"})]))
        self.assertEqual((turn.text, turn.reactions, llm.await_count), ("", [(1, "😂")], 1))
        events = len(self.log.events)
        turn, llm = await self.turn(answer("(silent)"))
        self.assertEqual((turn.text, turn.reactions, llm.await_count), ("", [], 1))
        self.assertEqual(len(self.log.events), events + 1)  # just the message — an empty turn isn't kept


class KeepingTests(_Case):
    def channel(self):
        return SimpleNamespace(id=300, name="bot-spam", guild=SimpleNamespace(id=400))

    def test_a_log_comes_back_after_a_restart(self):
        log = rh.log_for(self.channel())
        log.message("pip", "soup?", T0, to_bot=True, discord_id=900)
        log.add("rocky", text="Soup good.", calls=[], id=901, reasoning=[{"type": "reasoning.encrypted", "data": "x"}])
        log.ids["901"] = "you"
        log.asked = T0
        rh.save_log(log)
        rh.logs.clear()
        back = rh.log_for(self.channel())
        self.assertIsNot(back, log)
        self.assertEqual((back.space, back.ids, back.next_n, back.asked), ("400", {"900": 1, "901": "you"}, 2, T0))
        self.assertEqual(rh.render(back, "rocky"), rh.render(log, "rocky"))

    def test_memories_come_back_after_a_restart(self):
        rh.use(self.log, {"function": {"name": "remember", "arguments": '{"about": "pip", "fact": "likes soup"}'}}, {"n": 1})
        rh.save_memories()
        rh.memories.clear()
        rh.memories_loaded = False
        self.assertEqual(rh.memory("400")["facts"][0]["fact"], "likes soup")

    def test_the_first_sync_fills_in_history_before_and_later_ones_append(self):
        log = rh.log_for(self.channel())
        live = log.message("pip", "rocky hi", T0 + timedelta(minutes=5), to_bot=True, discord_id=905)
        history = [{"role": "user", "name": "kettle", "content": "soup?", "at": T0, "id": 900, "to_bot": True,
                    "reply": "Soup good.", "reply_id": 901, "reply_at": T0},
                   {"role": "user", "name": "moss", "content": "lol", "at": T0 + timedelta(minutes=1), "id": 902,
                    "to_bot": True, "reaction": "😂"},
                   {"role": "user", "name": "pip", "content": "rocky hi", "at": live["at"], "id": 905}]
        rh.sync(log, history)
        self.assertEqual([e["type"] for e in log.events], ["message", "rocky", "message", "rocky", "result", "message"])
        self.assertIs(log.events[-1], live)
        self.assertEqual(log.ids["901"], "you")
        before = rh.render(log, "rocky")
        rh.sync(log, history + [{"role": "user", "name": "kettle", "content": "back", "at": T0 + timedelta(minutes=9), "id": 909},
                                {"role": "user", "name": "kettle", "content": "old", "at": T0 - timedelta(days=1), "id": 899}])
        after = rh.render(log, "rocky")
        self.assertEqual([j.text_of(m.get("content")) for m in after[:len(before) - 1]],
                         [j.text_of(m.get("content")) for m in before[:-1]])
        self.assertIn(">back</message>", j.text_of(after[-1]["content"]))  # added at the end, not where it was
        self.assertNotIn(">old</message>", j.text_of(after[-1]["content"]))

    def test_a_deleted_message_is_marked_now_and_blanked_when_the_cache_is_rebuilt(self):
        rh.logs[300] = self.log
        q = self.said("pip", "my password is hunter2", 0, discord_id=900)
        self.log.add("edit", on=q["n"], text="my password is hunter3")
        self.said("kettle", "oops", 1, discord_id=901)
        self.log.add("rocky", text="Hunter. Is name.", calls=[], id=902)
        self.log.ids["902"] = "you"
        rh.use(self.log, {"function": {"name": "remember", "arguments": '{"about": "pip", "fact": "likes soup"}'}}, {"n": 1})
        before = [j.text_of(m.get("content")) for m in rh.render(self.log, "rocky")]
        rh.deleted(300, 900)
        rh.deleted(300, 902)
        rh.deleted(300, 900)  # once is enough
        rh.deleted(300, 12345)  # never in the log
        after = [j.text_of(m.get("content")) for m in rh.render(self.log, "rocky")]
        self.assertEqual(after[:len(before) - 1], before[:-1])  # nothing earlier changes yet
        self.assertIn("hunter2", after[2])  # Rocky still saw it said...
        self.assertTrue(after[-1].endswith('<deleted id="1"/>\n<deleted id="your message: Hunter. Is name."/>'))  # ...and gone
        rh.tidy(self.log)
        tidied = "\n".join(j.text_of(m.get("content")) for m in rh.render(self.log, "rocky"))
        self.assertNotIn("hunter", tidied.lower().replace("hunter. is name", ""))
        self.assertIn('<message id="1" from="pip" at="18:00" deleted="yes"/>', tidied)
        self.assertNotIn('<deleted id="1"/>', tidied)
        self.assertIn("[1] pip: likes soup", tidied)  # <memories> brought up to date at the same time

    async def test_after_an_hour_without_a_request_its_turn_tidies_first(self):
        for idle, tidied in [(timedelta(minutes=50), False), (timedelta(hours=2), True)]:
            with self.subTest(idle=idle):
                self.log = rh.Log("bot-spam", "400", 300)
                rh.logs[300] = self.log
                self.said("pip", "secret", 0, discord_id=900)
                rh.deleted(300, 900)
                self.log.asked = datetime.now(timezone.utc) - idle
                q = self.said("kettle", "rocky hi", 1, to_bot=True)
                llm = AsyncMock(return_value=answer("Hi."))
                with patch.object(j, "llm_message", llm):
                    await rh.take_turn(self.log, q)
                self.assertEqual("secret" not in j.text_of(llm.call_args.args[1][2]["content"]), tidied)
                self.assertGreater(self.log.asked, datetime.now(timezone.utc) - timedelta(minutes=1))

    def test_past_its_size_the_oldest_half_goes_at_a_message(self):
        for i in range(40):
            self.said("pip", "x" * 100, i, discord_id=1000 + i)
            self.log.add("rocky", text="", calls=[{"id": f"c{i}", "name": "react", "arguments": "{}"}])
            self.log.add("result", id=f"c{i}", content="Reacted.")
        rh.use(self.log, {"function": {"name": "remember", "arguments": '{"about": "pip", "fact": "likes soup"}'}}, {"n": 1})
        with patch.object(rh, "LOG_CHARS", rh.chars(self.log) - 1):
            self.assertTrue(rh.jump(self.log))
        self.assertEqual(self.log.events[0]["type"], "message")
        self.assertEqual(len(self.log.events), 60)
        self.assertNotIn("1000", self.log.ids)
        self.assertIn("1039", self.log.ids)
        self.assertIn("likes soup", rh.memories_text(self.log))
        self.assertFalse(rh.jump(self.log))


if __name__ == "__main__":
    unittest.main()
