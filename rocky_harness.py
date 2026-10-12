"""
Rocky as an agent: an LLM is Rocky, the channel is its conversation, and what it writes is posted.

Each channel is an append-only log of events. People's messages, reactions and edits come in as user turns, written as
XML with who, when and what they reply to. Rocky's turns are the assistant's: its text is its message, and its tool
calls (react, remember, forget) are kept with their results. Nothing in the log is rewritten, so every request starts
with the one before it and the model's prompt cache covers all but the newest events. A <memories> block before the
log is only rebuilt at a jump, when the log outgrows LOG_CHARS and its oldest half is dropped.

The way !model's LLMs work so far is a ghostwriter's: the chat as one transcript, then "Write rocky's reply to pip's
last message". That framing had Haiku sometimes describe a reply instead of writing it ("Rocky suggests something
concrete"), and talk about its tools. Here it doesn't write for Rocky; it is Rocky.
"""

import asyncio
import html
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

import aiohttp

import jev_bot as j

LOG_CHARS = 200_000         # a log's size before its oldest half goes — ~90K Haiku tokens, under its 100K price step
MAX_TOKENS = 1000           # room for its notes before tool calls; a reply is still cut to its first paragraph
FOLLOW_UPS = 2              # requests after one that called tools, for what it then says
CACHE = {"type": "ephemeral", "ttl": "1h"}
SAY = True                  # speak with a say tool, and keep its other text to itself (see SPEAKING)

HARNESS = """

This is #{channel} on Discord, as it happens. People's messages come to you as <message> elements: who sent it (from), when (at, {timezone} time), its number (id), and the message it replies to (reply_to). to="you" marks one sent to you — tagging you, or replying to you — and woke_you="yes" the one you're answering now; earlier ones may have been answered already, or have moved on. Reactions and edits show up as they happen. Your own earlier messages are your turns.

Your turn is what you write: it's posted in the channel as your reply to the message that woke you, so write only {bot}'s message, in his voice — never notes about the chat, about what {bot} would say, or about your tools. You can also:
- react: put an emoji on a message. Most of what's said to you that isn't a question or a request — "lol", "welp", "bad rocky", news, moans, thanks, polls — a friend would just react to, not answer: react, and say nothing. Write a message when someone asks you something, wants something from you, or you truly have something to add.
- remember: keep a fact for later chats in this server. About a person — food, pets, jobs, where they live, teams, trips, plans, big news, or anything someone asks you to remember — only from what they or a friend said. About {bot} — what you say you like or are — so you answer the same next time. One short fact each, written to follow its name ("is vegetarian"). Not small talk, not a summary of the chat, nothing already in <memories>.
- forget: drop a fact by its number, when someone says it's wrong, asks you to forget it, or it has changed (then remember the new one).
{speaking}"""

# How it speaks. Writing its message as its reply, and after any tool call in a turn of its own, Haiku still narrated in
# 3 of 57 eval turns ("Pip asked a plain question, so I answer that one"). With say, what it writes outside its tools
# is never posted, so thinking aloud does no harm.
SPEAKING = {
    False: "A response that uses tools is silent: nothing written alongside a tool call is posted. Once they've run you get another turn — write your message then, or nothing at all (after a react, usually nothing). Never mention reacting, remembering or forgetting.",
    True: "To speak, call say with your message — it's posted as your reply to the message that woke you, so don't start it with @ their name. Say one message at most. Nothing you write outside your tools is posted, so you can think there; your message itself is only {bot}'s words, never mention reacting, remembering or forgetting.",
}

SAY_TOOL = {"type": "function", "function": {
    "name": "say", "description": "Post your message in the channel, as your reply to the message that woke you.",
    "parameters": {"type": "object", "required": ["text"], "properties": {
        "text": {"type": "string", "description": "What you say, in your voice"}}}}}

TOOLS = [
    {"type": "function", "function": {
        "name": "react", "description": "Put an emoji on a message.",
        "parameters": {"type": "object", "required": ["emoji"], "properties": {
            "emoji": {"type": "string", "description": "One emoji"},
            "message": {"type": "integer", "description": "The message's id; leave out for the one that woke you"}}}}},
    {"type": "function", "function": {
        "name": "remember", "description": "Keep a fact for later chats in this server.",
        "parameters": {"type": "object", "required": ["about", "fact"], "properties": {
            "about": {"type": "string", "description": "The person's name as it is in the chat, your own name, or a short topic"},
            "fact": {"type": "string", "description": "Short, to follow the name: \"is vegetarian\""}}}}},
    {"type": "function", "function": {
        "name": "forget", "description": "Drop a remembered fact.",
        "parameters": {"type": "object", "required": ["number"], "properties": {
            "number": {"type": "integer", "description": "The fact's number in <memories>"}}}}},
]


# What happened in a channel, oldest first. Events are dicts with a "type":
#   message:  {"n", "author", "uid", "at", "text", "to_bot", "reply_to": n or "you"}  — "you": one of Rocky's
#   reaction: {"on": n or "you", "emoji", "author", "at"}
#   edit:     {"on": n, "text", "at"}
#   rocky:    {"text", "calls": [{"id", "name", "arguments"}]}  — one of Rocky's turns
#   result:   {"id", "content"}  — what came of a call, right after its turn
@dataclass
class Log:
    channel: str
    events: list = field(default_factory=list)
    next_n: int = 1
    memories: list = field(default_factory=list)  # [{"n", "about", "fact"}], kept apart from the log
    next_memory: int = 1

    def message(self, author, text, at, to_bot=False, reply_to=None, uid=None):
        e = {"type": "message", "n": self.next_n, "author": author, "uid": uid, "at": at, "text": text,
             "to_bot": to_bot, "reply_to": reply_to}
        self.next_n += 1
        self.events.append(e)
        return e

    def add(self, type, **fields):
        self.events.append({"type": type, **fields})


def attr(v):
    return html.escape(str(v), quote=True)

def when(at):
    return at.astimezone(j.TIMEZONE)

def snippet(text, n=40):
    return text if len(text) <= n else text[:n - 1] + "…"

# People's events since Rocky's last turn, as one user message: a <day> line when the date changes, then a line each
def people_turn(events, clock, rocky_said):
    lines = []
    for e in events:
        if e["type"] == "message":
            if (day := f"{when(e['at']):%a} {when(e['at']).day} {when(e['at']):%b}") != clock.get("day"):
                clock["day"] = day
                lines.append(f"<day>{day}</day>")
            to = (' to="you" woke_you="yes"' if e.get("woke") else ' to="you"' if e["to_bot"] else "")
            reply = ""
            if e["reply_to"] == "you":
                reply = f' reply_to="your message: {attr(snippet(rocky_said.get("last", "")))}"'
            elif e["reply_to"]:
                reply = f' reply_to="{e["reply_to"]}"'
            lines.append(f'<message id="{e["n"]}" from="{attr(e["author"])}" at="{when(e["at"]):%H:%M}"{to}{reply}>'
                         f'{html.escape(e["text"], quote=False)}</message>')
        elif e["type"] == "reaction":
            on = (f'your message: {snippet(rocky_said.get("last", ""))}' if e["on"] == "you" else e["on"])
            lines.append(f'<reaction from="{attr(e["author"])}" emoji="{attr(e["emoji"])}" on="{attr(on)}"/>')
        elif e["type"] == "edit":
            lines.append(f'<edit id="{e["on"]}">{html.escape(e["text"], quote=False)}</edit>')
    return "\n".join(lines)

def memories_text(log):
    if not log.memories:
        return "<memories>\nNothing yet.\n</memories>"
    return "<memories>\n" + "\n".join(f"[{m['n']}] {m['about']}: {m['fact']}" for m in log.memories) + "\n</memories>"

# The request: system (persona and how this works), <memories>, then the log as turns. Cache marks on the system
# prompt, the memories, and the end of the request — the next request starts with all of it.
def render(log, bot_name, cache=True):
    system = j.rocky_prompt(bot_name) + HARNESS.format(channel=log.channel, timezone=j.TIMEZONE.key, bot=bot_name,
                                                       speaking=SPEAKING[SAY].format(bot=bot_name))
    mark = lambda text: [{"type": "text", "text": text, **({"cache_control": CACHE} if cache else {})}]
    messages = [{"role": "system", "content": mark(system) if cache else system},
                {"role": "user", "content": mark(memories_text(log)) if cache else memories_text(log)}]
    pending, clock, rocky_said = [], {}, {}

    def flush():
        if pending:
            messages.append({"role": "user", "content": people_turn(pending, clock, rocky_said)})
            pending.clear()

    for e in log.events:
        if e["type"] == "rocky":
            flush()
            if SAY and e["text"]:  # an earlier message, as the say call it would have been
                said_id = f"say_{len(messages)}"
                messages.append({"role": "assistant", "content": None, "tool_calls": [
                    {"id": said_id, "type": "function", "function": {"name": "say", "arguments": json.dumps({"text": e["text"]})}}
                    ] + [{"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}}
                         for c in e["calls"]]})
                messages.append({"role": "tool", "tool_call_id": said_id, "content": "Posted."})
                rocky_said["last"] = e["text"]
                continue
            m = {"role": "assistant", "content": e["text"] or None}
            if e["calls"]:
                m["tool_calls"] = [{"id": c["id"], "type": "function",
                                    "function": {"name": c["name"], "arguments": c["arguments"]}} for c in e["calls"]]
            messages.append(m)
            if e["text"]:
                rocky_said["last"] = e["text"]
        elif e["type"] == "result":
            messages.append({"role": "tool", "tool_call_id": e["id"], "content": e["content"]})
        else:
            pending.append(e)
    flush()
    if cache and messages[-1]["role"] == "user" and isinstance(messages[-1]["content"], str):
        messages[-1] = {"role": "user", "content": mark(messages[-1]["content"])}
    return messages


@dataclass
class Turn:
    text: str = ""
    reactions: list = field(default_factory=list)  # [(message n or None for the waking one, emoji)]
    kept: list = field(default_factory=list)       # what remember and forget did
    requests: int = 0
    cost: float = 0.0
    prompt_tokens: int = 0
    cached_tokens: int = 0
    raw: list = field(default_factory=list)        # each response's message, for looking into


async def complete(session, model, messages, turn):
    spec = j.LLMS[model]
    body = {"model": spec["id"], "messages": messages, "tools": [SAY_TOOL, *TOOLS] if SAY else TOOLS, "max_tokens": MAX_TOKENS,
            "temperature": j.LLM_TEMPERATURE, "reasoning": {"enabled": False}, "usage": {"include": True}}
    if providers := spec.get("providers"):
        body["provider"] = {"order": providers, "require_parameters": True}
    for attempt in range(3):
        async with session.post(j.LLM_URL, json=body, timeout=aiohttp.ClientTimeout(total=60)) as r:
            data = await r.json(content_type=None)
        if r.status < 400 and "choices" in data:
            break
        j.log.warning(f"[HARNESS] {r.status} {attempt}: {str(data)[:300]}")
        await asyncio.sleep(1 + 2 * attempt)
    else:
        raise RuntimeError(f"no answer from {model}")
    usage = data.get("usage") or {}
    turn.requests += 1
    turn.cost += (usage.get("cost") or 0) + (usage.get("is_byok") and
                                             (usage.get("cost_details") or {}).get("upstream_inference_cost") or 0)
    turn.prompt_tokens += usage.get("prompt_tokens") or 0
    turn.cached_tokens += (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    choice = data["choices"][0]
    if choice.get("finish_reason") == "length":
        j.log.warning(f"[HARNESS] {model} ran out of tokens: {str(choice['message'])[:300]}")
    turn.raw.append(choice["message"])
    return choice["message"]

def use(log, call, waking):
    f = call.get("function") or {}
    try:
        args = json.loads(f.get("arguments") or "{}")
        if f.get("name") == "react":
            return ("react", args.get("message") or waking["n"], args["emoji"]), "Reacted. That's usually the whole turn."
        if f.get("name") == "remember":
            m = {"n": log.next_memory, "about": args["about"].strip().lstrip("@"), "fact": args["fact"].strip().rstrip(".")}
            log.next_memory += 1
            log.memories.append(m)
            return ("remember", m["n"], f"{m['about']}: {m['fact']}"), f"Remembered as [{m['n']}]."
        if f.get("name") == "forget":
            n = int(args["number"])
            gone = next((m for m in log.memories if m["n"] == n), None)
            if gone:
                log.memories.remove(gone)
            return ("forget", n, gone and f"{gone['about']}: {gone['fact']}"), "Forgot it." if gone else "No such fact."
        return None, f"No tool called {f.get('name')!r}."
    except (ValueError, KeyError, TypeError) as e:
        return None, f"That didn't work: {e!r}"

# What it wrote, as a message — or nothing: told it can say nothing after a react, Haiku wrote "(silent)" or "."
def said(content, bot_name):
    text = j.clean_llm(content or "", bot_name)
    return "" if not re.search(r"\w", text) or re.fullmatch(r"[(\[*].*[)\]*]", text) else text

# Rocky's turn after `waking` (already in the log): a request, and after any that called tools, another for what it
# says. Text alongside a tool call is the model thinking aloud ("A single lol, a quick react fits better") — it's
# neither posted nor kept, or it would see itself doing it.
async def take_turn(log, waking, model="haiku", bot_name="rocky", session=None):
    turn = Turn()
    for e in log.events:
        e.pop("woke", None)
    waking["woke"] = True
    own = session is None
    session = session or aiohttp.ClientSession(headers=j.HEADERS)
    try:
        for _ in range(1 + FOLLOW_UPS):
            message = await complete(session, model, render(log, bot_name, cache=bool(j.LLMS[model].get("cache_marks"))),
                                     turn)
            calls = [{"id": c["id"], "name": c["function"]["name"], "arguments": c["function"]["arguments"] or "{}"}
                     for c in message.get("tool_calls") or []]
            if SAY:  # one request: what it says is its say call; its other text was thinking, and isn't kept
                spoken = [c for c in calls if c["name"] == "say"][:1]
                # No tool at all: its text is its reply — 3 of 57 eval turns, all in Rocky's voice. With the tools
                # required instead, half its say calls came back with no text ({}).
                text = "" if calls else said(message.get("content"), bot_name)
                if spoken:
                    try:
                        text = said(json.loads(spoken[0]["arguments"]).get("text"), bot_name)
                    except ValueError:
                        pass
                others = [c for c in calls if c["name"] != "say"]
                if not text and not others:
                    break  # nothing said or done — an empty turn isn't one the model can be shown later
                log.add("rocky", text=text, calls=others)
                for c in others:
                    done, result = use(log, {"function": {"name": c["name"], "arguments": c["arguments"]}}, waking)
                    log.add("result", id=c["id"], content=result)
                    if done and done[0] == "react":
                        turn.reactions.append(done[1:])
                    elif done:
                        turn.kept.append(done)
                turn.text = text
                break
            if not calls:
                if text := said(message.get("content"), bot_name):
                    log.add("rocky", text=text, calls=[])
                    turn.text = text
                break
            log.add("rocky", text="", calls=calls)
            for c in calls:
                done, result = use(log, {"function": {"name": c["name"], "arguments": c["arguments"]}}, waking)
                log.add("result", id=c["id"], content=result)
                if done and done[0] == "react":
                    turn.reactions.append(done[1:])
                elif done:
                    turn.kept.append(done)
    finally:
        if own:
            await session.close()
    return turn


# A log from the bot's history entries (jev_bot's, or jev_eval's scenarios): people's messages, Rocky's replies under
# them, its reactions, and people's reactions to its replies. Rocky's reactions become react calls, as if it had made
# them here.
def from_history(channel, history, bot_name="rocky"):
    log = Log(channel)
    calls = 0
    for h in history or []:
        if h["role"] == "assistant":
            log.add("rocky", text=h["content"], calls=[])
            for emoji, n in (h.get("reactions") or {}).items():
                for name in list((h.get("reactors") or {}).get(emoji, {}).values())[:n] or ["someone"] * n:
                    log.add("reaction", on="you", emoji=emoji, author=name, at=h.get("at"))
            continue
        e = log.message(h["name"], j.unrender(h["content"]), h.get("at") or datetime.now(timezone.utc),
                        to_bot=h.get("to_bot", True))
        if "reply" in h:
            log.add("rocky", text=h["reply"], calls=[])
            for emoji, n in (h.get("reply_reactions") or {}).items():
                names = list((h.get("reply_reactors") or {}).get(emoji, {}).values())
                for name in names[:n] or ["someone"] * n:
                    log.add("reaction", on="you", emoji=emoji, author=name, at=h.get("at"))
        elif "reaction" in h:
            calls += 1
            log.add("rocky", text="", calls=[{"id": f"call_{calls}", "name": "react",
                                              "arguments": json.dumps({"emoji": h["reaction"], "message": e["n"]})}])
            log.add("result", id=f"call_{calls}", content="Reacted.")
    return log
