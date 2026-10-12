"""
Rocky as an agent: an LLM is Rocky, the channel is its conversation, and what it writes is posted. !model rocky, in the
channels in LONG_CONTEXT_CHANNEL_IDS.

Each channel is an append-only log of events. People's messages, reactions and edits come in as user turns, written as
XML with who, when and what they reply to. Rocky's turns are the assistant's: it thinks, then its text is its message,
and its tool calls (react, remember, forget) are kept with their results. Nothing in the log is rewritten, so every
request starts with the one before it and the model's prompt cache covers all but the newest events. A <memories>
block before the log is a snapshot, rebuilt — and deleted messages blanked — only when the cache would be rebuilt
anyway (tidy): at a jump, when the log outgrows LOG_CHARS and its oldest half goes, or after CACHE_HOURS without a
request.

The way !model's LLMs work elsewhere is a ghostwriter's: the chat as one transcript, then "Write rocky's reply to pip's
last message". That framing had Haiku sometimes describe a reply instead of writing it ("Rocky suggests something
concrete"), and talk about its tools. Here it doesn't write for Rocky; it is Rocky.

jev_bot hands itself over with bind() — run as a script it's __main__, so importing it here would load a second bot.
"""

import asyncio
import contextlib
import html
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

j = None

def bind(bot_module):
    global j
    j = bot_module

LOG_CHARS = 200_000         # a log's size before its oldest half goes — ~90K Haiku tokens, under its 100K price step
MAX_TOKENS = 4000           # its thinking, its message and its tool calls; the message is cut to its first paragraph
FOLLOW_UPS = 1              # requests after one that only kept or dropped memories, for what it then says
CACHE = {"type": "ephemeral", "ttl": "1h"}
CACHE_HOURS = 1             # CACHE's lifetime — after this long without a request, the next is read in full anyway
FACT_CHARS = 100            # a remembered fact's longest
SAVE_AFTER = 2              # seconds after a change before a log or the memories are written, so a burst writes once
# It thinks first. Without thinking, Haiku narrated in 3 of 57 eval turns ("Pip asked a plain question, so I answer
# that one"); given a say tool for its message instead, it didn't, but thinking with that tool it sometimes did nothing
# (5 of 57). Thinking and speaking its turn: no narration in 57, reply-or-react right as often, for 0.021¢ a turn
# where no thinking was 0.015¢, and ~2s slower.
REASONING = {"effort": "low"}
LOG_DIR = Path(__file__).parent / "rocky_logs"          # a channel's log each, gitignored
MEMORY_PATH = Path(__file__).parent / "rocky_memories.json"

HARNESS = """

This is #{channel} on Discord, as it happens. People's messages come to you as <message> elements: who sent it (from), when (at, {timezone} time), its number (id), and the message it replies to (reply_to). to="you" marks one sent to you — tagging you, or replying to you. woke_you="yes" marks each that woke you, and the newest of those is the one you're answering now; older ones have had their turn. Reactions, edits and deletions show up as they happen. Your own earlier messages are your turns.

Your turn is what you write: it's posted in the channel as your reply to the message that woke you. So write only {bot}'s message, in his voice — never notes about the chat or about what {bot} would say — and don't start it with @ their name. You can also use tools, alongside your message or instead of it. Write your message and make every tool call in the same response — don't wait to see what a tool did before you speak:
- react: put an emoji on a message. Most of what's said to you that isn't a question or a request — "lol", "welp", "bad rocky", news, moans, thanks, polls — a friend would just react to, not answer: react, and write nothing. But a question to you always gets an answer in words (a reaction as well, if you like), and so does a request; write a message too when you truly have something to add.
- remember: keep a fact for later chats in this server — whenever a message to you has one, as well as answering it. About a person — food, pets, jobs, where they live, teams, trips, plans, big news, or anything someone asks you to remember — only from what they or a friend said. About {bot} — what you say you like or are — so you answer the same next time. One short fact each, written to follow its name ("is vegetarian"). Not small talk, not a summary of the chat, nothing already in <memories>.
- forget: drop a fact by its number, when someone says it's wrong, asks you to forget it, or it has changed — and then remember the new one in the same response.
What you've remembered is in <memories>, and in your remember calls since. Never mention reacting, remembering or forgetting in your message."""

TOOLS = [
    {"type": "function", "function": {
        "name": "react", "description": "Put an emoji on a message.",
        "parameters": {"type": "object", "required": ["emoji"], "properties": {
            "emoji": {"type": "string", "description": "One emoji, or a server emoji by name, like :blobwave:"},
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


# Memories: facts kept with remember, a list for each server (and each DM) — every channel of a server shares them.
# {space: {"next": the next number, "facts": [{"n", "about", "fact", "at"}]}}, in MEMORY_PATH.
memories: dict[str, dict] = {}
memories_loaded = False
memories_save = None

def memory(space):
    global memories_loaded
    if not memories_loaded:
        memories_loaded = True
        try:
            memories.update(json.loads(MEMORY_PATH.read_text()))
        except FileNotFoundError:
            pass
        except Exception as e:  # unreadable — start again, and keep the file for a look
            j.log.warning(f"Loading {MEMORY_PATH.name} failed: {e}")
    return memories.setdefault(space, {"next": 1, "facts": []})

def save_memories():
    global memories_save
    memories_save = None
    try:
        tmp = MEMORY_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(memories, ensure_ascii=False, indent=1) + "\n")
        tmp.replace(MEMORY_PATH)
    except OSError as e:
        j.log.warning(f"Writing {MEMORY_PATH.name} failed: {e}")

def memories_changed():
    global memories_save
    if memories_save is None:
        try:
            memories_save = asyncio.get_running_loop().call_later(SAVE_AFTER, save_memories)
        except RuntimeError:  # no loop (a script) — now
            save_memories()


# What happened in a channel, oldest first. Events are dicts with a "type" and "at":
#   message:  {"n", "author", "uid", "text", "to_bot", "reply_to": n or "you", "you_said": that message of Rocky's,
#              "woke": it woke Rocky}
#   reaction: {"on": n or "you", "you_said", "emoji", "author", "removed"}
#   edit:     {"on": n, "text"}
#   deleted:  {"on": n or "you", "you_said"}  — and the message gets "deleted"; tidy() then blanks it ("gone")
#   rocky:    {"text", "calls": [{"id", "name", "arguments"}], "reasoning", "id": its Discord message}
#   result:   {"id", "content"}  — what came of a call, right after its turn
# ids: Discord message id to n, or to "you" for Rocky's own.
@dataclass
class Log:
    channel: str
    space: str = "test"
    channel_id: int = 0
    events: list = field(default_factory=list)
    next_n: int = 1
    ids: dict = field(default_factory=dict)
    shown: list = field(default_factory=list)  # the facts in <memories>: as they were at the last jump
    seeded: bool = False                       # filled from the channel's history, the first time it was needed
    asked: datetime = None                     # the last request, to know when the cache has gone

    def message(self, author, text, at, to_bot=False, reply_to=None, uid=None, discord_id=None, you_said=None):
        e = {"type": "message", "n": self.next_n, "author": author, "uid": uid, "at": at, "text": text,
             "to_bot": to_bot, "reply_to": reply_to, "you_said": you_said}
        self.next_n += 1
        self.events.append(e)
        if discord_id:
            self.ids[str(discord_id)] = e["n"]
        return e

    def add(self, type, **fields):
        e = {"type": type, "at": fields.pop("at", None) or datetime.now(timezone.utc), **fields}
        self.events.append(e)
        return e

    # <memories> as they are now: at a jump, or when someone wants a fact really gone. Everything after it in the
    # request is then read at full price once, so not for every remember — those are in the log as calls.
    def reorganise(self):
        self.shown = [dict(f) for f in memory(self.space)["facts"]]

    def discord_id(self, n):
        return next((int(i) for i, v in self.ids.items() if v == n), None)

    def rocky_text(self, discord_id):
        return next((e["text"] for e in reversed(self.events) if e["type"] == "rocky" and e.get("id") == discord_id), "")


def attr(v):
    return html.escape(str(v), quote=True)

def when(at):
    return at.astimezone(j.TIMEZONE)

def snippet(text, n=40):
    return text if len(text) <= n else text[:n - 1] + "…"

# People's events since Rocky's last turn, as one user message: a <day> line when the date changes, then a line each
def people_turn(events, clock):
    lines = []
    for e in events:
        if e["type"] == "message":
            if (day := f"{when(e['at']):%a} {when(e['at']).day} {when(e['at']):%b}") != clock.get("day"):
                clock["day"] = day
                lines.append(f"<day>{day}</day>")
            to = (' to="you" woke_you="yes"' if e.get("woke") else ' to="you"' if e["to_bot"] else "")
            reply = ""
            if e["reply_to"] == "you":
                reply = f' reply_to="your message: {attr(snippet(e.get("you_said") or ""))}"'
            elif e["reply_to"]:
                reply = f' reply_to="{e["reply_to"]}"'
            head = f'<message id="{e["n"]}" from="{attr(e["author"])}" at="{when(e["at"]):%H:%M}"{to}{reply}'
            lines.append(f'{head} deleted="yes"/>' if e.get("gone") else
                         f'{head}>{html.escape(e["text"], quote=False)}</message>')
        elif e["type"] == "reaction":
            on = f'your message: {snippet(e.get("you_said") or "")}' if e["on"] == "you" else e["on"]
            removed = ' removed="yes"' if e.get("removed") else ""
            lines.append(f'<reaction from="{attr(e["author"])}" emoji="{attr(e["emoji"])}" on="{attr(on)}"{removed}/>')
        elif e["type"] == "edit":
            lines.append(f'<edit id="{e["on"]}">{html.escape(e["text"], quote=False)}</edit>')
        elif e["type"] == "deleted":
            on = f'your message: {snippet(e.get("you_said") or "")}' if e["on"] == "you" else e["on"]
            lines.append(f'<deleted id="{attr(on)}"/>')
    return "\n".join(lines)

def memories_text(log):
    if not log.shown:
        return "<memories>\nNothing yet.\n</memories>"
    return "<memories>\n" + "\n".join(f"[{f['n']}] {f['about']}: {f['fact']}" for f in log.shown) + "\n</memories>"

# The request: system (persona and how this works), <memories>, then the log as turns. Cache marks on the system
# prompt, the memories, and the end of the request — the next request starts with all of it.
def render(log, bot_name, cache=True):
    system = j.rocky_prompt(bot_name) + HARNESS.format(channel=log.channel, timezone=j.TIMEZONE.key, bot=bot_name)
    mark = lambda text: [{"type": "text", "text": text, **({"cache_control": CACHE} if cache else {})}]
    messages = [{"role": "system", "content": mark(system) if cache else system},
                {"role": "user", "content": mark(memories_text(log)) if cache else memories_text(log)}]
    pending, clock = [], {}

    def flush():
        if pending:
            messages.append({"role": "user", "content": people_turn(pending, clock)})
            pending.clear()

    for e in log.events:
        if e["type"] == "rocky":
            flush()
            m = {"role": "assistant", "content": e["text"] or None}
            if e.get("reasoning"):  # its thinking, back as it came — signed, and not to be edited
                m["reasoning_details"] = e["reasoning"]
            if e["calls"]:
                m["tool_calls"] = [{"id": c["id"], "type": "function",
                                    "function": {"name": c["name"], "arguments": c["arguments"]}} for c in e["calls"]]
            messages.append(m)
        elif e["type"] == "result":
            messages.append({"role": "tool", "tool_call_id": e["id"], "content": e["content"]})
        else:
            pending.append(e)
    flush()
    # The last of people's turns: the end of the request, or, in the request after a tool call, where the one before
    # ended — marked only when it was last, that request after a tool call read only the system prompt and memories
    # from the cache, and paid for the whole log (~0.47¢ for 47K tokens, where a turn was ~0.05¢)
    last = max((i for i, m in enumerate(messages) if m["role"] == "user" and isinstance(m["content"], str)), default=None)
    if cache and last is not None:
        messages[last] = {"role": "user", "content": mark(messages[last]["content"])}
    return messages

# The log as text, for Context and the logs: what it saw, without the system prompt
def readable(log, bot_name, last=None):
    messages = render(log, bot_name, cache=False)[1:]
    lines = []
    for m in messages[-last:] if last else messages:
        if m["role"] == "user":
            lines.append(j.text_of(m["content"]))
        elif m["role"] == "assistant":
            calls = " ".join(f"[{c['function']['name']} {c['function']['arguments']}]" for c in m.get("tool_calls") or [])
            lines.append(f"{bot_name}: {m['content'] or ''} {calls}".rstrip())
        else:
            lines.append(f"[{m['content']}]")
    return "\n".join(lines)


@dataclass
class Turn:
    text: str = ""
    reactions: list = field(default_factory=list)  # [(message n, emoji)]
    kept: list = field(default_factory=list)       # what remember and forget did
    spoke: dict = None                             # the log's event for its message, to give it its Discord id
    requests: int = 0
    cost: float = 0.0
    prompt_tokens: int = 0
    cached_tokens: int = 0
    raw: list = field(default_factory=list)        # each response's message, for looking into


def use(log, call, waking):
    f = call.get("function") or {}
    try:
        args = json.loads(f.get("arguments") or "{}")
        if f.get("name") == "react":
            return ("react", args.get("message") or waking["n"], args["emoji"]), "Reacted."
        if f.get("name") == "remember":
            kept = memory(log.space)
            fact = {"n": kept["next"], "about": args["about"].strip().lstrip("@"), "fact": args["fact"].strip().rstrip("."),
                    "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            # Its thinking once spilled into a fact ("favourite film is Rocky — no, Project Hail Mary? Avoid; keep
            # simple"), and was kept twice
            if len(fact["fact"]) > FACT_CHARS or "?" in fact["fact"]:
                return None, f"Not kept: one short, settled fact, under {FACT_CHARS} characters."
            if same := next((x for x in kept["facts"] if (x["about"].lower(), x["fact"].lower()) ==
                             (fact["about"].lower(), fact["fact"].lower())), None):
                return None, f"Already remembered as [{same['n']}]."
            kept["next"] += 1
            kept["facts"].append(fact)
            memories_changed()
            return ("remember", fact["n"], f"{fact['about']}: {fact['fact']}"), f"Remembered as [{fact['n']}]."
        if f.get("name") == "forget":
            n, facts = int(args["number"]), memory(log.space)["facts"]
            gone = next((x for x in facts if x["n"] == n), None)
            if gone:
                facts.remove(gone)
                memories_changed()
            return ("forget", n, gone and f"{gone['about']}: {gone['fact']}"), "Forgot it." if gone else "No such fact."
        return None, f"No tool called {f.get('name')!r}."
    except (ValueError, KeyError, TypeError) as e:
        return None, f"That didn't work: {e!r}"

# What it wrote, as a message — or nothing: told it can say nothing, Haiku has written "(silent)" or "."
def said(content, bot_name):
    text = j.clean_llm(content or "", bot_name)
    return "" if not re.search(r"\w", text) or re.fullmatch(r"[(\[*].*[)\]*]", text) else text

# Rocky's turn after `waking` (already in the log): its message and tool calls, usually in one response. Only a turn
# that just kept or dropped memories gets another request, for what it then says. Its requests go through
# jev_bot.llm_message, which counts their cost on the trace.
# typing: the channel's typing() — shown only for that second request. Discord shows "rocky is typing" for ~10s after
# it's sent, and nothing stops it but a message, so shown while it decided, it lingered after a turn that only reacted.
async def take_turn(log, waking, model="rocky", bot_name="rocky", typing=None):
    turn = Turn()
    waking["woke"] = True  # and stays, so the turn it's in reads the same in every later request
    now = datetime.now(timezone.utc)
    if log.asked and (now - log.asked).total_seconds() > CACHE_HOURS * 3600:
        tidy(log)  # the cache has gone, so this request is read in full anyway
    log.asked = now
    token = j.trace.set({"cost": 0.0, "requests": 0}) if j.trace.get() is None else None
    t = j.trace.get()
    cost = t["cost"]
    try:
        for request in range(1 + FOLLOW_UPS):
            async with typing() if typing and request else contextlib.nullcontext():
                message = await j.llm_message(model, render(log, bot_name, cache=bool(j.LLMS[model].get("cache_marks"))),
                                              MAX_TOKENS, tools=TOOLS, reasoning=REASONING)
            if message is None:
                break
            turn.requests += 1
            turn.prompt_tokens += t.get("llm_prompt_tokens") or 0
            turn.cached_tokens += t.get("llm_cached_tokens") or 0
            turn.raw.append(message)
            calls = [{"id": c["id"], "name": c["function"]["name"], "arguments": c["function"]["arguments"] or "{}"}
                     for c in message.get("tool_calls") or []]
            text = said(message.get("content"), bot_name)
            if not text and not calls:
                break  # nothing said or done — an empty turn isn't one the model can be shown later
            e = log.add("rocky", text=text, calls=calls, reasoning=message.get("reasoning_details"))
            for c in calls:
                done, result = use(log, {"function": {"name": c["name"], "arguments": c["arguments"]}}, waking)
                log.add("result", id=c["id"], content=result)
                if done and done[0] == "react":
                    turn.reactions.append(done[1:])
                elif done:
                    turn.kept.append(done)
            if text:
                turn.text, turn.spoke = text, e
                break
            if all(c["name"] == "react" for c in calls):
                break
        turn.cost = t["cost"] - cost
    finally:
        if token is not None:
            j.trace.reset(token)
    return turn

# The log's oldest half goes once it's over LOG_CHARS: everything before a message, so no turn loses its results.
# <memories> is rebuilt then, so what was remembered in the dropped half isn't lost with it.
def chars(log):
    return sum(len(e.get("text") or "") + 60 + sum(len(c["arguments"]) for c in e.get("calls", [])) for e in log.events)

def jump(log):
    if chars(log) <= LOG_CHARS:
        return False
    cut = next((i for i in range(len(log.events) // 2, len(log.events)) if log.events[i]["type"] == "message"), None)
    if cut is None:
        return False
    log.events[:] = log.events[cut:]
    kept = {e["n"] for e in log.events if e["type"] == "message"}
    log.ids = {i: n for i, n in log.ids.items() if n == "you" or n in kept}
    tidy(log)
    j.log.info(f"[ROCKY] #{log.channel} jumped to {len(log.events)} events")
    return True

# What waits for the cache to be rebuilt anyway: <memories> as they are now, and deleted messages blanked, their
# edits and <deleted> marks gone with them. Until then Rocky has seen what was said, and that it was deleted.
def tidy(log):
    gone = {e["n"] for e in log.events if e["type"] == "message" and e.get("deleted")}
    for e in log.events:
        if e["type"] == "message" and e["n"] in gone:
            e["gone"], e["text"] = True, ""
    log.events[:] = [e for e in log.events if not (e["type"] in ("edit", "deleted") and e["on"] in gone)]
    log.reorganise()
    changed(log)


# Live logs, one for each channel in LONG_CHAT_CHANNELS, kept in LOG_DIR
logs: dict[int, Log] = {}
log_saves: dict[int, object] = {}

def plain(v):
    return v.isoformat() if isinstance(v, datetime) else v

def save_log(log):
    log_saves.pop(log.channel_id, None)
    saved = {"channel": log.channel, "space": log.space, "next_n": log.next_n, "ids": log.ids, "shown": log.shown,
             "seeded": log.seeded, "asked": plain(log.asked),
             "events": [{k: plain(v) for k, v in e.items()} for e in log.events]}
    try:
        LOG_DIR.mkdir(exist_ok=True)
        path = LOG_DIR / f"{log.channel_id}.json"
        path.with_suffix(".tmp").write_text(json.dumps(saved, ensure_ascii=False) + "\n")
        path.with_suffix(".tmp").replace(path)
    except OSError as e:
        j.log.warning(f"Writing the log for #{log.channel} failed: {e}")

def changed(log):
    if log.channel_id not in log_saves:
        try:
            log_saves[log.channel_id] = asyncio.get_running_loop().call_later(SAVE_AFTER, save_log, log)
        except RuntimeError:
            save_log(log)

# Everything waiting to be written, now: at shutdown, so a turn just before a restart isn't lost — the next request
# would then differ from the last, and read only what came before it from the cache
def flush():
    for log in [log for log in logs.values() if log.channel_id in log_saves]:
        log_saves[log.channel_id].cancel()
        save_log(log)
    if memories_save is not None:
        memories_save.cancel()
        save_memories()

def log_for(channel):
    if (log := logs.get(channel.id)) is not None:
        return log
    space = str(channel.guild.id) if getattr(channel, "guild", None) else f"dm-{channel.id}"
    log = Log(getattr(channel, "name", None) or "DM", space, channel.id)
    try:
        saved = json.loads((LOG_DIR / f"{channel.id}.json").read_text())
        log.next_n, log.ids, log.shown, log.seeded = saved["next_n"], saved["ids"], saved["shown"], saved["seeded"]
        log.asked = saved.get("asked") and datetime.fromisoformat(saved["asked"])
        log.events = [e | {"at": datetime.fromisoformat(e["at"])} if e.get("at") else e for e in saved["events"]]
    except FileNotFoundError:
        pass
    except Exception as e:  # unreadable — start again, from the channel's history
        j.log.warning(f"Loading the log for #{log.channel} failed: {e}")
    logs[channel.id] = log
    return log

# A message in the channel, as it arrives: people's and other apps' — not Rocky's own (its turns add those), commands,
# or side conversations (jev_bot keeps those out). Returns its event, or None.
def observe(m):
    log = log_for(m.channel)
    if (n := log.ids.get(str(m.id))) is not None:
        return next((e for e in log.events if e.get("n") == n), None)
    if m.author.id == j.bot.user.id or j.command(m) or not (text := j.message_text(m)):
        return None
    reply_to = you_said = None
    if m.reference and m.reference.message_id:
        reply_to = log.ids.get(str(m.reference.message_id))
        if reply_to == "you":
            you_said = log.rocky_text(m.reference.message_id)
    e = log.message(m.author.display_name, text, m.created_at, to_bot=j.should_respond(m), reply_to=reply_to,
                    uid=m.author.id, discord_id=m.id, you_said=you_said)
    changed(log)
    return e

# Someone's reaction on a message in a log, added or taken off
def reacted(channel_id, message_id, emoji, author, added=True):
    if (log := logs.get(channel_id)) is None or (on := log.ids.get(str(message_id))) is None:
        return
    log.add("reaction", on=on, you_said=log.rocky_text(message_id) if on == "you" else None, emoji=emoji,
            author=author or "someone", removed=not added)
    changed(log)

def edited(m):
    log = logs.get(m.channel.id)
    if log is None or not isinstance(n := log.ids.get(str(m.id)), int):
        return
    text = j.message_text(m)
    last = next((e["text"] for e in reversed(log.events)
                 if e["type"] == "message" and e["n"] == n or e["type"] == "edit" and e["on"] == n), None)
    if text and text != last:
        log.add("edit", on=n, text=text)
        changed(log)

# A message deleted: marked now, and blanked at the next tidy
def deleted(channel_id, message_id):
    if (log := logs.get(channel_id)) is None or (on := log.ids.get(str(message_id))) is None:
        return
    if on == "you":
        log.add("deleted", on="you", you_said=log.rocky_text(message_id))
    else:
        message = next((e for e in log.events if e["type"] == "message" and e["n"] == on), None)
        if message is None or message.get("deleted"):
            return
        message["deleted"] = True  # not shown until tidy, so nothing earlier in the request changes yet
        log.add("deleted", on=on)
    changed(log)

# One of Rocky's messages from outside a turn (written while another !model was chosen), so the log has it
def spoke_elsewhere(channel_id, text, sent):
    if (log := logs.get(channel_id)) is None:
        return
    log.add("rocky", text=text, calls=[], id=sent.id, at=sent.created_at)
    log.ids[str(sent.id)] = "you"
    changed(log)

# The first time a channel's log is needed, what the channel's history already holds goes in before what was seen live;
# after that (a restart), messages it never saw are added at the end. entries: jev_bot's history entries, oldest
# first. Nothing already in the log changes.
def sync(log, entries):
    missing = [h for h in entries if h.get("id") and str(h["id"]) not in log.ids and h.get("role") == "user"]
    seen = [e["at"] for e in log.events if e["type"] == "message"]
    if log.seeded:  # what came while it was offline, after everything it has
        missing = [h for h in missing if not seen or h["at"] > max(seen)]
    else:  # what came before it started — anything newer is still to be seen live
        missing = [h for h in missing if not seen or h["at"] < min(seen)]
    if missing:
        seed = Log(log.channel, log.space, log.channel_id, next_n=log.next_n)
        for h in missing:
            e = seed.message(h["name"], j.unrender(h["content"]), h["at"], to_bot=h.get("to_bot", False),
                             discord_id=h["id"])
            if "reply" in h:
                seed.add("rocky", text=h["reply"], calls=[], id=h.get("reply_id"), at=h.get("reply_at") or h["at"])
                if h.get("reply_id"):
                    seed.ids[str(h["reply_id"])] = "you"
            elif "reaction" in h:
                call = f"seed_{e['n']}"
                seed.add("rocky", text="", calls=[{"id": call, "name": "react",
                                                   "arguments": json.dumps({"emoji": h["reaction"], "message": e["n"]})}],
                         at=h["at"])
                seed.add("result", id=call, content="Reacted.", at=h["at"])
        log.next_n = seed.next_n
        log.ids |= seed.ids
        log.events[:] = log.events + seed.events if log.seeded else seed.events + log.events
    if not log.seeded:
        log.seeded = True
        log.reorganise()
    changed(log)

# The person being answered, tagged at the start of Rocky's message: it's a reply to them already
def unaddressed(text, author):
    return re.sub(rf"^@{re.escape(author)}(?![\w])[\s,:.!-]*", "", text, flags=re.I).strip() or text


# A log from the bot's history entries, for jev_eval's scenarios and replays: people's messages, Rocky's replies under
# them, its reactions, and people's reactions to its replies. Rocky's reactions become react calls, as if made here.
def from_history(channel, history, bot_name="rocky"):
    log = Log(channel)
    calls = 0
    for h in history or []:
        if h["role"] == "assistant":
            log.add("rocky", text=h["content"], calls=[], at=h.get("at"))
            for emoji, n in (h.get("reactions") or {}).items():
                for name in list((h.get("reactors") or {}).get(emoji, {}).values())[:n] or ["someone"] * n:
                    log.add("reaction", on="you", you_said=h["content"], emoji=emoji, author=name, at=h.get("at"))
            continue
        e = log.message(h["name"], j.unrender(h["content"]), h.get("at") or datetime.now(timezone.utc),
                        to_bot=h.get("to_bot", True))
        if "reply" in h:
            log.add("rocky", text=h["reply"], calls=[], at=h.get("reply_at") or h.get("at"))
            for emoji, n in (h.get("reply_reactions") or {}).items():
                names = list((h.get("reply_reactors") or {}).get(emoji, {}).values())
                for name in names[:n] or ["someone"] * n:
                    log.add("reaction", on="you", you_said=h["reply"], emoji=emoji, author=name, at=h.get("at"))
        elif "reaction" in h:
            calls += 1
            log.add("rocky", text="", calls=[{"id": f"call_{calls}", "name": "react",
                                              "arguments": json.dumps({"emoji": h["reaction"], "message": e["n"]})}])
            log.add("result", id=f"call_{calls}", content="Reacted.")
    return log
