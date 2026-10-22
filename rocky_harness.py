"""
Rocky as an agent: an LLM is Rocky, the channel is its conversation, and what it writes is posted. !model rocky, in every
channel. A !nocontext message opens a thread, a channel of its own with a log like any other, but without memories.

Each channel is an append-only log of events. People's messages, reactions and edits come in as user turns, written as
XML with who, when and what they reply to. Rocky's turns are the assistant's: it thinks, then its text is its message,
and its tool calls (react, remember, forget, note, bar, order, look) are kept with their results. Nothing in the log is
rewritten, so every request starts with the one before it and the model's prompt cache covers all but the newest
events. A <memories> block before the log is a snapshot, rebuilt — and deleted messages blanked — only when the cache
would be rebuilt anyway (tidy): at a jump, when the log outgrows LOG_TOKENS and its oldest half goes, or after
CACHE_HOURS without a request.

The way !model's LLMs work elsewhere is a ghostwriter's: the chat as one transcript, then "Write rocky's reply to pip's
last message". That framing had Haiku sometimes describe a reply instead of writing it ("Rocky suggests something
concrete"), and talk about its tools. Here it doesn't write for Rocky; it is Rocky.

jev_bot hands itself over with bind() — run as a script it's __main__, so importing it here would load a second bot.
"""

import asyncio
import base64
import contextlib
import html
import io
import json
import random
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from PIL import Image, UnidentifiedImageError

j = None

def bind(bot_module):
    global j
    j = bot_module

LOG_TOKENS = 90_000         # a log's size before its oldest half goes — under Haiku's 100K price step; set in /settings
MAX_LOG_TOKENS = 90_000     # the most /settings allows: past 100K, Haiku charges 5× for the whole prompt
CHARS_PER_TOKEN = 2.2       # Haiku's, in our chat — a log is measured in characters, LOG_TOKENS turned into them
EVENT_CHARS = 60            # what an event's tags and attributes add to its text, counting a log's characters
FILL = 0.5                  # a new log, filled from the channel's history up to this share of LOG_TOKENS
MAX_TOKENS = 8000           # its thinking, its message and its tool calls, only to stop a runaway: live, ≤300 a request
FOLLOW_UPS = 4              # requests after one with nothing to say yet, for what it then says — at the bar it has
                            # gone bar, react, bar, and at 2 ended its turn before speaking
CACHE = {"type": "ephemeral", "ttl": "1h"}
CACHE_HOURS = 1             # CACHE's lifetime — after this long without a request, the next is read in full anyway
FACT_CHARS = 100            # a remembered fact's longest
PICTURES = 4                # most pictures a look shows, from one message
PICTURE_SIDE = 1024         # a picture's longest side, shrunk to it — ~1400 tokens at most, square; text in a
                            # screenshot or meme still reads
PICTURE_BYTES = 20_000_000  # most of a picture downloaded
PICTURE_CHARS = 1400 * 2.2  # a picture in the log, counted as text for its size (chars): its most tokens, in chars
SAVE_AFTER = 2              # seconds after a change before a log or the memories are written, so a burst writes once
# It thinks first. Without thinking, Haiku narrated in 3 of 57 eval turns ("Pip asked a plain question, so I answer
# that one"); given a say tool for its message instead, it didn't, but thinking with that tool it sometimes did nothing
# (5 of 57). Thinking and speaking its turn: no narration in 57, reply-or-react right as often, for 0.021¢ a turn
# where no thinking was 0.015¢, and ~2s slower.
REASONING = {"effort": "low"}
LOG_DIR = Path(__file__).parent / "rocky_logs"          # a channel's log each, gitignored
MEMORY_PATH = Path(__file__).parent / "rocky_memories.json"
HERE = Path(__file__).parent
UPDATES_FROM = ["rocky_harness.py"]  # changes to Rocky: commits touching these come to its logs as <update>s

# Its last paragraph is for secrets. Without it, on a live-length log, "pick a number and don't tell us" got "Rocky pick
# seven" 4 times in 5, and a secret animal was named in its thinking once in 6, so asked later it made one up. Told to
# keep the secret in its thinking, it did whenever it thought — but at low effort it thought in only 4 of 8 replays of
# a live "pick a number", and at medium in none, so the others only reacted 🤫 with nothing picked. A note is a tool
# call, and those it makes without thinking: replayed with note, it noted a number 8 times in 8 and an animal 6 in 6,
# and named the noted one when asked in 12 of 14. Told to leave it out "until they ask for it", it then wouldn't say
# whether a guess was right — live, the right guess got 🤫, and "was that a yes" "Rocky not say yes or no", 6 times in 6
# replayed. Told to play fair, checking each guess against its note: "yes. Four. You got it" 6 in 6.
# The bar's paragraph: told only to "say what happened", it answered a closed bar with just a 🍺 in 3 of 3 replays;
# told to say it in words, it did in 8 of 8.
# <update>s: told to bring one up "only when it's what's being talked about or someone asks", it told "anything new
# with you today?" about the bar in 1 of 5 replays; told to when they ask what's new, 6 of 6. Asked about lunch, it
# announced it in none of 11.
# look: until it looks, a picture is a [photo] tag. Told to look and write nothing yet, on the live log it looked at a
# synthetic photo sent to it 6 times in 6 and said what was in it (pepperoni pizza; a sign's words); with no picture,
# it didn't look.
# Message before calls: a response ends at its first tool call, and told only to "make your calls and write your message
# in the same response", it reacted first and wrote nothing — in none of its first 628 live turns did it react and
# speak, even asked to prove it could. Told to write first, on a fresh chat it reacted and spoke in 4 of 4 asked to; on
# the live log, full of turns that only reacted, in 0 of 6, though "did you have a good weekend?" got words in 3 of 3
# where it had in 1 of 4.
# Several at once: two questions to it from two people before its turn, replayed on the live log. Told only the newest
# was its to answer (and the other not marked), it answered one of them in 3 of 9 — once the one it wasn't woken by —
# and both in none. Told to answer all of them, saying who's who, both in 3 of 9; the rest, on both, only reacted.
HARNESS = """

This is #{channel} on Discord, as it happens. People's messages come to you as <message> elements: who sent it (from), when (at, {timezone} time), its number (id), and the message it replies to (reply_to). to="you" marks one sent to you — tagging you, or replying to you. woke_you="yes" marks each that woke you. The ones since your last turn are what you're answering now — several, when they came at once: answer them all in your one message, saying who's who. Older ones have had their turn. Reactions, edits and deletions show up as they happen. Your own earlier messages are your turns. An <update> is a change to you, like a new tool, at the point it happened: tell them about it when they ask what's new with you, or it's what's being talked about, but don't announce it out of nowhere.

Your turn is what you write: it's posted in the channel as your reply to the first message that woke you. So write only {bot}'s message, in his voice — never notes about the chat or about what {bot} would say — and don't start it with @ their name. You can also use tools. Your response ends at your first call, so write your message first and then make your calls, in the same response — a call on its own, with no message, is only for what needs no answer. Don't wait to see what a tool did before you speak, except to look and at the bar (below):
- react: put an emoji on a message. Most of what's said to you that isn't a question or a request — "lol", "welp", "bad rocky", news, moans, thanks, polls — a friend would just react to, not answer: react, and write nothing. But a question to you always gets an answer in words, yes-or-no ones too, and so does a request; write a message too when you truly have something to add. To react and speak, write the message, then react: a reaction first leaves your turn with nothing said.
{memory_tools}- note: write something down for yourself.
- bar: see whether the bar is open, and what's on.
- order: order a drink at the bar, for you or for someone here.
- look: see the pictures in a message. A [photo], [gif] or [sticker], or a link, is all you know of a picture until you look — so when you're shown one, or asked about one, look and write nothing yet: you'll see it, and then answer. Never say what's in a picture you haven't looked at.
{memories}

The bar is {bar}, down the road. It keeps hours, and some days it doesn't open at all, or something's run out — so when you want a drink, or someone asks you to get one, call bar or order and write nothing yet: you'll see what the bar says, and then tell them in words what happened — open or not, ordered or not — not just a reaction.

Nobody sees your notes or your thinking: only your message and your reactions are seen. So when you're asked to keep something to yourself — a number you've picked, the answer in a guessing game — note the actual number or answer, and leave it out of your message. It's a game, so play it fair: check each guess against your note and say whether they've got it (yes, or no), and say the answer once they've got it or give up. Your notes stay with your turns, so you'll still know it later."""

# The parts of HARNESS for memories — a log without them (Log.space None) has neither, nor the tools
MEMORY_TOOLS = """- remember: keep a fact for later chats in this server — whenever a message to you has one, as well as answering it. About a person — food, pets, jobs, where they live, teams, trips, plans, big news, or anything someone asks you to remember — only from what they or a friend said. About {bot} — what you say you like or are — so you answer the same next time. One short fact each, written to follow its name ("is vegetarian"). Not small talk, not a summary of the chat, nothing already in <memories>.
- forget: drop a fact by its number, when someone says it's wrong, asks you to forget it, or it has changed — and then remember the new one in the same response.
"""
MEMORIES = "What you've remembered is in <memories>, and in your remember calls since. Never mention reacting, remembering, forgetting or noting in your message."
NO_MEMORIES = ("This thread is apart from the rest of the server: you have no memories here, and nothing said here is "
               "kept for later. Never mention reacting or noting in your message.")
MEMORY_CALLS = {"remember", "forget"}

TOOLS = [
    {"type": "function", "function": {
        "name": "react", "description": "Put an emoji on a message.",
        "parameters": {"type": "object", "required": ["emoji"], "properties": {
            "emoji": {"type": "string", "description": "One emoji, or a server emoji by name, like :blobwave:"},
            "message": {"type": "integer", "description": "The message's id; leave out for the first that woke you"}}}}},
    {"type": "function", "function": {
        "name": "remember", "description": "Keep a fact for later chats in this server.",
        "parameters": {"type": "object", "required": ["about", "fact"], "properties": {
            "about": {"type": "string", "description": "The person's name as it is in the chat, your own name, or a short topic"},
            "fact": {"type": "string", "description": "Short, to follow the name: \"is vegetarian\""}}}}},
    {"type": "function", "function": {
        "name": "forget", "description": "Drop a remembered fact.",
        "parameters": {"type": "object", "required": ["number"], "properties": {
            "number": {"type": "integer", "description": "The fact's number in <memories>"}}}}},
    {"type": "function", "function": {
        "name": "note", "description": "Write something down for yourself, like a secret you've been asked to keep. Nobody else sees it.",
        "parameters": {"type": "object", "required": ["text"], "properties": {
            "text": {"type": "string", "description": "What to keep, like \"picked 4\""}}}}},
    {"type": "function", "function": {
        "name": "bar", "description": "See whether the bar is open, and its menu.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "order", "description": "Order a drink at the bar.",
        "parameters": {"type": "object", "required": ["drink"], "properties": {
            "drink": {"type": "string", "description": "As it is on the menu"},
            "for": {"type": "string", "description": "Who it's for, by their name in the chat; leave out for you"}}}}},
    {"type": "function", "function": {
        "name": "look", "description": "See the pictures in a message: photos, gifs, stickers and links' previews.",
        "parameters": {"type": "object", "properties": {
            "message": {"type": "integer", "description": "The message's id; leave out for the first that woke you"}}}}},
]
# Calls whose result it needs before it speaks: what it wrote alongside one isn't posted
WAIT_FOR = {"bar", "order", "look"}


# The bar: a pub with a menu and hours, in TIMEZONE. Some days it doesn't open, and some days a drink has run out —
# picked from the date, so a day is the same however often Rocky asks.
BAR = "The Hail Mary"
MENU = {  # drink: price
    "Astrophage Amber": 5.40, "Tau Ceti Pale Ale": 5.80, "Eridani Stout": 6.10, "Blip-A Lager": 4.90,
    "Petrova Line Sour": 6.50, "Xenonite Cider": 5.20, "Ammonia-Free Lemonade": 2.80,
}
HOURS = {0: (16, 23), 1: (16, 23), 2: (16, 23), 3: (16, 23), 4: (12, 24), 5: (12, 24), 6: (12, 22)}  # Monday's 0
SHUT = ["a taumoeba leak in the cellar", "a private party", "the landlord's at a wedding", "a staff shortage",
        "pest control", "the card machine's down and so is the till", "the Astrophage tank's being drained",
        "a xenonite refit"]
SHUT_DAYS = 0.15            # a day's chance it doesn't open
SOLD_OUT = 0.4              # a day's chance a drink has run out

def clock():
    return datetime.now(timezone.utc)

def bar_today(day):
    pick = random.Random(day.isoformat())
    shut = pick.choice(SHUT) if pick.random() < SHUT_DAYS else None
    out = pick.choice(list(MENU)) if pick.random() < SOLD_OUT else None
    return shut, out

# (open, what the bar says): its menu if it's open, or why not and when it next opens
def bar_state(now=None):
    now = (now or clock()).astimezone(j.TIMEZONE)
    shut, out = bar_today(now.date())
    opens, closes = HOURS[now.weekday()]
    if shut:
        why = f"Closed today: {shut}."
    elif now.hour < opens:
        return False, f"Closed. Opens at {opens}:00."
    elif now.hour >= closes:
        why = "Closed for the night."
    else:
        menu = "; ".join(f"{d} £{p:.2f}" + (" (run out)" if d == out else "") for d, p in MENU.items())
        until = "midnight" if closes == 24 else f"{closes}:00"
        last = " Last orders!" if (now.hour, now.minute) >= (closes - 1, 45) else ""
        return True, f"{BAR} is open until {until}.{last} Menu: {menu}."
    for ahead in range(1, 8):  # when it next opens
        day = (now + timedelta(days=ahead)).date()
        if not bar_today(day)[0]:
            name = "tomorrow" if ahead == 1 else f"{day:%A}"
            return False, f"{why} Next open {name} from {HOURS[day.weekday()][0]}:00."
    return False, why

def menu_drink(asked):
    asked = re.sub(r"[^a-z]", "", asked.lower())
    return next((d for d in MENU if asked and (asked in re.sub(r"[^a-z]", "", d.lower()))), None)


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
#              "older": {"author": a name or "you", "text"} when what it replies to isn't in the log, "woke": it woke Rocky}
#   reaction: {"on": n or "you", "you_said", "emoji", "author", "removed"}
#   edit:     {"on": n, "text"}
#   deleted:  {"on": n or "you", "you_said"}  — and the message gets "deleted"; tidy() then blanks it ("gone")
#   rocky:    {"text", "calls": [{"id", "name", "arguments"}], "reasoning", "id": its Discord message,
#              "more_ids": the messages after it, when it was too long for one}
#   result:   {"id", "content", "on": a look's message n, "pictures": a look's, as data: URLs}  — what came of a call,
#             right after its turn
#   update:   {"commit", "subject"}  — a change to Rocky, deployed: a commit to UPDATES_FROM
# ids: Discord message id to n, or to "you" for Rocky's own.
@dataclass
class Log:
    channel: str
    space: str = "test"                        # whose memories it has: a server's, or a DM's — None for none
    channel_id: int = 0
    events: list = field(default_factory=list)
    next_n: int = 1
    ids: dict = field(default_factory=dict)
    shown: list = field(default_factory=list)  # the facts in <memories>: as they were at the last jump
    seeded: bool = False                       # filled from the channel's history, the first time it was needed
    asked: datetime = None                     # the last request, to know when the cache has gone
    code: str = None                           # the last commit it was told about (updates)
    keep: bool = True                          # saved in LOG_DIR — not one for a single answer (apart)

    def message(self, author, text, at, to_bot=False, reply_to=None, uid=None, discord_id=None, you_said=None,
                older=None):
        e = {"type": "message", "n": self.next_n, "author": author, "uid": uid, "at": at, "text": text,
             "to_bot": to_bot, "reply_to": reply_to, "you_said": you_said}
        if older:
            e["older"] = older
        self.next_n += 1
        self.events.append(e)
        if discord_id:
            self.ids[str(discord_id)] = e["n"]
        return e

    # after: an event to put it right after — its turn goes where its request ended, before what came meanwhile
    def add(self, type, after=None, **fields):
        e = {"type": type, "at": fields.pop("at", None) or datetime.now(timezone.utc), **fields}
        i = next((i + 1 for i in range(len(self.events) - 1, -1, -1) if self.events[i] is after), len(self.events))
        self.events.insert(i, e)
        return e

    # <memories> as they are now: at a jump, or when someone wants a fact really gone. Everything after it in the
    # request is then read at full price once, so not for every remember — those are in the log as calls.
    def reorganise(self):
        self.shown = [dict(f) for f in memory(self.space)["facts"]] if self.space else []

    def discord_id(self, n):
        return next((int(i) for i, v in self.ids.items() if v == n), None)

    # The channel message n is in: the log's, but for a thread's first, the message it's on, in the channel
    def channel_of(self, n):
        return next((e.get("in") for e in self.events if e["type"] == "message" and e["n"] == n), None) or self.channel_id

    def rocky_text(self, discord_id):
        return next((e["text"] for e in reversed(self.events) if e["type"] == "rocky" and
                     (e.get("id") == discord_id or discord_id in e.get("more_ids", []))), "")


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
            elif o := e.get("older"):
                whose = "your older message" if o["author"] == "you" else f'an older message from {o["author"]}'
                reply = f' reply_to="{attr(whose + ": " + o["text"])}"'
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
        elif e["type"] == "update":
            lines.append(f'<update>{html.escape(e["subject"], quote=False)}</update>')
    return "\n".join(lines)

def memories_text(log):
    if not log.shown:
        return "<memories>\nNothing yet.\n</memories>"
    return "<memories>\n" + "\n".join(f"[{f['n']}] {f['about']}: {f['fact']}" for f in log.shown) + "\n</memories>"

# The request: system (persona and how this works), <memories> if it has them, then the log as turns. Cache marks on the system
# prompt, the memories, and the end of the request — the next request starts with all of it.
# until: the last event to show — a turn's requests see the log as it was when it began, and what it's done since
def render(log, bot_name, cache=True, until=None):
    system = j.rocky_prompt(bot_name) + HARNESS.format(
        channel=log.channel, timezone=j.TIMEZONE.key, bot=bot_name, bar=BAR,
        memory_tools=MEMORY_TOOLS.format(bot=bot_name) if log.space else "", memories=MEMORIES if log.space else NO_MEMORIES)
    mark = lambda text: [{"type": "text", "text": text, **({"cache_control": CACHE} if cache else {})}]
    messages = [{"role": "system", "content": mark(system) if cache else system}]
    if log.space:
        messages.append({"role": "user", "content": mark(memories_text(log)) if cache else memories_text(log)})
    pending, clock = [], {}

    def flush():
        if pending:
            messages.append({"role": "user", "content": people_turn(pending, clock)})
            pending.clear()

    end = next((i + 1 for i in range(len(log.events) - 1, -1, -1) if log.events[i] is until), len(log.events))
    for e in log.events[:end]:
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
            content = e["content"]
            if e.get("pictures"):
                content = [{"type": "text", "text": content},
                           *({"type": "image_url", "image_url": {"url": url}} for url in e["pictures"])]
            messages.append({"role": "tool", "tool_call_id": e["id"], "content": content})
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
            calls = " ".join("[note]" if c["function"]["name"] == "note" else  # a secret: Context is anyone's to open
                             f"[{c['function']['name']} {c['function']['arguments']}]" for c in m.get("tool_calls") or [])
            lines.append(f"{bot_name}: {m['content'] or ''} {calls}".rstrip())
        else:
            lines.append(f"[{j.text_of(m['content'])}]")
    return "\n".join(lines)


@dataclass
class Turn:
    text: str = ""
    reactions: list = field(default_factory=list)  # [(message n, emoji)]
    kept: list = field(default_factory=list)       # what remember and forget did
    shown: list = field(default_factory=list)      # what it did that's shown under its message: kept, and orders
    spoke: dict = None                             # the log's event for its message, to give it its Discord id
    requests: int = 0
    events: list = field(default_factory=list)     # its events in the log — its turns and their calls' results


def use(log, call, waking):
    f = call.get("function") or {}
    try:
        args = json.loads(f.get("arguments") or "{}")
        if f.get("name") in MEMORY_CALLS and not log.space:
            return None, "No memories here."
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
        if f.get("name") == "note":
            # its call in the log is the note
            return None, "Noted." if str(args.get("text") or "").strip() else "Nothing to note."
        if f.get("name") == "bar":
            return None, bar_state()[1]
        if f.get("name") == "order":
            is_open, says = bar_state()
            if not is_open:
                return None, says
            drink, who = menu_drink(str(args.get("drink") or "")), str(args.get("for") or "").strip().lstrip("@")
            if not drink:
                return None, f"Not on the menu. {says}"
            if drink == bar_today(clock().astimezone(j.TIMEZONE).date())[1]:
                return None, f"The {drink}'s run out today. {says}"
            return ("order", drink, who or None), f"Ordered: {drink}{f' for {who}' if who else ''}, £{MENU[drink]:.2f}."
        return None, f"No tool called {f.get('name')!r}."
    except (ValueError, KeyError, TypeError) as e:
        return None, f"That didn't work: {e!r}"

# The pictures in a Discord message: its images, its embeds' (a gif's from Tenor, a link's preview) and its stickers.
# Fetched again when it looks: attachment links expire after a day or so.
async def pictures(log, n):
    if not isinstance(n, int) or not (discord_id := log.discord_id(n)):
        return None
    channel_id = log.channel_of(n)
    channel = j.bot.get_channel(channel_id) or await j.bot.fetch_channel(channel_id)
    m = await channel.fetch_message(discord_id)
    urls = [a.url for a in m.attachments if (a.content_type or "").startswith("image/")]
    urls += [x.url for e in m.embeds for x in (e.image, e.thumbnail) if x and x.url]
    urls += [s.url for s in m.stickers if str(getattr(s.format, "name", "")) != "lottie"]
    found = []
    async with j.aiohttp.ClientSession() as session:
        for url in dict.fromkeys(urls):
            if len(found) == PICTURES:
                break
            async with session.get(url, timeout=j.aiohttp.ClientTimeout(total=30)) as r:
                if r.status == 200 and (picture := shrunk(await body(r))):
                    found.append(picture)
    return found

# A response's body, up to PICTURE_BYTES. One content.read gives only what's arrived so far, and a photo cut short
# isn't a picture
async def body(r):
    data = bytearray()
    while len(data) < PICTURE_BYTES and (chunk := await r.content.read(PICTURE_BYTES - len(data))):
        data += chunk
    return bytes(data)

# A picture as a JPEG data: URL, no bigger than PICTURE_SIDE — a gif's first frame. None if it isn't one.
def shrunk(data):
    try:
        img = Image.open(io.BytesIO(data))
        img.seek(0)
        img = img.convert("RGBA")
    except (UnidentifiedImageError, OSError, ValueError):
        return None
    img.thumbnail((PICTURE_SIDE, PICTURE_SIDE))
    flat = Image.new("RGB", img.size, "white")  # a sticker's see-through parts
    flat.paste(img, mask=img.getchannel("A"))
    out = io.BytesIO()
    flat.save(out, "JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(out.getvalue()).decode()

# look: the result's text, and its pictures
async def look(log, arguments, waking):
    try:
        n = json.loads(arguments or "{}").get("message") or waking["n"]
    except (ValueError, AttributeError):
        n = waking["n"]
    try:
        found = await pictures(log, n)
    except j.discord.NotFound:
        return n, f"Message {n} was deleted.", []
    except Exception as e:
        j.log.warning(f"[ROCKY] looking at message {n} failed: {e!r}")
        return n, f"Couldn't see message {n}'s pictures.", []
    if found is None:
        return n, f"No message {n} to look at.", []
    if not found:
        return n, f"Message {n} has no pictures.", []
    return n, f"Message {n}'s {'picture' if len(found) == 1 else f'{len(found)} pictures'}:", found

# What it wrote, as a message — or nothing: told it can say nothing, Haiku has written "(silent)" or "."
def said(content, bot_name):
    text = j.clean_llm(content or "", bot_name)
    return "" if not re.search(r"\w", text) or re.fullmatch(r"[(\[*].*[)\]*]", text) else text

# Rocky's turn after `waking` (already in the log): its message and tool calls, usually in one response. Only a turn
# that just kept or dropped memories, noted, or went to the bar gets another request, for what it then says. Its
# requests go through jev_bot.llm_message, which counts their cost on the trace.
# typing: the channel's typing() — shown only for that second request. Discord shows "rocky is typing" for ~10s after
# it's sent, and nothing stops it but a message, so shown while it decided, it lingered after a turn that only reacted.
def tools_for(log):
    return TOOLS if log.space else [t for t in TOOLS if t["function"]["name"] not in MEMORY_CALLS]

async def take_turn(log, waking, model="rocky", bot_name="rocky", typing=None):
    turn = Turn()
    # Already in one of its turns' requests, before what it said: it's had its answer. Its messages from outside a turn
    # (spoke_elsewhere) don't count — a status posted while a message waited doesn't answer it
    after = next((i for i, e in enumerate(log.events) if e is waking), len(log.events))
    if any(e["type"] == "rocky" and not e.get("elsewhere") for e in log.events[after + 1:]):
        return turn
    # Every message to it since it last spoke is this turn's: one that came while it waited is in this request, and has
    # no turn of its own. Marked for good, so the turn it's in reads the same in every later request
    since = max((i for i, e in enumerate(log.events) if e["type"] == "rocky"), default=-1)
    for e in log.events[since + 1:]:
        if e["type"] == "message" and e["to_bot"] and not e.get("gone"):
            e["woke"] = True
    waking["woke"] = True
    now = datetime.now(timezone.utc)
    if log.asked and (now - log.asked).total_seconds() > CACHE_HOURS * 3600:
        tidy(log)  # the cache has gone, so this request is read in full anyway
    log.asked = now
    token = j.trace.set({"cost": 0.0, "requests": 0}) if j.trace.get() is None else None
    # Where its turn goes: what came while it was deciding comes after, has its own turn, and isn't in its follow-ups.
    # Put before, the next turn's request ended with Rocky's, and Haiku 5.5 refused it (no prefill)
    began = log.events[-1]
    try:
        for request in range(1 + FOLLOW_UPS):
            seen = turn.events[-1] if turn.events else began
            async with typing() if typing and request else contextlib.nullcontext():
                messages = render(log, bot_name, cache=bool(j.LLMS[model].get("cache_marks")), until=seen)
                # A request can't end with Rocky (Haiku 5.5 takes it as a prefill, and refuses it): when one of its messages
                # from outside a turn came after the one that woke it. Not a str, so the cache mark stays on people's turn
                if messages[-1]["role"] == "assistant":
                    messages.append({"role": "user", "content": [{"type": "text", "text": "(Your turn.)"}]})
                message = await j.llm_message(model, messages, MAX_TOKENS, tools=tools_for(log), reasoning=REASONING)
            if message is None:
                break
            turn.requests += 1
            calls = [{"id": c["id"], "name": c["function"]["name"], "arguments": c["function"]["arguments"] or "{}"}
                     for c in message.get("tool_calls") or []]
            text = said(message.get("content"), bot_name)
            # Written before it saw what the bar said — it'll say what happened next request. Left out of the log
            # too, so it isn't shown a message that was never posted
            waits = any(c["name"] in WAIT_FOR for c in calls)
            if waits:
                text = ""
            if not text and not calls:
                break  # nothing said or done — an empty turn isn't one the model can be shown later
            e = log.add("rocky", after=seen, text=text, calls=calls, reasoning=message.get("reasoning_details"))
            turn.events.append(e)
            for c in calls:
                if c["name"] == "look":
                    on, result, found = await look(log, c["arguments"], waking)
                    turn.events.append(log.add("result", after=turn.events[-1], id=c["id"], content=result, on=on,
                                               pictures=found))
                    continue
                done, result = use(log, {"function": {"name": c["name"], "arguments": c["arguments"]}}, waking)
                turn.events.append(log.add("result", after=turn.events[-1], id=c["id"], content=result))
                if done and done[0] == "react":
                    turn.reactions.append(done[1:])
                elif done:
                    if done[0] != "order":
                        turn.kept.append(done)
                    if done[2]:  # not a forget of no fact
                        turn.shown.append(done)
            if text:
                turn.text, turn.spoke = text, e
                break
            # A reaction answers; a memory or a note on its own doesn't, so it gets a request to say something, and
            # so do the bar and a look, for what it then says
            if not waits and any(c["name"] == "react" for c in calls) and all(c["name"] in ("react", "note") for c in calls):
                break
    finally:
        if token is not None:
            j.trace.reset(token)
    return turn

# The log's oldest half goes once it's over LOG_TOKENS: everything before a message, so no turn loses its results.
# <memories> is rebuilt then, so what was remembered in the dropped half isn't lost with it.
def chars(log):
    return sum(len(e.get("text") or "") + EVENT_CHARS + sum(len(c["arguments"]) for c in e.get("calls", []))
               + PICTURE_CHARS * len(e.get("pictures") or []) for e in log.events)

# A new log's size, read back from the channel's history: what a jump leaves, so it isn't cut on its first turn
def fill_chars():
    return LOG_TOKENS * CHARS_PER_TOKEN * FILL

# jev_bot's history entries as they'd count in a log: each message, and Rocky's reply or reaction under it
def entry_chars(entries):
    return sum(len(h.get("content") or "") + EVENT_CHARS
               + (len(h.get("reply") or "") + EVENT_CHARS if "reply" in h or "reaction" in h else 0) for h in entries)

def jump(log):
    if chars(log) <= LOG_TOKENS * CHARS_PER_TOKEN:
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
    for e in log.events:  # and what it saw of their pictures
        if e["type"] == "result" and e.get("on") in gone and e.get("pictures"):
            e["content"], e["pictures"] = f"Message {e['on']} was deleted.", []
    log.reorganise()
    changed(log)


# Live logs, one for each channel Rocky has taken a turn in, kept in LOG_DIR
logs: dict[int, Log] = {}

def has_log(channel_id):
    return channel_id in logs or (LOG_DIR / f"{channel_id}.json").exists()
log_saves: dict[int, object] = {}

def plain(v):
    return v.isoformat() if isinstance(v, datetime) else v

def save_log(log):
    log_saves.pop(log.channel_id, None)
    saved = {"channel": log.channel, "space": log.space, "next_n": log.next_n, "ids": log.ids, "shown": log.shown,
             "seeded": log.seeded, "asked": plain(log.asked), "code": log.code,
             "events": [{k: plain(v) for k, v in e.items()} for e in log.events]}
    try:
        LOG_DIR.mkdir(exist_ok=True)
        path = LOG_DIR / f"{log.channel_id}.json"
        path.with_suffix(".tmp").write_text(json.dumps(saved, ensure_ascii=False) + "\n")
        path.with_suffix(".tmp").replace(path)
    except OSError as e:
        j.log.warning(f"Writing the log for #{log.channel} failed: {e}")

def changed(log):
    if log.keep and log.channel_id not in log_saves:
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

# memories: False for a !nocontext thread's, new — it has nothing from before, so it's seeded already
def log_for(channel, memories=True):
    if (log := logs.get(channel.id)) is not None:
        return log
    space = str(channel.guild.id) if getattr(channel, "guild", None) else f"dm-{channel.id}"
    log = Log(getattr(channel, "name", None) or "DM", space if memories else None, channel.id, seeded=not memories)
    try:
        saved = json.loads((LOG_DIR / f"{channel.id}.json").read_text())
        log.next_n, log.ids, log.shown, log.seeded = saved["next_n"], saved["ids"], saved["shown"], saved["seeded"]
        log.space = saved.get("space", log.space)
        if not getattr(channel, "name", None):  # only its id: a thread's log, found by its first message
            log.channel = saved.get("channel", log.channel)
        log.asked = saved.get("asked") and datetime.fromisoformat(saved["asked"])
        log.events = [e | {"at": datetime.fromisoformat(e["at"])} if e.get("at") else e for e in saved["events"]]
        log.code = saved.get("code")
        updated(log)
    except FileNotFoundError:
        log.code = head()  # a new log starts as Rocky is now
    except Exception as e:  # unreadable — start again, from the channel's history
        j.log.warning(f"Loading the log for #{log.channel} failed: {e}")
    logs[channel.id] = log
    return log

# A !nocontext thread: every message in it is to Rocky, and its log has no memories
def no_memories(channel):
    return isinstance(channel, j.discord.Thread) and memoryless(channel.id)

# Read from its file once, for a log that isn't loaded: a log never gains or loses its memories
saved_memoryless: dict[int, bool] = {}

def memoryless(channel_id):
    if (log := logs.get(channel_id)) is not None:
        return log.space is None
    if channel_id not in saved_memoryless:
        try:
            saved_memoryless[channel_id] = json.loads((LOG_DIR / f"{channel_id}.json").read_text()).get("space", "") is None
        except FileNotFoundError:
            return False  # not yet: it may be a thread about to open
        except (OSError, ValueError):
            saved_memoryless[channel_id] = False
    return saved_memoryless[channel_id]

# The log of the !nocontext thread opened on a message, which has the message's id, or None
def thread_log(message_id):
    return log_for(SimpleNamespace(id=message_id, name=None, guild=None)) if memoryless(message_id) else None

# The logs a message is in: its channel's, and the !nocontext thread opened on it, which it starts
def logs_with(channel_id, message_id):
    found = [logs[channel_id]] if channel_id in logs else []
    if message_id != channel_id and (log := thread_log(message_id)):
        found.append(log)
    return found

# The Discord id of Rocky's answer to a thread's first message: its turn right after it, before anyone else spoke
def first_answer(log):
    for e in log.events[1:]:
        if e["type"] == "message":
            return None
        if e["type"] == "rocky" and e.get("text") and e.get("id"):
            return e["id"]
    return None

# A log for one answer, with nothing before it and no memories, never saved: a !nocontext message where there's no
# thread for it — in a DM, or when one couldn't be opened
def apart(channel):
    return Log(getattr(channel, "name", None) or "DM", None, channel.id, seeded=True, keep=False)

# Updates: commits to UPDATES_FROM since a log's last, as <update>s at its end — appended, so the cache keeps all
# before them. Only the subject, without its scope or PR number ("give Rocky a bar to order drinks from"): the repo is
# public, but bodies are long. A log from before updates were told gets every one since its first event, each where
# it happened — that rewrites the log, but only once, at the deploy that changed the prompt to tell of them, which
# rewrote the cache anyway.
def git(*args):
    try:
        return subprocess.run(["git", *args], cwd=HERE, capture_output=True, text=True, timeout=10, check=True).stdout
    except (OSError, subprocess.SubprocessError) as e:
        j.log.warning(f"git {args[0]} failed: {e}")
        return None

def head():
    return (git("rev-parse", "HEAD") or "").strip() or None

def updates(since_commit=None, since=None):
    span = [f"{since_commit}..HEAD"] if since_commit else [f"--since={since.isoformat()}", "HEAD"]
    found = []
    for line in (git("log", "--reverse", "--format=%H%x00%ct%x00%s", *span, "--", *UPDATES_FROM) or "").splitlines():
        commit, at, subject = line.split("\0", 2)
        found.append({"commit": commit, "at": datetime.fromtimestamp(int(at), timezone.utc),
                      "subject": re.sub(r"^[\w ,/-]+: |\s*\(#\d+\)$", "", subject)})
    return found

def updated(log):
    now = head()
    if not now or now == log.code:
        return False
    first = next((e["at"] for e in log.events if e.get("at")), None)
    new = updates(log.code) if log.code else updates(since=first) if first else []
    for u in new:
        if log.code:
            log.add("update", **u)
            continue
        # where it happened: before the first event after it, but never between a turn's calls and their results
        i = next((i for i, e in enumerate(log.events) if e.get("at") and e["at"] > u["at"]), len(log.events))
        while i < len(log.events) and log.events[i]["type"] == "result":
            i += 1
        log.events.insert(i, {"type": "update", **u})
    log.code = now
    changed(log)
    if new:
        j.log.info(f"[ROCKY] #{log.channel} told of {len(new)} update(s)")
    return bool(new)

# What a message replies to when the log doesn't have it — from before the log, or a command — as Discord sent it
# with the reply: who said it, and the start of it, since it's nowhere else Rocky can see. None if Discord didn't
# send it, or it's deleted or has no text.
def replied_outside(m):
    r = m.reference.resolved or j.bot._connection._get_message(m.reference.message_id)
    if r is None or isinstance(r, j.discord.DeletedReferencedMessage) or not (text := j.message_text(r)):
        return None
    return {"author": "you" if r.author.id == j.bot.user.id else r.author.display_name, "text": snippet(text, 200)}

# A message in the channel, as it arrives: people's and other apps' — not Rocky's own (its turns add those), commands,
# or !nocontext messages (jev_bot keeps those out of the channel's). Returns its event, or None. log: m's channel's,
# unless it's another's — a !nocontext message starts its thread's.
def observe(m, log=None):
    log = log or log_for(m.channel)
    if (n := log.ids.get(str(m.id))) is not None:
        return next((e for e in log.events if e.get("n") == n), None)
    if m.author.id == j.bot.user.id or j.command(m) or not (text := j.message_text(m)):
        return None
    reply_to = you_said = older = None
    if m.reference and m.reference.message_id:
        reply_to = log.ids.get(str(m.reference.message_id))
        if reply_to == "you":
            you_said = log.rocky_text(m.reference.message_id)
        elif reply_to is None and log.space:  # a !nocontext one shows nothing from outside it
            older = replied_outside(m)
    e = log.message(m.author.display_name, text, m.created_at, to_bot=j.should_respond(m), reply_to=reply_to,
                    uid=m.author.id, discord_id=m.id, you_said=you_said, older=older)
    if m.channel.id != log.channel_id:
        e["in"] = m.channel.id
    changed(log)
    return e

# Someone's reaction on a message in a log, added or taken off
def reacted(channel_id, message_id, emoji, author, added=True):
    for log in logs_with(channel_id, message_id):
        if (on := log.ids.get(str(message_id))) is not None:
            log.add("reaction", on=on, you_said=log.rocky_text(message_id) if on == "you" else None, emoji=emoji,
                    author=author or "someone", removed=not added)
            changed(log)

def edited(m):
    for log in logs_with(m.channel.id, m.id):
        if not isinstance(n := log.ids.get(str(m.id)), int):
            continue
        text = j.message_text(m)
        last = next((e["text"] for e in reversed(log.events)
                     if e["type"] == "message" and e["n"] == n or e["type"] == "edit" and e["on"] == n), None)
        if text and text != last:
            log.add("edit", on=n, text=text)
            changed(log)

# A message deleted: marked now, and blanked at the next tidy
def deleted(channel_id, message_id):
    for log in logs_with(channel_id, message_id):
        if (on := log.ids.get(str(message_id))) is None:
            continue
        if on == "you":
            log.add("deleted", on="you", you_said=log.rocky_text(message_id))
        else:
            message = next((e for e in log.events if e["type"] == "message" and e["n"] == on), None)
            if message is None or message.get("deleted"):
                continue
            message["deleted"] = True  # not shown until tidy, so nothing earlier in the request changes yet
            log.add("deleted", on=on)
        changed(log)

# One of Rocky's messages from outside a turn (written while another !model was chosen, or a status posted in
# STATUS_CHANNEL), so the log has it — loaded if nothing's been said there since a restart
def spoke_elsewhere(channel, text, sent):
    if not has_log(channel.id):
        return
    log = log_for(channel)
    log.add("rocky", text=text, calls=[], id=sent.id, at=sent.created_at, elsewhere=True)
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
                                                   "arguments": json.dumps({"emoji": h["reaction"], "message": e["n"]},
                                                                         ensure_ascii=False)}],
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

