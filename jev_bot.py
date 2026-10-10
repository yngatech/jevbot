"""
Jev Discord Bot — tournament sampling + history.

This is the version that produced "I depends on on situation of circumstances."
20K vocab, bucket tournament, empty descriptions, 3-turn history.
"""

import io
import os
import math
import re
import json
import time
import asyncio
import logging
import random
import signal
import contextvars
from pathlib import Path
from urllib.parse import urlsplit
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import aiohttp
import discord
from discord.ext import commands, tasks
from dotenv import load_dotenv

import why_chart

load_dotenv()

TOKEN = os.environ["DISCORD_TOKEN_JEV"]
OPENROUTER_KEY = os.environ["OPENROUTER_API_KEY"]
STATUS_CHANNEL = int(os.environ.get("STATUS_CHANNEL_ID") or 0)  # where each new status is also posted — unset for nowhere
TIMEZONE = ZoneInfo(os.environ.get("TIMEZONE") or "UTC")  # the clock an LLM's transcript shows times on (IANA name)
API_URL = "https://openrouter.ai/api/alpha/decisions"
MODEL = "~typesafe/jev-latest"
END = "<END>"

MAX_CHOICES = 255
QUESTIONS_PER_CALL = 20
TOP_PER_BUCKET = 2
MAX_WORDS = 30
MIN_WORDS = 2
HISTORY_TO_BOT = 6              # earlier messages to jev (mentions, pinged replies) in the transcript
HISTORY_CHATTER = 8             # earlier channel messages not aimed at jev in the transcript — 0 to leave them out
LLM_HISTORY = 200               # earlier channel messages, any kind, in an LLM's transcript (see !model) — ~300 lines with rocky's replies
HISTORY_SCAN = 400              # recent messages read to rebuild a channel's history after a restart
CATCH_UP_WINDOW = 30            # minutes — on startup, answer each channel's latest message to jev from this long ago that it missed
CHAIN_DEPTH = 20                # Discord replies followed back from a message, looking for a !nocontext it carries on — and messages kept per side conversation
SIDE_TALKS = 50                 # side conversations (!nocontext and the replies under it) kept, the latest
STOP_THRESHOLD = 0.5            # let jev stop earlier — the good part is always the first half
REPEAT_PENALTY = 1.5
REPEAT_WINDOW = 8
CONTENT_PENALTY = 2.5
CONTENT_PENALTY_CAP = 4
STOP_PENALTY = 1.6
STOP_PENALTY_CAP = 6
# Repeats across replies: jev copies itself — "Dunno" went 0.32 → 0.61 as its own "Dunno? Dunno?" replies filled
# the transcript, and one "Private? Private?" made five. Each of its last ECHO_TURNS answers (replies or reactions)
# that said a word divides it by ECHO_PENALTY. Replayed over a day's logs, 2.0 took replies opening with "dunno"
# from 15 of 40 to 8 and kept the ones it clearly meant; 2.5 dropped "Dunno" for "Yes" at 0.12 vs 0.09 after one
ECHO_TURNS = 3
ECHO_PENALTY = 2.0
# Words that say jev has nothing to say. Funny once in a while, but they opened half its replies, and the echo
# penalty only moved jev from one to the next: "dunno" went, "no" (a stopword, so never echoed) took over, "No? No?"
# three times running. So they count as one word across replies: each of jev's last ECHO_TURNS answers that said
# any of them divides them all by NOTHING_PENALTY, instead of ECHO_PENALTY. The first is free, and a run breaks
# into saying something ("century egg?" "Yuck", "toast with butter? jam?" "Jam"). Replayed over two days' logs,
# 3.0 took replies opening with one from 59 of 130 to 35; 2.0 left 49, 4.0 left 28
NOTHING = {"dunno", "no", "crickets", "chirping", "silence", "silent", "empty", "quiet", "blank", "shrug", "huh",
           "ignoring"}
NOTHING_PENALTY = 3.0
VOCAB_SIZE = 10_000             # vocab.txt is ordered most common first — every word costs ~7 input tokens on every step
REACT_THRESHOLD = 0.4           # react when P(message is a question/request for jev) is below this — questions ~0.8-0.98, chatty ~0.03-0.45
STATUS_EVERY = 180              # minutes between new statuses (~$0.02-0.04 each) — 0 to leave the status alone
STATUS_MIN_WORDS = 4            # a short status is just "Fine thanks" — the soup comes from making it keep going
STATUS_MAX_WORDS = 12
STATUS_CHAT = 8                 # recent messages from the latest active channel, in view for every other status — 0 for none
STATUS_ATTEMPTS = 3             # keep the previous status if this many candidates fail the meaning check
STATUS_THRESHOLD = 0.6          # P(the finished status makes sense on its own), allowing jev's broken wording
# Bare "Next word?" reads as "which word fits this?" — jev described its reply ("empty", "silent", "garbled")
# instead of continuing it, but "Next word of {bot_name}'s reply?" didn't help live and made <END> far likelier
NEXT_WORD = "Next word?"

STOPWORDS = set(
    "a an the and or but if of to in on at by for with from as is are was were be been "
    "being it its this that these those i you he she they we me him her them us my your "
    "his their our not no so then than there here when where which who what how all any "
    "some each into over under about above below up down out off again more most very "
    "can will just do does did have has had would could should may might must".split()
)
NO_SPACE_BEFORE = set(".,!?;:)\"'\n")
NEWLINE = "\\n"  # vocab.txt's line break token — jev sees it literally, Discord gets a real newline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("jev")

# One JSON line per message jev handles: what it saw, what it considered, what it did and what it cost.
# Holds what people said in the server, so it stays local (gitignored) — delete old days freely.
LOG_DIR = Path(__file__).parent / "logs"
trace: contextvars.ContextVar[dict | None] = contextvars.ContextVar("trace", default=None)

def note(**fields):
    if (t := trace.get()) is not None:
        t.update(fields)

def write_trace(t):
    try:
        LOG_DIR.mkdir(exist_ok=True)
        with open(LOG_DIR / f"{t['at'][:10]}.jsonl", "a") as f:
            f.write(json.dumps(t, ensure_ascii=False, default=str) + "\n")
    except OSError as e:
        log.warning(f"Writing {LOG_DIR} failed: {e}")

# History: recent messages per channel, oldest first. "to_bot" marks the ones that mentioned jev.
channel_history: dict[int, list[dict]] = defaultdict(list)
dm_channels: set[int] = set()  # channels in channel_history that are DMs, kept out of the status

# The last `to_bot` messages to jev and the last `chatter` other messages, in order
def recent(entries, to_bot=None, chatter=None):
    limits = {True: HISTORY_TO_BOT if to_bot is None else to_bot, False: HISTORY_CHATTER if chatter is None else chatter}
    keep = []
    for e in reversed(entries):
        kind = e.get("to_bot", True)
        if limits[kind] > 0:
            limits[kind] -= 1
            keep.append(e)
    return keep[::-1]

# What the transcript shows: recent() of the channel — or for an LLM, its last `limit` messages — minus messages to
# jev it never answered but has answered something after since — left behind (sent while it was offline, say), they
# pull jev back to their topic. One being answered right now is "pending", so a quick reaction to a later message
# doesn't hide it.
def shown(entries, limit=None):
    keep, answered_since = [], False
    for e in reversed(entries):
        if e.get("to_bot", True):
            if "reply" in e or "reaction" in e:
                answered_since = True
            elif answered_since and not e.get("pending"):
                continue
        keep.append(e)
    return recent(keep[::-1]) if limit is None else keep[::-1][-limit:]

# What a channel's history keeps: enough for both transcripts — Jev's can reach further back for messages to jev
# when there's a lot of chatter. +1: the message being answered.
def kept(entries):
    keep = {id(e) for e in recent(entries, HISTORY_TO_BOT + 1, HISTORY_CHATTER)} | {id(e) for e in entries[-(LLM_HISTORY + 1):]}
    return [e for e in entries if id(e) in keep]

def add_history(ch_id, entry):
    h = channel_history[ch_id]
    h.append(entry)
    channel_history[ch_id] = kept(h)
    return entry

# Side conversations: "@jev !nocontext ..." and every Discord reply under it, on any branch, by the !nocontext
# message's id. A reply in one sees all of it and nothing else, and they're never in channel_history, so the rest of
# the channel (and statuses) never see them.
side_talk: dict[int, list[dict]] = {}
side_of: dict[int, int] = {}  # id of each message in one, people's and jev's — to the one it's under

def add_side(root, entry):
    talk = side_talk.setdefault(root, [])
    if all(e["id"] != entry["id"] for e in talk):
        talk.append(entry)
        talk.sort(key=lambda e: e["at"])  # a chain followed back after a restart comes in late
        del talk[:-CHAIN_DEPTH]
    side_of[entry["id"]] = root
    if "reply_id" in entry:
        side_of[entry["reply_id"]] = root
    while len(side_talk) > SIDE_TALKS:
        gone = next(iter(side_talk))
        del side_talk[gone]
        for i in [i for i, r in side_of.items() if r == gone]:
            del side_of[i]
    return entry

# The history entries a message would be in: its side conversation's, or else the channel's
def entries_with(ch_id, message_id):
    root = side_of.get(message_id)
    return side_talk.get(root, []) if root is not None else channel_history.get(ch_id, [])

# Vocab
VOCAB_PATH = Path(__file__).parent / "vocab.txt"
CUSTOM_VOCAB_PATH = Path(__file__).parent / "custom_vocab.txt"
BANNED = {"unanswered"}
ALL_WORDS = [w for w in VOCAB_PATH.read_text().split("\n") if w and w.lower() not in BANNED]
# Punctuation is at the end of vocab.txt, so these are kept past the cutoff — without them jev spells out "period"
# when it wants a full stop. Not quotes or brackets: jev scatters them unpaired ("I?' remember' her")
PUNCTUATION = [".", ",", "!", "?", NEWLINE]
SENTENCE_ENDS = {".", "!", "?"}  # count as votes to end — see loom()
BASE_VOCAB = ALL_WORDS[:VOCAB_SIZE] + [w for w in PUNCTUATION if w not in ALL_WORDS[:VOCAB_SIZE]]
log.info(f"Loaded {len(BASE_VOCAB)} vocab words")

# Server words/phrases: one per line, "#" comments, multi-word phrases are picked as a single unit
def load_custom_vocab():
    if not CUSTOM_VOCAB_PATH.exists():
        return []
    seen = {w.lower() for w in BASE_VOCAB}
    custom = []
    for line in CUSTOM_VOCAB_PATH.read_text().split("\n"):
        w = " ".join(line.split("#", 1)[0].split())
        if w and w.lower() not in BANNED and w.lower() not in seen:
            seen.add(w.lower())
            custom.append(w)
    return custom

CUSTOM_VOCAB = load_custom_vocab()
BASE_VOCAB += CUSTOM_VOCAB
log.info(f"Loaded {len(CUSTOM_VOCAB)} custom vocab words")

# Reactions: unicode emoji from emoji.txt, plus the server's own emoji by their :name:
EMOJI_PATH = Path(__file__).parent / "emoji.txt"
BASE_EMOJI = [e for e in EMOJI_PATH.read_text().split("\n") if e]
log.info(f"Loaded {len(BASE_EMOJI)} reaction emoji")

# An emoji as jev sees it: unicode as itself, the server's own by :name:
def emoji_text(e):
    if isinstance(e, str):
        return e
    if isinstance(e, discord.PartialEmoji) and e.is_unicode_emoji():
        return e.name
    return f":{e.name}:"

def emoji_vocabulary(guild):
    emoji = {e: e for e in BASE_EMOJI}
    if guild:
        emoji |= {f":{e.name}:": e for e in guild.emojis if e.is_usable()}
    return emoji


def is_word(w):
    return w.replace(" ", "").isalnum()


def render(tokens):
    out = ""
    for t in tokens:
        t = "\n" if t == NEWLINE else t
        if not out or out.endswith(("\n", " ")) or t in NO_SPACE_BEFORE:
            out += t
        else:
            out += " " + t
    return re.sub(r"(^|[.!?]\s+|\n)([a-z])", lambda m: m.group(1) + m.group(2).upper(), out.strip(" "))


# One line per turn in the transcript — newlines go back to the token jev chose them as
def unrender(text):
    return text.replace("\n", NEWLINE)


def vocabulary(message):
    seen = set(BASE_VOCAB)
    # Apostrophes only inside a word, and no single letters — "'s" on its own would come out as "He 's" or "He s"
    extra = [w for w in re.findall(r"[a-z]+(?:'[a-z]+)*", message.lower())
             if len(w) > 1 and w not in seen and not seen.add(w)]
    return BASE_VOCAB + extra + [END]


# Words that say the same thing, so penalty() counts them as repeats of each other — otherwise jev dodges it by
# switching between them ("Hi hey hi hey", "Yeah yes yea yes yea yeah"). Only fillers and greetings: words that
# merely look or sound alike ("Google goo woo") or mean nearly the same ("skinny thin") are left alone.
SIMILAR = ["yes yeah yea yep yup ya yah yeh ye yas", "no nah nope naw", "hi hey hello hiya heya howdy hai yo",
           "ok okay k kk okey", "lol lmao lmfao haha hahaha rofl", "dunno idk", "what wat wut",
           "thanks thx ty", "bye cya goodbye", "hmm hm hmmm", "um uh erm uhh umm", "wow whoa woah"]
SAME = {w: g.split()[0] for g in SIMILAR for w in g.split()}
# Hesitation words compete individually, but still count as repeats of each other.
UNPOOLED = NOTHING | {"um", "hmm"}

def same(word):
    return SAME.get(word.lower(), word)

# Emoji that say the same as a word, so a run of "Dunno" replies counts against 🤷 and a run of 🤷 against "Dunno"
EMOJI_SAYS = {"🤷": "dunno"}

def said_as(token):
    return EMOJI_SAYS.get(token) or same(token).lower()

def said_in(text):
    return {said_as(w) for w in re.findall(r"[a-z]+(?:'[a-z]+)*", text.lower())}

# jev's last ECHO_TURNS answers in view — replies and reactions — each as the set of things it said
def recent_answers(history):
    answers = []
    for h in history or []:
        if h["role"] == "assistant":
            answers.append(h["content"])
        elif "reply" in h:
            answers.append(h["reply"])
        elif "reaction" in h:
            answers.append(h["reaction"])
    return [{said_as(a)} | said_in(a) for a in answers[-ECHO_TURNS:]]


# recent: recent_answers() — a word jev said in them is a repeat too, apart from ones in `exempt`, said_in() the
# message it's answering: repeating what someone just said is fair game
def penalty(reply, word, recent=(), exempt=()):
    reply, word = [same(w) for w in reply], same(word)
    local = reply[-REPEAT_WINDOW:].count(word) + 2 * (reply[-1:] == [word])
    p = REPEAT_PENALTY ** local
    seen = reply.count(word)
    alpha = word.replace(" ", "").isalpha()
    if alpha and word.lower() not in STOPWORDS:
        p *= CONTENT_PENALTY ** min(seen, CONTENT_PENALTY_CAP)
    elif alpha:
        p *= STOP_PENALTY ** min(seen, STOP_PENALTY_CAP)
    if said_as(word) not in exempt:
        if said_as(word) in NOTHING:
            p *= NOTHING_PENALTY ** sum(bool(a & NOTHING) for a in recent)
        elif alpha and word.lower() not in STOPWORDS:
            p *= ECHO_PENALTY ** sum(said_as(word) in a for a in recent)
    return p


async def post(session, state, questions):
    body = {"model": MODEL, "state": state, "questions": questions}
    for attempt in range(3):
        try:
            async with session.post(API_URL, json=body, timeout=aiohttp.ClientTimeout(total=30)) as r:
                if r.status < 400:
                    data = await r.json()
                    if (t := trace.get()) is not None:
                        t["cost"] += data.get("usage", {}).get("cost", 0)
                        t["requests"] += 1
                    await set_credit(True)
                    return data.get("answers", {})
                if r.status == 402:  # out of credit — retrying won't help
                    log.warning(f"API 402: {(await r.text())[:300]}")
                    note(error="out of credit")
                    await set_credit(False)
                    return {}
                log.warning(f"API {r.status} {attempt}: {(await r.text())[:300]}")
                await asyncio.sleep(1 + 2 * attempt)
        except Exception as e:
            log.warning(f"API err {attempt}: {e}")
            await asyncio.sleep(1 + 2 * attempt)
    return {}


def by_prob(ans):
    return sorted(ans.get("probabilities", {}).items(), key=lambda kv: -kv[1])


def choice_q(words, instructions):
    return {"type": "choice", "instructions": instructions, "criteria": {w: "" for w in words}}


# done_state: what the "is the reply complete?" question sees, if not state
async def next_word(session, state, vocab, rng, instructions, done_state=None):
    shuffled = list(vocab)
    rng.shuffle(shuffled)
    buckets = [shuffled[i:i + MAX_CHOICES] for i in range(0, len(shuffled), MAX_CHOICES)]
    groups = [buckets[i:i + QUESTIONS_PER_CALL] for i in range(0, len(buckets), QUESTIONS_PER_CALL)]

    results = await asyncio.gather(
        *(post(session, state, {f"b{gi * QUESTIONS_PER_CALL + i}": choice_q(b, instructions)
                                for i, b in enumerate(g)})
          for gi, g in enumerate(groups)),
        post(session, done_state or state, {"complete": {"type": "noul", "instructions": "Is the reply complete?"}}),
    )

    complete_noul = results[-1].get("complete", {}).get("noul", 0)

    finalists = []
    for group_answers in results[:-1]:
        for ans in group_answers.values():
            finalists += [w for w, p in by_prob(ans)[:TOP_PER_BUCKET] if p > 0]

    if END not in finalists:
        finalists.append(END)

    runoff = await post(session, state, {"final": choice_q(finalists[:MAX_CHOICES], instructions)})
    probs = runoff.get("final", {}).get("probabilities", {})

    return probs, complete_noul


# People's reactions to one of jev's replies, with readable mentions of the reactors: " (😂 by @Moss, @Pip; 💀×3)".
# The count is left out when every reactor is named, and kept when some names are missing.
def reacted(counts, reactors=None):
    parts, named = [], False
    for e, n in (counts or {}).items():
        names = [f"@{name}" for name in (reactors or {}).get(e, {}).values() if name]
        text = e if n == 1 or len(names) >= n else f"{e}×{n}"
        parts.append(text + (" by " + ", ".join(names) if names else ""))
        named = named or bool(names)
    return " (" + ("; " if named else " ").join(parts) + ")" if parts else ""  # "; " only to keep names apart

# marked: mark messages addressed to jev for the question check, including pinged replies without a textual
# mention. Keep existing mentions in place rather than adding a second one.
# reactions: jev's own past reactions; their_reactions: people's reactions to jev's replies; reactors: who made them
# at: when the message was sent, to show each turn's time — for an LLM only: Jev would pick the times as words
def transcript(message, author, bot_name, history, words, reactions=True, marked=False, their_reactions=None, at=None,
               reactors=True):
    their_reactions = reactions if their_reactions is None else their_reactions
    def addressed(text, to_bot=True):
        mentioned = re.search(rf"(?<!\w)@{re.escape(bot_name)}(?!\w)", text)
        return f"@{bot_name} {text}" if marked and to_bot and not mentioned else text
    turns, day, minute = [], None, None
    # Like an IRC log: "14:32 pip: ..." when the minute changes, after a "--- Sat 10 Oct" line when the day does.
    # Turns in the same minute go without — a time on every line made a 300-line transcript a third longer. A turn
    # without a time (a reply of jev's from before replies kept theirs) goes in as it is.
    def turn(line, when):
        nonlocal day, minute
        if at and when:
            local = when.astimezone(TIMEZONE)
            if local.date() != day:
                day = local.date()
                turns.append(f"--- {local:%a} {local.day} {local:%b}")
            if f"{local:%H:%M}" != minute:
                minute = f"{local:%H:%M}"
                line = f"{minute} {line}"
        turns.append(line)
    if history:
        for h in history:
            name = bot_name if h["role"] == "assistant" else h["name"]
            text = unrender(h["content"])
            if h["role"] == "assistant" and their_reactions:
                text += reacted(h.get("reactions"), reactors and h.get("reactors"))
            turn(f"{name}: {addressed(text, h['role'] == 'user' and h.get('to_bot', True))}", h.get("at"))
            # jev's answer, if it gave one — without it every earlier question looks unanswered, and jev goes back
            # to them or describes the silence ("crickets"). Past reactions, jev's and people's to its replies,
            # stay out of the question check — jev copies emoji it sees there.
            if "reply" in h:
                turn(f"{bot_name}: {unrender(h['reply'])}"
                     f"{reacted(h.get('reply_reactions'), reactors and h.get('reply_reactors')) if their_reactions else ''}",
                     h.get("reply_at"))
            elif reactions and "reaction" in h:
                turns.append(f"{bot_name}: {h['reaction']}")
    turn(f"{author}: {addressed(unrender(message))}", at)
    turns.append(f"{bot_name}: {unrender(render(words))}")
    return "\n".join(turns)


HEADERS = {"Authorization": f"Bearer {OPENROUTER_KEY}", "Content-Type": "application/json"}


# The emoji jev reacts with instead of replying, or None to reply
async def choose_reaction(message, author, bot_name, emoji, history=None):
    rng = random.Random()
    # Without past reactions — a run of them reads as a habit to keep up, and jev copies the last emoji
    state = transcript(message, author, bot_name, history, [], reactions=False, marked=True)
    note(asked_transcript=state)  # for Context
    shuffled = list(emoji)
    rng.shuffle(shuffled)
    buckets = [shuffled[i:i + MAX_CHOICES] for i in range(0, len(shuffled), MAX_CHOICES)]
    questions = {f"b{i}": choice_q(b, "Reaction?") for i, b in enumerate(buckets[:QUESTIONS_PER_CALL - 1])}
    # Asked as "is it a question?" — "should jev react instead?" scored everything ~0.4-0.55, so no threshold separated them
    questions["asked"] = {"type": "noul", "instructions": f"Is {author}'s last message a question or request for {bot_name}?"}

    async with aiohttp.ClientSession(headers=HEADERS) as session:
        answers = await post(session, state, questions)
        asked = answers.pop("asked", {}).get("noul", 1)
        note(asked=asked)
        if asked >= REACT_THRESHOLD:
            log.info(f"  asked={asked:.2f} reply")
            return None

        if len(answers) > 1:
            finalists = [w for ans in answers.values() for w, p in by_prob(ans)[:TOP_PER_BUCKET] if p > 0]
            runoff = await post(session, state, {"final": choice_q(finalists[:MAX_CHOICES], "Reaction?")})
            probs = runoff.get("final", {}).get("probabilities", {})
        else:
            probs = next(iter(answers.values()), {}).get("probabilities", {})

    # Its own answers count against an emoji, as they do against words: jev's past reactions are hidden from it,
    # but 🤷 still came first in a channel full of its "Dunno? Dunno?" replies
    recent = recent_answers(history)
    scored = {w: p / ECHO_PENALTY ** sum(said_as(w) in a for a in recent) for w, p in probs.items() if p > 0}
    reaction = max(scored, key=scored.get, default=None)
    log.info(f"  asked={asked:.2f} {reaction}")
    # The likeliest before the penalty, with their raw probability and score — the reaction is the best score
    likeliest = sorted(scored, key=probs.get, reverse=True)[:5]
    note(reaction_candidates=[[w, round(probs[w], 3), round(scored[w], 3)] for w in likeliest])
    return reaction


# Most SIMILAR groups vote together; the group with the most votes wins, and its best-scoring word is said.
# Hesitation groups (um/uh/erm and hmm/hm) vote apart, so their sum can't crowd out a substantive word.
# Ending tokens ".", "!" and "?" count with <END>. Apart, they split the vote to end and a word won
# instead: where master ended, jev picked <END> 31% of the time with them in the vocab and 78% without ("Dunno
# google" went on "for", "on", "type"). So they count together, and the reply ends with the mark if jev liked that
# best ("Dunno forgot?"). Not the words that say nothing: pooled, "dunno idk" and "no nah nope" won where jev meant
# something else ("idk" over "hello" at 19%), which undoes NOTHING_PENALTY.
def vote(word, stoppable):
    if stoppable and (word == END or word in SENTENCE_ENDS):
        return END
    return word if said_as(word) in UNPOOLED else same(word).lower()

# The word jev says from scored ({word: score}), whether it ends the reply, and the others that voted with it
def pick(scored, stoppable):
    votes = defaultdict(float)
    for w, s in scored.items():
        votes[vote(w, stoppable)] += s
    best = max(votes, key=votes.get)
    group = sorted((w for w in scored if vote(w, stoppable) == best), key=scored.get, reverse=True)
    return group[0], best == END, group[1:]


# Words jev picks one at a time to continue state(words), until it stops, runs out, or has max_words (MAX_WORDS).
# It can't stop before min_words (MIN_WORDS) real words. done_state(words), if given, is what the "is the reply
# complete?" question sees instead of state(words). recent and exempt go to penalty().
async def loom(state, vocab, instructions, max_words=None, min_words=None, done_state=None, recent=(), exempt=()):
    rng = random.Random()
    words = []
    steps = []
    note(steps=steps, stop="max words")

    async with aiohttp.ClientSession(headers=HEADERS) as session:
        for step in range(max_words or MAX_WORDS):
            probs, complete = await next_word(session, state(words), vocab, rng, instructions,
                                              done_state and done_state(words))
            if not probs:
                note(stop="no answer")
                break

            said = sum(1 for w in words if is_word(w))
            stoppable = said >= (min_words or MIN_WORDS)
            if stoppable and complete >= STOP_THRESHOLD:
                log.info(f"  noul={complete:.2f} stop")
                note(stop=f"done={complete:.2f}")
                break

            scored = {}
            for w, p in probs.items():
                if p <= 0: continue
                if w in NO_SPACE_BEFORE and words[-1:] == [w]: continue
                if w == END and not stoppable: continue
                # Nothing opens with punctuation: leading newlines get stripped anyway, and a leading "?" took over
                # whenever jev's favourite first word was held back ("? Public? Public?", "? No? No?")
                if w in PUNCTUATION and not said: continue
                scored[w] = p / penalty(words, w, recent, exempt)

            if not scored:
                note(stop="nothing left")
                break

            ranked = sorted(scored.items(), key=lambda kv: -kv[1])
            word, ends, pooled = pick(scored, stoppable)

            top3 = [(w, probs.get(w, 0)) for w, _ in ranked[:3]]
            log.info(f"  [{step+1:2d}] {word:12s}  {' '.join(f'{w}:{p:.0%}' for w,p in top3)}  done={complete:.2f}"
                     + ("  ends" if ends else ""))
            # The best-scoring candidates, then every other one at 1% or more, each as [word, probability, score].
            # Asked to pick from 33 countries jev said "No? Idk": "idk" at 23% scored below "no" at 4% after the echo
            # penalty, and the countries split their vote at ~3% each — the top five showed neither.
            likely = [w for w, _ in sorted(scored.items(), key=lambda kv: -probs[kv[0]]) if probs[w] >= 0.01]
            steps.append({"word": word, "done": round(complete, 3), "ends": ends, "pooled": pooled,
                          "top": [[w, round(probs[w], 3), round(scored[w], 3)]
                                  for w in dict.fromkeys([w for w, _ in ranked[:5]] + likely)]})

            if word == END:
                note(stop="<END>")
                break
            words.append(word)
            if ends:
                note(stop=f"<END> as {word}")
                break

    return words


# llm_history: the longer history an LLM gets, if it isn't `history`; at: when the message was sent, for its times
async def generate_reply(message, author, bot_name, history=None, llm_history=None, at=None):
    if (name := model_name) != "jev":
        return await llm_reply(name, message, author, bot_name, history if llm_history is None else llm_history, at)
    # Every word and name people used in the transcript, not just the message being replied to — lets jev say what
    # it can see. Not jev's own words: from a broken reply that would add "garbled" and "unclear" back for reuse.
    vocab = vocabulary(" ".join([f"{h['name']} {h['content']}" for h in history or [] if h["role"] == "user"]
                                + [f"{author} {message}"]))
    # Who reacted is left to the LLMs: names didn't change what Jev said (jev_eval's *-laughed-names), and Jev sends
    # the transcript with every word it asks for
    note(transcript=transcript(message, author, bot_name, history, [], reactors=False))
    # People's reactions stay out of the "complete?" question: with 😂×4 on jev's last reply in view it read a
    # two-word reply as done, and jev stopped at "Dunno forgot" where it had gone on to "Dunno forgot liar bitch"
    words = await loom(lambda words: transcript(message, author, bot_name, history, words, reactors=False),
                       vocab, NEXT_WORD.format(bot_name=bot_name),
                       done_state=lambda words: transcript(message, author, bot_name, history, words,
                                                            their_reactions=False),
                       recent=recent_answers(history), exempt=said_in(message))
    return render(words).strip() or "..."


# How statuses start, taken in turn. Each is the rest of a diary entry — asked "how are you feeling?" jev says
# "Fine thanks", asked "what are you thinking about?" "I dunno anything", and as a status "Is online"
STATUS_STARTS = [["i", "feel"], ["i'm", "thinking", "about"], ["i", "wonder"]]

# The last STATUS_CHAT messages people sent in the channel that heard from someone most recently, one per line
def recent_chat():
    active = [h for ch, h in channel_history.items() if h and ch not in dm_channels]  # the status is public
    if not STATUS_CHAT or not active:
        return ""
    h = max(active, key=lambda h: h[-1]["at"])[-STATUS_CHAT:]
    return "\n".join(f"{e['name']}: {unrender(e['content'])}" for e in h)

def status_state(bot_name, start, chat, words):
    return (f"{chat}\n\n" if chat else "") + f"{bot_name}'s diary, today: {render(start + words)}"


async def generate_status(bot_name, start, chat=""):
    if (name := model_name) != "jev":
        return await llm_status(name, bot_name, start, chat)
    vocab = [w for w in vocabulary(chat) if w != NEWLINE]  # a status is one line
    note(transcript=status_state(bot_name, start, chat, []))
    words = await loom(lambda words: status_state(bot_name, start, chat, words), vocab, NEXT_WORD,
                       max_words=STATUS_MAX_WORDS, min_words=STATUS_MIN_WORDS)
    return render(start + words)[:128] if words else None  # 128: Discord's custom status limit


# The judge sees only the finished status, so unseen chat cannot rescue an otherwise meaningless sentence.
STATUS_RUBRIC = '''Judge a word-by-word generated diary status posted on its own. Its readers cannot see the conversation that inspired it. Accept a status when a reader can recover a feeling, thought, topic, or question from the words themselves. It may be silly, vague, repetitive, philosophical, or grammatically broken. It need not be informative, original, or polished.
Ignore spelling mistakes, broken grammar, weird punctuation, repetition, and a few stray words. Read the whole status. Extra words can add a compatible thought, emphasis, humour, or uncertainty; they must not turn the status into uninterpretable question fragments or unrelated words. Do not rescue a broken sentence by imagining what its writer was responding to. A clear opening alone does not rescue a tail that loses the thought.
An explicit feeling can stand alone: feeling strange and not knowing why is meaningful. A self-contained philosophical thought or understandable question can also stand alone. Conversely, "how it happens" with no identifiable subject, "what this means" with no intelligible thought, or a chain of "what", "is", "about", and "means" may only look like a response to unseen context. Such fragments do not make a useful standalone status.
Examples:
Accept: "I'm thinking about dinner hi for i about tonight what we have fridge and of" — tonight's dinner and what's in the fridge.
Accept: "I feel hot weird strange? Why dunno?" — feels hot and strange, doesn't know why.
Accept: "I'm thinking about something about life meaning is of matter matters." — life's meaning and what matters.
Accept: "I wonder what does mean it? Means meaning itself is itself." — a philosophical thought about meaning itself.
Accept: "I wonder what is happening?.? Dunno confusion confused" — wonders what's happening and expresses confusion.
Accept: "I'm thinking about about boxing about boxing." — repetitive but clearly about boxing.
Borderline: "I feel good. And also else." — a feeling followed by empty filler.
Borderline: "I'm thinking about something about thing of thing." — too vague to identify a topic.
Reject: "I wonder what is how mean means" — no recoverable question.
Reject: "I feel fine. Period. Sans. Fat" — a feeling followed by unrelated words.
Reject: "I'm thinking about about what the about of the of" — empty connective words.
'''
STATUS_QUESTION = ('Does this work as a meaningful standalone diary status under the rubric? Judge only the actual '
                   'words. Do not assume any unseen source conversation. Borderline statuses may pass if they '
                   'contain a recoverable thought.')


async def status_score(mood):
    async with aiohttp.ClientSession(headers=HEADERS) as session:
        answers = await post(session, STATUS_RUBRIC + '\nStatus to judge: ' + json.dumps(mood),
                             {"acceptable": {"type": "noul", "instructions": STATUS_QUESTION}})
    answer = answers.get("acceptable", {})
    score = answer.get("noul") if isinstance(answer, dict) else None
    # Missing or malformed decisions must not publish a candidate or spend more on regeneration.
    return score if type(score) in (int, float) and math.isfinite(score) and 0 <= score <= 1 else None


async def generate_filtered_status(bot_name, start, chat=""):
    parent = trace.get()
    attempts = []
    note(status_attempts=attempts)
    for _ in range(STATUS_ATTEMPTS):
        attempt = {"status": None, "score": None, "accepted": False, "cost": 0.0, "requests": 0}
        attempts.append(attempt)
        token = trace.set(attempt)
        try:
            mood = await generate_status(bot_name, start, chat)
            attempt["status"] = mood
            if not mood or has_credit is False:
                return None
            score = await status_score(mood)
            attempt["score"] = score
            if score is None or has_credit is False:
                return None
            attempt["accepted"] = score >= STATUS_THRESHOLD
            log.info(f"[STATUS] candidate {len(attempts)}/{STATUS_ATTEMPTS}: {mood} "
                     f"(meaning={score:.2f}, {'accepted' if attempt['accepted'] else 'rejected'})")
            if attempt["accepted"]:
                # !why and Context keep using the accepted generation, while the full retry history stays local.
                if parent is not None:
                    parent.update({k: attempt[k] for k in ("transcript", "steps", "stop", "llm", "llm_model", "llm_tokens")
                                   if k in attempt})
                    parent["status_score"] = score
                return mood
        finally:
            trace.reset(token)
            if parent is not None:
                parent["cost"] += attempt["cost"]
                parent["requests"] += attempt["requests"]
                if "error" in attempt:
                    parent["error"] = attempt["error"]
    return None


# Instead of Jev, an ordinary LLM can write the replies and statuses, told to talk like Rocky, the Eridian engineer
# from Project Hail Mary who the bot is named after — !model switches. Jev still decides between replying and
# reacting, and picks the emoji. Tried on jev_eval's conversations: DeepSeek V4 Pro gave the best Rocky ("Is called
# Biscuit. Small predator, no respect for hot liquid.") for ~$0.0005 a reply, against Jev's ~$0.014.
#   shuffle: the example lines in a fresh order every call — Claude ignores temperature, so the same chat got the same
#     reply word for word; DeepSeek is prompt-cached, which a shuffle would break (4.7× the cost)
#   question_dice: the chance a reply may end a question with ", question?" — a model can't count how often it said
#     it, and Haiku put it on 40 of 46 replies even when told to only use it for real questions
#   tail: said at the end of the request, where Claude heeds it — in the system prompt, Haiku still wrote 16 words
#   logprobs: ask for each token's top alternatives, for !why — only some of a model's providers give them
LLM_URL = "https://openrouter.ai/api/v1/chat/completions"
LLMS = {
    "deepseek": {"id": "deepseek/deepseek-v4-pro", "logprobs": True},
    "haiku": {"id": "anthropic/claude-haiku-5.5", "shuffle": True, "question_dice": 0.33,
              "tail": " Like Rocky: a few words, one short sentence at most."},
    "kimi": {"id": "moonshotai/kimi-k2-0905"},
}
LLM_MAX_TOKENS = 60             # only to stop a runaway reply — Rocky decides how much to say
LLM_TEMPERATURE = 1.0
LLM_FREQUENCY_PENALTY = 0.5
LLM_TOP_LOGPROBS = 5
LLM_ATTEMPTS = 2                # tries for a reply that isn't empty (Kimi sometimes says nothing)

# Real lines of Rocky's from the book and film, one per line ("#" comments), shown to the LLM as examples. Gitignored:
# they're quotes from copyrighted works, so they stay out of this public repo. Without the file it goes by the rules.
ROCKY_LINES_PATH = Path(__file__).parent / "rocky_lines.txt"
ROCKY_LINES = ([l.strip() for l in ROCKY_LINES_PATH.read_text().split("\n") if l.strip() and not l.startswith("#")]
               if ROCKY_LINES_PATH.exists() else [])
log.info(f"Loaded {len(ROCKY_LINES)} Rocky lines")

# Each rule answers something the first tries got wrong: "What is scone, question?" in a third of replies (he was
# learning English in the book), ", question?" on everything, replies to everyone in the chat at once
ROCKY = """You are {bot}, a bot in a Discord server of friends. You talk like Rocky, the Eridian engineer from Project Hail Mary, who you're named after.

How {bot} talks:
- Compact, concrete sentences. Drops articles and helper verbs ("I make new one", "You are friend", "Is good"), but the thought is always clear: an observation, an opinion, a reason, a plan or a request.
- A real question can end with ", question?": "Why stupid, question?". Only when he truly asks something — never filler like "Why you ask, question?" — and not if his own previous message used it. Very rarely a firm conclusion ends with ", statement.".
- Repetition has a job — excitement, distress, urgency: "Amaze, amaze, amaze!", "Bad, bad, bad." Plain answers are said once.
- Blunt, sometimes bossy, curious, competitive, teasing, now and then sarcastic. Earnest, practical warmth for his friends.
- He has lived in this server a long time and knows everyday human things: food, films, music, games, weather, jobs. He only asks what a word means when it is truly strange slang or an idiom ("No understand word."), and takes idioms literally.
- Answers what the friends actually said, with an opinion. If he truly doesn't know, "I not know." — rarely.
- He's in a Discord chat, not on a spaceship: talks about whatever the chat is about. No Grace, Erid, Astrophage or space unless someone brings it up.
- Says one thing, to the last message only — not a reply to everyone in the chat. Usually a few words or one short sentence, like the lines below; at most two short sentences.
- Never emoji, never says he's a bot or AI. Don't copy the lines below word for word or repeat your earlier replies."""

LLM_STATUS = """Write {bot}'s new Discord custom status, as the rest of a diary entry that starts "{start}". In {bot}'s voice; a reader with no context should get a feeling, thought or question from it. One line, at most 12 words after the opener. Output the whole status, starting with "{start}"."""

def rocky_prompt(bot_name, shuffle=False):
    prompt = ROCKY.format(bot=bot_name)
    if ROCKY_LINES:
        lines = random.sample(ROCKY_LINES, len(ROCKY_LINES)) if shuffle else ROCKY_LINES
        prompt += "\n\nReal Rocky lines from the book and film:\n\n" + "\n".join(lines)
    return prompt

# The reply on its own: no "rocky:" in front, no quotes around it, and only the first paragraph — an LLM sometimes
# goes on to write the next turns of the chat
def clean_llm(text, bot_name):
    text = re.sub(rf"^{re.escape(bot_name)}:\s*", "", (text or "").strip(), flags=re.I)
    return text.split(f"\n{bot_name}:")[0].split("\n\n")[0].strip().strip('"').strip()

# The LLM's answer to messages, and its tokens with their top alternatives ([{"token", "p", "top": [[token, p]]}])
# when the model gives them. Costs go on the trace like Jev's.
# A response's logprobs as [{"token", "p", "top": [[token, p]]}]. Some providers give a token the previous one's
# alternatives again ("ers" with "isk"'s list, after "Wh" "isk") — those show only the token itself. The end of the
# reply comes as a token too ("<｜end▁of▁sentence｜>"), and isn't something rocky said.
END_TOKEN = re.compile(r"<[^<>\s]*end[^<>\s]*>", re.IGNORECASE)

def llm_tokens(content):
    tokens = []
    for x in content or []:
        top = [[y["token"], round(math.exp(y["logprob"]), 4)] for y in x.get("top_logprobs", [])]
        if tokens and x["token"] not in dict(top) and [w for w, _ in top] == [w for w, _ in tokens[-1]["top"]]:
            top = []
        tokens.append({"token": x["token"], "p": round(math.exp(x["logprob"]), 4), "top": top})
    while tokens and END_TOKEN.fullmatch(tokens[-1]["token"]):
        tokens.pop()
    return tokens

async def llm(name, messages, max_tokens=LLM_MAX_TOKENS):
    spec = LLMS[name]
    body = {"model": spec["id"], "messages": messages, "max_tokens": max_tokens, "temperature": LLM_TEMPERATURE,
            "frequency_penalty": LLM_FREQUENCY_PENALTY, "reasoning": {"enabled": False}, "usage": {"include": True}}
    if spec.get("logprobs"):
        # Only to providers that give them — some of DeepSeek's don't
        body |= {"logprobs": True, "top_logprobs": LLM_TOP_LOGPROBS, "provider": {"require_parameters": True}}
    async with aiohttp.ClientSession(headers=HEADERS) as session:
        for attempt in range(3):
            try:
                async with session.post(LLM_URL, json=body, timeout=aiohttp.ClientTimeout(total=60)) as r:
                    if r.status == 402:  # out of credit — retrying won't help
                        log.warning(f"LLM 402: {(await r.text())[:300]}")
                        note(error="out of credit")
                        await set_credit(False)
                        return "", []
                    data = await r.json(content_type=None)
                    if r.status >= 400 or "choices" not in data:
                        log.warning(f"LLM {r.status} {attempt}: {str(data)[:300]}")
                        await asyncio.sleep(1 + 2 * attempt)
                        continue
            except Exception as e:
                log.warning(f"LLM err {attempt}: {e}")
                await asyncio.sleep(1 + 2 * attempt)
                continue
            if (t := trace.get()) is not None:
                t["cost"] += data.get("usage", {}).get("cost") or 0
                t["requests"] += 1
            await set_credit(True)
            choice = data["choices"][0]
            return choice["message"].get("content") or "", llm_tokens((choice.get("logprobs") or {}).get("content"))
    return "", []

# name: one of LLMS — passed in, since !model can switch while a reply is being written
async def llm_reply(name, message, author, bot_name, history=None, at=None):
    spec = LLMS[name]
    state = transcript(message, author, bot_name, history, [], at=at or datetime.now(timezone.utc))
    note(transcript=state, llm=name, llm_model=spec["id"], history=history or [])  # what it saw, not Jev's share
    chat = state.rsplit("\n", 1)[0]  # without its own empty turn, which the request asks for instead
    # Says who it's answering: asked for "rocky's next message", Haiku kept opening with the name its earlier replies
    # did ("Binja, ...") when someone else asked. Not quoting the message — it echoed a name in it back.
    ask = (f"The chat so far, times in {TIMEZONE.key}:\n\n{chat}\n\n"
           f"Write {bot_name}'s reply to {author}'s last message. Output only the message."
           + spec.get("tail", ""))
    if (dice := spec.get("question_dice")) is not None and random.random() >= dice:
        ask += ' This time, no ", question?" tag.'
    messages = [{"role": "system", "content": rocky_prompt(bot_name, spec.get("shuffle"))},
                {"role": "user", "content": ask}]
    for _ in range(LLM_ATTEMPTS):
        text, tokens = await llm(name, messages)
        if reply := clean_llm(text, bot_name):
            if tokens:
                note(llm_tokens=tokens)
            log.info(f"  {name}: {reply}")
            return reply
        if has_credit is False:
            break
    return "..."

async def llm_status(name, bot_name, start, chat=""):
    spec = LLMS[name]
    opener = render(start)
    ask = (f"Recent chat in the server:\n{chat}\n\n" if chat else "") + LLM_STATUS.format(bot=bot_name, start=opener)
    note(transcript=ask, llm=name, llm_model=spec["id"])
    text, tokens = await llm(name, [{"role": "system", "content": rocky_prompt(bot_name, spec.get("shuffle"))},
                              {"role": "user", "content": ask}])
    mood = " ".join(clean_llm(text, bot_name).split())  # one line
    if not mood:
        return None
    if tokens:
        note(llm_tokens=tokens)
    if not mood.lower().startswith(opener.lower()):
        mood = f"{opener} {mood[0].lower()}{mood[1:]}"
    return mood[:128]  # Discord's custom status limit

# Which model writes: "jev" or one of LLMS. Kept in MODEL_PATH (gitignored), so a restart keeps what !model chose.
MODEL_PATH = Path(__file__).parent / "model.json"

def load_model():
    try:
        name = json.loads(MODEL_PATH.read_text())["model"]
    except FileNotFoundError:
        return "jev"
    except Exception as e:
        log.warning(f"Loading {MODEL_PATH.name} failed: {e}")
        return "jev"
    if name != "jev" and name not in LLMS:
        log.warning(f"{MODEL_PATH.name}: unknown model {name!r}, using jev")
        return "jev"
    return name

def save_model(name):
    try:
        MODEL_PATH.write_text(json.dumps({"model": name}) + "\n")
    except OSError as e:
        log.warning(f"Writing {MODEL_PATH.name} failed: {e}")

model_name = load_model()
log.info(f"Model: {model_name}")


# DMs: only from the Discord user IDs in dm_users.txt (gitignored, one per line, "#" comments) — every reply costs
# credit, and nobody else sees what's said in private. Restart the bot to pick up changes.
DM_USERS_PATH = Path(__file__).parent / "dm_users.txt"

def load_dm_users():
    if not DM_USERS_PATH.exists():
        return set()
    users = set()
    for line in DM_USERS_PATH.read_text().split("\n"):
        if w := line.split("#", 1)[0].strip():
            try:
                users.add(int(w))
            except ValueError:
                log.warning(f"{DM_USERS_PATH.name}: {w!r} isn't a user ID")
    return users

DM_USERS = load_dm_users()
log.info(f"Loaded {len(DM_USERS)} DM user(s)")

# Discord
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)
gen_lock = asyncio.Lock()

# The Why and Context actions (below) are registered with Discord each start, so a change to them shows up. If that
# fails, the ones registered last time stay — no reason to keep jev offline.
async def setup_hook():
    try:
        synced = await bot.tree.sync()
        log.info(f"Synced {len(synced)} app command(s)")
    except discord.HTTPException as e:
        log.warning(f"Registering app commands failed: {e}")
bot.setup_hook = setup_hook

# Stopping: the first Ctrl-C (or SIGINT/SIGTERM) takes no new messages and lets the ones being answered finish —
# catch_up() answers what came in meanwhile on the next start. Another Ctrl-C quits at once. A signal sent to the
# whole process group (a service manager stopping it, say) reaches Python twice under `uv run` — directly and
# forwarded by uv — so signals within a second of the first count as one. A terminal's Ctrl-C arrives once.
stopping = asyncio.Event()
handling: set[asyncio.Task] = set()

# Out of OpenRouter credit, jev shows as idle with a status saying so — otherwise it just answers "..." and nobody
# knows why. None until known; any request that goes through, or a check every CREDIT_CHECK_EVERY, puts it back.
CREDIT_CHECK_EVERY = 300        # seconds
# What jev answers with meanwhile. The .gif itself, not its klipy page — that unfurls as a "KLIPY: … View & Share" card.
NO_MONEY = [
    "https://static2.klipy.com/ii/2711dd8a75a85be822d136ec94899b3f/6c/19/i4pB3OVh.gif",             # wallet
    "https://static2.klipy.com/ii/925f17378dd1893b674a723c07535afe/85/6c/wfANYRWk.gif",             # wallet penacony
    "https://static2.klipy.com/ii/39f2394ae36df6e199be9eb7c9fa1012/b4/ac/scOGuksw.gif",             # donald duck
    "https://static2.klipy.com/ii/d6b0ce929193df3c242ac34b5654d2ce/9e/c1/5zMKqCkz.gif",             # no money broke
    "https://static2.klipy.com/ii/935d7ab9d8c6202580a668421940ec81/fd/f0/8IO0ioVm.gif",             # al bundy
    "https://static2.klipy.com/ii/4493325008d34b7bf8cd6813cd5c1619/7c/91/P9M6TXIqsKJx.gif",         # we have no money
]
has_credit: bool | None = None
status: discord.CustomActivity | None = None  # update_status()'s latest, shown whenever jev has credit
credit_check: asyncio.Task | None = None

async def set_credit(ok):
    global has_credit, credit_check
    if ok == has_credit:
        return
    has_credit = ok
    log.info(f"[CREDIT] {'back' if ok else 'out of credit'}")
    await show_credit()
    if not ok and not (credit_check and not credit_check.done()):
        credit_check = asyncio.create_task(wait_for_credit())

async def show_credit():
    if not bot.is_ready():
        return  # on_ready shows it
    try:
        if has_credit is False:
            await bot.change_presence(status=discord.Status.idle, activity=discord.CustomActivity("out of credit"))
        else:
            await bot.change_presence(status=discord.Status.online, activity=status)
    except Exception as e:
        log.warning(f"Changing status failed: {e}")

# What's left to spend: the account's credit, or the key's own limit if that's lower. None if the check failed.
async def credit_left():
    try:
        async with aiohttp.ClientSession(headers=HEADERS, timeout=aiohttp.ClientTimeout(total=30)) as session:
            async with session.get("https://openrouter.ai/api/v1/credits") as r:
                credits = (await r.json())["data"]
            async with session.get("https://openrouter.ai/api/v1/key") as r:
                key = (await r.json())["data"]
    except Exception as e:
        log.warning(f"Checking credit failed: {e}")
        return None
    left = credits["total_credits"] - credits["total_usage"]
    if key.get("limit_remaining") is not None:
        left = min(left, key["limit_remaining"])
    return left

async def wait_for_credit():
    while has_credit is False:
        await asyncio.sleep(CREDIT_CHECK_EVERY)
        if (left := await credit_left()) is not None and left > 0:
            await set_credit(True)

# The role Discord creates for the bot (same name, e.g. "@rocky") — mentioning it counts as mentioning the bot
def bot_role(m):
    return m.guild.self_role if m.guild else None

# Ignore bot mentions when parsing commands.
def strip_mention(m):
    mention = rf"<@!?{bot.user.id}>" + (f"|{re.escape(role.mention)}" if (role := bot_role(m)) else "")
    return re.sub(mention, "", m.content).strip()

# Links, e.g. a GIF from Discord's picker (https://klipy.com/gifs/azumanga-daioh-sakai) — shown as a tag with the
# embed's title (a tweet's has none, so its author and the start of its text), or the link's site and path words while there's no embed.
# Raw, "https" went into the vocab, and jev picked it ("Yeah yes https https too is").
LINK = re.compile(r"<?(https?://[^\s<>]+)>?")
GIF_HOSTS = {"klipy.com", "tenor.com", "giphy.com"}
SECOND_LEVEL = {"co", "com", "org", "net", "gov", "ac"}
EMBED_WORDS = 20

def slug_words(text):
    return [w for w in re.split(r"[/\-_.+]", text) if w.isalpha()]

# The start of a tweet's text: its first line, plain (the embed's is markdown), cut at EMBED_WORDS words
def embed_text(description):
    line = (description or "").strip().split("\n")[0]
    line = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", line)  # [@someone](https://x.com/someone) -> @someone
    words = re.sub(r"\\(.)|\*\*", r"\1", line).split()
    return " ".join(words[:EMBED_WORDS]) + (" ..." if len(words) > EMBED_WORDS else "")

def link_tag(url, embed=None):
    u = urlsplit(url)
    host = (u.hostname or "").removeprefix("www.")
    gif = ".".join(host.split(".")[-2:]) in GIF_HOSTS or (embed is not None and embed.type == "gifv")
    if embed is not None and embed.title:
        words = embed.title
    elif embed is not None and (author := re.sub(r" \(@\S+\)$", "", embed.author.name or "")):
        words = author + (f": {text}" if (text := embed_text(embed.description)) else "")
    elif gif:  # just the slug: klipy/tenor/giphy's path boilerplate and IDs say nothing about the GIF
        words = " ".join(w for w in slug_words(u.path.rsplit("/", 1)[-1]) if w.lower() != "gif")
    else:
        words = " ".join([h for h in host.split(".")[:-1] if h not in SECOND_LEVEL] + slug_words(u.path))
    kind = "gif" if gif else "link"
    return f"[{kind}: {words}]" if words else f"[{kind}]"

# The embed Discord made for url — its URL can differ (x.com comes back as twitter.com), so matched by path,
# or taken as is when there's one link and one embed (a YouTube short's embed is a watch?v= link)
def embed_for(url, m, only):
    embeds = [e for e in m.embeds if e.url]
    path = urlsplit(url).path.rstrip("/")
    return (next((e for e in embeds if urlsplit(e.url).path.rstrip("/") == path), None)
            or (embeds[0] if only and len(embeds) == 1 else None))

def attachment_tag(a):
    if a.content_type == "image/gif":
        return "[gif]"
    kind = (a.content_type or "").split("/")[0]
    return {"image": "[photo]", "video": "[video]", "audio": "[audio]"}.get(kind, "[file]")

# "@jev !nocontext ..." answers with no history in view. Taken out of the text, so jev never sees it, and "nocontext"
# never reaches the vocab from the message or, later, from the history.
NO_CONTEXT = re.compile(r"(?<!\S)!nocontext(?!\S)[ \t]*", re.IGNORECASE)

def no_context(m):
    return should_respond(m) and NO_CONTEXT.search(strip_mention(m)) is not None

# The Discord replies m is the end of, oldest first, without m — up to CHAIN_DEPTH back, or to a message that's gone
async def reply_chain(m):
    chain, cur = [], m
    while len(chain) < CHAIN_DEPTH and cur.reference and cur.reference.message_id:
        r = cur.reference.resolved
        if not isinstance(r, discord.Message):  # Discord only sends the first one up with m
            r = bot._connection._get_message(cur.reference.message_id)
        if r is None:
            try:
                r = await m.channel.fetch_message(cur.reference.message_id)
            except discord.HTTPException:  # deleted, or no Read Message History
                break
        chain.append(r)
        cur = r
    return chain[::-1]

# History entries for a reply chain, jev's replies under what they answered. Its other messages in it (a !why chart)
# are left out.
async def chain_entries(chain):
    entries = []
    for x in chain:
        if x.author.id != bot.user.id:
            if entry := history_entry(x):
                entries.append(entry)
        elif (entries and x.reference and entries[-1]["id"] == x.reference.message_id and "reply" not in entries[-1]
              and x.content and x.content not in NO_MONEY):
            entries[-1]["reply"], entries[-1]["reply_id"], entries[-1]["reply_at"] = x.content, x.id, x.created_at
            await attach_reactions(entries[-1], x)
    return entries

# The side conversation m is in, by the id of its !nocontext message — m's own, if it has one — or None. One jev
# hasn't seen (from before a restart, say) is found by following m's replies back, and starts with that chain.
async def side_root(m):
    if no_context(m):
        return m.id
    if not (m.reference and m.reference.message_id):
        return None
    parent = m.reference.message_id
    if (root := side_of.get(parent)) is not None:
        return root
    if any(parent in (e["id"], e.get("reply_id")) for e in channel_history.get(m.channel.id, [])):
        return None  # a reply to the channel's conversation
    chain = await reply_chain(m)
    start = next((i for i in reversed(range(len(chain))) if no_context(chain[i])), None)
    if start is None:
        return None
    root = chain[start].id
    for x in chain[start:]:
        side_of[x.id] = root
    for entry in await chain_entries(chain[start:]):
        add_side(root, entry)
    return root

# m as jev sees it: mentions as readable names, without !nocontext if it's to jev, links as tags, and a tag for each attachment
# and sticker — otherwise a photo on its own is an empty message, dropped or read as "hello"
def message_text(m):
    text = m.clean_content
    if should_respond(m):
        text = NO_CONTEXT.sub("", text)
    only = len(LINK.findall(text)) == 1
    text = LINK.sub(lambda l: link_tag(l[1], embed_for(l[1], m, only)), text)
    tags = [attachment_tag(a) for a in m.attachments] + [f"[sticker: {s.name}]" for s in m.stickers]
    return " ".join([text.strip(), *tags]).strip()

# The message of jev's that m is a Discord reply to, if any
def replied_to_bot(m):
    r = m.reference and m.reference.resolved
    return r if isinstance(r, discord.Message) and r.author.id == bot.user.id else None

def should_respond(m):
    if m.author.bot: return False
    if m.guild is None: return m.author.id in DM_USERS  # in a DM, every message is to jev
    if bot.user in m.mentions: return True
    if (role := bot_role(m)) and role in m.role_mentions: return True
    return False  # replying with the author ping off doesn't ask jev to respond

# channel_history is in memory, so rebuild it from Discord the first time a channel talks to jev after a restart
history_loaded: dict[int, asyncio.Task] = {}

# !why on its own in a message, and !model with or without a model's name — the command, or None
COMMANDS = {"!why", "!model"}

def command(m):
    if m.author.bot:
        return None
    c = strip_mention(m).lower().split()
    if c and c[0] in COMMANDS and (len(c) == 1 or c[0] == "!model" and len(c) == 2):
        return c[0]
    return None

# m as a history entry, or None for messages that never go in one (bots, including jev itself, empty ones, and
# commands — often a reply to jev, so after a restart one came back as a message jev never answered)
def history_entry(m):
    if m.author.bot or command(m):
        return None
    to_bot = should_respond(m)
    content = message_text(m) or ("hello" if to_bot else "")
    if not content:
        return None
    entry = {"role": "user", "name": m.author.display_name, "content": content,
             "at": m.created_at, "id": m.id, "to_bot": to_bot}
    if to_bot and (r := next((r for r in m.reactions if r.me), None)):
        entry["reaction"] = emoji_text(r.emoji)
    return entry

# People's reactions to a message of jev's, emoji to count — not jev's own
def reactions_to(m):
    return {emoji_text(r.emoji): n for r in m.reactions if (n := r.count - r.me)}


# Server nickname lookups for reactors, by (guild id, user id). The lookup itself is kept, so a history rebuild
# loading every reply's reactions at once asks Discord about each person once. A live reaction refreshes it.
reactor_names = {}

async def member_name(guild, user_id):
    try:
        return (guild.get_member(user_id) or await guild.fetch_member(user_id)).display_name
    except discord.HTTPException:
        return None  # left the server — their own name will do

# A reactor's name as the server shows it, like the mentions elsewhere in the transcript. Without the members
# intent Discord hands back plain users, whose display name is the global one, so the member is looked up.
async def reactor_name(guild, user_id, user=None):
    if isinstance(user, discord.Member):
        reactor_names.pop((user.guild.id, user_id), None)
        return user.display_name
    if guild:
        if (guild.id, user_id) not in reactor_names:
            reactor_names[guild.id, user_id] = asyncio.ensure_future(member_name(guild, user_id))
        if name := await reactor_names[guild.id, user_id]:
            return name
    try:
        user = user or bot.get_user(user_id) or await bot.fetch_user(user_id)
    except discord.HTTPException:
        return None
    return user.display_name

# Loading history needs Discord's reaction-user endpoint; live events maintain the same map by user ID.
# Keep counts even when Discord cannot return users, so the feedback is still visible.
async def attach_reactions(entry, m, prefix="reply_"):
    counts = reactions_to(m)
    if not counts:
        return
    entry[prefix + "reactions"] = counts
    reactors = entry[prefix + "reactors"] = {}
    guild = getattr(m, "guild", None)

    async def load(r):
        users = reactors[emoji_text(r.emoji)] = {}
        try:
            async for user in r.users():
                if user.id != bot.user.id:
                    users[str(user.id)] = await reactor_name(guild, user.id, user)
        except discord.HTTPException:
            log.warning("Could not load reaction users for message %s", m.id)

    await asyncio.gather(*(load(r) for r in m.reactions if emoji_text(r.emoji) in counts))

async def load_history(first):
    ch = first.channel.id
    known = {e["id"] for e in channel_history[ch]}  # chatter already recorded live since startup
    known |= side_of.keys()
    scanned, found, replies = [], [], {}
    try:
        async for m in first.channel.history(limit=HISTORY_SCAN, before=first):
            scanned.append(m)
    except Exception as e:  # no Read Message History permission — start empty, like before
        log.warning(f"Loading history for {ch} failed: {e}")
    for m in reversed(scanned):  # oldest first, so a side conversation's !nocontext comes before the replies under it
        if (root := m.id if no_context(m) else m.reference and side_of.get(m.reference.message_id)) is not None:
            side_of[m.id] = root
        if m.author.id == bot.user.id and m.reference and m.content and m.content not in NO_MONEY:
            replies[m.reference.message_id] = m  # jev's reply, to go back under the message it answered
        elif m.id not in known and (entry := history_entry(m)):
            found.append((entry, root))
    talk, reacting = [], []
    for entry, root in found:
        if r := replies.get(entry["id"]):
            entry["reply"], entry["reply_id"], entry["reply_at"] = r.content, r.id, r.created_at
            reacting.append(attach_reactions(entry, r))
        if root is None:
            talk.append(entry)
        else:
            add_side(root, entry)
    await asyncio.gather(*reacting)  # all at once: someone is waiting on the reply this history is for
    channel_history[ch] = kept(sorted(channel_history[ch] + talk, key=lambda e: e["at"]))
    log.info(f"Loaded {len(channel_history[ch])} history entries for {ch}")

@bot.event
async def on_ready():
    log.info(f"jev online as {bot.user} | vocab {len(BASE_VOCAB)} | {NEXT_WORD!r}")
    if has_credit is None and (left := await credit_left()) is not None:
        await set_credit(left > 0)
    await show_credit()  # a reconnect starts over as online, with no status — put the latest back
    await catch_up()
    # on_ready runs again after a reconnect, so only start it the first time
    if STATUS_EVERY and not update_status.is_running():
        update_status.start()

# A new custom status for jev, shown in every server and on its profile. The starts take turns, and every other
# status has recent chat in view (and its words in the vocab) — so it can say what people said, anywhere jev is.
# 3 starts, alternating chat: each start gets a turn with and without it.
# The latest is kept in STATUS_PATH, so a restart shows it again instead of paying for a new one, and the next
# comes when it's due — with the next start, not "I feel" every time.
# One is only made once someone has said something in a server since the last (or since jev started), so a quiet
# night doesn't cost anything or fill STATUS_CHANNEL with statuses talking to nobody.
STATUS_PATH = Path(__file__).parent / "status.json"
status_turn = 0      # which start and chat the next status gets
status_at = None     # when the kept one was made
heard = False        # whether someone has spoken since then

def load_status():
    global status, status_turn, status_at
    try:
        saved = json.loads(STATUS_PATH.read_text())
        status = discord.CustomActivity(name=saved["status"])
        status_turn, status_at = saved["turn"] + 1, datetime.fromisoformat(saved["at"])
        log.info(f"[STATUS] kept from {saved['at']}: {saved['status']}")
    except FileNotFoundError:
        pass
    except Exception as e:  # unreadable — make a new one
        log.warning(f"Loading {STATUS_PATH.name} failed: {e}")

def save_status(mood, turn, at):
    try:
        STATUS_PATH.write_text(json.dumps({"status": mood, "turn": turn, "at": at}, ensure_ascii=False) + "\n")
    except OSError as e:
        log.warning(f"Writing {STATUS_PATH.name} failed: {e}")

if STATUS_EVERY:
    load_status()

@tasks.loop(minutes=STATUS_EVERY or 1)
async def update_status():
    global status, status_turn, heard
    if has_credit is False:  # showing "out of credit" — and every request would fail anyway
        return
    if not heard:
        log.info("[STATUS] nobody's spoken since the last, keeping it")
        return
    heard = False  # before making it, so a message that comes meanwhile counts for the next
    bot_name = bot.user.display_name
    t = {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "status": None, "bot_name": bot_name,
         "cost": 0.0, "requests": 0}
    trace.set(t)
    start = time.monotonic()
    try:
        async with gen_lock:  # after any reply in progress, not alongside it
            n, status_turn = status_turn, status_turn + 1
            chat = recent_chat() if n % 2 else ""
            mood = await generate_filtered_status(bot_name, STATUS_STARTS[n % len(STATUS_STARTS)], chat)
        if not mood:  # no acceptable candidate, or the API didn't answer — keep the old status
            heard = True
            return
        status = discord.CustomActivity(name=mood)
        save_status(mood, n, t["at"])
        await show_credit()
        t["status"] = mood
        await post_status(mood)
        log.info(f"[STATUS] {mood} (${t['cost']:.5f})")
    except Exception as e:  # keep the old status and try again next time — an uncaught error would end the loop
        log.error(f"Status error: {e}", exc_info=True)
        heard = True
        t["error"] = repr(e)
    finally:
        t["seconds"] = round(time.monotonic() - start, 1)
        write_trace(t)

# A status is gone when the next comes, so each is also posted in STATUS_CHANNEL, to keep and react to. jev's own
# messages never enter the history, so a posted status isn't in the next one's recent chat. Its id is logged, so
# !why and Context can find it.
async def post_status(mood):
    if not STATUS_CHANNEL:
        return
    try:
        channel = bot.get_channel(STATUS_CHANNEL) or await bot.fetch_channel(STATUS_CHANNEL)
        sent = await channel.send(mood, allowed_mentions=discord.AllowedMentions.none())
        note(status_id=sent.id)
    except Exception as e:  # the status itself is already showing
        log.warning(f"Posting the status in {STATUS_CHANNEL} failed: {e}")

@update_status.before_loop
async def wait_for_status():  # the kept status stays until it's due
    if status_at and (due := status_at + timedelta(minutes=STATUS_EVERY)) > datetime.now(timezone.utc):
        log.info(f"[STATUS] next at {due.isoformat(timespec='seconds')}")
        await asyncio.sleep((due - datetime.now(timezone.utc)).total_seconds())

# Messages sent while jev was offline (a restart, an outage) never reach on_message. On (re)connect, answer each
# channel's latest message to jev from the last CATCH_UP_WINDOW minutes since jev last replied or reacted there —
# only the latest, so a long outage doesn't bring a burst of replies to messages people have moved on from.
async def catch_up():
    since = datetime.now(timezone.utc) - timedelta(minutes=CATCH_UP_WINDOW)
    channels = [ch for guild in bot.guilds for ch in [*guild.text_channels, *guild.threads]
                if (perms := ch.permissions_for(guild.me)).read_messages and perms.read_message_history]
    for user_id in DM_USERS:
        try:
            channels.append(await (await bot.fetch_user(user_id)).create_dm())
        except discord.HTTPException as e:
            log.warning(f"Opening a DM with {user_id} failed: {e}")
    missed = []
    for ch in channels:
        try:
            found = [m async for m in ch.history(limit=HISTORY_SCAN, after=since, oldest_first=False)]
        except discord.HTTPException as e:
            log.warning(f"Catching up on {ch.id} failed: {e}")
            continue
        mine = [m for m in found if m.author.id == bot.user.id]
        answered = {m.reference.message_id for m in mine if m.reference}
        handled = [m for m in found if m.id in answered or any(r.me for r in m.reactions)]
        # Only what came after jev last replied or reacted here — anything older was left behind, not missed,
        # and answering it would walk back one more old message on every reconnect
        last = max((m.created_at for m in mine + handled), default=since)
        unanswered = [m for m in found if m.created_at > last and should_respond(m) and not command(m)]
        if unanswered:
            log.info(f"[CATCH UP] #{ch} missed {len(unanswered)} message(s) to jev, answering the latest")
            missed.append(max(unanswered, key=lambda m: m.created_at))
    for m in sorted(missed, key=lambda m: m.created_at):
        if stopping.is_set():
            break
        await respond(m, caught_up=True)

@bot.event
async def on_message(m):
    global heard
    if m.guild is None and m.author.id not in DM_USERS:
        return  # a DM from anyone else, or jev's own — not even kept as history
    if m.guild and not m.author.bot:  # not jev's own posts, or each status would earn the next
        heard = True
    if cmd := command(m):
        if not stopping.is_set():
            await {"!why": why, "!model": model_command}[cmd](m)
        return
    if not should_respond(m):
        # Not for jev, but part of the conversation it might be asked about
        if HISTORY_CHATTER and (entry := history_entry(m)):
            if (root := await side_root(m)) is not None:
                add_side(root, entry)
            else:
                add_history(m.channel.id, entry)
        return
    if stopping.is_set():
        return
    await respond(m)


# Discord often adds a link's embed just after the message arrives, as an edit — and people fix typos
@bot.event
async def on_message_edit(before, after):
    entry = next((e for e in entries_with(after.channel.id, after.id) if e.get("id") == after.id), None)
    if entry and (content := message_text(after)):
        entry["content"] = content


# !why: what jev weighed for one of its answers, from the logs — the reply the !why is a Discord reply to (or the
# message it reacted to), or on its own jev's latest answer in the channel, as why_chart's chart. Costs nothing: no
# API calls.
WHY_DAYS = 7  # how many days of logs it looks back through

# The chart's font has no emoji, so a reaction's are drawn from images: Twemoji's, which Discord's are, and the
# server's own from Discord. Kept for as long as jev runs; one that can't be fetched is shown by name instead.
TWEMOJI = "https://cdn.jsdelivr.net/gh/jdecked/twemoji@17.0.3/assets/72x72/{}.png"
emoji_images: dict[str, bytes] = {}

def emoji_url(e, guild):
    if e.startswith(":"):
        custom = guild and discord.utils.get(guild.emojis, name=e.strip(":"))
        return custom and f"https://cdn.discordapp.com/emojis/{custom.id}.png?size=64"  # .png: animated ones too
    # Twemoji's file names: the code points, without the variation selector unless it's a sequence joined by ZWJ
    return TWEMOJI.format("-".join(f"{ord(c):x}" for c in (e if "\u200d" in e else e.replace("\ufe0f", ""))))

async def emoji_image(session, e, guild):
    if (url := emoji_url(e, guild)) and url not in emoji_images:
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as r:
                r.raise_for_status()
                emoji_images[url] = await r.read()
        except (aiohttp.ClientError, asyncio.TimeoutError) as ex:
            log.warning(f"Fetching {e} failed: {ex.status if isinstance(ex, aiohttp.ClientResponseError) else repr(ex)}")
    return emoji_images.get(url)

# Whether a log entry is for something jev did in the channel: an answer there, or in STATUS_CHANNEL a status
def in_channel(t, channel_id):
    if "channel_id" in t:
        return t["channel_id"] == channel_id
    return bool(t.get("status")) and channel_id == STATUS_CHANNEL

# The latest log entry in the channel that has(), or the one for `target` if given — by default an answer or status
# with candidates to show
def find_trace(channel_id, target=None, has=lambda t: t.get("steps") or t.get("reaction_candidates") or t.get("llm")):
    for path in sorted(LOG_DIR.glob("*.jsonl"), reverse=True)[:WHY_DAYS]:
        for line in reversed(path.read_text().splitlines()):
            try:
                t = json.loads(line)
            except ValueError:
                continue
            if not in_channel(t, channel_id) or not has(t):
                continue
            if target is None:
                return t
            if target.author.id == bot.user.id:
                # Entries from before reply_id (or status_id) was logged: the text instead
                if "status" in t:
                    if t.get("status_id", target.id) == target.id and t["status"] == target.content:
                        return t
                elif t.get("reply_id", target.id) == target.id and t.get("reply") == target.content:
                    return t
            elif t.get("message_id") == target.id or ("message_id" not in t
                                                      and t.get("message") == (message_text(target) or "hello")):
                return t
    return None

# One why_chart panel per word jev picked. Entries from before scores were logged get them from penalty() again,
# with the history they logged. A status's words come after its start, which it was given.
def why_panels(t):
    if "status" in t:
        recent, exempt = (), ()  # generate_status() doesn't pass them
        words = list(next((s for s in STATUS_STARTS if t["status"].startswith(render(s))), []))
    else:
        recent, exempt = recent_answers(t.get("history")), said_in(t["message"])
        words = []
    panels = []
    for s in t["steps"]:
        rows = [[w, p, rest[0] if rest else p / penalty(words, w, recent, exempt)] for w, p, *rest in s["top"]]
        rows.sort(key=lambda r: -r[2])
        keep = rows[:8] + [r for r in rows[8:] if r[0] == s["word"]]
        panels.append({"so_far": render(words), "picked": s["word"], "ends": s.get("ends", False),
                       "pooled": s.get("pooled", []), "rows": keep})
        if s["word"] != END:
            words.append(s["word"])
    return panels

# One why_chart panel per token an LLM wrote (the first WHY_TOKENS), with the alternatives it gave, likeliest first
WHY_TOKENS = 24

def llm_why_panels(tokens):
    panels, so_far = [], ""
    for t in tokens[:WHY_TOKENS]:
        top = dict(t["top"])
        top.setdefault(t["token"], t["p"])  # sampled from outside its top few
        rows = sorted(([w, p, p] for w, p in top.items()), key=lambda r: -r[1])
        panels.append({"so_far": so_far, "picked": t["token"], "ends": False, "pooled": [], "rows": rows})
        so_far += t["token"]
    return panels

# The message a command is a Discord reply to: None if it isn't a reply, False (having said so) if it can't be seen
async def command_target(m):
    if not (m.reference and m.reference.message_id):
        return None
    target = m.reference.resolved
    if not isinstance(target, discord.Message):  # not in discord.py's cache, or deleted
        try:
            target = await m.channel.fetch_message(m.reference.message_id)
        except discord.HTTPException:
            await m.reply("Can't see that message", mention_author=False)
            return False
    return target

# The question check that chose between replying and reacting, and how close it came to going the other way
def asked_note(t, bot_name):
    if (asked := t.get("asked")) is None:
        return None
    side = f"reacts below {REACT_THRESHOLD:.0%}" if t.get("reaction") else f"replies from {REACT_THRESHOLD:.0%}"
    # Rounded down, so 39.6% doesn't show as 40% on the reacting side of 40%
    return f"Question for {bot_name}? {math.floor(asked * 100)}%, {side}"

# How !why and Context answer: the !why command with a Discord reply to it, the Why and Context actions (right-click
# a message, Apps) with a message only whoever asked can see. Both quote what people said, so no pings from mentions.
def reply_to(m):
    async def send(content=None, file=None):
        await m.reply(content, file=file, mention_author=False, allowed_mentions=discord.AllowedMentions.none())
    return send

def answer_privately(interaction):
    async def send(content=None, file=None):
        await interaction.followup.send(content or discord.utils.MISSING, file=file or discord.utils.MISSING,
                                        ephemeral=True, allowed_mentions=discord.AllowedMentions.none())
    return send

# The chart's file name, from what it's about, so saved charts don't all overwrite why.png:
# "why-day-good-rain-make-race.png", "why-reacted-to-lol-that-cat.png"
def why_filename(text, reacted=False):
    words = re.findall(r"[a-z0-9]+", text.lower())[:6]
    return "-".join(["why"] + (["reacted", "to"] if reacted else []) + words)[:60] + ".png"

async def why(m):
    if (target := await command_target(m)) is not False:
        await explain_why(m.channel.id, m.guild, target, reply_to(m))

# !why's answer for `target` (None: the latest in the channel), through send(content=None, file=None)
async def explain_why(channel_id, guild, target, send):
    t = await asyncio.to_thread(find_trace, channel_id, target)
    if t is None:
        await send("Nothing logged for that")
        return
    bot_name = t.get("bot_name") or (guild.me if guild else bot.user).display_name
    if t.get("llm") and not t.get("reaction"):
        if not t.get("llm_tokens"):
            await send(f"{t['llm']} doesn't say how likely its words were, so there's nothing to chart"
                       " — Context shows what it saw")
            log.info(f"[WHY] no tokens from {t['llm']}")
            return
        said = t.get("reply") or t.get("status") or ""
        n = len(t["llm_tokens"])
        model = t.get("llm_model") or LLMS.get(t["llm"], {}).get("id", t["llm"])  # from before llm_model was logged
        note = f"Model: {model}" + (f" — its first {WHY_TOKENS} of {n} tokens" if n > WHY_TOKENS else "")
        png = await asyncio.to_thread(why_chart.render, bot_name, said, llm_why_panels(t["llm_tokens"]), note=note,
                                      tokens=True)
        await send(file=discord.File(io.BytesIO(png), why_filename(said)))
        log.info(f"[WHY] token chart for {said!r}")
    elif t.get("steps"):
        said = t.get("reply") or t.get("status") or ""
        png = await asyncio.to_thread(why_chart.render, bot_name, said, why_panels(t), note=asked_note(t, bot_name))
        await send(file=discord.File(io.BytesIO(png), why_filename(said)))
        log.info(f"[WHY] chart for {t.get('reply') or t.get('status')!r}")
    elif isinstance(t["reaction_candidates"][0], str):
        # Entries from before their probabilities were logged: just the emoji, likeliest first
        await send(f"Reacted {t.get('reaction')} to \"{t['message'][:80]}\" — {' '.join(t['reaction_candidates'])}")
        log.info(f"[WHY] reaction {t.get('reaction')}")
    else:
        rows = sorted(t["reaction_candidates"], key=lambda r: -r[2])  # [emoji, probability, score], best score first
        async with aiohttp.ClientSession() as session:
            images = await asyncio.gather(*(emoji_image(session, e, guild) for e, _, _ in rows))
        panel = {"so_far": t.get("author") or "", "picked": t.get("reaction"), "ends": False, "rows": rows}
        png = await asyncio.to_thread(why_chart.render, bot_name, "", [panel], reacted_to=t["message"],
                                      images={e: img for (e, _, _), img in zip(rows, images) if img},
                                      note=asked_note(t, bot_name))
        await send(file=discord.File(io.BytesIO(png), why_filename(t["message"], reacted=True)))
        log.info(f"[WHY] chart for reaction {t.get('reaction')}")


# Context: the transcript jev had in view for one of its answers, from the logs — found like !why's. For a reply,
# the one it picked words from; for a reaction, the question check's, which chose reacting (messages to jev marked
# "@jev", no past reactions); for a status, its diary entry, with any recent chat. Entries logged without one (out
# of credit, or from before asked_transcript) get it from their history again — which is logged at the end, so can
# show a reaction or reply that came in meanwhile.
CONTEXT_LIMIT = 2000  # Discord's message length — a longer transcript goes as a file

# A code block in someone's message would open or end ours
def unfence(text):
    return text.replace("```", "`\u200b``")

def logged_context(t):
    if "status" in t:
        return t["transcript"]
    if t.get("reaction"):
        return t.get("asked_transcript") or transcript(t["message"], t["author"], t["bot_name"], t["history"], [],
                                                       reactions=False, marked=True)
    return t.get("transcript") or transcript(t["message"], t["author"], t["bot_name"], t["history"], [])

# The Context action's answer for `target`, through send(content=None, file=None)
async def explain_context(channel_id, guild, target, send):
    t = await asyncio.to_thread(find_trace, channel_id, target,
                                lambda t: "history" in t and (t.get("reply") or t.get("reaction"))
                                or t.get("status") and "transcript" in t)
    if t is None:
        await send("Nothing logged for that")
        return
    # Statuses from before bot_name was logged: the name now
    bot_name = t.get("bot_name") or (guild.me if guild else bot.user).display_name
    if "status" in t:
        what = "its status"
        head = f"What {bot_name} saw before the status \"{unfence(t['status'])}\""
    else:
        what = f"reacting {t['reaction']} to" if t.get("reaction") else "replying to"
        head = (f"What {bot_name} saw before {what} \"{unfence(t['message'][:80])}\""
                + (" (!nocontext)" if t.get("no_context") else ""))
    text = logged_context(t)
    body = f"{head}:\n```\n{unfence(text)}\n```"
    if len(body) <= CONTEXT_LIMIT:
        await send(body)
    else:
        await send(head, file=discord.File(io.BytesIO(text.encode()), "context.txt"))
    log.info(f"[CONTEXT] {what} {(t.get('message') or t['status'])[:40]!r}")


# Right-clicking a message for Why or Context: that message, as if !why were a Discord reply to it.
# Charts can take longer than the 3s Discord waits for an answer, so it's deferred ("jev is thinking...") first.
@bot.tree.context_menu(name="Why")
async def why_action(interaction: discord.Interaction, target: discord.Message):
    await interaction.response.defer(ephemeral=True, thinking=True)
    await explain_why(interaction.channel_id, interaction.guild, target, answer_privately(interaction))

@bot.tree.context_menu(name="Context")
async def context_action(interaction: discord.Interaction, target: discord.Message):
    await interaction.response.defer(ephemeral=True, thinking=True)
    await explain_context(interaction.channel_id, interaction.guild, target, answer_privately(interaction))


# !model: which model writes the replies and statuses, or with a name, switch to it — for everyone, and kept
# across restarts
async def model_command(m):
    global model_name
    names = ["jev", *LLMS]
    args = strip_mention(m).lower().split()[1:]
    if not args:
        text = f"Writing with **{model_name}**. `!model <name>` switches: {', '.join(names)}"
    elif args[0] not in names:
        text = f"No model called {args[0]!r} — {', '.join(names)}"
    else:
        model_name = args[0]
        save_model(model_name)
        text = f"Now writing with **{model_name}**" + (f" ({LLMS[model_name]['id']})" if model_name in LLMS else "")
        log.info(f"[MODEL] {m.author} switched to {model_name}")
    await m.reply(text, mention_author=False, allowed_mentions=discord.AllowedMentions.none())


# Messages jev has taken on, so each is answered once: at startup, catch_up() can find one on_message is already
# answering (its reply not sent yet, so it looks missed), or the other way round. The latest only — catch_up() looks
# back CATCH_UP_WINDOW minutes.
taken: dict[int, None] = {}

async def respond(m, **extra):
    if m.id in taken:
        log.info(f"[SKIP] already answering {m.id}")
        return
    taken[m.id] = None
    if len(taken) > 1000:
        del taken[next(iter(taken))]
    handling.add(task := asyncio.current_task())
    try:
        await respond_traced(m, **extra)
    finally:
        handling.discard(task)


async def respond_traced(m, **extra):
    c = message_text(m) or "hello"
    log.info(f"[IN] {m.author}: {c[:80]}")
    # discord.py runs each event in its own task, so this trace only sees this message's requests
    t = {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "channel": getattr(m.channel, "name", None),
         "channel_id": m.channel.id, "message_id": m.id, "author": m.author.display_name, "message": c,
         "cost": 0.0, "requests": 0, **extra}
    trace.set(t)
    start = time.monotonic()
    try:
        await handle(m, c)
    finally:
        t["seconds"] = round(time.monotonic() - start, 1)
        write_trace(t)


async def handle(m, c):
    if m.guild is None:
        dm_channels.add(m.channel.id)  # before its history loads, so a status made meanwhile can't show it
    # Shared task, so messages arriving while it loads wait for it instead of loading twice
    if m.channel.id not in history_loaded:
        history_loaded[m.channel.id] = asyncio.create_task(load_history(m))
    await history_loaded[m.channel.id]
    root = await side_root(m)
    # Snapshot before waiting on gen_lock — messages that arrive meanwhile must not shift this one's history
    if root is None:
        entry = add_history(m.channel.id, history_entry(m))
        h, long = shown(channel_history[m.channel.id][:-1]), shown(channel_history[m.channel.id][:-1], LLM_HISTORY)
    else:
        h = list(side_talk.get(root, []))  # nothing yet for a !nocontext message
        long = list(h)
        entry = add_side(root, history_entry(m))
        note(no_context=True)
    entry["pending"] = True
    # Server nickname, so the transcript uses the name people call the bot by
    bot_name = (m.guild.me if m.guild else bot.user).display_name
    # The reply of jev's someone is answering, if it isn't already shown under the message it answered
    if not no_context(m) and (r := replied_to_bot(m)) and r.content:
        reply_entry = None
        for x in (h, long):
            if all(e.get("reply_id") != r.id for e in x):
                if reply_entry is None:
                    reply_entry = {"role": "assistant", "name": bot_name, "content": r.content, "at": r.created_at}
                    await attach_reactions(reply_entry, r, prefix="")
                x.append(reply_entry)
    note(bot_name=bot_name, history=h)
    try:
        # Kept out of history: jev would see the link in its transcript and start talking about it
        if has_credit is False:
            gif = random.choice(NO_MONEY)
            note(reply=gif)
            log.info(f"[OUT] out of credit: {gif}")
            await m.reply(gif, mention_author=False)
            return
        # Decided before typing() — a reaction sends no message, so the typing indicator would linger
        emoji = emoji_vocabulary(m.guild)
        if reaction := await choose_reaction(c, m.author.display_name, bot_name, emoji, history=h):
            try:
                await m.add_reaction(emoji[reaction])
                entry["reaction"] = reaction  # kept on its message, so it stays in order and doesn't use a history slot
                note(reaction=reaction)
                log.info(f"[REACT] {reaction} (${trace.get()['cost']:.5f})")
                return
            except discord.HTTPException as e:  # no Add Reactions permission, or an emoji Discord doesn't know
                log.warning(f"React {reaction} failed, replying instead: {e}")
        async with m.channel.typing():
            async with gen_lock:
                r = await generate_reply(c, m.author.display_name, bot_name, history=h, llm_history=long,
                                         at=m.created_at)
        if has_credit is False and r == "...":  # ran out before the first word
            r = random.choice(NO_MONEY)
        note(reply=r)
        log.info(f"[OUT] {r} (${trace.get()['cost']:.5f})")
        # No pings from whatever an LLM writes ("@everyone")
        sent = await m.reply(r, mention_author=False, allowed_mentions=discord.AllowedMentions.none())
        note(reply_id=sent.id)  # for !why
        if r not in NO_MONEY:
            entry["reply"], entry["reply_id"], entry["reply_at"] = r, sent.id, sent.created_at  # shown under it from now on
            if root is not None:
                side_of[sent.id] = root  # replies to it carry on the side conversation
    except Exception as e:
        log.error(f"Error: {e}", exc_info=True)
        note(error=repr(e))
        try: await m.reply("...", mention_author=False)
        except: pass
    finally:
        entry.pop("pending", None)

# Keeps people's reactions to jev's replies current in the history — a reply shown later carries them
async def on_reaction_change(p, change):
    if p.user_id == bot.user.id:
        return
    entry = next((e for e in entries_with(p.channel_id, p.message_id) if e.get("reply_id") == p.message_id), None)
    if entry is None:
        return
    counts = entry.setdefault("reply_reactions", {})
    e = emoji_text(p.emoji)
    users = entry.setdefault("reply_reactors", {}).setdefault(e, {})
    uid = str(p.user_id)
    counts[e] = counts.get(e, 0) + change
    if change > 0:
        users[uid] = None
    else:
        users.pop(uid, None)
    if counts[e] <= 0:
        del counts[e]
        entry["reply_reactors"].pop(e, None)
    if change > 0:
        guild = bot.get_guild(p.guild_id) if p.guild_id else None
        name = await reactor_name(guild, p.user_id, p.member)  # None if it can't be found — the count still shows
        if uid in users:
            users[uid] = name

@bot.event
async def on_raw_reaction_add(p):
    await on_reaction_change(p, 1)

@bot.event
async def on_raw_reaction_remove(p):
    await on_reaction_change(p, -1)

@bot.event
async def on_raw_reaction_clear(p):
    for e in entries_with(p.channel_id, p.message_id):
        if e.get("reply_id") == p.message_id:
            e.pop("reply_reactions", None)
            e.pop("reply_reactors", None)

@bot.event
async def on_raw_reaction_clear_emoji(p):
    for e in entries_with(p.channel_id, p.message_id):
        if e.get("reply_id") == p.message_id:
            e.get("reply_reactions", {}).pop(emoji_text(p.emoji), None)
            e.get("reply_reactors", {}).pop(emoji_text(p.emoji), None)

async def main():
    first_signal = None
    def on_signal():
        nonlocal first_signal
        if first_signal is None:
            first_signal = time.monotonic()
            stopping.set()
        elif time.monotonic() - first_signal > 1:
            log.warning("[STOP] quitting now, without finishing")
            os._exit(1)
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, on_signal)

    async with bot:
        running = asyncio.create_task(bot.start(TOKEN))
        await asyncio.wait([running, asyncio.create_task(stopping.wait())], return_when=asyncio.FIRST_COMPLETED)
        if running.done():
            running.result()  # login failed or the connection gave up — raise it
        if handling:
            log.info(f"[STOP] finishing {len(handling)} message(s) in progress — Ctrl-C again to quit now")
            await asyncio.gather(*handling, return_exceptions=True)
        log.info("[STOP] done")


if __name__ == "__main__":
    asyncio.run(main())
