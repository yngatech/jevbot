"""
Jev Discord Bot — tournament sampling + history.

This is the version that produced "I depends on on situation of circumstances."
20K vocab, bucket tournament, empty descriptions, 3-turn history.
"""

import os
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

import aiohttp
import discord
from discord.ext import commands, tasks
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.environ["DISCORD_TOKEN_JEV"]
OPENROUTER_KEY = os.environ["OPENROUTER_API_KEY"]
API_URL = "https://openrouter.ai/api/alpha/decisions"
MODEL = "~typesafe/jev-latest"
END = "<END>"

MAX_CHOICES = 255
QUESTIONS_PER_CALL = 20
TOP_PER_BUCKET = 2
MAX_WORDS = 30
MIN_WORDS = 2
HISTORY_TO_BOT = 6              # earlier messages to jev (mentions, replies) in the transcript
HISTORY_CHATTER = 8             # earlier channel messages not aimed at jev in the transcript — 0 to leave them out
HISTORY_SCAN = 100              # recent messages read to rebuild a channel's history after a restart
CATCH_UP_WINDOW = 30            # minutes — on startup, answer each channel's latest message to jev from this long ago that it missed
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
VOCAB_SIZE = 10_000             # vocab.txt is ordered most common first — every word costs ~7 input tokens on every step
REACT_THRESHOLD = 0.4           # react when P(message is a question/request for jev) is below this — questions ~0.8-0.98, chatty ~0.03-0.45
STATUS_EVERY = 180              # minutes between new statuses (~$0.02-0.04 each) — 0 to leave the status alone
STATUS_MIN_WORDS = 6            # a short status is just "Fine thanks" — the soup comes from making it keep going
STATUS_MAX_WORDS = 12
STATUS_CHAT = 8                 # recent messages from the latest active channel, in view for every other status — 0 for none
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

# History: recent messages per channel, oldest first. "to_bot" marks the ones that mentioned or replied to jev.
channel_history: dict[int, list[dict]] = defaultdict(list)

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

# What the transcript shows: recent() of the channel, minus messages to jev it never answered but has answered
# something after since — left behind (sent while it was offline, say), they pull jev back to their topic.
# One being answered right now is "pending", so a quick reaction to a later message doesn't hide it.
def shown(entries):
    keep, answered_since = [], False
    for e in reversed(entries):
        if e.get("to_bot", True):
            if "reply" in e or "reaction" in e:
                answered_since = True
            elif answered_since and not e.get("pending"):
                continue
        keep.append(e)
    return recent(keep[::-1])

def add_history(ch_id, entry):
    h = channel_history[ch_id]
    h.append(entry)
    channel_history[ch_id] = recent(h, HISTORY_TO_BOT + 1, HISTORY_CHATTER)  # +1: the message being answered
    return entry

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
        if said_as(word) not in exempt:
            p *= ECHO_PENALTY ** sum(said_as(word) in a for a in recent)
    elif alpha:
        p *= STOP_PENALTY ** min(seen, STOP_PENALTY_CAP)
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


# People's reactions to one of jev's replies, as " (😂×3 💀)" — a laugh jev can see, and maybe chase
def reacted(counts):
    return " (" + " ".join(e if n == 1 else f"{e}×{n}" for e, n in counts.items()) + ")" if counts else ""

# marked: show messages to jev as "name: @jev ..." (the mention is stripped otherwise). Tested live: with channel
# chatter in view it's what tells the question check which messages were for jev, but for picking words it made
# jev describe more and stop sooner, so only the question check uses it.
# reactions: jev's own past reactions; their_reactions: people's reactions to jev's replies
def transcript(message, author, bot_name, history, words, reactions=True, marked=False, their_reactions=None):
    their_reactions = reactions if their_reactions is None else their_reactions
    to = f"@{bot_name} " if marked else ""
    turns = []
    if history:
        for h in history:
            name = bot_name if h["role"] == "assistant" else h["name"]
            text = unrender(h["content"])
            if h["role"] == "assistant" and their_reactions:
                text += reacted(h.get("reactions"))
            turns.append(f"{name}: {to if h['role'] == 'user' and h.get('to_bot', True) else ''}{text}")
            # jev's answer, if it gave one — without it every earlier question looks unanswered, and jev goes back
            # to them or describes the silence ("crickets"). Past reactions, jev's and people's to its replies,
            # stay out of the question check — jev copies emoji it sees there.
            if "reply" in h:
                turns.append(f"{bot_name}: {unrender(h['reply'])}{reacted(h.get('reply_reactions')) if their_reactions else ''}")
            elif reactions and "reaction" in h:
                turns.append(f"{bot_name}: {h['reaction']}")
    turns.append(f"{author}: {to}{unrender(message)}")
    turns.append(f"{bot_name}: {unrender(render(words))}")
    return "\n".join(turns)


HEADERS = {"Authorization": f"Bearer {OPENROUTER_KEY}", "Content-Type": "application/json"}


# The emoji jev reacts with instead of replying, or None to reply
async def choose_reaction(message, author, bot_name, emoji, history=None):
    rng = random.Random()
    # Without past reactions — a run of them reads as a habit to keep up, and jev copies the last emoji
    state = transcript(message, author, bot_name, history, [], reactions=False, marked=True)
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
            word = ranked[0][0]
            # ".", "!" and "?" are jev ending as much as <END> is. Apart, they split the vote to end and a word won
            # instead: where master ended, jev picked <END> 31% of the time with them in the vocab and 78% without
            # ("Dunno google" went on "for", "on", "type"). So they count together, and the reply ends with the
            # mark if jev liked that best ("Dunno forgot?")
            ending = {w: s for w, s in scored.items() if w == END or w in SENTENCE_ENDS} if stoppable else {}
            ends = sum(ending.values()) > max((s for w, s in scored.items() if w not in ending), default=0)
            if ends:
                word = max(ending, key=ending.get)

            top3 = [(w, probs.get(w, 0)) for w, _ in ranked[:3]]
            log.info(f"  [{step+1:2d}] {word:12s}  {' '.join(f'{w}:{p:.0%}' for w,p in top3)}  done={complete:.2f}"
                     + ("  ends" if ends else ""))
            # Raw probabilities of the best-scoring candidates, so penalties' effect on the pick is visible
            steps.append({"word": word, "done": round(complete, 3), "ends": ends,
                          "top": [[w, round(probs[w], 3)] for w, _ in ranked[:5]]})

            if word == END:
                note(stop="<END>")
                break
            words.append(word)
            if ends:
                note(stop=f"<END> as {word}")
                break

    return words


async def generate_reply(message, author, bot_name, history=None):
    # Every word and name people used in the transcript, not just the message being replied to — lets jev say what
    # it can see. Not jev's own words: from a broken reply that would add "garbled" and "unclear" back for reuse.
    vocab = vocabulary(" ".join([f"{h['name']} {h['content']}" for h in history or [] if h["role"] == "user"]
                                + [f"{author} {message}"]))
    note(transcript=transcript(message, author, bot_name, history, []))
    # People's reactions stay out of the "complete?" question: with 😂×4 on jev's last reply in view it read a
    # two-word reply as done, and jev stopped at "Dunno forgot" where it had gone on to "Dunno forgot liar bitch"
    words = await loom(lambda words: transcript(message, author, bot_name, history, words),
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
    active = [h for h in channel_history.values() if h]
    if not STATUS_CHAT or not active:
        return ""
    h = max(active, key=lambda h: h[-1]["at"])[-STATUS_CHAT:]
    return "\n".join(f"{e['name']}: {unrender(e['content'])}" for e in h)

def status_state(bot_name, start, chat, words):
    return (f"{chat}\n\n" if chat else "") + f"{bot_name}'s diary, today: {render(start + words)}"


async def generate_status(bot_name, start, chat=""):
    vocab = [w for w in vocabulary(chat) if w != NEWLINE]  # a status is one line
    note(transcript=status_state(bot_name, start, chat, []))
    words = await loom(lambda words: status_state(bot_name, start, chat, words), vocab, NEXT_WORD,
                       max_words=STATUS_MAX_WORDS, min_words=STATUS_MIN_WORDS)
    return render(start + words)[:128] if words else None  # 128: Discord's custom status limit


# Discord
intents = discord.Intents.default()
intents.message_content = True
intents.dm_messages = False  # no DMs — every reply costs credit, and nobody sees what's said in private
bot = commands.Bot(command_prefix="!", intents=intents)
gen_lock = asyncio.Lock()

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

# A mention used as a name ("@rocky's seeing…") becomes the bot's name — dropped, it left "'s" behind, which jev
# then picked as a word
def strip_mention(m):
    mention = rf"<@!?{bot.user.id}>" + (f"|{re.escape(role.mention)}" if (role := bot_role(m)) else "")
    c = re.sub(rf"(?:{mention})(?=')", (m.guild.me if m.guild else bot.user).display_name, m.content)
    return re.sub(mention, "", c).strip()

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

# m as jev sees it: without the mention if it's to jev, links as tags, and a tag for each attachment and sticker —
# otherwise a photo on its own is an empty message, dropped or read as "hello"
def message_text(m):
    text = strip_mention(m) if should_respond(m) else m.content
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
    if bot.user in m.mentions: return True
    if (role := bot_role(m)) and role in m.role_mentions: return True
    return replied_to_bot(m) is not None

# channel_history is in memory, so rebuild it from Discord the first time a channel talks to jev after a restart
history_loaded: dict[int, asyncio.Task] = {}

# m as a history entry, or None for messages that never go in one (bots, including jev itself, and empty ones)
def history_entry(m):
    if m.author.bot:
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

async def load_history(first):
    ch = first.channel.id
    known = {e["id"] for e in channel_history[ch]}  # chatter already recorded live since startup
    found, replies = [], {}
    try:
        async for m in first.channel.history(limit=HISTORY_SCAN, before=first):
            if m.author.id == bot.user.id and m.reference and m.content and m.content not in NO_MONEY:
                replies[m.reference.message_id] = m  # jev's reply, to go back under the message it answered
            elif m.id not in known and (entry := history_entry(m)):
                found.append(entry)
    except Exception as e:  # no Read Message History permission — start empty, like before
        log.warning(f"Loading history for {ch} failed: {e}")
    for entry in found:
        if r := replies.get(entry["id"]):
            entry["reply"], entry["reply_id"] = r.content, r.id
            if counts := reactions_to(r):
                entry["reply_reactions"] = counts
    channel_history[ch] = recent(sorted(channel_history[ch] + found, key=lambda e: e["at"]))
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
STATUS_PATH = Path(__file__).parent / "status.json"
status_turn = 0      # which start and chat the next status gets
status_at = None     # when the kept one was made

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
    global status, status_turn
    if has_credit is False:  # showing "out of credit" — and every request would fail anyway
        return
    bot_name = bot.user.display_name
    t = {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "status": None, "cost": 0.0, "requests": 0}
    trace.set(t)
    start = time.monotonic()
    try:
        async with gen_lock:  # after any reply in progress, not alongside it
            n, status_turn = status_turn, status_turn + 1
            chat = recent_chat() if n % 2 else ""
            mood = await generate_status(bot_name, STATUS_STARTS[n % len(STATUS_STARTS)], chat)
        if not mood:  # the API didn't answer — keep the old status rather than a bare "I feel"
            return
        status = discord.CustomActivity(name=mood)
        save_status(mood, n, t["at"])
        await show_credit()
        t["status"] = mood
        log.info(f"[STATUS] {mood} (${t['cost']:.5f})")
    except Exception as e:  # keep the old status and try again next time — an uncaught error would end the loop
        log.error(f"Status error: {e}", exc_info=True)
        t["error"] = repr(e)
    finally:
        t["seconds"] = round(time.monotonic() - start, 1)
        write_trace(t)

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
    missed = []
    for guild in bot.guilds:
        for ch in [*guild.text_channels, *guild.threads]:
            perms = ch.permissions_for(guild.me)
            if not (perms.read_messages and perms.read_message_history):
                continue
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
            unanswered = [m for m in found if m.created_at > last and should_respond(m)]
            if unanswered:
                log.info(f"[CATCH UP] #{ch} missed {len(unanswered)} message(s) to jev, answering the latest")
                missed.append(max(unanswered, key=lambda m: m.created_at))
    for m in sorted(missed, key=lambda m: m.created_at):
        if stopping.is_set():
            break
        await respond(m, caught_up=True)

@bot.event
async def on_message(m):
    if not should_respond(m):
        # Not for jev, but part of the conversation it might be asked about
        if HISTORY_CHATTER and (entry := history_entry(m)):
            add_history(m.channel.id, entry)
        return
    if stopping.is_set():
        return
    await respond(m)


# Discord often adds a link's embed just after the message arrives, as an edit — and people fix typos
@bot.event
async def on_message_edit(before, after):
    entry = next((e for e in channel_history.get(after.channel.id, []) if e.get("id") == after.id), None)
    if entry and (content := message_text(after)):
        entry["content"] = content


async def respond(m, **extra):
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
         "channel_id": m.channel.id, "author": m.author.display_name, "message": c, "cost": 0.0, "requests": 0, **extra}
    trace.set(t)
    start = time.monotonic()
    try:
        await handle(m, c)
    finally:
        t["seconds"] = round(time.monotonic() - start, 1)
        write_trace(t)


async def handle(m, c):
    # Shared task, so messages arriving while it loads wait for it instead of loading twice
    if m.channel.id not in history_loaded:
        history_loaded[m.channel.id] = asyncio.create_task(load_history(m))
    await history_loaded[m.channel.id]
    entry = add_history(m.channel.id, history_entry(m))
    entry["pending"] = True
    # Server nickname, so the transcript uses the name people call the bot by
    bot_name = (m.guild.me if m.guild else bot.user).display_name
    # Snapshot before waiting on gen_lock — messages that arrive meanwhile must not shift this one's history
    h = shown(channel_history[m.channel.id][:-1])
    # The reply of jev's someone is answering, if it isn't already shown under the message it answered
    if (r := replied_to_bot(m)) and r.content and all(x.get("reply_id") != r.id for x in h):
        h.append({"role": "assistant", "name": bot_name, "content": r.content, "reactions": reactions_to(r)})
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
                r = await generate_reply(c, m.author.display_name, bot_name, history=h)
        if has_credit is False and r == "...":  # ran out before the first word
            r = random.choice(NO_MONEY)
        note(reply=r)
        log.info(f"[OUT] {r} (${trace.get()['cost']:.5f})")
        sent = await m.reply(r, mention_author=False)
        if r not in NO_MONEY:
            entry["reply"], entry["reply_id"] = r, sent.id  # shown under this message from now on
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
    entry = next((e for e in channel_history.get(p.channel_id, []) if e.get("reply_id") == p.message_id), None)
    if entry is None:
        return
    counts = entry.setdefault("reply_reactions", {})
    e = emoji_text(p.emoji)
    counts[e] = counts.get(e, 0) + change
    if counts[e] <= 0:
        del counts[e]

@bot.event
async def on_raw_reaction_add(p):
    await on_reaction_change(p, 1)

@bot.event
async def on_raw_reaction_remove(p):
    await on_reaction_change(p, -1)

@bot.event
async def on_raw_reaction_clear(p):
    for e in channel_history.get(p.channel_id, []):
        if e.get("reply_id") == p.message_id:
            e.pop("reply_reactions", None)

@bot.event
async def on_raw_reaction_clear_emoji(p):
    for e in channel_history.get(p.channel_id, []):
        if e.get("reply_id") == p.message_id:
            e.get("reply_reactions", {}).pop(emoji_text(p.emoji), None)

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
