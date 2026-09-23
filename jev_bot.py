"""
Jev Discord Bot — tournament sampling + history.

This is the version that produced "I depends on on situation of circumstances."
20K vocab, bucket tournament, empty descriptions, 3-turn history.
"""

import os
import re
import asyncio
import logging
import random
from pathlib import Path
from collections import defaultdict

import aiohttp
import discord
from discord.ext import commands
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
MAX_HISTORY = 3
STOP_THRESHOLD = 0.5            # let jev stop earlier — the good part is always the first half
REPEAT_PENALTY = 1.5
REPEAT_WINDOW = 8
CONTENT_PENALTY = 2.5
CONTENT_PENALTY_CAP = 4
STOP_PENALTY = 1.6
STOP_PENALTY_CAP = 6

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

# History
channel_history: dict[int, list[dict]] = defaultdict(list)

def add_history(ch_id, role, name, content):
    h = channel_history[ch_id]
    h.append({"role": role, "name": name, "content": content})
    if len(h) > MAX_HISTORY:
        channel_history[ch_id] = h[-MAX_HISTORY:]

# Vocab
VOCAB_PATH = Path(__file__).parent / "vocab.txt"
CUSTOM_VOCAB_PATH = Path(__file__).parent / "custom_vocab.txt"
BANNED = {"unanswered"}
BASE_VOCAB = [w for w in VOCAB_PATH.read_text().split("\n") if w and w.lower() not in BANNED]
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
    extra = [w for w in re.findall(r"[A-Za-z']+", message.lower())
             if w not in seen and not seen.add(w)]
    return BASE_VOCAB + extra + [END]


def penalty(reply, word):
    local = reply[-REPEAT_WINDOW:].count(word) + 2 * (reply[-1:] == [word])
    p = REPEAT_PENALTY ** local
    seen = reply.count(word)
    alpha = word.replace(" ", "").isalpha()
    if alpha and word.lower() not in STOPWORDS:
        p *= CONTENT_PENALTY ** min(seen, CONTENT_PENALTY_CAP)
    elif alpha:
        p *= STOP_PENALTY ** min(seen, STOP_PENALTY_CAP)
    return p


async def post(session, state, questions):
    body = {"model": MODEL, "state": state, "questions": questions}
    for attempt in range(3):
        try:
            async with session.post(API_URL, json=body, timeout=aiohttp.ClientTimeout(total=30)) as r:
                if r.status < 400:
                    return (await r.json()).get("answers", {})
                await asyncio.sleep(1 + 2 * attempt)
        except Exception as e:
            log.warning(f"API err {attempt}: {e}")
            await asyncio.sleep(1 + 2 * attempt)
    return {}


def choice_q(words):
    return {"type": "choice", "instructions": "Next word?", "criteria": {w: "" for w in words}}


async def next_word(session, state, vocab, rng):
    shuffled = list(vocab)
    rng.shuffle(shuffled)
    buckets = [shuffled[i:i + MAX_CHOICES] for i in range(0, len(shuffled), MAX_CHOICES)]
    groups = [buckets[i:i + QUESTIONS_PER_CALL] for i in range(0, len(buckets), QUESTIONS_PER_CALL)]

    results = await asyncio.gather(
        *(post(session, state, {f"b{gi * QUESTIONS_PER_CALL + i}": choice_q(b)
                                for i, b in enumerate(g)})
          for gi, g in enumerate(groups)),
        post(session, state, {"complete": {"type": "noul", "instructions": "Is the reply complete?"}}),
    )

    complete_noul = results[-1].get("complete", {}).get("noul", 0)

    finalists = []
    for group_answers in results[:-1]:
        for ans in group_answers.values():
            if "probabilities" not in ans:
                continue
            ranked = sorted(ans["probabilities"].items(), key=lambda kv: -kv[1])
            finalists += [w for w, p in ranked[:TOP_PER_BUCKET] if p > 0]

    if END not in finalists:
        finalists.append(END)

    runoff = await post(session, state, {"final": choice_q(finalists[:MAX_CHOICES])})
    probs = runoff.get("final", {}).get("probabilities", {})

    return probs, complete_noul


async def generate_reply(message, author, bot_name, history=None):
    rng = random.Random()
    vocab = vocabulary(f"{author} {message}")  # lets jev say the author's name
    words = []

    headers = {"Authorization": f"Bearer {OPENROUTER_KEY}", "Content-Type": "application/json"}

    async with aiohttp.ClientSession(headers=headers) as session:
        for step in range(MAX_WORDS):
            turns = []
            if history:
                for h in history:
                    name = bot_name if h["role"] == "assistant" else h["name"]
                    turns.append(f"{name}: {unrender(h['content'])}")
            turns.append(f"{author}: {unrender(message)}")
            turns.append(f"{bot_name}: {unrender(render(words))}")
            state = "\n".join(turns)

            probs, complete = await next_word(session, state, vocab, rng)
            if not probs:
                break

            stoppable = sum(1 for w in words if is_word(w)) >= MIN_WORDS
            if stoppable and complete >= STOP_THRESHOLD:
                log.info(f"  noul={complete:.2f} stop")
                break

            scored = {}
            for w, p in probs.items():
                if p <= 0: continue
                if w in NO_SPACE_BEFORE and words[-1:] == [w]: continue
                if w == END and not stoppable: continue
                scored[w] = p / penalty(words, w)

            if not scored: break

            ranked = sorted(scored.items(), key=lambda kv: -kv[1])
            word = ranked[0][0]

            top3 = [(w, probs.get(w, 0)) for w, _ in ranked[:3]]
            log.info(f"  [{step+1:2d}] {word:12s}  {' '.join(f'{w}:{p:.0%}' for w,p in top3)}  done={complete:.2f}")

            if word == END: break
            words.append(word)

    return render(words).strip() or "..."


# Discord
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)
gen_lock = asyncio.Lock()

def strip_mention(c, bid):
    return re.sub(rf"<@!?{bid}>", "", c).strip()

def should_respond(m):
    if m.author.bot: return False
    if bot.user in m.mentions: return True
    if m.reference and m.reference.resolved:
        r = m.reference.resolved
        if isinstance(r, discord.Message) and r.author.id == bot.user.id: return True
    return False

@bot.event
async def on_ready():
    log.info(f"jev online as {bot.user} | vocab {len(BASE_VOCAB)}")

@bot.event
async def on_message(m):
    if not should_respond(m): return
    c = strip_mention(m.content, bot.user.id) or "hello"
    log.info(f"[IN] {m.author}: {c[:80]}")
    add_history(m.channel.id, "user", m.author.display_name, c)
    # Snapshot before waiting on gen_lock — messages that arrive meanwhile must not shift this one's history
    # Only user messages in history — jev's own broken output poisons follow-ups
    h = [x for x in channel_history[m.channel.id][:-1] if x["role"] == "user"]
    # Server nickname, so the transcript uses the name people call the bot by
    bot_name = (m.guild.me if m.guild else bot.user).display_name
    try:
        async with m.channel.typing():
            async with gen_lock:
                r = await generate_reply(c, m.author.display_name, bot_name, history=h)
        log.info(f"[OUT] {r}")
        await m.reply(r, mention_author=False)
    except Exception as e:
        log.error(f"Error: {e}", exc_info=True)
        try: await m.reply("...", mention_author=False)
        except: pass

if __name__ == "__main__":
    bot.run(TOKEN, log_handler=None)
