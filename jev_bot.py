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
HISTORY_SCAN = 100              # recent messages read to rebuild a channel's history after a restart
STOP_THRESHOLD = 0.5            # let jev stop earlier — the good part is always the first half
REPEAT_PENALTY = 1.5
REPEAT_WINDOW = 8
CONTENT_PENALTY = 2.5
CONTENT_PENALTY_CAP = 4
STOP_PENALTY = 1.6
STOP_PENALTY_CAP = 6
VOCAB_SIZE = 10_000             # vocab.txt is ordered most common first — every word costs ~7 input tokens on every step
REACT_THRESHOLD = 0.4           # react when P(message is a question/request for jev) is below this — questions ~0.8-0.98, chatty ~0.03-0.45
# Bare "Next word?" reads as "which word fits this?" — jev described its reply ("empty", "silent", "garbled")
# instead of continuing it. Set back to "Next word?" to compare.
NEXT_WORD = "Next word of {bot_name}'s reply?"

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
    entry = {"role": role, "name": name, "content": content}
    h.append(entry)
    if len(h) > MAX_HISTORY:
        channel_history[ch_id] = h[-MAX_HISTORY:]
    return entry

# Vocab
VOCAB_PATH = Path(__file__).parent / "vocab.txt"
CUSTOM_VOCAB_PATH = Path(__file__).parent / "custom_vocab.txt"
BANNED = {"unanswered"}
BASE_VOCAB = [w for w in VOCAB_PATH.read_text().split("\n") if w and w.lower() not in BANNED][:VOCAB_SIZE]
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


async def next_word(session, state, vocab, rng, instructions):
    shuffled = list(vocab)
    rng.shuffle(shuffled)
    buckets = [shuffled[i:i + MAX_CHOICES] for i in range(0, len(shuffled), MAX_CHOICES)]
    groups = [buckets[i:i + QUESTIONS_PER_CALL] for i in range(0, len(buckets), QUESTIONS_PER_CALL)]

    results = await asyncio.gather(
        *(post(session, state, {f"b{gi * QUESTIONS_PER_CALL + i}": choice_q(b, instructions)
                                for i, b in enumerate(g)})
          for gi, g in enumerate(groups)),
        post(session, state, {"complete": {"type": "noul", "instructions": "Is the reply complete?"}}),
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


def transcript(message, author, bot_name, history, words, reactions=True):
    turns = []
    if history:
        for h in history:
            name = bot_name if h["role"] == "assistant" else h["name"]
            turns.append(f"{name}: {unrender(h['content'])}")
            if reactions and "reaction" in h:  # a single emoji, unlike jev's replies, doesn't poison follow-ups
                turns.append(f"{bot_name}: {h['reaction']}")
    turns.append(f"{author}: {unrender(message)}")
    turns.append(f"{bot_name}: {unrender(render(words))}")
    return "\n".join(turns)


HEADERS = {"Authorization": f"Bearer {OPENROUTER_KEY}", "Content-Type": "application/json"}


# The emoji jev reacts with instead of replying, or None to reply
async def choose_reaction(message, author, bot_name, emoji, history=None):
    rng = random.Random()
    # Without past reactions — a run of them reads as a habit to keep up, and jev copies the last emoji
    state = transcript(message, author, bot_name, history, [], reactions=False)
    shuffled = list(emoji)
    rng.shuffle(shuffled)
    buckets = [shuffled[i:i + MAX_CHOICES] for i in range(0, len(shuffled), MAX_CHOICES)]
    questions = {f"b{i}": choice_q(b, "Reaction?") for i, b in enumerate(buckets[:QUESTIONS_PER_CALL - 1])}
    # Asked as "is it a question?" — "should jev react instead?" scored everything ~0.4-0.55, so no threshold separated them
    questions["asked"] = {"type": "noul", "instructions": f"Is {author}'s last message a question or request for {bot_name}?"}

    async with aiohttp.ClientSession(headers=HEADERS) as session:
        answers = await post(session, state, questions)
        asked = answers.pop("asked", {}).get("noul", 1)
        if asked >= REACT_THRESHOLD:
            log.info(f"  asked={asked:.2f} reply")
            return None

        finalists = [w for ans in answers.values() for w, p in by_prob(ans)[:TOP_PER_BUCKET] if p > 0]
        if len(answers) > 1:
            runoff = await post(session, state, {"final": choice_q(finalists[:MAX_CHOICES], "Reaction?")})
            finalists = [w for w, p in by_prob(runoff.get("final", {})) if p > 0]

    log.info(f"  asked={asked:.2f} {' '.join(finalists[:3])}")
    return finalists[0] if finalists else None


async def generate_reply(message, author, bot_name, history=None):
    rng = random.Random()
    # Every word and name in the transcript, not just the message being replied to — lets jev say what it can see
    vocab = vocabulary(" ".join([f"{h['name']} {h['content']}" for h in history or []] + [f"{author} {message}"]))
    instructions = NEXT_WORD.format(bot_name=bot_name)
    words = []

    async with aiohttp.ClientSession(headers=HEADERS) as session:
        for step in range(MAX_WORDS):
            state = transcript(message, author, bot_name, history, words)

            probs, complete = await next_word(session, state, vocab, rng, instructions)
            if not probs:
                break

            said = sum(1 for w in words if is_word(w))
            stoppable = said >= MIN_WORDS
            if stoppable and complete >= STOP_THRESHOLD:
                log.info(f"  noul={complete:.2f} stop")
                break

            scored = {}
            for w, p in probs.items():
                if p <= 0: continue
                if w in NO_SPACE_BEFORE and words[-1:] == [w]: continue
                if w == END and not stoppable: continue
                if w == NEWLINE and not said: continue  # leading newlines get stripped anyway — don't spend steps on them
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

# The role Discord creates for the bot (same name, e.g. "@rocky") — mentioning it counts as mentioning the bot
def bot_role(m):
    return m.guild.self_role if m.guild else None

def strip_mention(m):
    c = re.sub(rf"<@!?{bot.user.id}>", "", m.content)
    if role := bot_role(m):
        c = c.replace(role.mention, "")
    return c.strip()

def should_respond(m):
    if m.author.bot: return False
    if bot.user in m.mentions: return True
    if (role := bot_role(m)) and role in m.role_mentions: return True
    if m.reference and m.reference.resolved:
        r = m.reference.resolved
        if isinstance(r, discord.Message) and r.author.id == bot.user.id: return True
    return False

# channel_history is in memory, so rebuild it from Discord the first time a channel talks to jev after a restart
history_loaded: dict[int, asyncio.Task] = {}

async def load_history(first):
    found = []
    try:
        async for m in first.channel.history(limit=HISTORY_SCAN, before=first):
            if should_respond(m):
                found.append(m)
                if len(found) == MAX_HISTORY: break
    except Exception as e:  # no Read Message History permission — start empty, like before
        log.warning(f"Loading history for {first.channel.id} failed: {e}")
    for m in reversed(found):
        entry = add_history(first.channel.id, "user", m.author.display_name, strip_mention(m) or "hello")
        if r := next((r for r in m.reactions if r.me), None):
            entry["reaction"] = r.emoji if isinstance(r.emoji, str) else f":{r.emoji.name}:"
    log.info(f"Loaded {len(found)} history entries for {first.channel.id}")

@bot.event
async def on_ready():
    log.info(f"jev online as {bot.user} | vocab {len(BASE_VOCAB)} | {NEXT_WORD!r}")

@bot.event
async def on_message(m):
    if not should_respond(m): return
    c = strip_mention(m) or "hello"
    log.info(f"[IN] {m.author}: {c[:80]}")
    # Shared task, so messages arriving while it loads wait for it instead of loading twice
    if m.channel.id not in history_loaded:
        history_loaded[m.channel.id] = asyncio.create_task(load_history(m))
    await history_loaded[m.channel.id]
    entry = add_history(m.channel.id, "user", m.author.display_name, c)
    # Snapshot before waiting on gen_lock — messages that arrive meanwhile must not shift this one's history
    # Only user messages in history — jev's own broken output poisons follow-ups
    h = [x for x in channel_history[m.channel.id][:-1] if x["role"] == "user"]
    # Server nickname, so the transcript uses the name people call the bot by
    bot_name = (m.guild.me if m.guild else bot.user).display_name
    try:
        # Decided before typing() — a reaction sends no message, so the typing indicator would linger
        emoji = emoji_vocabulary(m.guild)
        if reaction := await choose_reaction(c, m.author.display_name, bot_name, emoji, history=h):
            try:
                await m.add_reaction(emoji[reaction])
                entry["reaction"] = reaction  # kept on its message, so it stays in order and doesn't use a history slot
                log.info(f"[REACT] {reaction}")
                return
            except discord.HTTPException as e:  # no Add Reactions permission, or an emoji Discord doesn't know
                log.warning(f"React {reaction} failed, replying instead: {e}")
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
