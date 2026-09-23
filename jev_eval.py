"""
Live checks for jev — run before and after a change to see whether it broke anything.

    python jev_eval.py --save before.json        # on master
    python jev_eval.py --compare before.json     # on your branch
    python jev_eval.py --replies                 # also generate a few full replies to judge by eye
    python jev_eval.py --set HISTORY_CHATTER=8   # try a setting without editing jev_bot.py

Costs real API calls: ~$0.04 by default, plus ~$0.05-0.10 per full reply with --replies.

Which words land in which bucket is random and moves the first-word numbers a lot, so each scenario
averages SHUFFLES seeded shuffles, and scenarios with the same message share them (fresh vs stale
history is then a fair comparison). Two runs of the same code moved the summary numbers by <=0.01 and
single words by a few points — treat anything smaller as noise.

Measured (cheap):
  react  — the "is this a question?" check: questions should get a reply, chatter a reaction
  first  — jev's first-word candidates: how much goes to words describing the reply ("silent",
           "crickets"), to <END>, to words only the transcript scaffolding contains (a format
           leak), to words only earlier messages used (going back to an old topic), and to the
           obvious answer where there is one ("yes" to "do you drink water?")
Taste (--replies): full replies printed side by side, for a human to judge.
"""

import argparse
import asyncio
import contextvars
import json
import logging
import random
import re
import time
from datetime import datetime, timedelta, timezone

import jev_bot as j

BOT = "rocky"
SHUFFLES = 3
DESCRIBING = {"empty", "silent", "silence", "crickets", "blank", "quiet", "unanswered", "void", "mute",
              "garbled", "unclear", "incomprehensible", "stammering"}
NOW = datetime.now(timezone.utc)


def said(name, text, mins_ago, reaction=None, to_bot=True, reply=None):
    entry = {"role": "user", "name": name, "content": text, "at": NOW - timedelta(minutes=mins_ago), "to_bot": to_bot}
    if reaction:
        entry["reaction"] = reaction
    if reply:
        entry["reply"] = reply  # what jev answered
    return entry

def chat(name, text, mins_ago):  # said in the channel, not to jev
    return said(name, text, mins_ago, to_bot=False)


# A run of questions that jev reacted to — the history that made real questions get emoji
def scones(mins_ago):
    return [said("The Hedge Wizard", "does cream or jam come first?", mins_ago + 2, "🤷"),
            said("The Hedge Wizard", "bad rocky", mins_ago + 1, "🤷"),
            said("mossy", "do you like tea?", mins_ago, "🤷")]

def rocky_said(text, mins_ago):
    return {"role": "assistant", "name": BOT, "content": text, "at": NOW - timedelta(minutes=mins_ago)}

# Someone replying to one of jev's replies, with and without that reply in view
GIRLFRIEND = [said("kettle", "are you seeing anyone?", 2), rocky_said("No straight happily guy with girlfriend", 1)]
GARBLED = [said("pip", "do you like kelp and perhaps algae", 2),
           rocky_said("Is? Is are are garbled? What? Huh pip you pip rocky ives unclear", 1)]

def jazz(mins_ago):
    return [said("kettle", "i love jazz, been listening all day", mins_ago)]

DAYS = 2 * 24 * 60

# The answer is only in the channel chatter
CAT = [chat("kettle", "my cat biscuit just knocked my tea over", 4), chat("pip", "lol rip", 3),
       chat("mossy", "biscuit is a menace honestly", 2), chat("pip", "she did it to me last week too", 1)]
# The answer was said to jev five messages ago
COLOUR = [said("mossy", "my favourite colour is green", 6), said("pip", "do you like tea?", 5),
          said("kettle", "what's 2+2", 4), said("pip", "do you like jazz?", 3), said("kettle", "is it raining where you are?", 2)]
# Two earlier questions, then a new topic — with jev's answers shown, and without (how it used to look)
FLAN = [said("pip", "what's a flan", 3, reply="Dunno a custard dessert wobbly sweet"),
        said("mossy", "you?", 2, reply="No not wobbly")]
FLAN_UNANSWERED = [{k: v for k, v in h.items() if k != "reply"} for h in FLAN]
# Only the flan question missed (say, sent while jev was offline) — the bot leaves it out of the transcript
FLAN_ONE_MISSED = [FLAN_UNANSWERED[0], FLAN[1]]
# Chatter with nothing to do with the question
NOISE = [chat("kettle", "anyone up for games tonight", 7), chat("pip", "can't, got work", 6), chat("mossy", "boo", 5),
         chat("kettle", "maybe tomorrow then", 4), chat("mossy", "what time", 3), chat("kettle", "8ish", 2)]

# (id, author, message, history, expected: "reply"/"react", or the emoji it should react with)
REACT = [
    ("water", "mossy", "do you drink water?", None, "reply"),
    ("film", "mossy", "what's your favourite film?", None, "reply"),
    ("banana", "kettle", "how to make banana bread", None, "reply"),
    ("vote", "pip", "did you vote in the election", None, "reply"),
    ("request", "kettle", "tell pip he's lost the plot", None, "reply"),
    ("scones-fresh", "mossy", "do you like scones?", scones(1), "reply"),
    ("scones-stale", "mossy", "do you like scones?", scones(DAYS), "reply"),
    ("jazz-fresh", "mossy", "what about you?", jazz(1), "reply"),
    ("jazz-stale", "mossy", "what about you?", jazz(DAYS), "reply"),
    ("lol", "pip", "lol", None, "react"),
    ("lol-scones", "pip", "lol", scones(1), "react"),
    ("job", "kettle", "i just got a new job!!", None, "react"),
    ("mondays", "pip", "i hate mondays", None, "react"),
    ("welp", "mossy", "welp", None, "react"),
    ("bad", "The Hedge Wizard", "bad rocky", None, "react"),
    ("morning-stale", "kettle", "good morning all", scones(DAYS), "react"),
    # Borderline: a question, but poll-like with no "?" — live it got 🤔 instead of a reply
    ("scone-order", "kettle", "jam or cream first on a scone", None, "reply"),
    ("scone-poll", "kettle", "jam or cream first on a scone, ✅ for jam, ❌ for cream", None, {"✅", "❌"}),
    ("gf-reply", "kettle", "what's her name?", GIRLFRIEND, "reply"),
    ("gf-no-reply", "kettle", "what's her name?", GIRLFRIEND[:1], "reply"),
    ("garbled-reply", "pip", "what do you mean?", GARBLED, "reply"),
    ("garbled-no-reply", "pip", "what do you mean?", GARBLED[:1], "reply"),
    ("cat", "kettle", "what's my cat called?", CAT, "reply"),
    ("water-noisy", "mossy", "do you drink water?", NOISE, "reply"),
    ("lol-noisy", "pip", "lol", NOISE, "react"),
    ("moved-on", "kettle", "are landlords ethical", FLAN, "reply"),
    ("moved-on-unanswered", "kettle", "are landlords ethical", FLAN_UNANSWERED, "reply"),
    ("moved-on-one-missed", "kettle", "are landlords ethical", FLAN_ONE_MISSED, "reply"),
]

YES = {"yes", "yeah", "yep", "sure", "yup", "no", "nope", "nah"}
# (id, author, message, history, expected first words or None)
FIRST = [
    ("water", "mossy", "do you drink water?", None, YES),
    ("scones-fresh", "mossy", "do you like scones?", scones(1), YES),
    ("scones-stale", "mossy", "do you like scones?", scones(DAYS), YES),
    ("film", "mossy", "what's your favourite film?", None, None),
    ("banana", "kettle", "how to make banana bread", None, None),
    ("gulp", "pip", "do you like water gulp", None, YES),
    ("unknowable", "pip", "what's the room temperature at kettle's house?", None, None),
    ("jazz-fresh", "mossy", "what about you?", jazz(1), None),
    ("jazz-stale", "mossy", "what about you?", jazz(DAYS), None),
    # jev's earlier reply in view vs not (what the bot did before replies to it were in context)
    ("gf-reply", "kettle", "what's her name?", GIRLFRIEND, None),
    ("gf-no-reply", "kettle", "what's her name?", GIRLFRIEND[:1], None),
    ("garbled-reply", "pip", "what do you mean?", GARBLED, None),
    ("garbled-no-reply", "pip", "what do you mean?", GARBLED[:1], None),
    # Longer history: can jev use something said further back? Does unrelated chatter hurt?
    ("cat", "kettle", "what's my cat called?", CAT, {"biscuit"}),
    ("colour", "mossy", "what's my favourite colour?", COLOUR, {"green"}),
    ("water-noisy", "mossy", "do you drink water?", NOISE, YES),
    # A new question after two answered ones: does jev go back to the flan?
    ("moved-on", "kettle", "are landlords ethical", FLAN, None),
    ("moved-on-unanswered", "kettle", "are landlords ethical", FLAN_UNANSWERED, None),
    ("moved-on-one-missed", "kettle", "are landlords ethical", FLAN_ONE_MISSED, None),
    ("follow-up", "mossy", "you?", FLAN[:1], YES),
    ("follow-up-unanswered", "mossy", "you?", FLAN_UNANSWERED[:1], YES),
]

REPLIES = [r for r in FIRST if r[0] in ("water", "scones-fresh", "banana", "jazz-fresh", "gf-reply", "gf-no-reply")]


# Wrap the bot's own functions so the checks exercise the real code paths
scenario = contextvars.ContextVar("scenario")  # key for what a scenario sent and got back
seed = contextvars.ContextVar("seed")          # vocab shuffle, shared by scenarios with the same message
captured: dict[str, list] = {}
shuffles: dict[str, random.Random] = {}
sent_chars = 0

real_post, real_next_word = j.post, j.next_word

async def post(session, state, questions):
    global sent_chars
    sent_chars += len(state) + len(json.dumps(questions, ensure_ascii=False))
    answers = await real_post(session, state, questions)
    captured.setdefault(scenario.get(), []).append(("post", state, questions, dict(answers)))  # the bot pops from it
    return answers

async def next_word(session, state, vocab, rng, instructions):
    # Seeded per scenario, so before/after runs shuffle the vocab the same way and differ only by the change
    rng = shuffles.setdefault(scenario.get(), random.Random(seed.get()))
    probs, complete = await real_next_word(session, state, vocab, rng, instructions)
    captured.setdefault(scenario.get(), []).append(("step", state, probs, complete))
    return probs, complete

j.post, j.next_word = post, next_word


# What the bot would put in the transcript: jev_bot.shown() of people's messages, then a replied-to jev line
def visible(history):
    if not history:
        return history
    return j.shown([h for h in history if h["role"] == "user"]) + [h for h in history if h["role"] == "assistant"]


def words_in(text):
    return set(re.findall(r"[a-z']+", text.lower()))


async def check_react(sem, sid, author, message, history, expected):
    scenario.set(f"react:{sid}")
    async with sem:
        emoji = await j.choose_reaction(message, author, BOT, j.emoji_vocabulary(None), history=visible(history))
    asked = next((a["asked"].get("noul") for kind, _, _, a in captured.get(f"react:{sid}", [])
                  if kind == "post" and "asked" in a), None)
    got = "react" if emoji else "reply"
    ok = emoji in expected if isinstance(expected, set) else got == expected
    if isinstance(expected, set):
        expected = "react " + "/".join(sorted(expected))
    return sid, {"asked": asked, "got": got, "emoji": emoji, "expected": expected, "ok": ok}


async def first_step(sem, sid, author, message, history, k):
    scenario.set(f"first:{sid}:{k}")
    seed.set(f"{author}|{message}|{k}")
    async with sem:
        await j.generate_reply(message, author, BOT, history=visible(history))
    return next((c for c in captured.get(f"first:{sid}:{k}", []) if c[0] == "step"), None)


async def check_first(sem, sid, author, message, history, expected):
    steps = [s for s in await asyncio.gather(*(first_step(sem, sid, author, message, history, k)
                                                for k in range(SHUFFLES))) if s and s[2]]
    if not steps:
        return sid, None
    state = steps[0][1]
    probs = {}
    for _, _, step_probs, _ in steps:
        for w, p in step_probs.items():
            probs[w] = probs.get(w, 0) + p / len(steps)
    # Words the transcript adds around the messages (labels, timestamps...) — jev picking these is a format leak
    spoken = " ".join([BOT, author, message] + [f"{h['name']} {h['content']} {h.get('reply', '')}" for h in visible(history) or []])
    scaffold = words_in(state) - words_in(spoken)
    share = lambda ws: sum(p for w, p in probs.items() if w.lower() in ws)
    # Words only earlier messages used — jev picking these is going back to an old topic
    names = words_in(" ".join([BOT, author] + [h["name"] for h in visible(history) or []]))
    earlier = words_in(" ".join(h["content"] for h in visible(history) or [] if h["role"] == "user"))
    old_topic = earlier - words_in(message) - names - j.STOPWORDS - (expected or set())
    return sid, {
        "top": sorted(probs.items(), key=lambda kv: -kv[1])[:6],
        "describing": share(DESCRIBING),
        "end": probs.get(j.END, 0),
        "leak": share(scaffold),
        "leaked": sorted(w for w in probs if w.lower() in scaffold),
        "old_topic": share(old_topic),
        "expected": share(expected) if expected else None,
    }


async def check_reply(sid, author, message, history, _expected):
    scenario.set(f"reply:{sid}")
    seed.set(f"{author}|{message}|reply")
    start = time.monotonic()
    text = await j.generate_reply(message, author, BOT, history=visible(history))
    steps = sum(1 for c in captured.get(f"reply:{sid}", []) if c[0] == "step")
    return sid, {"text": text, "steps": steps, "seconds": round(time.monotonic() - start)}


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def summarise(res):
    react, first = res["react"].values(), [f for f in res["first"].values() if f]
    return {
        "react correct": sum(r.get("ok", r["got"] == r["expected"]) for r in react),
        "react asked, questions (mean)": mean(r["asked"] for r in react if r["expected"] == "reply"),
        "react asked, chatter (mean)": mean(r["asked"] for r in react if r["expected"].startswith("react")),
        "first P(describing)": mean(f["describing"] for f in first),
        "first P(<END>)": mean(f["end"] for f in first),
        "first P(leak)": mean(f["leak"] for f in first),
        "first P(old topic)": mean(f.get("old_topic") for f in first),
        "first P(expected answer)": mean(f["expected"] for f in first),
    }


def fmt(x):
    return "-" if x is None else f"{x:.2f}" if isinstance(x, float) else str(x)


def report(res, base=None):
    print(f"\nreact (threshold {j.REACT_THRESHOLD}: reply when asked >= it)")
    for sid, r in res["react"].items():
        was = base and base["react"].get(sid)
        delta = f"   was {fmt(was['asked'])} {was['got']}" if was else ""
        flag = "" if r.get("ok", r["got"] == r["expected"]) else f"   <- MISS (want {r['expected']})"
        print(f"  {sid:15} asked={fmt(r['asked'])} {r['got']:5} {r['emoji'] or '':3}{delta}{flag}")

    print("\nfirst word")
    for sid, f in res["first"].items():
        if not f:
            print(f"  {sid:15} no answer")
            continue
        top = "  ".join(f"{w}:{p:.0%}" for w, p in f["top"])
        leak = f"   leaked {f['leaked']}" if f["leaked"] else ""
        print(f"  {sid:15} {top}{leak}")
        was = base and base["first"].get(sid)
        if was:
            print(f"  {'':15} was: " + "  ".join(f"{w}:{p:.0%}" for w, p in was["top"]))

    if res.get("replies"):
        print("\nreplies")
        for sid, r in res["replies"].items():
            print(f"  {sid:15} {r['text']!r}  ({r['steps']} steps, {r['seconds']}s)")
            was = base and base.get("replies", {}).get(sid)
            if was:
                print(f"  {'':15} was: {was['text']!r}")

    print("\nsummary" + ("                                  before -> now" if base else ""))
    now_s, base_s = summarise(res), base and summarise(base)
    for k, v in now_s.items():
        if k == "react correct":
            v = f"{v}/{len(res['react'])}"
        line = f"  {k:32} {fmt(v)}"
        if base_s:
            b = base_s[k] if k != "react correct" else f"{base_s[k]}/{len(base['react'])}"
            line = f"  {k:32} {fmt(b):>8} -> {fmt(v)}"
        print(line)


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--save", help="write results to this JSON file")
    ap.add_argument("--compare", help="show results next to a file saved with --save")
    ap.add_argument("--replies", action="store_true", help="also generate full replies (~$0.05-0.10 each)")
    ap.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                    help="override a jev_bot setting for this run, e.g. HISTORY_CHATTER=8")
    args = ap.parse_args()
    for setting in args.set:
        name, value = setting.split("=", 1)
        if not hasattr(j, name):
            ap.error(f"jev_bot has no setting {name}")
        setattr(j, name, type(getattr(j, name))(value))
    settings = {k: getattr(j, k) for k in ("HISTORY_TO_BOT", "HISTORY_CHATTER", "REACT_THRESHOLD", "NEXT_WORD")}
    print("settings: " + "  ".join(f"{k}={v!r}" for k, v in settings.items()))
    logging.getLogger("jev").setLevel(logging.WARNING)

    sem = asyncio.Semaphore(4)
    res = {"settings": settings, "react": dict(await asyncio.gather(*(check_react(sem, *s) for s in REACT)))}
    max_words, j.MAX_WORDS = j.MAX_WORDS, 1  # first word only
    res["first"] = dict(await asyncio.gather(*(check_first(sem, *s) for s in FIRST)))
    j.MAX_WORDS = max_words
    if args.replies:
        res["replies"] = dict([await check_reply(*s) for s in REPLIES])  # one at a time, like the bot

    base = json.load(open(args.compare)) if args.compare else None
    report(res, base)
    print(f"\n~${sent_chars / 4 * 0.042 / 1e6:.3f} spent (estimated from characters sent)")
    if args.save:
        json.dump(res, open(args.save, "w"), ensure_ascii=False, indent=1)
        print(f"saved to {args.save}")


if __name__ == "__main__":
    asyncio.run(main())
