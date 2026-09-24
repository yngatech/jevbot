"""
Offline checks for penalty() and pick(): near-synonyms count as repeats and vote together, interesting phrases
don't. No API calls, no cost.

The replays are real replies from the logs: each step's top candidates with their raw probabilities, re-scored
with today's penalty() given the words picked before it. Only the first step that changes means anything —
after it jev would have seen a different reply, so its later candidates are unknown (`jev_eval.py --replies`
shows what it goes on to say).

    python test_penalty.py      # or: uv run --with pytest --with-requirements requirements.txt pytest test_penalty.py
"""

import os

os.environ.setdefault("DISCORD_TOKEN_JEV", "test")
os.environ.setdefault("OPENROUTER_API_KEY", "test")

from jev_bot import END, MIN_WORDS, is_word, penalty, pick, recent_answers, render, said_in  # noqa: E402

# reply: [(word picked, [(candidate, raw probability), ...]), ...] — "top" from logs/*.jsonl
REPLAYS = {
    "Hi hey hi hey": [
        ("hi", [("hi", .23), ("dunno", .09), ("silence", .06), ("hey", .06), ("crickets", .05)]),
        ("hey", [("hey", .07), ("dunno", .06), ("blue", .04), ("hi", .33), ("likewise", .03)]),
        ("hi", [("hi", .21), ("likewise", .05), ("blue", .04), ("dunno", .04), ("you", .03)]),
        ("hey", [("hey", .16), ("pink", .04), ("you", .04), ("what", .04), ("dunno", .03)]),
        (END, [(END, .05), ("sorry", .04), ("rocky", .03), ("red", .03), ("hello", .02)]),
    ],
    "Yeah yes https https too is": [
        ("yeah", [("yeah", .26), ("yes", .12), ("crickets", .09), ("yea", .06), ("yup", .06)]),
        ("yes", [("yes", .15), ("yea", .06), ("too", .04), ("yep", .04), ("yeah", .2)]),
        ("https", [("https", .07), ("yeah", .2), ("and", .05), ("beeves", .04), ("yea", .04)]),
        ("https", [("https", .56), ("yes", .12), ("link", .03), ("yea", .03), ("com", .03)]),
        ("too", [("too", .05), ("yes", .15), ("yeah", .12), ("is", .03), (END, .03)]),
        ("is", [("is", .04), ("yea", .04), ("beeves", .04), ("yeah", .12), ("yep", .03)]),
        (END, [(END, .05), ("yea", .04), ("yeah", .11), ("too", .1), ("yes", .1)]),
    ],
    "Yes yeah": [
        ("yes", [("yes", .34), ("yeah", .1), ("dunno", .05), ("i", .03), ("bad", .02)]),
        ("yeah", [("yeah", .04), ("dunno", .03), ("right", .03), ("is", .03), ("sim", .02)]),
        (END, [(END, .17), ("yes", .14), ("and", .03), ("dunno", .03), ("correct", .03)]),
    ],
    # Funny ones that must come out the same
    "Google goo woo": [
        ("google", [("google", .19), ("dunno", .1), ("wat", .05), ("what", .04), ("look", .04)]),
        ("goo", [("goo", .05), ("google", .29), ("what", .03), ("using", .03), ("look", .02)]),
        ("woo", [("woo", .21), (END, .06), ("glue", .05), ("go", .05), ("wat", .04)]),
        (END, [(END, .06), ("whoosh", .06), ("whoa", .04), ("wow", .03), ("dunno", .03)]),
    ],
    "He 's annoying": [
        ("he", [("he", .03), ("yeah", .03), ("anyway", .02), ("him", .02), ("shush", .02)]),
        ("'s", [("'s", .06), ("annoying", .06), ("keeps", .05), ("him", .05), ("is", .05)]),
        ("annoying", [("annoying", .09), ("bothering", .07), ("is", .05), ("he's", .04), ("keeps", .03)]),
        (END, [(END, .06), ("he's", .05), ("him", .04), ("is", .04), ("just", .03)]),
    ],
    "Yes no is no": [
        ("yes", [("yes", .35), ("no", .09), ("yeah", .07), ("absolutely", .07), ("is", .06)]),
        ("no", [("no", .08), ("yes", .42), ("absolutely", .04), ("question", .03), ("yeah", .03)]),
        ("is", [("is", .08), ("yes", .24), ("question", .06), ("no", .27), (END, .04)]),
        ("no", [("no", .17), ("question", .07), ("bad", .05), ("yes", .15), ("is", .16)]),
        (END, [(END, .05), ("happens", .04), ("is", .09), ("yes", .13), ("no", .37)]),
    ],
    "Dunno a know is mean means knows skinny thin young gay": [
        ("dunno", [("dunno", .32), ("is", .09), ("crickets", .06), ("a", .05), ("huh", .04)]),
        ("a", [("a", .1), ("is", .09), ("young", .05), ("know", .05), ("skinny", .04)]),
        ("know", [("know", .21), ("idea", .06), ("i", .05), ("mean", .05), ("dunno", .17)]),
        ("is", [("is", .04), ("mean", .04), ("skinny", .04), ("dunno", .13), ("know", .28)]),
        ("mean", [("mean", .08), ("know", .24), ("means", .05), ("knows", .04), ("knew", .04)]),
        ("means", [("means", .11), ("meant", .07), ("know", .18), (END, .04), ("and", .03)]),
        ("knows", [("knows", .05), ("skinny", .03), ("mean", .08), ("thin", .02), ("words", .02)]),
        ("skinny", [("skinny", .05), ("young", .03), ("is", .07), ("means", .08), ("knows", .17)]),
        ("thin", [("thin", .09), ("skinny", .29), ("gay", .03), ("twink", .03), ("young", .03)]),
        ("young", [("young", .06), ("gay", .05), ("and", .05), ("skinny", .15), ("is", .08)]),
        ("gay", [("gay", .04), (END, .04), ("guy", .03), ("is", .07), ("young", .17)]),
        (END, [(END, .07), ("is", .11), ("yes", .04), ("guy", .03), ("and", .02)]),
    ],
}


# Each step's pick under today's penalty(), the way loom() picks: <END> only once MIN_WORDS words are said
def replay(steps):
    picks, words = [], []
    for picked, top in steps:
        stoppable = sum(1 for w in words if is_word(w)) >= MIN_WORDS
        scored = {w: p / penalty(words, w) for w, p in top if w != END or stoppable}
        picks.append(pick(scored, stoppable)[0])
        if picked != END:
            words.append(picked)
    return picks


# (step, new pick) of the first step that picks differently, or None
def first_change(steps):
    for i, (new, (old, _)) in enumerate(zip(replay(steps), steps)):
        if new != old:
            return i + 1, new
    return None


def test_variants_count_as_repeats():
    assert penalty(["hi"], "hey") == penalty(["hi"], "hi")
    assert penalty(["yeah", "yes"], "yea") == penalty(["yeah", "yeah"], "yeah")
    assert penalty(["Yeah"], "yep") == penalty(["yeah"], "yeah")

def test_lookalikes_are_not_repeats():
    assert penalty(["google"], "goo") == 1
    assert penalty(["google", "goo"], "woo") == 1
    assert penalty(["skinny"], "thin") == 1
    assert penalty(["yes"], "no") == 1

def test_variant_loops_break():
    assert first_change(REPLAYS["Hi hey hi hey"]) == (2, "dunno")
    # yes, yeah, yea and yep had 47% between them, so even penalised they outvote "too": a repeat, not a variant
    assert first_change(REPLAYS["Yeah yes https https too is"]) == (2, "yeah")
    assert first_change(REPLAYS["Yes yeah"]) == (2, "dunno")

def test_funny_replies_unchanged():
    for reply in ["He 's annoying", "Yes no is no",
                  "Dunno a know is mean means knows skinny thin young gay"]:
        assert first_change(REPLAYS[reply]) is None, reply

# Where jev copied itself from reply to reply: first-step candidates from the logs, with the answers it had in view
ECHOES = {
    "Dunno? Dunno?": (["Dunno? Dunno?", "Dunno yes", "Dunno? Dunno"],
                      [("dunno", .3), ("?", .06), ("shrugging", .05), ("stop", .05), ("idk", .04)]),
    "Private? Private?": (["Private? Private?", "Private? Private?"],
                          [("private", .21), ("?", .12), ("my", .08), ("grok", .07), ("computer", .06)]),
    # "Dunno google" goes too, but to "Google": crickets and 🤷 were in view as well
    "Dunno google": (["Crickets chirping gay guy is twink", "🤷", "Dunno yeah kinda"],
                     [("dunno", .2), ("crickets", .13), ("google", .07), ("silence", .06), ("penis", .04)]),
    # Clearly meant: a dunno after one other, and "four" for 2+2 with dunno everywhere
    "Dunno maybe positive": (["He 's annoying", "🤣", "No dunno ninja"],
                             [("dunno", .3), ("no", .1), ("i", .07), ("yes", .05), ("nope", .05)]),
    # "century egg?" / "toast with butter? jam?" after runs of answers that said nothing
    "Century egg": (["No? No?", "No? No"], [("no", .31), ("yuck", .06), ("nope", .04), ("not", .04), ("nor", .03)]),
    "Jam": (["Crickets chirping"], [("no", .08), ("jam", .08), ("empty", .06), ("nay", .05), ("nope", .03)]),
    "Four dunno": (["Dunno? Dunno?", "Dunno? Dunno?", "Dunno gone"], [("four", .86), ("dunno", .08), ("idk", .03)]),
}

def first_word(reply, answers=None, message=""):
    answers, top = ECHOES[reply] if answers is None else (answers, ECHOES[reply][1])
    history = [{"role": "user", "name": "x", "content": "?", "reply": a} for a in answers]
    exempt = said_in(message)
    return max(top, key=lambda wp: wp[1] / penalty([], wp[0], recent_answers(history), exempt))[0]

def test_echoes_across_replies():
    assert penalty([], "idk", recent_answers([{"role": "assistant", "name": "jev", "content": "Dunno"}])) > 1
    assert penalty([], "idk", recent_answers([{"role": "user", "name": "x", "content": "?", "reaction": "🤷"}])) > 1
    assert penalty([], "google", recent_answers([{"role": "assistant", "name": "jev", "content": "Dunno"}])) == 1

def test_echo_runs_break():
    assert first_word("Dunno? Dunno?") == "?"
    assert first_word("Private? Private?") == "?"
    assert first_word("Dunno google") == "google"

def test_meant_answers_kept():
    assert first_word("Dunno maybe positive") == "dunno"
    assert first_word("Four dunno") == "four"
    assert first_word("Dunno? Dunno?", message="idk what to do") == "dunno"  # said to jev: fair game

# jev's answers said nothing ("No? No?", "Crickets chirping"): the other words that say nothing count against it too
def test_nothing_words_count_together():
    crickets = recent_answers([{"role": "assistant", "name": "jev", "content": "Crickets chirping"}])
    assert penalty([], "no", crickets) == penalty([], "dunno", crickets) > 1
    assert penalty([], "jam", crickets) == 1
    assert penalty([], "no", recent_answers([{"role": "assistant", "name": "jev", "content": "No? No?"}])) > 1
    assert penalty([], "no", crickets, said_in("no dunno idk")) == 1  # said to jev: fair game

def test_nothing_runs_break():
    assert first_word("Century egg", message="century egg?") == "yuck"
    assert first_word("Jam", message="toast with butter? jam?") == "jam"
    assert first_word("Century egg", ["Yes rock"], "century egg?") == "no"  # the first is free
    assert first_word("Century egg", ["No? No"], "century egg?") == "no"  # one before: "no" at 31% still clearly meant

def test_only_last_answers_count():
    old = ["Dunno", "Dunno", "Dunno"]
    assert first_word("Dunno? Dunno?", old + ["Yes", "No", "Hi"]) == "dunno"

# Its ending goes on: whoa 4% and wow 3% outvote stopping at 6%. The sound-alikes still don't pool
def test_google_goo_woo_whoa():
    assert first_change(REPLAYS["Google goo woo"]) == (4, "whoa")

# Near-synonyms vote together, so a split doesn't hand it to another word
def test_similar_words_pool_their_vote():
    assert pick({"um": .08, "uh": .07, "erm": .05, "cat": .1}, True) == ("um", False, ["uh", "erm"])
    assert pick({"hmm": .06, "hm": .05, "cat": .1}, True)[0] == "hmm"
    assert pick({"Yeah": .06, "yes": .05, "cat": .1}, True)[0] == "Yeah"
    assert pick({"um": .08, "cat": .1}, True) == ("cat", False, [])
    assert pick({"google": .06, "goo": .05, "cat": .1}, True)[0] == "cat"  # lookalikes aren't the same word

# From the logs: "Closed.? Yes?" — yes 7%, yep 2%, yea 1%, yup 1% against "closed" at 8%
def test_split_yes_wins():
    top = {"closed": .08, "yes": .07, "no": .05, "is": .04, "yep": .02, "yea": .01, "yup": .01, "nope": .01}
    assert pick(top, False) == ("yes", False, ["yep", "yea", "yup"])

# ...but the words that say nothing don't: "Hello world am positive." kept "hello" at 19% over idk 18% + dunno 8%
def test_nothing_words_do_not_pool():
    top = {"hello": .19, "idk": .18, "i": .09, "dunno": .08, "no": .05, "nope": .01}
    assert pick(top, False)[0] == "hello"
    assert pick({"no": .06, "nah": .05, "cat": .1}, True)[0] == "cat"

def test_endings_pool():
    assert pick({END: .04, ".": .03, "?": .02, "cat": .06}, True) == (END, True, [".", "?"])
    assert pick({END: .02, "?": .05, "cat": .06}, True) == ("?", True, [END])
    assert pick({".": .04, "?": .03, "cat": .06}, False) == ("cat", False, [])  # can't stop yet


if __name__ == "__main__":
    for reply, steps in REPLAYS.items():
        change = first_change(steps)
        if change:
            i, new = change
            print(f"{reply!r:58} -> {render([w for w, _ in steps[:i - 1]] + [new])!r}...")
        else:
            print(f"{reply!r:58}    unchanged")
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
    print("ok")
