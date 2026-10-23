# rocky

Discord bot named after Rocky, the Eridian engineer from *Project Hail Mary*. It has two writers, and `!model` switches between them for everyone:

- **`jev`** (the default): [TypeSafe's Jev](https://openrouter.ai/~typesafe/jev-latest), a decision model that "cannot generate text", made to talk by asking it "next word?" over and over until it has loomed a broken sentence ("Band is from california in san los angeles. Las angels.").
- **`rocky`**: Claude Haiku on OpenRouter as Rocky himself ("Is called Biscuit. Small predator, no respect for hot liquid."), an agent with its own log of each channel, tools and memories.

Forked from [lyramakesmusic/jevbot](https://github.com/lyramakesmusic/jevbot).

## Setup

Create `.env`:

```
DISCORD_TOKEN_JEV=your_discord_bot_token
OPENROUTER_API_KEY=your_openrouter_key
OPENROUTER_API_KEY_EVAL=another_openrouter_key
STATUS_CHANNEL_ID=channel_to_post_statuses_in
TIMEZONE=Europe/London
ROCKY_VIEWER_PORT=8350
```

- `OPENROUTER_API_KEY` is the bot's. `jev_eval.py` and `rocky_replay.py` use `OPENROUTER_API_KEY_EVAL`, so each key can have its own credit limit and its usage shows up separately. With your own Anthropic keys on OpenRouter (BYOK), give each OpenRouter key its own Anthropic key from its own Claude Console workspace, so tests show up apart there too and can't use up the bot's rate limit.
- `STATUS_CHANNEL_ID` (optional): each new status is also posted there (see Status).
- `TIMEZONE` (an IANA name, UTC if unset): the clock the times in Rocky's log and the bar's hours are on.
- `ROCKY_VIEWER_PORT` (off if unset): serves the page of Rocky's logs on that port while the bot runs (see Rocky's logs).

Optional files next to `jev_bot.py`, all gitignored. Lists are one entry per line, `#` for comments; restart the bot to pick up changes.

- `rocky_lines.txt`: real Rocky lines, shown to the LLM as examples. A line can start with a note in brackets on what had just happened (`(a friend uses slang he doesn't know) No understand word.`), which it's told isn't his words. They're quotes from copyrighted works, so they stay out of the repo. Without it Rocky goes by the voice rules alone.
- `rocky_body.txt`: the book's narration of what Rocky looks like and is made of, a paragraph per passage with blank lines between, so he knows drawings of himself. Shown as narration, apart from his lines.
- `dm_users.txt`: Discord user IDs whose DMs get answered, no mention needed. Other DMs are ignored. What's said in DMs never makes it into a status.
- `admins.txt`: Discord user IDs who can change `/settings`, as well as the bot's owner in Discord's developer portal.
- `custom_vocab.txt`: server-specific words and phrases for Jev (see Vocab).

In the Discord developer portal, enable **Message Content Intent**. The bot needs **Read Message History** (to rebuild history and catch up after a restart), **Add Reactions** (without it Jev replies instead of reacting) and **Create Public Threads** (for `!nocontext`).

```bash
python3 run.py
```

`run.py` runs the bot (`uv run --with-requirements requirements.txt jev_bot.py`; pass another command after `--`) and keeps it up to date: every minute it fetches, and when the branch's upstream has new commits that fast-forward onto a clean checkout it pulls and restarts the bot, so merging to `master` deploys. Local edits or commits are left alone. If the bot exits on its own it's restarted, backing off if it keeps dying. Ctrl-C passes on to the bot; a third Ctrl-C kills it.

Ctrl-C (or SIGTERM) to the bot lets replies in progress finish: it stops taking new messages and exits once it's done. Ctrl-C again quits without finishing. Messages sent while it's offline never reach it, so on startup it looks back `CATCH_UP_WINDOW` (30) minutes in each channel and answers the latest message to it since it last replied or reacted there, one per channel.

## Talking to it

Mention the bot, or reply to one of its messages with the reply ping on. A reply with the ping off is just chatter unless it mentions the bot or its role. Other bots' and apps' posts are seen as chatter and never trigger a reply.

Put `!nocontext` in a message to the bot (`@rocky !nocontext what's your favourite colour?`) and it opens a thread on that message, named from it, and answers there with none of the channel's history in view. Every message in the thread is to the bot, no tag needed, and nothing said there reaches the channel's history, Rocky's log of it, or statuses. If the message is a Discord reply, what it replies to isn't shown. Rocky's log of the thread has no memories: no `<memories>` block, no `remember` or `forget`, and its prompt says nothing said there is kept. `!nocontext` again inside the thread just carries on. In a DM, or when the thread can't be opened, it answers that message on its own, with nothing before it.

Rocky can ping people by writing `@` and their full display name (`@Pip Meadow, ...`). The name must end at punctuation, a line break or the end of the reply, so `@Moss, ...` pings Moss but `@Moss knows` stays text. Names match regardless of case; bare names, ambiguous names, code and URLs stay text. Only people in its context can be pinged: speakers, mentioned users and reactors. Roles, `@everyone`, `@here`, raw IDs and automatic reply pings stay off, and Jev, statuses and the Why/Context answers never ping anyone.

## Commands

- `!model` says which writer is on; `!model jev` or `!model rocky` switches, for everyone. Kept in `model.json` (gitignored), so a restart keeps it. Works without a ping.
- `/settings` shows how often a new status is made (`STATUS_EVERY`) and Rocky's context window (`LOG_TOKENS`), visible only to you. Admins get a menu under each: a status every 1 to 24 hours, or off, and a window of 25K to 90K tokens (past 100K Haiku charges 5× for the whole prompt). A smaller window takes effect after Rocky's next turn, when the oldest half of its log goes; a much smaller one can take a few turns, halving each time. Kept in `settings.json` (gitignored). The menus stop working after 15 minutes or a restart. Works in a DM too.
- `!why` posts a chart of what Jev weighed for its latest reply in the channel: each word's candidates, how likely the model thought each was, and their scores after the repeat penalties. As a Discord reply to one of its replies it charts that one, or to a message it reacted to, the emoji it weighed. In the status channel it charts the statuses posted there the same way. Emoji are drawn from [Twemoji](https://github.com/jdecked/twemoji) (the set Discord uses) and the server's own emoji. For a reply Rocky wrote there's nothing to chart: Haiku doesn't say how likely its words were.

Right-click (long-press on mobile) a reply, a message the bot reacted to, or a posted status and pick from **Apps** for answers only you see:

- **Why**: `!why` for that message.
- **Context**: the transcript the bot was given, ending in its own empty turn, with the model and message count, and for an LLM turn its output tokens, input against the model's window, and cached versus uncached input, each with a text bar. A long transcript is attached as `context.txt`.
- **Stats**: the model, total recorded cost in USD (question checks, status checks and every logged attempt, including costs billed to a linked provider key), and context usage of the final request.
- **Diagnostics**: request counts and purposes, elapsed time, model IDs, billing, cache and output details, retries, and why an LLM request stopped early (a refusal is also logged as a warning with its reason, provider and generation id, never what was said).

These all read `logs/` (Why looks back `WHY_DAYS`, 7) and make no model requests. Everyone who can use apps in a channel sees the actions; a server can limit them in **Server Settings → Integrations**. The bot registers them each time it starts, so an open Discord client may need reloading (Ctrl+R) the first time. Why, Stats and Diagnostics share an in-memory cache of up to `EXPLANATION_CACHE_ENTRIES` (100) answers; charts are uploaded once and their signed Discord URLs reused until `EXPLANATION_EXPIRY_MARGIN` (60) seconds before they expire.

## Rocky

`!model rocky` is Haiku as Rocky, in every channel and DM. Each channel is Rocky's own conversation, kept by `rocky_harness.py` as an append-only log in `rocky_logs/` (gitignored).

- **The log**: people's messages, reactions and edits come in as user turns, written as XML with who, when and what they reply to (`<message id="12" from="pip" at="18:01" to="you" woke_you="yes">…</message>`); Rocky's turns are the assistant's. It takes everything said in the channel, and after a restart what it missed is added at the end. A channel gets a log the first time Rocky is woken there, filled from its history back to half of `LOG_TOKENS` (`FILL`) or `FILL_SCAN` (3000) messages, leaving out `!nocontext` messages and reactions to the bot's old replies.
- **Waking**: a tag or a reply wakes it. It thinks first (low effort), then what it writes is posted as its reply. A question always gets an answer in words.
- **Tools**, called in the same response:
  - `react` puts an emoji on any message in the log: what it does instead of answering "lol", news or thanks.
  - `remember` keeps a short fact about someone, itself or a topic; `forget` drops one.
  - `bar` and `order` are for drinks at The Hail Mary, a made-up pub with a menu and prices, hours on the `TIMEZONE` clock, and days it doesn't open or a drink has run out (picked from the date, so the same all day).
  - `look` shows it the pictures in a message: its images, its embeds' (a Tenor gif, a link preview) and its stickers, up to 4, fetched again from Discord and shrunk to 1024px (a gif's first frame). Until then a picture is only its `[photo]` tag.
  - For `bar`, `order` and `look`, what Rocky writes waits for the result, so it can't promise a pint the bar won't serve.
- **Memories**: facts are kept in `rocky_memories.json` (gitignored), shared by a server's channels, each DM its own. They're shown in a `<memories>` block before the log, a snapshot only rebuilt when the cache is (below); facts kept or forgotten since are in the log as its calls.
- **The cache**: nothing earlier in the log is ever rewritten, so each request starts with the one before it and the prompt cache covers all but the newest events. The cache is only rebuilt when the log outgrows `LOG_TOKENS` (90K, counted at `CHARS_PER_TOKEN`, 2.2 characters a token; set in `/settings`) and its oldest half goes, or after an hour without a request, when it has expired anyway. A picture Rocky looked at counts ~1400 tokens. A deleted message is marked straight away (`<deleted id="12"/>`), since Rocky already saw it, and blanked, with its edits, the next time the cache is rebuilt.
- **Updates**: when the bot starts on new code, each log is told of commits to `rocky_harness.py` since it last was, by subject alone (`<update>give Rocky a bar to order drinks from</update>`), so Rocky knows what changed if asked but doesn't announce it.

On `jev_eval.py`'s chats, a turn costs ~0.036¢ and takes ~4s to reply, with 86% of the prompt read from the cache.

## Jev

Jev is a non-autoregressive decision model: it answers questions with calibrated probabilities, not text. The bot gives it a 10K word vocabulary and asks "next word?" until it forms a reply.

1. **Tournament sampling**: the vocab is shuffled into 255-word buckets, all scored in parallel, and the top 2 from each bucket compete in a runoff.
2. **Stopping**: a separate question asks "is the reply complete?", and Jev stops when it thinks it's done. It also stops when it picks `<END>`; `.`, `!` and `?` count as votes for that, and a reply that ends this way ends on the mark if Jev liked it best ("Dunno forgot?").
3. **Penalties**: content words are penalised 2.5× per reuse and stopwords 1.6×, which prevents "is are I is are" loops.
   - Fillers and greetings that say the same thing (`SIMILAR`: yes/yeah/yea, hi/hey/hello, …) count as repeats of each other, and most of them vote together, so yes at 8%, yeah at 7% and yep at 5% beat a word at 10%. Sound-alikes ("Google goo woo") are left alone.
   - Jev copies itself from one reply to the next, so each of its last 3 answers (`ECHO_TURNS`) that said a word divides it by 2 (`ECHO_PENALTY`), unless the message it's answering used that word too.
   - Words that say it has nothing to say (`NOTHING`: dunno/idk, no/nope, crickets, silence, …) count as one word, divided by 3 (`NOTHING_PENALTY`): one is free, but a run of "Dunno", "Crickets chirping", "No? No?" breaks into an answer.
4. **History**: the last 6 messages to Jev (`HISTORY_TO_BOT`) and the last 8 other messages in the channel (`HISTORY_CHATTER`), labelled with display names, with mentions as readable `@name` and `#channel`. Its own replies and reactions are shown under the messages they answered, and people's reactions to its replies after them (`😂×2 💀`). Every word and name in those messages is added to the vocab, so Jev can repeat them.
5. **Reactions**: before replying, Jev is asked whether the message is a question or request for it. If not, a chatty one-liner ("lol", "i just got a new job!!") gets a 😂 or 🎉 instead, picked from `emoji.txt` and the server's custom emoji. Its recent answers count against an emoji the same way, with 🤷 counted as "dunno".

Jev can't see images or follow links, so they're shown as tags: a link by its embed's title, or its site and path words until Discord adds the embed (`[gif: Azumanga Daioh Sakai: 100% Real, True, Based]`, `[link: <headline>]`), attachments and stickers as `[photo]`, `[video]`, `[sticker: name]`, and standalone embeds as `[embed: ...]`, cut to `EMBED_WORDS` (20) words. Rocky sees the same tags, and can `look`.

History lives in memory, so after a restart a channel's is rebuilt from its last `HISTORY_SCAN` (400) messages the first time someone talks to the bot there. Who reacted is kept in `reactors.json` (gitignored), so a rebuild only asks Discord about an emoji whose count has changed.

Some things it has said:

- "I love jazz because its improvised and freedom."
- "No because overkill. Overkill!.!.!"
- "I depends on on situation of circumstances."
- "Yuck no ugh spit! Gag gagging ing"

### Vocab

`vocab.txt` is a 20K word list (from [bewinxed/jevgpt](https://github.com/bewinxed/jevgpt)) with slurs removed, most common first. Jev only uses the first `VOCAB_SIZE` (10K), since every word is paid for on every step, plus `. , ! ?` and `\n` from the end (`PUNCTUATION`). The vocab is the content filter, so edit it freely, but a word added past the cutoff is never used: put new words in `custom_vocab.txt`, where multi-word phrases are chosen as one unit. Words from the message Jev is replying to are always added.

### Cost

Tournament sampling does ~4 API calls per word. Jev costs $0.042 per million input tokens (output is free), ~$0.003 per word with the 10K vocab, so ~$0.01-0.10 per reply depending on length.

## Status

Every `STATUS_EVERY` (180) minutes the bot makes a new custom status, written as the rest of a diary entry that starts "I feel", "I'm thinking about" or "I wonder", in turn ("I feel something what weird is strange means dunno"). The current writer makes it: Jev looms it, Rocky writes it in one Haiku request.

- Jev judges whether each candidate makes sense on its own, with none of the chat in view. Broken grammar is fine; disconnected words and fragments are not. A score of at least `STATUS_THRESHOLD` (0.60) accepts it. It tries up to `STATUS_ATTEMPTS` (3) candidates, then keeps the previous status.
- Every other status has the last `STATUS_CHAT` (8) messages from the most recently active channel in view, and their words in the vocab ("I'm thinking about going snooker gonna out with mates"). Set `STATUS_CHAT = 0` to use the vocab alone.
- A new one is only made once someone has said something since the last, so a quiet stretch costs nothing.
- Each new status is posted in `STATUS_CHANNEL_ID`, if set, so the good ones aren't lost.
- The latest is kept in `status.json` (gitignored), so a restart shows it again.
- Out of credit, it shows "out of credit". Just after starting it shows as idle with "waking up" until it has caught up (about 20 seconds).

A Jev candidate costs ~$0.02-0.04, plus the small judgement. Set `STATUS_EVERY = 0`, or turn it off in `/settings`, to leave the status alone.

## Logs

Every message the bot handles is written as one JSON line to `logs/YYYY-MM-DD.jsonl`: who said what, the history and transcript the writer saw, the question score, the reaction or each word Jev picked (with its top candidates, their probabilities and scores, and the done score), every API call with its purpose and billing (for LLM requests also why it stopped, the provider and the generation id), and the cost, including requests billed to our own Anthropic key, which OpenRouter reports as free (`byok_cost`). Status entries keep every candidate in `status_attempts`. A Rocky turn keeps its events from Rocky's log in `rocky_events`. `logs/` holds what people said in the server, so it's gitignored: keep it local and delete old days whenever.

### Rocky's logs

`rocky_viewer.py` is a web page for looking into Rocky: a channel's log as Rocky sees it, with its thinking folded away, tool calls and results, and each turn's cost, tokens, cache reads and time; every turn it has taken, newest first, from `logs/`; and its memories, now and as in its `<memories>` snapshot. The open page picks up changes every few seconds.

```bash
python rocky_viewer.py                    # http://127.0.0.1:8350, this checkout's files
python rocky_viewer.py --root ../jevbot   # another checkout's
```

With `ROCKY_VIEWER_PORT` set, the bot serves it too. It listens on 127.0.0.1 only and shows what people said, so it's for this machine; to open it to others, put something in front that checks who's asking first (Cloudflare Tunnel with Cloudflare Access, say), never the bare port.

### Reactions

`reactions.py` reports how people reacted to what the bot posted. It reads `logs/` for replies and statuses, then fetches their reactions from Discord as they are now. Replies are broken down by writer and Jev's by length, with statuses apart and the most-reacted posts at the end. 👎 is a Rocky meme as often as a complaint, so it's counted apart: "besides 👎" is the share of posts with some other reaction. "audience" is how many other people spoke in the channel in the 30 minutes after a post (`AUDIENCE_MINUTES`), so a quiet evening can be told apart from a flop.

```bash
python reactions.py                 # everything in logs/
python reactions.py --days 7 --top 20
python reactions.py --logs ../jevbot/logs   # another checkout's logs
```

It runs alongside the bot on the same token and costs nothing. The most-reacted list prints what was said, so keep its output local.

## Testing

Tests run offline:

```bash
uv run --with-requirements requirements.txt python -m unittest
```

They use synthetic logs, Discord messages and API results: `test_harness.py` for Rocky's log and turns, `test_llm.py` for `!model` and LLM statuses, `test_mentions.py` for routing, history, catch-up and pings, `test_explanations.py` for Why, Context, Stats and Diagnostics, `test_status.py`, `test_settings.py`, `test_viewer.py`, `test_reactions.py` and `test_replay.py`. `test_penalty.py` replays the logged steps of real replies through `penalty()` and checks that the variant loops change while funny ones stay the same.

`jev_eval.py` runs a fixed set of conversations against the live Jev API and reports whether the question check sends questions to a reply and chatter to a reaction, and how Jev's first-word candidates split between real answers, `<END>`, words that describe the reply ("silent", "crickets"), and words only the transcript's formatting contains. A run costs ~$0.21 on `OPENROUTER_API_KEY_EVAL`; add `--replies` for a few full replies to judge by eye (~$0.05-0.10 each).

```bash
python jev_eval.py --save before.json     # on master
python jev_eval.py --compare before.json  # on your branch
```

`rocky_replay.py` replays messages to Rocky on a copy of the live channel log, its memories, `rocky_lines.txt` and `rocky_body.txt`, and prints its tool calls and results, reactions and what it said. Nothing is written back, and its thinking isn't printed. In a worktree it reads the files of the checkout its `.env` links to.

```bash
python rocky_replay.py "@rocky how are you doing?" "lol rocky"
python rocky_replay.py -n 3 --file scenarios.txt       # one message a line
python rocky_replay.py --image cat.png "@rocky what's this"  # sent with a synthetic picture, for look
python rocky_replay.py --at "2026-10-16 19:00" "rocky get us a beer"
python rocky_replay.py --fresh "@rocky get me a stout"  # a short synthetic chat instead of the live log
python rocky_replay.py "@rocky soup?" --with "fern: @rocky tea?"  # another to it, there before its turn
```

A turn on the live log is ~$0.007 and ~70K input tokens on `OPENROUTER_API_KEY_EVAL`. Run them one at a time: the first of a run writes the log to the cache, which counts towards Anthropic's input tokens a minute. Its output can quote the live log, so keep it local.

## Credits

- [TypeSafe AI](https://typesafe.ai) for Jev
- [bewinxed/jevgpt](https://github.com/bewinxed/jevgpt) for the tournament sampling architecture and vocab
- Built by [lyra](https://twitter.com/_lyraaaa_) + clod
