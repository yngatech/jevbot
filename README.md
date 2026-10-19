# jevbot

Discord bot that makes [TypeSafe's Jev](https://openrouter.ai/~typesafe/jev-latest) talk — a decision model that "cannot generate text," loomed word-by-word into broken sentences.

Jev is a non-autoregressive decision model. It answers questions with calibrated probabilities, not text. This bot gives it a 10K word vocabulary and asks "next word?" repeatedly via tournament sampling until it forms a reply.

## How it works

1. **Tournament sampling**: 10K vocab shuffled into 255-word buckets, all scored in parallel
2. **Runoff**: Top-2 from each bucket compete in a final round  
3. **Completeness judge**: A separate `noul` question asks "is the reply complete?" — jev stops when it thinks it's done. It also stops when it picks `<END>`, and `.`, `!` and `?` count as votes for that: counted apart, they split the vote to end and a word won instead. A reply that ends this way ends on the mark if jev liked it best ("Dunno forgot?")
4. **Penalty system**: Content words penalized 2.5x per reuse, stopwords 1.6x — prevents "is are I is are" loops. Fillers and greetings that say the same thing (`SIMILAR`: yes/yeah/yea, hi/hey/hello, …) count as repeats of each other, so "Hi hey hi hey" is penalised like "Hi hi hi hi"; sound-alikes ("Google goo woo") are left alone. Most of these groups also vote together, so yes at 8%, yeah at 7% and yep at 5% beat a word at 10% instead of splitting the vote, and jev says the likeliest of them. Hesitation words (um/uh/erm and hmm/hm) and the words that say nothing (`NOTHING`, below) vote apart. Jev also copies itself from one reply to the next ("Dunno? Dunno?" four times running), so each of its last 3 answers (`ECHO_TURNS`) that said a word divides it by 2 (`ECHO_PENALTY`), unless the message it's answering used that word too. Words that say it has nothing to say (`NOTHING`: dunno/idk, no/nope, crickets, silence, empty, …) count as one word here, and by 3 (`NOTHING_PENALTY`): one is free, but a run of "Dunno", "Crickets chirping", "No? No?" breaks into an answer
5. **History**: The last 6 messages to jev (`HISTORY_TO_BOT`) and the last 8 other messages in the channel (`HISTORY_CHATTER`) are included as context — or, when an LLM is writing (see `!model`), the last 200 channel messages of any kind (`LLM_HISTORY`), about 300 lines with jev's replies. In the channels in `LONG_CONTEXT_CHANNEL_IDS`, `!model rocky` sees the channel as its own log instead (see `!model`); there the history keeps all `HISTORY_SCAN` messages a rebuild reads, so Rocky's log starts with all of them. Jev and the question check keep the short window, since Jev sends its history with every request it makes for a word. An LLM's transcript also shows when each message was sent, on the `TIMEZONE` clock, like an IRC log: the time whenever the minute changes, and a line where the day does (`--- Sat 10 Oct`, `02:40 kettle: rocky you still awake?`), so it can tell a late night or a long gap; Jev's doesn't, since it would pick the times as words. Each message is labelled with the sender's display name, with mentions preserved as readable `@name` and `#channel` text using Discord's formatter, and jev's own turn labelled with the bot's server nickname; jev's replies and emoji reactions are shown under the messages they answered, so earlier questions don't look ignored (jev's own words aren't added to the vocab, though). People's reactions to jev's replies are shown after them — to an LLM with reactors' display names as readable mentions (`rocky: No not wobbly (😂 by @Moss, @Pip; 💀 by @Fern)`), by their server nicknames like other mentions, and kept up to date as they come and go. Names are loaded from Discord when rebuilding history; if they can't be fetched, the emoji and counts still appear (`😂×3 by @Moss` when only some are known). They're kept in `reactors.json` (gitignored), up to date with live reactions, so a rebuild only asks Discord about an emoji whose count has changed since: asking about every reply made rebuilding a busy channel take ~25 seconds, with the reply waiting on it. Jev gets the emoji and counts alone (`😂×2 💀`): names didn't change what it said, and it sends the transcript with every word it asks for. This lets jev see what landed — they're left out of the question check, where jev would copy the emoji. Every word and sender name in those messages is added to the vocab, so jev can repeat them. Other bots' and apps' posts are included as chatter but never trigger replies; jev's own posts stay out. Standalone embeds include their title or author, description and fields in a plain `[embed: ...]` tag, truncated to `EMBED_WORDS` (20) words; link-preview embeds are represented only by their link tag. Jev can't see images or follow links, so a link is shown as a tag with its embed's title, or its site and path words until Discord adds the embed (`https://klipy.com/gifs/azumanga-daioh-sakai` → `[gif: Azumanga Daioh Sakai: 100% Real, True, Based]`, a BBC article → `[link: <headline>]`, a tweet → `[link: <author>]`), and attachments and stickers as tags (`[photo]`, `[video]`, `[sticker: name]`). History lives in memory, so after a restart it is rebuilt from the channel's last `HISTORY_SCAN` (400) messages the first time someone talks to jev there (needs **Read Message History**)
6. **Reactions**: Before replying, jev is asked whether the message is a question or request for it — if not, chatty one-liners ("lol", "i just got a new job!!") get a 😂 or 🎉 instead of a reply. Its recent answers count against an emoji the same way, with 🤷 counted as "dunno", so a run of "Dunno" replies doesn't turn into a run of shrugs

7. **Status**: Every `STATUS_EVERY` (180) minutes jev looms a new custom status, written as the rest of a diary entry that starts "I feel", "I'm thinking about" or "I wonder", in turn ("I feel something what weird is strange means dunno"). Before showing or posting a candidate, jev judges whether its finished text makes sense on its own, with none of the source chat in view. Broken grammar, spelling, punctuation and repetition are allowed; disconnected words and fragments that need unseen context are rejected. A score of at least `STATUS_THRESHOLD` (0.60) accepts it unchanged. It tries up to `STATUS_ATTEMPTS` (3) candidates with the same opener and chat, then keeps the previous status if all fail. An empty generation or missing judgement also keeps the previous status and stops retries. Every other status has the last `STATUS_CHAT` (8) messages from the most recently active channel in view, and their words in the vocab ("I'm thinking about going snooker gonna out with mates"). The status shows in every server jev is in, so those can repeat what people said; set `STATUS_CHAT = 0` to use the vocab alone. While jev is out of credit it shows "out of credit" instead, and makes none. Just after starting, or reconnecting, it shows as idle with "waking up" until it has caught up on messages it missed (about 20 seconds), since it's slow to answer meanwhile, then online with its status. A new one is only made once someone has said something in a server since the last, so a quiet stretch keeps the old one and costs nothing; once one is due, the next message brings it. The latest is kept in `status.json` (gitignored), so a restart shows it again and makes the next one when it's due, with the next opener. Each new status is also posted in the channel `STATUS_CHANNEL_ID` in `.env`, if it's set, so the good ones aren't lost when the next comes; jev's own messages stay out of its history, so the post doesn't feed the next status. ~$0.02-0.04 per generated candidate, plus the small judgement cost; retries can cost up to three candidates. Set `STATUS_EVERY = 0` to leave the status alone. Admins can change the interval with `/settings` (see below), and it counts from the last status straight away

## Output examples

- "I love jazz because its improvised and freedom."
- "Rock.? Yeah"  
- "No because overkill. Overkill!.!.!"
- "I depends on on situation of circumstances."
- "Yuck no ugh spit! Gag gagging ing"
- "Band is from california in san los angeles. Las angels."

## Setup

```bash
pip install -r requirements.txt
```

Create `.env`:
```
DISCORD_TOKEN_JEV=your_discord_bot_token
OPENROUTER_API_KEY=your_openrouter_key
OPENROUTER_API_KEY_EVAL=another_openrouter_key
STATUS_CHANNEL_ID=channel_to_post_statuses_in
TIMEZONE=Europe/London
LONG_CONTEXT_CHANNEL_IDS=channel_ids_for_rocky
ROCKY_VIEWER_PORT=8350
```

For `!model`'s LLMs, put real Rocky lines in `rocky_lines.txt` (gitignored, next to `jev_bot.py`), one per line (`#` for comments). A line can start with a note in brackets on what had just happened (`(a friend uses slang he doesn't know) No understand word.`), so the LLM sees when he says it; it's told the notes aren't his words. They're quotes from copyrighted works, so they stay out of the repo. Without it the LLMs go by the voice rules alone. Restart the bot to pick up changes.

`TIMEZONE` (an IANA name, UTC if unset) is the clock the times in an LLM's transcript are shown on.

`LONG_CONTEXT_CHANNEL_IDS` (comma-separated, none if unset) are the channels where `!model rocky` is Rocky itself, with the channel as its conversation (see `!model`).

`ROCKY_VIEWER_PORT` (off if unset) serves the page of Rocky's logs on that port while the bot runs (see Rocky's logs).

`OPENROUTER_API_KEY` is the bot's; `jev_eval.py` and `rocky_replay.py` use `OPENROUTER_API_KEY_EVAL` instead, so each key can have its own credit limit on OpenRouter and their usage shows up separately. With your own Anthropic keys on OpenRouter (BYOK), give each OpenRouter key its own Anthropic key, from its own Claude Console workspace, so tests show up apart there too and can't use up the bot's rate limit.

Enable **Message Content Intent** in Discord developer portal.

```bash
python3 run.py
```

`run.py` runs the bot (`uv run --with-requirements requirements.txt jev_bot.py`; pass another command after `--`) and keeps it up to date: every minute it fetches, and when the branch's upstream has new commits that fast-forward onto a clean checkout it pulls and restarts the bot. Local edits or commits are left alone. If the bot exits on its own it's restarted, backing off if it keeps dying. Ctrl-C passes on to the bot; a third Ctrl-C kills it.

## Usage

Mention jev or reply to jev's messages with the reply ping enabled. A reply with the ping off won't trigger a response unless you mention jev or its bot role in the message.

When an LLM is writing (see `!model`), the bot can mention people in the conversation by writing `@` followed by their full display name, including names with spaces (`@Pip Meadow, ...`). The full name must end at punctuation, a line break, or the end of the reply: `@Moss, ...` can ping Moss, but `@Moss Fernando` or `@Moss knows` stays text when only Moss is known. Names match without regard to case; bare names, ambiguous names, code, and URLs stay text. Only people in the LLM's context are eligible: speakers, mentioned users, and reactors. `!nocontext` uses only its side conversation. Roles, `@everyone`, `@here`, raw user IDs, and automatic reply pings remain disabled. Jev's word picker gets no mention choices and cannot ping anyone. Status posts and Why/Context answers still allow no pings.

`!why` posts a chart of what jev weighed for its latest reply in the channel: for each word, the candidates, how likely the model thought each was, and their scores after the repeat penalties. Send it as a Discord reply to one of jev's replies to chart that one instead, or to a message jev reacted to for a chart of the emoji it weighed. In the status channel (`STATUS_CHANNEL_ID`) it works the same for the statuses jev posted there: on its own it charts the latest, or send it as a reply to one; the words after the opener ("I feel") are charted. The chart's font has no emoji, so those are drawn from images: [Twemoji](https://github.com/jdecked/twemoji)'s, the set Discord uses, and the server's own emoji from Discord; one that can't be fetched is shown by name. It reads `logs/` (the last `WHY_DAYS`, 7, days) and costs nothing.

For a reply or status an LLM wrote (see `!model`), `!why` charts its tokens instead — the pieces of words an LLM writes in — each with the few alternatives it weighed and how likely it thought them, the one it said in gold. It's sampled, so not always the likeliest. Only the DeepSeek models and Kimi K3 give these (only some of their providers do, so requests go to those); for Haiku or Kimi K2 it says there's nothing to chart.

**Context** shows the transcript the bot was given, ending in its own empty turn. Right-click (long-press on mobile) a reply, a message it reacted to, or a posted status and pick **Apps → Context**; only you see the answer. It's one embed: the transcript as selectable text, then the model and chat message count, and for an LLM reply or status the output tokens, input tokens against the model's window, and cached versus uncached input when recorded, each with a bar drawn in text — no image. These use OpenRouter's reported token counts. Cached input includes both instructions and earlier chat, so it is not a prompt/chat split and is not added again to the window usage. Jev decisions and older logs without token usage show just the transcript, model and message count. A long transcript is attached as `context.txt` instead, which Discord shows above the embed and can open without downloading; it's uploaded again each time rather than cached, since a link to an earlier upload only downloads. Like `!why`, it reads `logs/` and costs nothing.

**Stats** is a compact summary: the model, total recorded cost in USD, and context usage with the chat message count when available. Pick **Apps → Stats** on a reply, a message the bot reacted to, or a posted status; only you see it. Cost includes the question check, status meaning checks and all logged generation attempts, including rejected status candidates and costs billed to a linked provider key. Context usage belongs to the final generation request; earlier attempts and Jev checks are not added to it. Missing usage is shown as “Not recorded.”

**Diagnostics** shows request counts, elapsed time, model IDs, billing and cache/output details. New logs also name each request's purpose: deciding whether to reply or react, writing a reply or status, and checking that a status makes sense. All requests go through OpenRouter; a linked Anthropic or other provider key can pay for the generation at its provider. HTTP failures and retries are counted when logged; unreported costs stay unavailable. An LLM request that stopped before finishing shows why, such as a refusal or running out of tokens, and a refusal is also logged as a warning with its reason, provider and generation id, never what was said. Older messages retain their totals and recorded usage, with a note when request purposes and retries were not recorded. Pick **Apps → Diagnostics**; it is private and makes no model requests.

`!why` is also an action, **Apps → Why**, as if `!why` were a Discord reply to that message, answered with a message only you can see — `!why` itself posts the chart for everyone. Everyone who can use apps in a channel sees the Why, Context, Stats and Diagnostics actions; a server can limit them per role or channel in **Server Settings → Integrations**. jev registers them with Discord each time it starts, and a Discord client that was already open may need reloading (Ctrl+R) to show them the first time.

Why, Stats and Diagnostics share an in-memory cache of up to `EXPLANATION_CACHE_ENTRIES` (100) answers, dropping the least recently used first, and clear it when jev restarts. Short text and embeds stay in the cache; charts are uploaded once, then only their signed Discord attachment URLs are kept, and repeated charts use image embeds. Requests arriving together share preparation and the first upload. URLs keep their full signing parameters, including for private responses; an unavailable attachment or a URL within `EXPLANATION_EXPIRY_MARGIN` (60) seconds of expiry is generated and uploaded again. Each request still checks the logs, so asking for the latest follows new answers and deleted logs stop being shown.

`!model` says which model is writing jev's replies and statuses; `!model <name>` switches, for everyone: `jev` (the default), `deepseek` (DeepSeek V4 Pro), `flash` (DeepSeek V4.1 Flash), `haiku`, `kimi` (Kimi K2) or `k3` (Kimi K3). The five are ordinary LLMs on OpenRouter, told to talk like Rocky, the Eridian engineer from *Project Hail Mary* who the bot is named after ("Is called Biscuit. Small predator, no respect for hot liquid."), with real lines of his from `rocky_lines.txt` as examples. Jev still decides whether to reply or react, and picks the emoji. The choice is kept in `model.json` (gitignored), so a restart keeps it. Like `!why`, it works without a ping.

`!model rocky` is Haiku as Rocky itself, in the channels in `LONG_CONTEXT_CHANNEL_IDS` (elsewhere it writes like `haiku`). There the channel is Rocky's conversation, kept by `rocky_harness.py` as an append-only log: people's messages, reactions and edits come to it as user turns, written as XML with who, when and what they reply to (`<message id="12" from="pip" at="18:01" to="you" woke_you="yes">…</message>`), and its own turns are the assistant's. Woken by a tag or a reply, it thinks first (low effort), then what it writes is posted as its reply, and in the same response it can call tools: `react` puts an emoji on any message in the log (what it does instead of answering "lol", news or thanks — Jev's reply-or-react check isn't used), `remember` keeps a short fact about someone, itself or a topic, and `forget` drops one. `bar` and `order` are for drinks at The Hail Mary, a made-up pub with a menu and prices, hours on the `TIMEZONE` clock, and days it doesn't open or a drink has run out (picked from the date, so the same all day). For those two, what Rocky writes waits for what the bar says: it gets another request to say what happened, so it can't promise a pint the bar won't serve. `look` shows it the pictures in a message — its images, its embeds' (a Tenor gif's, a link's preview) and its stickers, up to 4, fetched again from Discord since attachment links expire, and shrunk to 1024px (a gif's first frame) — and like the bar it waits to see them before it speaks; until then a picture is only its `[photo]` tag. What it saw stays in the log as the call's result, counted at ~1400 tokens a picture towards `LOG_TOKENS`, and goes with a deleted message's text at the next tidy. A question always gets an answer in words. Nothing earlier in the log is ever rewritten, so each request starts with the one before it and the prompt cache covers all but the newest events — on jev_eval's chats, 0.036¢ a turn with ~4s to reply, 86% of the prompt read from the cache. Its facts are in a `<memories>` block before the log, shared by a server's channels (and each DM its own), a snapshot only rebuilt when the cache would be rebuilt anyway: when the log outgrows its context window (`LOG_TOKENS`, 90K tokens, under Haiku's 100K price step; counted as 2.2 characters a token, `CHARS_PER_TOKEN`, and set in `/settings`) and its oldest half goes, or after an hour without a request, when the cache has expired. Facts kept or forgotten since are in the log as its `remember` and `forget` calls. A deleted message is marked straight away (`<deleted id="12"/>`) — Rocky still saw what it said — and blanked at that same point (`<message id="12" from="pip" at="18:01" deleted="yes"/>`), with its edits and the mark, so taking it out never costs a cache rewrite of its own. The first time a channel's log is needed it's filled from the channel's history, and after a restart what it missed is added at the end. When the bot starts on new code, each log is told of commits to `rocky_harness.py` since it last was, by subject alone, at its end (`<update>give Rocky a bar to order drinks from</update>`), so Rocky knows when it changed: it says so when asked what's new, but doesn't announce it. A log from before then got every one since it began, each where it happened. Logs are in `rocky_logs/` and memories in `rocky_memories.json` (both gitignored). Asked to write Rocky's reply to a transcript, as the other models are, Haiku sometimes described the reply instead ("Rocky suggests something concrete") — as Rocky, with thinking, it didn't in 57 eval turns. Side conversations (`!nocontext`) are still answered the old way, and never go in the log.

Put `!nocontext` in a message to jev (`@jev !nocontext what's your favourite colour?`) and it answers with none of the channel's history in view, just that message. That message and every Discord reply under it, on any branch and from anyone, are a side conversation: a reply in it sees all of it (each branch sees the others) and nothing from the rest of the channel, and the rest of the channel — other questions to jev, and its statuses — never sees it. Jev’s replies in it end with an inline linked `↳` pointing to the starting `!nocontext` message; the marker stays out of its context. Mention jev outside it to get the channel back. jev keeps the latest `SIDE_TALKS` (50) side conversations, up to `CHAIN_DEPTH` (20) messages each; after a restart it finds them again in the last `HISTORY_SCAN` messages, or by following a reply's chain back.

DMs are ignored, except from the Discord user IDs listed in `dm_users.txt` (gitignored, create it next to `jev_bot.py`), one per line (`#` for comments) — every message they DM jev gets answered, no mention needed. What's said in DMs never makes it into jev's status. Restart the bot to pick up changes.

`/settings` shows two settings, answered so only you can see: how often jev makes a new status (`STATUS_EVERY`), and Rocky's context window, how far back `!model rocky` remembers a channel (`LOG_TOKENS`). Admins also get a menu under each, and picking from one saves it and counts straight away: a status every hour up to every 24 hours, or off, and a context window of 25K to 200K tokens. A smaller window takes effect after Rocky's next turn, when the oldest half of its log goes and its cache is rewritten; a much smaller one can take a few turns, halving each time. Changes are kept in `settings.json` (gitignored), so a restart keeps them. The menus stop working after 15 minutes or a restart; `/settings` again brings new ones. Admins are the Discord user IDs in `admins.txt` (gitignored, next to `jev_bot.py`, the same format as `dm_users.txt`), plus the bot's owner in Discord's developer portal. `/settings` works in a DM with the bot too. Restart the bot to pick up changes to `admins.txt`.

Ctrl-C (or SIGTERM) lets replies in progress finish first: jev stops taking new messages (it catches up on them next start) and exits once it's done. Ctrl-C again quits without finishing.

Messages sent while jev is offline (say, during a restart) never reach it, so on startup it looks back `CATCH_UP_WINDOW` (30) minutes in each channel and answers the latest message to it since it last replied or reacted there — one per channel. Needs **Read Message History**.

## Cost

Tournament sampling does ~4 API calls per word. Via OpenRouter, Jev costs $0.042 per million input tokens (output is free), which comes to ~$0.003 per word with the 10K vocab, so ~$0.01-0.10 per reply depending on length.

With `!model`, and the 200-message history, a reply costs ~$0.0015 with Haiku (its median since the history grew) — and with DeepSeek ~$0.001-0.0015 when its provider's cache hits and ~$0.0045 when it doesn't (its replies before it was pinned to one provider had a median of ~$0.004), ~$0.00005 with Flash, ~$0.001 with Kimi K2, ~$0.001-0.002 with K3, plus Jev's reply-or-react check (~$0.0001). A status is about the same, plus Jev's meaning check.

## Logs

Every message jev handles is written as one JSON line to `logs/YYYY-MM-DD.jsonl`: who said what, the history and transcript jev saw, the question score, request purposes and reported billing in `api_calls` (for LLM requests also why each stopped, the provider OpenRouter sent it to and its generation id), the reaction or each word it picked (with its top candidates, each with its probability and its score after penalties, and the done score), and what the API calls actually cost — including a request OpenRouter sent on our own Anthropic key (its BYOK), which OpenRouter reports as free; that part is also logged on its own as `byok_cost`. Status entries also keep each candidate in `status_attempts`, with its judgement score, acceptance, generation trace and cost; the top-level word steps and transcript belong to the accepted candidate so `!why` and Context still show what was posted. The total cost includes all attempts and judgements, including when no new status was accepted. A `!model rocky` turn also keeps its events from Rocky's log in `rocky_events`: its responses, with their thinking and tool calls, and its calls' results. The console's `[OUT]` and `[REACT]` lines show the cost too. `logs/` holds what people said in the server, so it's gitignored — keep it local and delete old days whenever.

## Rocky's logs

`rocky_viewer.py` is a web page for looking into `!model rocky`: a channel's log as Rocky sees it, with people's messages, reactions, edits and deletions, and its turns with their thinking (folded away), tool calls and results, and what each turn cost, its tokens and how much was read from the cache, and how long it took; every turn it has taken, newest first, from `logs/`, which goes back past a log's jumps; and its memories, now and as they are in its `<memories>` snapshot. It reads only the files the bot writes, so it shows a log a couple of seconds after it changes and a turn once it's finished, and the open page picks up changes every few seconds.

```bash
python rocky_viewer.py                    # http://127.0.0.1:8350, this checkout's files
python rocky_viewer.py --root ../jevbot   # another checkout's
```

With `ROCKY_VIEWER_PORT` set, the bot serves it too, for as long as it runs. Either way it listens on 127.0.0.1 only, and the page runs no script but its own, showing what people said as text. The logs hold what people said in the server, so it's for this machine; to open it to others, put something in front that checks who's asking first (Cloudflare Tunnel with Cloudflare Access, say), never the bare port.

## Reactions

`reactions.py` reports how people reacted to what jev posted. It reads `logs/` for jev's replies and statuses, then fetches the reactions on them from Discord as they are now, because reactions aren't logged and people keep adding them after jev moves on. Replies are broken down by who wrote them (Jev, or the LLM picked with `!model`) and Jev's by length, with statuses in a table of their own and the most-reacted posts listed at the end. 👎 is a Rocky meme as often as a complaint, so it's counted apart: "besides 👎" is the share of posts with some other reaction. "audience" is how many people other than jev spoke in the channel in the 30 minutes after a post (`AUDIENCE_MINUTES`), so a quiet evening can be told apart from a flop.

```bash
python reactions.py                 # everything in logs/
python reactions.py --days 7 --top 20
python reactions.py --logs ../jevbot/logs   # another checkout's logs
```

It reads one page of channel history per 100 messages over the stretch the posts cover, through Discord's API only, so it runs alongside the bot on the same token. It costs nothing. Posts that have since been deleted are left out. The most-reacted list prints what jev said, so keep its output local, like `logs/`.

## Testing changes

`jev_eval.py` runs a fixed set of conversations against the live Jev API and reports what's measurable: whether the question check sends questions to a reply and chatter to a reaction, and how jev's first-word candidates split between real answers, `<END>`, words that describe the reply ("silent", "crickets"), and words that only the transcript's formatting contains. Save a run on master, then compare your branch against it:

```bash
python jev_eval.py --save before.json     # on master
python jev_eval.py --compare before.json  # on your branch
```

A run costs ~$0.21, on `OPENROUTER_API_KEY_EVAL`, and prints what OpenRouter billed for it. Add `--replies` to also generate a few full replies to judge by eye (~$0.05-0.10 each) — whether they're funny is still up to you.

`rocky_replay.py` replays messages to `!model rocky` on a copy of the live channel log, its memories and `rocky_lines.txt`, and prints what Rocky did: its tool calls and their results, its reactions, and what it said. Nothing is written back, and its thinking isn't printed. Run it on master and on your branch to compare:

```bash
python rocky_replay.py "@rocky how are you doing?" "lol rocky"
python rocky_replay.py -n 3 --file scenarios.txt       # one message a line
python rocky_replay.py --image cat.png "@rocky what's this"  # sent with a synthetic picture, for look
python rocky_replay.py --at "2026-10-16 19:00" "rocky get us a beer"
python rocky_replay.py --fresh "@rocky get me a stout"  # a short synthetic chat instead of the live log
python rocky_replay.py "@rocky soup?" --with "fern: @rocky tea?"  # another to it, there before its turn
```

In a worktree it reads the files of the checkout its `.env` links to. A turn on the live log is ~$0.007 and ~70K input tokens, on `OPENROUTER_API_KEY_EVAL`, one request at a time — the first of a run writes the log to the cache, which counts towards Anthropic's input tokens a minute, so don't run many at once. Its output can quote the live log (what Rocky reacted to), so keep it local. `test_replay.py` checks it offline (`python -m unittest test_replay`).

`test_penalty.py` needs no API: it replays the logged steps of real replies through `penalty()` and checks that the variant loops change while funny ones stay the same (`python test_penalty.py`).

`test_status.py` exercises status generation, judging, retries and publishing with synthetic API results and Discord writes (`python -m unittest test_status`). It checks that rejected or unscored candidates keep the previous status, that accepted candidates retain their original wording and diagnostic trace, and that quiet and out-of-credit periods make no attempts.

`test_viewer.py` checks `rocky_viewer.py`'s answers offline with synthetic logs: a channel's log, memories and turns, turns read from a day's file as it grows (a line still being written left for later), a page at a time, nothing sent when nothing has changed, and no file served but the page's own (`python -m unittest test_viewer`).

`test_reactions.py` checks `reactions.py` offline: which log entries count as posts, matching them to synthetic Discord messages, leaving out jev's own reactions, and keeping 👎 apart (`python -m unittest test_reactions`).

`test_mentions.py` exercises live message routing and startup catch-up with synthetic Discord messages (`python -m unittest test_mentions`). It checks that unpinged replies stay as chatter, other apps' posts and truncated standalone embeds appear in live and rebuilt history without triggering replies, embed edits update context, link previews keep their tags without duplication, mentions and allowed DMs still trigger responses, diagnostic commands work without a ping, and the Why and Context actions answer only whoever asked. Outgoing pings are checked for LLMs only, including full-name boundaries, code and URLs, ambiguity, side-thread restarts, and consistent live and rebuilt replies. In a `LONG_CONTEXT_CHANNEL_IDS` channel, the other LLMs see the last 200 messages like anywhere else, and a rebuild keeps all it read for Rocky's log.

`test_explanations.py` exercises the Why, Context, Stats and Diagnostics answers and their cache with synthetic logs and Discord sends (`python -m unittest test_explanations`). It renders real charts and checks URL reuse across commands and private actions without retaining attachment bytes, expiry and availability checks, fresh uploads when an embed is rejected, shared uploads during concurrent requests and cancellation, eviction, retries after rendering or upload errors, and updates when answers, logs or custom emoji change. Action checks exercise compact private summaries, exact window/cache fields and text bars with short and freshly attached transcripts, named request purposes and billing, failed request attempts, old logs, Jev reactions and posted statuses, missing usage, zero cost, out-of-credit replies, and missing or mismatched logs.

## Vocab

`vocab.txt` is a 20K word list (from [bewinxed/jevgpt](https://github.com/bewinxed/jevgpt)) with slurs removed, ordered from most to least common. Jev only uses the first `VOCAB_SIZE` (10K) words, since every word in the vocab is paid for on every step, plus `. , ! ?` and `\n` from the end of the list (`PUNCTUATION`). Words can be added or removed freely — the vocab IS the content filter — but a word added past the cutoff is never used, so put new words in `custom_vocab.txt`. Words from the message jev is replying to are always added, so it can repeat a rarer word someone just used.

Server-specific words and phrases go in `custom_vocab.txt` (gitignored, create it next to `jev_bot.py`), one per line (`#` for comments). Multi-word phrases are chosen as a single unit, and entries already in the vocab are skipped. Restart the bot to pick up changes.

Reactions come from `emoji.txt` (one unicode emoji per line) plus the server's own custom emoji, which jev sees by their `:name:`. Like the word vocab, edit the list to change what jev can react with. The bot needs the **Add Reactions** permission; without it jev replies instead.

## Credits

- [TypeSafe AI](https://typesafe.ai) for Jev
- [bewinxed/jevgpt](https://github.com/bewinxed/jevgpt) for the tournament sampling architecture and vocab
- Built by [lyra](https://twitter.com/_lyraaaa_) + clod
