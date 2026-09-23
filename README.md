# jevbot

Discord bot that makes [TypeSafe's Jev](https://openrouter.ai/~typesafe/jev-latest) talk — a decision model that "cannot generate text," loomed word-by-word into broken sentences.

Jev is a non-autoregressive decision model. It answers questions with calibrated probabilities, not text. This bot gives it a 10K word vocabulary and asks "next word?" repeatedly via tournament sampling until it forms a reply.

## How it works

1. **Tournament sampling**: 10K vocab shuffled into 255-word buckets, all scored in parallel
2. **Runoff**: Top-2 from each bucket compete in a final round  
3. **Completeness judge**: A separate `noul` question asks "is the reply complete?" — jev stops when it thinks it's done
4. **Penalty system**: Content words penalized 2.5x per reuse, stopwords 1.6x — prevents "is are I is are" loops
5. **History**: Last 3 user messages included as context, each labelled with the sender's display name, and jev's own turn labelled with the bot's server nickname; jev's own broken output is excluded (it poisons follow-ups), but its emoji reactions are kept. Every word and sender name in those messages is added to the vocab, so jev can repeat them. History lives in memory, so after a restart it is rebuilt from the channel's recent messages the first time someone talks to jev there (needs **Read Message History**)
6. **Reactions**: Before replying, jev is asked whether the message is a question or request for it — if not, chatty one-liners ("lol", "i just got a new job!!") get a 😂 or 🎉 instead of a reply

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
```

Enable **Message Content Intent** in Discord developer portal.

```bash
python jev_bot.py
```

## Usage

Mention jev or reply to jev's messages. Replies only — it won't respond to messages that don't involve it.

## Cost

Tournament sampling does ~4 API calls per word. Via OpenRouter, Jev costs $0.042 per million input tokens (output is free), which comes to ~$0.003 per word with the 10K vocab, so ~$0.01-0.10 per reply depending on length.

## Vocab

`vocab.txt` is a 20K word list (from [bewinxed/jevgpt](https://github.com/bewinxed/jevgpt)) with slurs removed, ordered from most to least common. Jev only uses the first `VOCAB_SIZE` (10K) words, since every word in the vocab is paid for on every step. Words can be added or removed freely — the vocab IS the content filter — but a word added past the cutoff is never used, so put new words in `custom_vocab.txt`. Words from the message jev is replying to are always added, so it can repeat a rarer word someone just used.

Server-specific words and phrases go in `custom_vocab.txt` (gitignored, create it next to `jev_bot.py`), one per line (`#` for comments). Multi-word phrases are chosen as a single unit, and entries already in the vocab are skipped. Restart the bot to pick up changes.

Reactions come from `emoji.txt` (one unicode emoji per line) plus the server's own custom emoji, which jev sees by their `:name:`. Like the word vocab, edit the list to change what jev can react with. The bot needs the **Add Reactions** permission; without it jev replies instead.

## Credits

- [TypeSafe AI](https://typesafe.ai) for Jev
- [bewinxed/jevgpt](https://github.com/bewinxed/jevgpt) for the tournament sampling architecture and vocab
- Built by [lyra](https://twitter.com/_lyraaaa_) + clod
