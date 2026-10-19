# Agent instructions

## The live bot

- `run.py` pulls `master` within a minute of a merge and restarts the bot, so merging deploys. After a merge, check the bot came back up and that the change behaves as expected.
- In a worktree, `.env` and `custom_vocab.txt` are symlinks to the live checkout's files. Never write test data to them: they're gitignored and have no copy to restore from. Put test files in `.scratch/` or set env vars inline.

## Privacy

- This repo is public. Never put real chat messages, usernames or server emoji names in PRs, commits, tests or fixtures. Use synthetic conversations, such as `jev_eval.py`'s scenarios. You can validate against real logs locally, but describe what you found without quoting them.
- Rocky's thinking is kept in its logs as Anthropic's signed blocks (`reasoning_details`, each with a `text` and a `signature`). Never print a block's `text`, and never edit a block: Anthropic's safeguards treat reading or tampering with it as extracting the model's reasoning. Its other fields, lengths and counts are fine. Strip it before printing logs: `jq 'del(.. | .reasoning?, .reasoning_details?)'`.
- `rocky_lines.txt` holds copyrighted quotes. Never print or commit it.

## Testing

- Tests run offline: `uv run --with-requirements requirements.txt python -m unittest`.
- `rocky_replay.py` replays messages to Rocky on the live log, using the eval key. Run replays one at a time: each one writes the whole log to Anthropic's cache, which counts against the account's per-minute input limit.

## Rocky

- When Rocky misbehaves, prefer clearer wording in its prompt over code that inspects a turn and retries it. Replay the new wording, report the results honestly, then ship it and watch how it goes live.

## Pull requests

- Start every PR description with a GitHub note holding a one- or two-sentence tl;dr: what was wrong or missing, and what the PR changes. The rest of the description follows below it.

  ```markdown
  > [!NOTE]
  > **tl;dr:** Rocky's reply was dropped when a second message arrived while it was still replying. Rocky's turn now stays where it began, and messages that arrive during it get their own turn.
  ```
