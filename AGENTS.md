# Agent instructions

## The live bot

- `run.py` pulls `master` within a minute of a merge and restarts the bot, so merging deploys. After a merge, check the bot came back up and that the change behaves as expected.
- In a worktree, `.env` and `custom_vocab.txt` are symlinks to the live checkout's files. Never write test data to them: they're gitignored and have no copy to restore from. Put test files in `.scratch/` or set env vars inline.

## Privacy

- This repo is public. Never put real chat messages, usernames or server emoji names in PRs, commits, tests or fixtures. Use synthetic conversations, such as `jev_eval.py`'s scenarios. You can validate against real logs locally, but describe what you found without quoting them.
- Rocky's thinking is kept in its logs as Anthropic's thinking blocks (`reasoning_details`, each with a summarised `text` and an encrypted `signature`). Never print or quote a block's `text` in an agent session: an agent doing that was stopped by a `reasoning_extraction` refusal, which Anthropic uses for requests that try to get a model's reasoning into its response text ([docs](https://platform.claude.com/docs/en/build-with-claude/thinking#summarized-thinking)). Lengths, counts and the other fields are fine. Strip the blocks before printing logs: `jq 'del(.. | .reasoning?, .reasoning_details?)'`.
- Never edit a thinking block. The harness sends them back with Rocky's log, and Anthropic rejects a modified block with a 400 ([docs](https://platform.claude.com/docs/en/build-with-claude/thinking#preserving-thinking-blocks)).
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
