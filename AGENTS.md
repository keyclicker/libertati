# AGENTS.md

Guidance for AI coding agents working in this repository. Read
`README.md` first for what the project is and how it behaves.

## Commands

```sh
uv sync                      # install (dev deps included)
uv run pytest                # tests (pytest-asyncio, auto mode)
uv run ruff check .          # lint (I, UP, B, D; google docstrings)
uv run ruff format --check . # formatting
uv run ty check              # type check
uv run libertati             # run the bot (needs .env + settings.toml)
uv run libertati-spy         # dev TUI over the live context DB
```

CI runs all four checks plus tests; keep every one green.

## Architecture map

One package, `src/libertati/`, no sub-packages:

- `bot.py` — aiogram wiring, entry point `run()`: handlers persist every
  message and push events to the agent; background loops for wakeups,
  heartbeats and dream handover.
- `agent.py` — `Agent(ModelLoop)`: the single waking loop; event queue,
  turn lock, context-window trimming/restore invariants.
- `loop.py` — `ModelLoop`: one Responses-API round (model call + tool
  execution + usage bookkeeping), shared by waking and dreaming loops.
- `dream.py` — `Dreamer(ModelLoop)` and `DreamGate`: the offline
  reflection session; budget, cooldown and idle triggers.
- `tools.py` — every function-tool schema and its handler (`Toolbox`).
- `db.py` — aiosqlite persistence: messages/chats/users, append-only
  agent context, turns, usage, dreams, wakeups.
- `memory.py` — `Mind`: the five markdown mind files under
  `data/memory/`.
- `config.py` — pydantic-settings `Settings` (env > .env >
  settings.toml > defaults).
- `prompts.py` / `prompts.toml` — user-editable model prompts; every
  key in `Prompts` is required and validated at startup.
- `chats.py` — chat approval registry (`data/chats.toml`).
- `clock.py` — the only place timestamp formats live.
- `transcript.py` — the only place stored messages are rendered for the
  model (history tools and the context events carry).
- `spy.py` — standalone read-only TUI; must not import aiogram/openai
  at module level (keeps `libertati-spy` startup fast).
- `render.py` — how the spy lays out one context item: a layout per
  kind, each falling back to the generic body, plus `searchable()`,
  the text a spy search matches against. Pure functions over decoded
  items; no database, no terminal.

## Invariants to preserve

- **Only tools act.** The agent's plain text output is private; sending
  anything to Telegram goes through a tool handler that also persists
  the outgoing message via `db.save_message(..., outgoing=True)`.
- **Context window validity.** The in-memory window must start at a
  `role: user` event and never end with an unanswered `function_call`
  or a bare `reasoning` item — the API rejects such input and the agent
  wedges. All trimming goes through `_trim_to_boundary` /
  `_trim_dangling`; keep new code paths behind them.
- **Append-only history.** Every context item is written through to the
  `context` table; the DB is the source of truth, the window a view.
- **`store=False` everywhere.** Nothing is stored server-side;
  encrypted reasoning must ride along in the context.
- **Dream context is throwaway.** A dream persists nothing except mind
  files, its ledger row and the wake-up event.
- **Timestamps**: UTC in the DB (`clock.utc_stamp`, matches SQLite's
  `datetime('now')`), the configured timezone for anything the model
  sees (`clock.format_local`).
- **Idle clock**: `Agent.last_active` moves only on activity events or
  outward tool calls — heartbeat-only turns must not reset it, or idle
  dreams become unreachable.

## Adding an agent tool

Three places, all in `tools.py`: a strict-mode JSON schema in the right
`*_TOOLS` list, an async `_handler(self, args)` on `Toolbox`, and an
entry in `self._handlers`. A tool that reads or acts on a specific chat
must also list its chat-id argument(s) in `GATED_CHAT_ARGS` so the
approval registry is enforced before the handler runs (a test asserts
this for every schema parameter named like a chat id). Provider-side
strict schemas are not a security boundary: `Toolbox.run` validates the
same schemas locally before dispatch, so keep new schemas within its
supported subset. Optional parameters are `["type", "null"]` and still
listed in `required`. Handlers return plain strings (JSON for lists)
and never raise — `Toolbox.run` converts exceptions to `error: …`
strings. Decide whether the dreaming loop may use it: dream tools come
from `DREAM_API_TOOLS`, and execution is gated by `DREAM_TOOL_NAMES`.
A new tool in `MESSAGING_TOOLS` counts as outward activity by default
and so resets the dream idle clock; if it only looks something up, list
it in `READ_ONLY_MESSAGING_TOOLS`.

## Conventions

- Docstrings on every module, class and function (ruff `D`, google
  style), moderate verbosity. Comments explain *why*, not what.
- Section banner comments (`# ===== Name =====`) group related code in
  the larger files; keep new code in the right section.
- Config lives in `settings.toml` (shipped, every non-secret key listed
  explicitly with a comment) mirrored by defaults in `config.py` — keep
  the two in the same order. Secrets only in `.env` (`LIBERTATI_*`).
- New `prompts.toml` keys must be added to `Prompts` and `load_prompts`
  in `prompts.py` (all keys required) and to the tests.
- Schema changes: `SCHEMA` uses `CREATE TABLE IF NOT EXISTS`, which
  never alters existing tables — add columns for existing DBs via
  `Database._ensure_column` in `connect()`.
- Tests are plain functions with plain asserts; async tests need no
  decorator (asyncio_mode = auto). Real SQLite via the `db` fixture in
  `conftest.py`; minimal hand-rolled fakes (`SimpleNamespace`, small
  classes) instead of mock frameworks; aiogram objects built with
  `Message.model_validate`.
- Commits: Conventional Commits, imperative, ≤72-char lines (see the
  global user instructions for the full rules and trailers).

## Gotchas

- `Agent.__new__(Agent)` in tests skips `__init__`; when `_process`,
  `_turn` or `_remember` grow new attribute reads, update the fixtures
  that hand-build agents — `make_turn_agent` / `make_processing_agent`
  in `test_agent.py`, and `test_dream.py`.
- Telegram messages are sent with legacy Markdown and fall back to
  plain text when parsing fails (`_markdown_send`); `@username`
  underscores are pre-escaped.
- SQLite string ops (LIKE, NOCASE) fold case for ASCII only; message
  search uses the custom `casefold` SQL function registered in
  `Database.connect`.
- Same-second message bursts are real: any query ordering by `date`
  needs `message_id` as a tiebreaker.
- The three background loops in `bot.py` must survive transient
  errors — log and continue, never let the loop die.
- Spy search matches `render.searchable()` — the headline plus the
  rendered body — so a layout that moves text between the two changes
  what is findable. Highlight numbering follows the same order
  (headline first), and `spy.Match.index` is an index into it.
