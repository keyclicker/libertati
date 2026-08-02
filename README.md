# libertati

An agentic Telegram bot: one LLM brain living across all chats.

Unlike a classic request/response bot, libertati runs a single persistent
agent loop. Every incoming message from every chat (plus timers and
heartbeats) is appended to one shared context timeline, and the model
decides — deliberately, via a `send_message` tool — whether, where and
when to reply. Plain assistant output is ignored and sends nothing.

## How it works

- **Single brain** (`agent.py`): events queue up and are processed one
  batch at a time on the OpenAI Responses API; the full context history
  is persisted append-only in SQLite, with a capped tail window sent to
  the model. The model/tool round engine itself lives in `loop.py` and
  is shared with the dreaming loop.
- **Tools** (`tools.py`): `send_message`, `get_recent_messages`, and
  `schedule_wakeup` — the agent can set alarms for its future self.
- **Dreaming** (`dream.py`): after a stretch of idleness — or when the
  agent calls `dream` itself — the waking loop pauses and a
  differently-prompted loop runs one long offline session. It wanders
  with read-only and web tools, writes a reflection into `DREAMS.md`,
  folds `INBOX.md` into a rewritten `MEMORY.md`, and may revise
  `SOUL.md`. None of a dream's context is persisted; the agent wakes to
  a `[dream ended …]` event carrying its summary. Budgeted per 24h.
- **Wiring** (`bot.py`): aiogram handlers persist every message and push
  it as an event; background loops deliver due wakeups, periodic
  heartbeat status digests, and hand the agent over to a dream.
- **Storage** (`db.py`): SQLite (WAL) with full raw Telegram payloads,
  append-only model context, and exact API token/cache usage.
- **Mind** (`memory.py`): four markdown files under `data/memory/`,
  editable by hand at any time. `SOUL.md` is the personality, re-read
  and attached to the instructions every turn. `INBOX.md` is where the
  `remember` tool drops raw dated facts. `MEMORY.md` is the curated
  long-term store the `recall` tool answers questions from with a
  one-shot extraction call (it is never inlined into the agent's
  context) — only a dream rewrites it. `DREAMS.md` is the dream
  journal. Every soul rewrite is snapshotted under `soul/` first.
- **Prompts** (`prompts.toml`): user-editable agent, roleplay,
  web-search, dream and memory-helper instructions loaded at startup.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12+.

```sh
cp .env.example .env      # fill in bot token + API key
$EDITOR settings.toml     # model, timezone, heartbeat, persona
uv run libertati
```

Any OpenAI-compatible endpoint works (OpenAI, OpenRouter, local) — see
`settings.toml` for `base_url`/`model` examples. All settings can be
overridden with `LIBERTATI_*` environment variables.

## Docker

```sh
docker compose up -d --build
```

The compose file bind-mounts `./settings.toml` (read-only) and `./data`
(SQLite), and runs hardened: read-only rootfs, no capabilities, non-root.

## Development

```sh
uv run ruff check .       # lint (incl. docstring rules)
uv run ty check           # type check
uv run pytest             # tests
uv run libertati-ctx      # live colored tail of the agent's context
uv run libertati-ctx --live   # paged full-screen viewer (j/k, g/G, q)
```

The full-screen status shows authoritative token/cache counts from the
latest API call plus a separate rough estimate for the next effective
context after reasoning and pruning policy are applied.
