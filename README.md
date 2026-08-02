# libertati

An agentic Telegram bot: one LLM brain living across all chats.

Unlike a classic request/response bot, libertati runs a single persistent
agent loop. Every incoming message from every chat (plus timers and
heartbeats) is appended to one shared context timeline, and the model
decides — deliberately, via a `send_message` tool — whether, where and
when to reply. Plain text output is private thinking and sends nothing.

## How it works

- **Single brain** (`agent.py`): events queue up and are processed one
  batch at a time on the OpenAI Responses API; the full context history
  is persisted append-only in SQLite, with a capped tail window sent to
  the model.
- **Tools** (`tools.py`): `send_message`, `get_recent_messages`, and
  `schedule_wakeup` — the agent can set alarms for its future self.
- **Wiring** (`bot.py`): aiogram handlers persist every message and push
  it as an event; background loops deliver due wakeups and periodic
  heartbeat status digests.
- **Storage** (`db.py`): SQLite (WAL) with full raw Telegram payloads.
- **Mind** (`memory.py`): three markdown files under `data/memory/`,
  editable by hand at any time. `SOUL.md` is the personality, re-read
  and attached to the instructions every turn. `MEMORY.md` is the
  long-term fact store: the `remember` tool appends to it, the `recall`
  tool answers questions from it with a one-shot extraction call (it is
  never inlined into the agent's context). `DIARY.md` is reserved for
  the upcoming Dreaming loop.
- **Prompts** (`prompts.toml`): user-editable agent, roleplay, web-search
  and memory-helper instructions loaded at startup.

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
