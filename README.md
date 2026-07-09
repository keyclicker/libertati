# libertati

An **agentic Telegram bot** that role-plays a persona ("Ana Tati"), remembers the people and
groups it talks to, reads the news, and lives a little life of its own — messaging people on a
randomized **heartbeat** and **dreaming** once a day to reflect and update its memory.

It's a full rewrite of the original 2023 GPT-3.5 bot: modern async stack, `uv`-managed,
Docker + GitHub Actions, dual Telegram backends, and a test suite.

## Features

- **Two Telegram backends** behind one interface:
  - `bot` mode — the HTTP Bot API via [aiogram](https://aiogram.dev).
  - `account` mode — the standard MTProto protocol as a real user account via
    [Telethon](https://docs.telethon.dev).
- **OpenAI-driven agent** (SDK v1+, function calling). Model and `base_url` are configurable, so
  any OpenAI-compatible endpoint works.
- **Persistent, indexed history** in SQLite with an FTS5 full-text index and reply-thread
  reconstruction.
- **Self-maintained markdown memory** the bot reads and writes via tools:
  - `user/<handle>.md` — per-user facts, tone, running summary.
  - `group/<slug>.md` — per-group dynamics, in-jokes, topics.
  - `self.md` — the bot's evolving self-concept / persona journal.
  - `world.md` — a running digest of news it has read.
  - `social.md` — relationship graph and open threads.
  - `todo.md` — follow-ups to raise with people on the next heartbeat.
  - `diary/<date>.md` — nightly dream reflections.
- **News reading** from configurable RSS/Atom feeds (a tool + a scheduled refresh).
- **Heartbeat** twice a day at randomized times — the bot may proactively message someone.
- **Dreaming** once a day — it reviews history, reflects, and updates its memory.
- **Responds to all messages** by default (configurable).

## Architecture

```
src/libertati/
  config.py            pydantic-settings configuration
  main.py              entrypoint: wires everything, runs the client + scheduler
  telegram/            backend-agnostic client (base) + aiogram & telethon adapters + factory
  storage/             SQLite db + history store (FTS5) + markdown memory store
  llm/                 OpenAI client, tool schemas/dispatch, prompts, the Agent
  news/                RSS reader
  scheduler/           APScheduler jobs (heartbeat, dream, news) + runner
```

The app depends only on the `TelegramClient` abstraction, so the two backends are
interchangeable. The `Agent` runs a bounded tool-call loop over the tools in `llm/tools.py`.

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)

## Setup

```bash
uv sync                 # create the venv and install deps
cp .env.example .env    # then fill in your keys
```

Configuration is entirely via environment variables (prefix `LIBERTATI_`), loaded from `.env`.
See `.env.example` for every option.

### Bot API mode

1. Create a bot with [@BotFather](https://t.me/BotFather), get the token.
2. Set in `.env`:
   ```
   LIBERTATI_TELEGRAM_MODE=bot
   LIBERTATI_BOT_TOKEN=123456:your-token
   LIBERTATI_OPENAI_API_KEY=sk-...
   ```
3. Run: `uv run libertati`

> Note: to receive all group messages in bot mode, disable privacy mode for the bot in BotFather
> (`/setprivacy` → Disable).

### Account (MTProto) mode

1. Get `api_id` / `api_hash` from <https://my.telegram.org/apps>.
2. Set in `.env`:
   ```
   LIBERTATI_TELEGRAM_MODE=account
   LIBERTATI_TG_API_ID=12345
   LIBERTATI_TG_API_HASH=your-hash
   LIBERTATI_TG_SESSION=libertati
   LIBERTATI_OPENAI_API_KEY=sk-...
   ```
3. Authorize once to create the session file (interactive login):
   ```bash
   uv run python -m libertati.login
   ```
4. Run: `uv run libertati`

## Running the checks

```bash
uv run ruff check .     # lint
uv run mypy src         # type check
uv run pytest           # tests (no network / no real API calls)
```

## Docker

```bash
docker compose up --build
```

`./data` (SQLite) and `./memory` (markdown) are mounted as volumes so state survives restarts.
In account mode the `libertati.session` file is mounted too.

## CI/CD

- **`.github/workflows/ci.yml`** — runs ruff, mypy and pytest on every push/PR.
- **`.github/workflows/docker.yml`** — on push to `main` and on `v*` tags, builds the image and
  pushes it to `ghcr.io/<owner>/libertati` using the built-in `GITHUB_TOKEN`.

## Configuration reference

| Variable | Default | Description |
| --- | --- | --- |
| `LIBERTATI_TELEGRAM_MODE` | `bot` | `bot` or `account` |
| `LIBERTATI_BOT_TOKEN` | – | Bot API token (bot mode) |
| `LIBERTATI_TG_API_ID` / `_TG_API_HASH` | – | MTProto credentials (account mode) |
| `LIBERTATI_TG_SESSION` | `libertati` | Telethon session name |
| `LIBERTATI_OPENAI_API_KEY` | – | OpenAI key (required) |
| `LIBERTATI_OPENAI_BASE_URL` | – | Override for OpenAI-compatible endpoints |
| `LIBERTATI_OPENAI_MODEL` | `gpt-4o-mini` | Chat model |
| `LIBERTATI_DB_PATH` | `data/libertati.db` | SQLite path |
| `LIBERTATI_MEMORY_DIR` | `memory` | Markdown memory root |
| `LIBERTATI_RESPOND_TO_ALL` | `true` | Reply to every message |
| `LIBERTATI_HEARTBEAT_ENABLED` | `true` | Enable the twice-daily heartbeat |
| `LIBERTATI_DREAM_ENABLED` | `true` | Enable daily dreaming |
| `LIBERTATI_NEWS_REFRESH_HOURS` | `6` | News refresh interval |
| `LIBERTATI_NEWS_FEEDS` | (built-in list) | Comma-separated RSS/Atom feeds |

## License

MIT
