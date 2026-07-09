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
- **Random Telegram browsing** (account mode) — a few times a day it reads other chats/channels,
  notes interesting things to `reading.md`, and may bring them up in conversation. The model can
  also read a channel on demand via the `read_telegram` tool.
- **Heartbeat** twice a day at randomized times — the bot may proactively message someone.
- **Dreaming** once a day — it reviews history, reflects, and updates its memory.
- **Responds to all messages** by default, or only in an **allowlist of groups** you specify.

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

## Restricting which groups it talks in

By default the bot replies everywhere it can see messages. To limit *proactive replies* to
specific chats, set an allowlist (chat ids and/or public `@usernames`):

```
LIBERTATI_ALLOWED_CHATS=@mygroup,-1001234567890
```

It still quietly logs messages it sees elsewhere (needed for memory/history) — it just won't talk
there. An empty allowlist means "everywhere", the default.

## Random browsing (account mode)

When running as a user account, the bot can periodically read other Telegram chats/channels and
discuss what it finds:

```
LIBERTATI_TELEGRAM_MODE=account
LIBERTATI_BROWSE_ENABLED=true
LIBERTATI_BROWSE_CHANNELS=@somechannel,@another   # optional; else samples your own chats
LIBERTATI_BROWSE_PUBLIC_ONLY=true                 # never surface private-group content elsewhere
LIBERTATI_BROWSE_TIMES_PER_DAY=3
```

It reads a source, appends anything interesting to `memory/reading.md` (which is part of its
prompt, so it can reference it in normal chats), and — if there's something worth saying — posts
into one of its allowed/active chats. Browsing is **account-mode only**; in bot mode it's ignored
with a warning, because the HTTP Bot API can't fetch channel history.

## Triggering routines manually

Run any background routine once without waiting for the scheduler:

```bash
uv run python -m libertati.trigger heartbeat   # think, maybe message someone
uv run python -m libertati.trigger dream       # reflect + update memory + write diary
uv run python -m libertati.trigger browse      # read a channel + maybe discuss it
uv run python -m libertati.trigger news        # refresh world.md from feeds
```

(`libertati-trigger <routine>` is installed as a console script too.)

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
| `LIBERTATI_ALLOWED_CHATS` | (empty = all) | Chats it may proactively talk in (ids/`@usernames`) |
| `LIBERTATI_HEARTBEAT_ENABLED` | `true` | Enable the twice-daily heartbeat |
| `LIBERTATI_DREAM_ENABLED` | `true` | Enable daily dreaming |
| `LIBERTATI_NEWS_REFRESH_HOURS` | `6` | News refresh interval |
| `LIBERTATI_NEWS_FEEDS` | (built-in list) | Comma-separated RSS/Atom feeds |
| `LIBERTATI_BROWSE_ENABLED` | `false` | Randomly read other chats/channels (account mode) |
| `LIBERTATI_BROWSE_CHANNELS` | (empty) | Channels to read; empty = sample own dialogs |
| `LIBERTATI_BROWSE_PUBLIC_ONLY` | `true` | Only read/discuss public `@channels` |
| `LIBERTATI_BROWSE_TIMES_PER_DAY` | `3` | How many random browses per day |

## License

MIT
