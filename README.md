# libertati

An **agentic Telegram bot** that role-plays a persona ("Ana Tati"), remembers the people and
groups it talks to, reads the news, and lives a little life of its own — messaging people on a
randomized **heartbeat** and **dreaming** once a day to reflect and update its memory.

It's a full rewrite of the original 2023 GPT-3.5 bot: modern async stack, `uv`-managed,
Docker + GitHub Actions, and a test suite.

## Features

- **Telegram Bot API** via [aiogram](https://aiogram.dev) — no MTProto credentials or
  session files needed.
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
- **Random channel browsing** — a few times a day it reads configured public channels via the
  `t.me/s/` web preview (no extra credentials), notes interesting things to `reading.md`, and may
  bring them up in conversation. The model can also read a channel on demand via the
  `read_telegram` tool.
- **Heartbeat** twice a day at randomized times — the bot may proactively message someone.
- **Dreaming** once a day — it reviews history, reflects, and updates its memory.
- **Human-like replying** — in groups it always answers when addressed (mentioned or replied-to)
  but stays out of most ambient chatter, and pauses briefly (scaled by reply length) before
  sending, so it reads like a person rather than an always-on bot.
- **Bounded memory & context** — memory files are capped on disk (oldest lines trimmed) and
  clipped per-file in the prompt; each reply mixes the focused reply thread with a small window
  of recent messages, so token cost stays predictable as history and notes grow.
- **Responds to all messages** by default, or only in an **allowlist of groups** you specify.
- **Structured event log** (JSONL) tracing every turn — LLM calls (tokens/latency), tool calls,
  memory writes and messages, correlated by turn id — so its real behavior can be analyzed and tuned.

## Architecture

```
src/libertati/
  config.py            pydantic-settings configuration
  main.py              entrypoint: wires everything, runs the client + scheduler
  telegram/            client interface (base) + aiogram adapter + t.me/s channel reader
  storage/             SQLite db + history store (FTS5) + markdown memory store
  llm/                 OpenAI client, tool schemas/dispatch, prompts, the Agent
  news/                RSS reader
  scheduler/           APScheduler jobs (heartbeat, dream, news, browse) + runner
  observability.py     structured JSONL event log (per-turn behavioral telemetry)
```

The app depends only on the `TelegramClient` abstraction, so test doubles are cheap. The
`Agent` runs a bounded tool-call loop over the tools in `llm/tools.py`.

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)

## Setup

```bash
uv sync                 # create the venv and install deps
cp .env.example .env    # then fill in your tokens
```

Configuration is split in two:

- **`config.toml`** — all non-secret settings (behaviour, browsing, persona, logging), checked
  into the repo with sane defaults. Keys are the field names from `config.py`.
- **`.env`** — secrets only (bot token, OpenAI key). Any `LIBERTATI_`-prefixed environment
  variable also works and overrides `config.toml`.

1. Create a bot with [@BotFather](https://t.me/BotFather), get the token.
2. Set in `.env`:
   ```
   LIBERTATI_BOT_TOKEN=123456:your-token
   LIBERTATI_OPENAI_API_KEY=sk-...
   ```
3. Run: `uv run libertati`

> Note: to receive all group messages, disable privacy mode for the bot in BotFather
> (`/setprivacy` → Disable).

## Restricting which groups it talks in

By default the bot replies everywhere it can see messages. To limit *proactive replies* to
specific chats, set an allowlist (chat ids and/or public `@usernames`):

```
LIBERTATI_ALLOWED_CHATS=@mygroup,-1001234567890
```

It still quietly logs messages it sees elsewhere (needed for memory/history) — it just won't talk
there. An empty allowlist means "everywhere", the default.

## Random channel browsing

The bot can periodically read public Telegram channels via the `t.me/s/<username>` web preview
(no extra credentials needed) and discuss what it finds:

```
LIBERTATI_BROWSE_ENABLED=true
LIBERTATI_BROWSE_CHANNELS=@somechannel,@another   # required: which channels to read
LIBERTATI_BROWSE_TIMES_PER_DAY=3
```

It reads a channel, appends anything interesting to `memory/reading.md` (which is part of its
prompt, so it can reference it in normal chats), and — if there's something worth saying — posts
into one of its allowed/active chats. Only **public channels with the web preview enabled** are
readable; private chats/groups are not.

## Triggering routines manually

Run any background routine once without waiting for the scheduler:

```bash
uv run python -m libertati.trigger heartbeat   # think, maybe message someone
uv run python -m libertati.trigger dream       # reflect + update memory + write diary
uv run python -m libertati.trigger browse      # read a channel + maybe discuss it
uv run python -m libertati.trigger news        # refresh world.md from feeds
```

(`libertati-trigger <routine>` is installed as a console script too.)

## Observability — the event log

To understand (and fix) how the bot actually behaves, every turn is traced to a JSONL event log
(`data/events.jsonl` by default). Each line is one event, and everything within a decision shares a
`turn` id, so a single `grep` reconstructs the whole thing:

```json
{"ts":…,"turn":"t1","type":"turn_start","kind":"respond","chat_id":555,"thread_len":2,"memory_bytes":180}
{"ts":…,"turn":"t1","type":"llm_call","model":"gpt-4o-mini","total_tokens":812,"latency_ms":430,"n_tool_calls":1}
{"ts":…,"turn":"t1","type":"tool_call","name":"read_memory","ok":true,"latency_ms":1,"result_len":60}
{"ts":…,"turn":"t1","type":"reply","preview":"та нічо, живу","reply_len":13}
{"ts":…,"turn":"t1","type":"turn_end","kind":"respond","llm_calls":1,"tool_calls":1,"turn_tokens":812}
```

Event types include `inbound` / `skip` / `outbound` (message flow), `llm_call` / `llm_error`
(model, token usage, latency), `tool_call` (name, args, timing, result size), `memory_write`
(path, mode, size delta — the signal for memory growth/poisoning), `tool_loop_cap`, and the
`heartbeat_result` / `dream_result` / `browse_result` outcomes.

Handy queries:

```bash
# everything that happened in one decision
grep '"turn":"t42"' data/events.jsonl | jq .
# token spend per turn
jq 'select(.type=="turn_end") | {turn, kind, turn_tokens}' data/events.jsonl
# how memory is changing over time
jq 'select(.type=="memory_write") | {path, mode, delta}' data/events.jsonl
# tool usage frequency
jq -r 'select(.type=="tool_call") | .name' data/events.jsonl | sort | uniq -c
```

Set `LIBERTATI_LOG_MESSAGE_CONTENT=false` to record only lengths/shapes (no message text) for
privacy, or `LIBERTATI_EVENT_LOG_ENABLED=false` to turn it off. Run with `LIBERTATI_LOG_LEVEL=DEBUG`
to also see the per-iteration tool decisions on stdout.

## Running the checks

```bash
uv run ruff check .     # lint
uv run ty check src     # type check
uv run pytest           # tests (no network / no real API calls)
```

## Docker

```bash
docker compose up --build
```

`./data` (SQLite) and `./memory` (markdown) are mounted as volumes so state survives restarts.

## CI/CD

- **`.github/workflows/ci.yml`** — runs ruff, ty and pytest on every push/PR.
- **`.github/workflows/docker.yml`** — on push to `main` and on `v*` tags, builds the image and
  pushes it to `ghcr.io/<owner>/libertati` using the built-in `GITHUB_TOKEN`.

See [`DEPLOYMENT.md`](DEPLOYMENT.md) for how to configure and run it on a server.

## Configuration reference

Each variable can be set in `config.toml` (lowercase key without the `LIBERTATI_` prefix, e.g.
`group_reply_chance`) or as an environment variable / in `.env` (full `LIBERTATI_` name; wins
over the TOML). Keep tokens in `.env` only.

| Variable | Default | Description |
| --- | --- | --- |
| `LIBERTATI_BOT_TOKEN` | – | Bot API token (required) |
| `LIBERTATI_OPENAI_API_KEY` | – | OpenAI key (required) |
| `LIBERTATI_OPENAI_BASE_URL` | – | Override for OpenAI-compatible endpoints |
| `LIBERTATI_OPENAI_MODEL` | `gpt-4o-mini` | Chat model |
| `LIBERTATI_DB_PATH` | `data/libertati.db` | SQLite path |
| `LIBERTATI_MEMORY_DIR` | `memory` | Markdown memory root |
| `LIBERTATI_MAX_THREAD_CHARS` | `5000` | Reply-thread budget (chars) |
| `LIBERTATI_RECENT_CONTEXT_MESSAGES` | `20` | Recent messages scanned for extra context |
| `LIBERTATI_RECENT_CONTEXT_CHARS` | `1200` | Budget for recent messages added beyond the thread |
| `LIBERTATI_MEMORY_MAX_FILE_CHARS` | `4000` | Disk cap per memory file (oldest lines trimmed) |
| `LIBERTATI_MEMORY_CONTEXT_FILE_CHARS` | `700` | Per-file clip when injecting memory into the prompt |
| `LIBERTATI_RESPOND_TO_ALL` | `true` | Master switch for replying at all |
| `LIBERTATI_GROUP_REPLY_CHANCE` | `0.05` | Chance of replying to *ambient* group messages (always replies when addressed) |
| `LIBERTATI_TYPING_DELAY_ENABLED` | `true` | Human-like pause before sending |
| `LIBERTATI_TYPING_DELAY_MAX_SECONDS` | `5.0` | Cap on that pause |
| `LIBERTATI_ALLOWED_CHATS` | (empty = all) | Chats it may proactively talk in (ids/`@usernames`) |
| `LIBERTATI_HEARTBEAT_ENABLED` | `true` | Enable the twice-daily heartbeat |
| `LIBERTATI_DREAM_ENABLED` | `true` | Enable daily dreaming |
| `LIBERTATI_NEWS_REFRESH_HOURS` | `6` | News refresh interval |
| `LIBERTATI_NEWS_FEEDS` | (built-in list) | Comma-separated RSS/Atom feeds |
| `LIBERTATI_BROWSE_ENABLED` | `false` | Randomly read public channels (t.me/s web preview) |
| `LIBERTATI_BROWSE_CHANNELS` | (empty) | Public `@channels` to read (required for browsing) |
| `LIBERTATI_BROWSE_TIMES_PER_DAY` | `3` | How many random browses per day |
| `LIBERTATI_EVENT_LOG_ENABLED` | `true` | Write the structured JSONL event log |
| `LIBERTATI_EVENT_LOG_PATH` | `data/events.jsonl` | Where the event log goes |
| `LIBERTATI_LOG_MESSAGE_CONTENT` | `true` | Include (truncated) text in the event log |

## License

MIT
