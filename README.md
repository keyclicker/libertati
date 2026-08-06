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
- **Tools** (`tools.py`): the only way the agent acts. Messaging
  (`send_message`, `send_sticker`, `forward_message`, `react`,
  `edit_message`, `delete_message`, `list_stickers`), alarms for its
  future self (`schedule_wakeup`, `list_wakeups`, `cancel_wakeup`),
  reading its own history (`list_chats`, `get_chat_info`,
  `list_chat_speakers`, `list_topics`, `get_recent_messages`,
  `get_unread_messages_count`, `get_message_thread`,
  `search_messages`), memory (`remember`, `recall`,
  `summarize_memory`) and `dream`. Every chat-scoped tool is checked
  against the approval registry before it runs.
- **Dreaming** (`dream.py`): after a stretch of idleness — or when the
  agent calls `dream` itself — the waking loop pauses and a
  differently-prompted loop runs one long offline session. It wanders
  with read-only and web tools, writes a reflection into `DREAMS.md`,
  folds `INBOX.md` into a rewritten `MEMORY.md`, and may revise
  `SOUL.md`. Its context is recorded in `dream_context` for the viewer
  but never read back — the agent wakes to a `[dream ended …]` event
  carrying its summary, and nothing else carries over. Budgeted per 24h.
- **Wiring** (`bot.py`): aiogram handlers persist every message and push
  it as an event; background loops deliver due wakeups, periodic
  heartbeat status digests, and hand the agent over to a dream.
- **Storage** (`db.py`): SQLite (WAL) with full raw Telegram payloads,
  append-only model context (waking and dreaming kept apart), and exact
  API token/cache usage per turn or dream.
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

Chat approval defaults on. First message from a chat creates a `false`
entry in `data/chats.toml`; review it, flip that entry to `true`, then
send another message. Disable `chat_approval` only when every Telegram
user who can reach the bot belongs to one trusted group.

`settings.toml` carries every non-secret setting explicitly, grouped by
area — nothing is hidden in code defaults. Any OpenAI-compatible endpoint
works (OpenAI, OpenRouter, local); set `base_url`/`model` there. All
settings can be overridden with `LIBERTATI_*` environment variables.

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
uv run libertati-spy      # live full-screen view of the agent's context
```

`libertati-spy` tails the context table as it is written. Every item is
laid out by kind: chat events split into chat, speaker and text, tool
calls list their arguments (long ones as their own block), a
`get_recent_messages` result reads as one row per message rather than as
a JSON dump, and a web search lists the queries it ran. Anything that
does not fit its shape falls back to the plain body. Navigation is
vim-like: `j`/`k` and `ctrl+e`/`ctrl+y` by line, `ctrl+d`/`ctrl+u` by
half a screen, `ctrl+f`/`ctrl+b` by screen, `g`/`G` for the ends,
`/`, `?`, `n`, `N` to search (smartcase, highlighted), `f` to un-truncate
long bodies, `q` to quit. Older history pages in as you scroll up.

Search covers the whole stored context, not the part currently on
screen: the pattern is indexed occurrence by occurrence out of the
database, `n`/`N` step through those occurrences (paging history in when
the next one is older than what is loaded), the hit under the cursor is
underlined apart from the other highlights, and the status line counts
your position (`match 4/109`). A body long enough to be truncated is
shown in full when the pattern matched inside it, and a wrap around the
ends says so the way vim does.

`d` switches to a dream's context and back. While you are at the bottom
and have not pressed `d`, a starting dream is followed on its own and let
go again on waking, so leaving the viewer open shows the dream as it
happens. `uv run libertati-spy --dream 3` reopens a past dream.

The status shows authoritative input/cache counts from the latest API
call next to the projected size of the next one — that projection is the
last measured input plus an estimate of everything appended since, so
the instructions/tool overhead comes from real numbers rather than a
guess. Per-item counts are estimated from byte length (scaled for JSON
framing, multi-byte text and base64 reasoning blobs), not tokenized.
