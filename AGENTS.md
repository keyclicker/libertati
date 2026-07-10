# AGENTS.md

Agentic Telegram bot ("Ana Tati"). Python 3.12, `uv`, async.

## Commands
- `uv sync` — install
- `uv run ruff check .` — lint
- `uv run mypy src` — types
- `uv run pytest` — tests (no network)
- `uv run libertati` — run

## Layout
`src/libertati/`: `telegram/` (backends), `storage/` (SQLite+FTS, markdown memory),
`llm/` (client, tools, prompts, agent), `scheduler/` (jobs), `news/`, `observability.py`.

## Conventions
- Config via `LIBERTATI_`-prefixed env vars (`config.py`); no hardcoded secrets.
- New collaborators get an optional `events`/deps param defaulting to a no-op — never break call sites.
- Keep it green: ruff + mypy + pytest must pass before committing.
- Persona/prompts live in `llm/prompts.py`.
