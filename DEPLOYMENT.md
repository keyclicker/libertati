# Deployment

How libertati is built, published, and run on a server.

## Pipeline at a glance

```
push/PR ──▶ CI (ci.yml)            ruff + ty + pytest on every branch & PR
push main ─▶ Docker (docker.yml)   build image ──▶ push to GHCR
push v* tag ┘                                        │
                                                     ▼
                                   ghcr.io/keyclicker/libertati:<tag>
                                                     │
server ── docker compose pull + up ──────────────────┘
```

- **`.github/workflows/ci.yml`** runs on every push and PR: `uv sync --frozen`, then
  `ruff check`, `ty check src`, `pytest`. Keep it green — it's the gate before anything ships.
- **`.github/workflows/docker.yml`** runs on push to `main` and on `v*` tags. It builds the
  image and pushes it to the GitHub Container Registry (GHCR) using the built-in `GITHUB_TOKEN`
  (no secrets to configure). Image tags produced:
  - `main` — moving tag for the latest `main` build
  - `<version>` — e.g. `1.2.0` from a `v1.2.0` git tag
  - `sha-<commit>` — immutable, per-commit

Cutting a release is just a tag:

```bash
git tag v1.0.0 && git push origin v1.0.0
```

## One-time server setup

Requirements: Docker + the Compose plugin.

```bash
git clone https://github.com/keyclicker/libertati.git
cd libertati
cp .env.example .env      # then fill in real values (see below)
```

### 1. Configure secrets (`.env`)

All config is environment variables with the `LIBERTATI_` prefix; `docker-compose.yml` loads
`.env` via `env_file`. At minimum set:

```
LIBERTATI_OPENAI_API_KEY=sk-...
LIBERTATI_BOT_TOKEN=123456:your-token
```

See `README.md` for the full configuration reference. `.env` is gitignored — never commit it.

> `LIBERTATI_DB_PATH` and `LIBERTATI_MEMORY_DIR` are overridden to `/data` and `/memory` by
> `docker-compose.yml`, which mounts `./data` and `./memory` as host volumes so state survives
> restarts and image upgrades. Don't repoint those in `.env`.

### 2. Authenticate to GHCR (only if the package is private)

The image is public by default only if you set the GHCR package visibility to public in the repo's
**Packages** settings. If it stays private, log the server in with a
[Personal Access Token](https://github.com/settings/tokens) that has `read:packages`:

```bash
echo "$GHCR_PAT" | docker login ghcr.io -u <github-username> --password-stdin
```

## Run it

```bash
docker compose pull      # fetch the published image from GHCR
docker compose up -d     # start detached; restart policy is "unless-stopped"
docker compose logs -f   # tail logs
```

## Updating / redeploy

New `main` build or new tag → on the server:

```bash
docker compose pull && docker compose up -d
```

To auto-update, add a watcher like [Watchtower](https://containrrr.dev/watchtower/), or wire a
deploy step into CI that SSHes to the server and runs the two commands above (put the host, user,
and SSH key in GitHub **Actions secrets**, and only trigger it after the Docker job succeeds).

## Data, backups, observability

- **State to back up:** `./data` (SQLite history + `events.jsonl`) and `./memory` (the markdown
  memory the bot maintains). A periodic `tar`/rsync of those two directories is a full backup.
- **Behavioral log:** `./data/events.jsonl` is the structured per-turn event log — see the
  Observability section in `README.md` for `jq` recipes. Set `LIBERTATI_LOG_MESSAGE_CONTENT=false`
  to keep message text out of it.
- **Log level:** `LIBERTATI_LOG_LEVEL=DEBUG` surfaces per-iteration tool/loop decisions on stdout.

## Troubleshooting

- **`denied` / `unauthorized` on `docker compose pull`** — the GHCR package is private and the
  server isn't logged in; do step 2, or make the package public.
- **Bot silent in groups** — disable BotFather privacy mode (`/setprivacy` → Disable)
  so it receives all group messages, and remember it also only replies to ambient chatter with
  probability `LIBERTATI_GROUP_REPLY_CHANCE` (always when addressed).
- **State lost after redeploy** — the `./data` / `./memory` volumes aren't mounted; check
  `docker-compose.yml` and that the directories exist and are writable.
