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
# Pre-create the state dirs owned by the container user (uid 10001) — if Docker
# creates them on first run they end up root-owned and the bot can't write:
mkdir -p data memory && sudo chown -R 10001:10001 data memory
```

### 1. Configure secrets (`.env`) and settings (`config.toml`)

Secrets go in `.env` (loaded via `env_file`, gitignored — never commit it):

```
LIBERTATI_OPENAI_API_KEY=sk-...
LIBERTATI_BOT_TOKEN=123456:your-token
```

Everything else lives in `config.toml`, which `docker-compose.yml` mounts read-only into the
container — edit it on the server and restart, no rebuild needed. `LIBERTATI_*` environment
variables override the TOML. See `README.md` for the full configuration reference.

> `LIBERTATI_DB_PATH`, `LIBERTATI_EVENT_LOG_PATH` and `LIBERTATI_MEMORY_DIR` are overridden to
> the `/data` and `/memory` volumes by the image/compose file, which mount `./data` and
> `./memory` on the host so state survives restarts and image upgrades. Don't repoint those in
> `.env` or `config.toml`.

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
- **`PermissionError` on `/data` or `/memory` at startup** — the host dirs are root-owned
  (Docker creates missing bind-mount dirs as root). Fix:
  `sudo chown -R 10001:10001 data memory` (the container runs as uid 10001).
- **State lost after redeploy** — the `./data` / `./memory` volumes aren't mounted; check
  `docker-compose.yml` and that the directories exist and are writable.
