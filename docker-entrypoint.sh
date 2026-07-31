#!/bin/sh
# Runs as root: fix ownership of the bind-mounted state dirs (Docker creates
# missing host dirs as root), then drop privileges to the app user.
set -e

for dir in /data /memory; do
    if [ -d "$dir" ] && [ "$(stat -c %u "$dir")" != "10001" ]; then
        chown -R app:app "$dir"
    fi
done

exec setpriv --reuid app --regid app --init-groups "$@"
