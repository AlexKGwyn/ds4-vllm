#!/usr/bin/env bash
# Full DS4 cluster teardown: serve process -> stranded vllm serve procs -> ray on
# both boxes. The stop half of ds4-cluster-restart.sh, for callers that need the
# stack DOWN rather than restarted. Idempotent: every step is a no-op when its
# target is already gone.
#
# The explicit process reap exists because what the pid file tracks is a
# `podman exec` wrapper, not the vllm serve inside the container: killing the
# wrapper alone strands the server, still holding the API port.
set -uo pipefail

# Teardown must work even with a broken/missing config -- fall back to defaults.
eval "$("$HOME/ds4-config" "$HOME/ds4-config.yaml" 2>/dev/null)" 2>/dev/null || true
WORKER_IP=${DS4_WORKER_IP:-192.168.100.2}
CTR=${DS4_CONTAINER:-vllm}
RUN_DIR=${DS4_RUN_DIR:-${XDG_RUNTIME_DIR:-/tmp}/ds4-vllm}
PIDFILE=$RUN_DIR/serve.pid

if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  kill "$(cat "$PIDFILE")" 2>/dev/null
fi
rm -f "$PIDFILE"
sleep 2

# The bracket keeps this grep from matching its own command line.
for p in $(ps -eo pid,cmd --no-headers | grep "bin/[v]llm serve" | awk '{print $1}'); do
  echo "[cluster-down] reaping vllm serve pid=$p"
  kill "$p" 2>/dev/null; sleep 3
  kill -0 "$p" 2>/dev/null && { kill -9 "$p" 2>/dev/null; sleep 2; }
done

timeout 60 podman exec -u 1000:1000 -w "$HOME" "$CTR" \
  bash -lc 'ray stop --force >/dev/null 2>&1' 2>/dev/null
timeout 60 ssh -o BatchMode=yes -o ConnectTimeout=6 "$WORKER_IP" \
  "podman exec -u 1000:1000 -w \$HOME $CTR bash -lc 'ray stop --force >/dev/null 2>&1'" 2>/dev/null

echo "[cluster-down] done"
exit 0
