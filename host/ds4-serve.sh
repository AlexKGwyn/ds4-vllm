#!/usr/bin/env bash
# ds4-serve.sh — bring the 2-box DS4 cluster up, down, or report on it.
#
# This is the entry point. Run it from box1 (the Ray head); box2 is driven over
# ssh. There is deliberately no systemd here: the stack is a script you run, and
# whether it should also be a unit is your call, not this repo's.
# examples/systemd/ds4-vllm.service is one way to wrap it if you want that.
#
#   ds4-serve.sh start     # full bring-up: teardown -> ray both boxes -> serve -> verify
#   ds4-serve.sh stop      # full teardown, both boxes
#   ds4-serve.sh restart   # stop then start
#   ds4-serve.sh status    # API health, serve/ray process counts, pid file
#   ds4-serve.sh logs [-f] # tail the serve log
#
# Supervision is a pid file plus a log file, both under the usual XDG dirs and
# overridable with DS4_RUN_DIR / DS4_LOG_DIR:
#
#   ${XDG_RUNTIME_DIR:-/tmp}/ds4-vllm/serve.pid
#   ${XDG_STATE_HOME:-$HOME/.local/state}/ds4-vllm/serve.log
#
# NOTE what the pid file actually holds: the `podman exec` WRAPPER, not the
# `vllm serve` inside the container. Killing the wrapper on its own leaves the
# server running and holding the API port, which is why stop/start go through
# ds4-cluster-{down,restart}.sh — they reap surviving `vllm serve` processes and
# refuse to bring the stack back up until none remain.
set -uo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
RESTART=${DS4_RESTART_SH:-$HOME/ds4-cluster-restart.sh}
DOWN=${DS4_DOWN_SH:-$HOME/ds4-cluster-down.sh}
# Fall back to the copies next to this script, so it also works from a checkout.
[ -x "$RESTART" ] || RESTART=$HERE/ds4-cluster-restart.sh
[ -x "$DOWN" ] || DOWN=$HERE/ds4-cluster-down.sh

RUN_DIR=${DS4_RUN_DIR:-${XDG_RUNTIME_DIR:-/tmp}/ds4-vllm}
LOG_DIR=${DS4_LOG_DIR:-${XDG_STATE_HOME:-$HOME/.local/state}/ds4-vllm}
PIDFILE=$RUN_DIR/serve.pid
SERVE_LOG=$LOG_DIR/serve.log

# Port comes from the site config when it is readable; the default matches
# ds4-config.yaml's api_port.
eval "$("$HOME/ds4-config" "$HOME/ds4-config.yaml" 2>/dev/null)" 2>/dev/null || true
PORT=${DS4_API_PORT:-1234}

usage() { sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }

case "${1:-}" in
  start)
    exec "$RESTART"
    ;;
  stop)
    exec "$DOWN"
    ;;
  restart)
    "$DOWN"
    exec "$RESTART"
    ;;
  status)
    code=$(curl -s -o /dev/null -m 5 -w "%{http_code}" \
             "http://127.0.0.1:$PORT/v1/models" 2>/dev/null)
    echo "api            : ${code:-unreachable} (http://127.0.0.1:$PORT)"
    if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
        echo "supervisor pid : $(cat "$PIDFILE") (running)"
    else
        echo "supervisor pid : not running"
    fi
    # The bracket keeps these greps from matching their own command line.
    echo "vllm serve     : $(ps -eo cmd --no-headers | grep -c 'bin/[v]llm serve') (want 1 when up)"
    echo "ray idle procs : $(ps -eo cmd --no-headers | grep -c '[r]ay::IDLE')"
    echo "serve log      : $SERVE_LOG"
    [ "$code" = "200" ] || exit 1
    ;;
  logs)
    [ -f "$SERVE_LOG" ] || { echo "no log yet at $SERVE_LOG" >&2; exit 1; }
    if [ "${2:-}" = "-f" ]; then exec tail -f "$SERVE_LOG"; else exec tail -n 200 "$SERVE_LOG"; fi
    ;;
  *)
    usage
    ;;
esac
