#!/usr/bin/env bash
# ds4-rccl-bench.sh — RCCL all-reduce latency bench across BOTH DS4 boxes over
# a chosen transport env (default: ib). Run on box1 (the ray head); box2 is
# driven over ssh. Adapted from neuhaus/ds4-vllm feature/infiniband-mlx4 to
# this site's $HOME-deployed script layout.
#
# Measures the per-op all-reduce latency RCCL delivers for the shapes the vLLM
# communicator sees (decode ~48 KiB, prefill ~4 MiB). Reference points: the
# odl_ar2 decode all-reduce runs ~90-110 us/op in-engine; the decode step
# chains ~160 all-reduces per token, so per-op latency is the TP bottleneck.
#
# Usage: ds4-rccl-bench.sh [transport]   (transport = ib | rdma | tcp; the
# matching ~/ds4-cluster-env.<transport>.sh must exist on BOTH boxes)
set -uo pipefail

TRANSPORT=${1:-ib}
eval "$("$HOME/ds4-config" "${DS4_CONFIG_PATH:-$HOME/ds4-config.yaml}")"
HEAD_IP=${DS4_HEAD_IP:?ds4-config.yaml: head_ip missing}
WORKER_IP=${DS4_WORKER_IP:?ds4-config.yaml: worker_ip missing}
CTR=${DS4_CONTAINER:-vllm}
PORT=${DS4_BENCH_PORT:-29600}
ITERS=${DS4_BENCH_ITERS:-2000}
CENV=$HOME/ds4-cluster-env.$TRANSPORT.sh
BENCH=$HOME/ds4-rccl-bench.py

box2() { timeout "${2:-120}" ssh -o BatchMode=yes "$WORKER_IP" "$1"; }
inbox() { timeout "${2:-120}" podman exec -u 1000:1000 -w "$HOME" "$CTR" bash -lc "$1"; }

# rdma_hca must reach the env file on both ranks (same mechanism as bringup).
ENVPASS="export DS4_RDMA_HCA=${DS4_RDMA_HCA:-};"

[ -f "$CENV" ] || { echo "!! $CENV missing (transport=$TRANSPORT)"; exit 1; }
[ -f "$BENCH" ] || { echo "!! $BENCH missing"; exit 1; }
inbox true 20 >/dev/null 2>&1 || { echo "!! box1 $CTR not exec-able"; exit 1; }
box2 "podman exec $CTR true" 20 >/dev/null 2>&1 || { echo "!! box2 $CTR not exec-able"; exit 1; }

echo "== RCCL bench: transport=$TRANSPORT $HEAD_IP:$PORT, $ITERS iters, hca=${DS4_RDMA_HCA:-<env default>} =="

# rank1 on box2 in the background, rank0 on box1 in the foreground; both
# source the transport env so RCCL's net selection matches serving bringup.
box2 "podman exec -u 1000:1000 -w \$HOME $CTR bash -lc '$ENVPASS source $CENV; exec python3 \$HOME/ds4-rccl-bench.py --rank 1 --master $HEAD_IP --port $PORT --iters $ITERS'" 600 2>/dev/null \
  >/tmp/ds4-rccl-bench-rank1.log &
RANK1=$!

inbox "$ENVPASS source $CENV; exec python3 $BENCH --rank 0 --master $HEAD_IP --port $PORT --iters $ITERS" 600
RC0=$?
wait "$RANK1" 2>/dev/null
echo "--- rank1 (box2) ---"
cat /tmp/ds4-rccl-bench-rank1.log
rm -f /tmp/ds4-rccl-bench-rank1.log

[ "$RC0" -eq 0 ] || { echo "!! rank0 (box1) bench failed"; exit 1; }
echo "== done — med us/op on the ~48 KiB row is the decode-relevant number =="
