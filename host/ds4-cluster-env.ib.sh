#!/usr/bin/env bash
# Native-InfiniBand transport env for the DS4 vLLM cluster (`transport: ib` in
# ds4-config.yaml). Runs every TP=2 collective through RCCL's built-in IB net
# over the ConnectX-3 FDR fabric (mlx4_0, back-to-back cable; the `opensm`
# podman container on box1 keeps both ports ACTIVE -- `rdma link` must show
# state ACTIVE before bringup). Control-plane sockets (ray, NCCL/GLOO
# rendezvous) stay on thunderbolt0 with head_ip/worker_ip 192.168.100.x, so
# the Thunderbolt cable is still required; only the verbs data path is IB.
# The IB cards sit in x4-limited slots: ~27.8 Gb/s wire ceiling (ib_write_bw),
# ~1 us raw write latency.
source "$HOME/ds4-cluster-env.sh"
# Pin the exact HCA:port (rdma_hca in the site config). The base env's
# usb4_rdma prefix would miss the mlx4 device entirely.
export NCCL_IB_HCA=${DS4_RDMA_HCA:-mlx4_0:1}
# Index 0 is the native-IB link-local GID. The base env's index 1 is the
# RoCEv2-IPv4 GID, which does not exist on an IB link -- RCCL's
# ncclCommInitRank fails at init with it.
export NCCL_IB_GID_INDEX=0
export NCCL_IB_DISABLE=0
# The base env pins NCCL_PROTO=LL, tuned for when RCCL carried the tiny decode
# all-reduces. On this transport ib_ar2 owns everything <=1MiB and RCCL only
# sees prefill-sized ops, where LL's flag-byte inflation halves the x4 link:
# measured 4MiB AR med 2810us (LL) vs 1572us (auto: LL small/Simple large).
# Unset = RCCL picks per size.
unset NCCL_PROTO
# USB4/OdinLink custom all-reduces off; ib_ar2 (libib_ar2, tbv_ar2 ported to
# mlx4 native IB) carries the decode collective: 48 KiB med ~48us standalone
# vs RCCL-over-IB's ~59. Prefill (>1MiB) still goes to RCCL. Set 0 to fall
# back to RCCL for every collective.
export DS4_TBV_AR=0
export DS4_TBV_AR2=0
export DS4_ODL_AR2=0
export DS4_IB_AR2=${DS4_IB_AR2:-1}
export DS4_IB_AR2_HCA=${DS4_IB_AR2_HCA:-mlx4}

# Live profiling: enables /start_profile & /stop_profile API endpoints; traces
# land in ~/vllm-profiles (shared into the container). Inert unless invoked.
export VLLM_TORCH_PROFILER_DIR=/home/alex/vllm-profiles
export VLLM_SERVER_DEV_MODE=1
# Single-kernel tiny-batch MoE routing (5 launches -> 1; ~8ms/step CPU).
export DS4_TINY_ROUTING=1

# --- decode-tail kernel tuning (same block as the odl env, so transport A/B
# comparisons measure the fabric, not the kernel dispatch path) ---------------
export DS4_FAST_TRITON=1
export DS4_MTP_FAST_REPLAY=1
export PYTORCH_TUNABLEOP_ENABLED=1
export PYTORCH_TUNABLEOP_TUNING=0
export PYTORCH_TUNABLEOP_FILENAME=$HOME/ds4-tunableop.csv
