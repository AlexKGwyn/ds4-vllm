#!/usr/bin/env bash
# OdinLink-transport env for the DS4 vLLM cluster (`transport: odl` in
# ds4-config.yaml). Runs the 2-box cluster with RCCL's net plugin backed by
# the odl_tb5 kernel driver (/dev/odl_tb5_*) instead of the usb4_rdma ibverbs
# HCA. IB is disabled; the decode all-reduce goes
# through odl_ar2 and every other collective through the plugin. Bring the
# link up first: odinlink/README.md.
source "$HOME/ds4-cluster-env.sh"
export NCCL_IB_DISABLE=1
unset  NCCL_IB_HCA NCCL_IB_GID_INDEX
# RCCL dlopens the plugin at this exact path (built into the image by the
# odinlink-build stage); libodl_tb5.so.0 resolves from the plugin's rpath.
export NCCL_NET_PLUGIN=/usr/local/lib/odinlink/librccl_net_odl_tb5.so
export LD_LIBRARY_PATH="/usr/local/lib/odinlink${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# Bootstrap/control sockets: tbnet stays loaded under OdinLink, so the default
# control plane is still thunderbolt0 (192.168.100.x). Override per box via
# control_iface_head/control_iface_worker in ds4-config.yaml if the control
# plane rides another interface.
export NCCL_SOCKET_IFNAME=${DS4_CONTROL_IFACE:-thunderbolt0}
export GLOO_SOCKET_IFNAME=${DS4_CONTROL_IFACE:-thunderbolt0}
export DS4_TBV_AR=0
export DS4_TBV_AR2=0
# odl_ar2: decode all-reduce over the OdinLink stream API
# (rootfs odl_ar2.py + in-image libodl_ar2.so). The rendezvous rides TCP:
# rank1 listens on odl_rank1_ip (defaults to worker_ip via the restart
# script), rank0 connects out.
export DS4_ODL_AR2=1
export ODL2_RANK1_IP=${DS4_ODL_RANK1_IP:-192.168.100.2}
export ODL2_PORT=${DS4_ODL_PORT:-18541}
export ODL2_DEV=${DS4_ODL_DEV:-0}
# RCCL fragments each collective across its channel count; 26 channels over
# one OdinLink device means 26x the per-message syscall overhead on prefill
# allreduces. Decode is unaffected (odl_ar2 path).
export NCCL_MAX_NCHANNELS=4
# Debug logging is off. Re-enable NCCL_DEBUG=INFO to trace plugin selection
# or init failures.

# This file adds only what is specific to the OdinLink transport. Everything
# else -- DS4_W8A8_BF16_DIRECT, DS4_MTP_MAXSEQS and the rest -- lives in the
# base env above in `${VAR:-default}` form; do not restate those knobs here
# (see the note in ds4-cluster-env.rdma.sh).
