#!/usr/bin/env python3
"""ds4-rccl-bench.py — RCCL (NCCL backend) all-reduce latency probe for the DS4
TP=2 cluster, run across BOTH boxes over the InfiniBand fabric.

Measures the per-op latency RCCL delivers for the exact shapes the vLLM
communicator's all_reduce sees, and prints it against the number to beat --
the custom USB4 tbv_ar2 all-reduce measured ~105 us/op on the decode
collective (~48 KiB). On gfx1151 (no GPUDirect) the op is host-staged, so the
number will NOT be the ~1.15 us wire latency; what matters is whether it is
well under 105 us, because the decode step chains ~160 of them per token.

Launch via ds4-rccl-bench.sh (or by hand on both boxes):

    rank0 (box1): python3 ds4-rccl-bench.py --rank 0 --master <head_ip>
    rank1 (box2): python3 ds4-rccl-bench.py --rank 1 --master <head_ip>

Run inside the vllm container with the rdma cluster-env sourced so
NCCL_IB_HCA / NCCL_IB_GID_INDEX point at the mlx4 fabric.
"""
import argparse
import os
import statistics
import time

import torch
import torch.distributed as dist


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--master", required=True)
    ap.add_argument("--port", type=int, default=int(os.environ.get("DS4_BENCH_PORT", "29600")))
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--warmup", type=int, default=100)
    # decode collective ~48 KiB (bf16), prefill ~4 MiB.
    ap.add_argument("--sizes", type=int, nargs="+", default=[49152, 4194304])
    args = ap.parse_args()

    torch.cuda.init()
    dist.init_process_group(backend="nccl", init_method=f"tcp://{args.master}:{args.port}",
                            rank=args.rank, world_size=2)
    rank = dist.get_rank()
    hca = os.environ.get("NCCL_IB_HCA", "?")
    print(f"[bench] rank{rank} backend=nccl hca={hca} peer={args.master} "
          f"fabric={'IB' if os.environ.get('NCCL_IB_DISABLE', '0') == '0' else 'TCP'}",
          flush=True)

    t = torch.empty(args.sizes[-1] // 2, dtype=torch.bfloat16, device="cuda")
    for size in args.sizes:
        nbytes = size
        t = torch.empty(nbytes // 2, dtype=torch.bfloat16, device="cuda")
        for _ in range(args.warmup):
            dist.all_reduce(t)
        torch.cuda.synchronize()

        times_us = []
        for _ in range(args.iters):
            torch.cuda.synchronize()
            st = time.perf_counter_ns()
            dist.all_reduce(t)
            torch.cuda.synchronize()
            times_us.append((time.perf_counter_ns() - st) / 1000.0)

        med = statistics.median(times_us)
        mean = statistics.mean(times_us)
        mn = min(times_us)
        gbps = 2 * nbytes / (mean / 1e6) / 1e9
        if rank == 0:
            print(f"[bench] {nbytes:>8} B  iters={len(times_us)}  "
                  f"med={med:7.2f} us  mean={mean:7.2f} us  min={mn:7.2f} us  "
                  f"~{gbps:5.2f} GB/s (decode budget: ~160 ops/token, "
                  f"tbv_ar2 baseline 105 us/op)", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()