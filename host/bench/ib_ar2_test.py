#!/usr/bin/env python3
"""ib_ar2_test.py — standalone 2-box correctness + latency probe for ib_ar2.

Run inside the vllm container on both boxes (rank1 first; rank0 drives):
    box2: python3 ib_ar2_test.py --rank 1
    box1: python3 ib_ar2_test.py --rank 0

Correctness: bit-exact against a local fp32-accumulate reference (the kernel
adds with float accumulation and casts back, same as the reference here).
Latency: per-op wall time with a stream sync per op, decode shape first.
"""
import argparse
import statistics
import time

import torch

from ib_ar2 import IbAllReduce2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--sizes", type=int, nargs="+", default=[49152, 8192, 1048576])
    args = ap.parse_args()

    torch.cuda.init()
    ar = IbAllReduce2(args.rank)

    # --- correctness: deterministic distinct payload per rank ---------------
    n = 49152 // 2
    base = torch.arange(n, dtype=torch.float32, device="cuda")
    mine = ((base * (1.0 + args.rank) + args.rank) / 977.0).bfloat16()
    other = ((base * (1.0 + (1 - args.rank)) + (1 - args.rank)) / 977.0).bfloat16()
    expected = (mine.float() + other.float()).bfloat16()
    out = torch.empty_like(mine)
    assert ar.eligible(mine), "test tensor not eligible?"
    ar.all_reduce_out(mine, out)
    torch.cuda.synchronize()
    bad = (out != expected).sum().item()
    print(f"[rank{args.rank}] correctness: {'EXACT' if bad == 0 else f'{bad}/{n} MISMATCHED'}",
          flush=True)
    if bad:
        raise SystemExit(1)

    # --- latency ------------------------------------------------------------
    for nbytes in args.sizes:
        t = torch.ones(nbytes // 2, dtype=torch.bfloat16, device="cuda")
        o = torch.empty_like(t)
        for _ in range(args.warmup):
            ar.all_reduce_out(t, o)
        torch.cuda.synchronize()
        times_us = []
        for _ in range(args.iters):
            torch.cuda.synchronize()
            st = time.perf_counter_ns()
            ar.all_reduce_out(t, o)
            torch.cuda.synchronize()
            times_us.append((time.perf_counter_ns() - st) / 1000.0)
        med = statistics.median(times_us)
        p99 = statistics.quantiles(times_us, n=100)[98]
        print(f"[rank{args.rank}] {nbytes:>8} B  med={med:7.2f} us  min={min(times_us):7.2f} us  "
              f"p99={p99:7.2f} us  (rccl-over-IB 48KiB med ~59, odl_ar2 in-engine ~90-110)",
              flush=True)


if __name__ == "__main__":
    main()
