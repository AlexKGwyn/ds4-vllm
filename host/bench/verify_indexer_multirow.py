#!/usr/bin/env python3
"""GPU gate for the K-stationary multi-row decode indexer (DS4_IDX_MULTIROW).

Runs INSIDE the vllm container as a single process (never against the live
workers). Builds a real-layout paged fp8 indexer cache ([pages, bs*(dim+4)]
uint8 with per-token fp32 scale bytes), B=6 rows sharing one page list with
context lengths sl-5..sl (the flatten-path decode shape), and checks that
fp8_paged_mqa_logits_tl with the multi-row path is torch.equal to the per-row
path (DS4_IDX_MULTIROW=0) on the full logits buffer, incl. -inf placement.
Then times both at several compressed-context lengths.

    python3 verify_indexer_multirow.py [--ctx 4000,25600,131072]
"""
import argparse, importlib, os, sys, time
import torch

H, DIM, BS = 64, 128, 64


def make_case(sl_max, B=6, device="cuda"):
    torch.manual_seed(0)
    npg = (sl_max + BS - 1) // BS
    # paged cache: fp8 e4m3 values + fp32 scale per token, 576... here dim+4 per token
    vals = (torch.randn(npg * BS, DIM, device=device) * 0.5).to(torch.float8_e4m3fn)
    vb = vals.view(torch.uint8)
    vb[0, :4] = torch.tensor([0x7F, 0xFF, 0x01, 0x81], dtype=torch.uint8, device=device)  # NaN, -NaN, subnormals
    vb[1, :2] = torch.tensor([0x00, 0x80], dtype=torch.uint8, device=device)  # +0, -0
    scales = (torch.rand(npg * BS, device=device) * 2 + 0.1).float()
    flat = torch.empty((npg, BS * (DIM + 4)), dtype=torch.uint8, device=device)
    flat[:, :BS * DIM] = vals.view(torch.uint8).view(npg, BS * DIM)
    flat[:, BS * DIM:] = scales.view(torch.uint8).view(npg, BS * 4)
    kv_cache = flat.view(npg, BS, 1, DIM + 4)
    # rows: one request's 6 positions, contexts sl-5..sl, identical page list
    cls = torch.tensor([sl_max - (B - 1) + j for j in range(B)], dtype=torch.int32, device=device)
    bt = torch.arange(npg, dtype=torch.int32, device=device)[None, :].repeat(B, 1)
    q = torch.randn(B, 1, H, DIM, device=device).to(torch.float8_e4m3fn)
    w = torch.rand(B, H, device=device).float()
    return q, kv_cache, w, cls, bt


def run(mode, q, kv, w, cls, bt, max_model_len):
    """mode: '0' per-row TileLang (reference), '1' multi-row TileLang, 'hip' HIP WMMA."""
    os.environ["DS4_IDX_MULTIROW"] = "1" if mode == "1" else "0"
    os.environ["DS4_IDX_HIP"] = "1" if mode == "hip" else "0"
    if "ds4_tl_indexer" in sys.modules:
        del sys.modules["ds4_tl_indexer"]
    m = importlib.import_module("ds4_tl_indexer")
    assert m._IDX_MULTIROW == (mode == "1")
    assert m._IDX_HIP == (mode == "hip"), "HIP scorer library not loadable"
    return m, m.fp8_paged_mqa_logits_tl(q, kv, w, cls, bt, max_model_len)


def timeit(m, q, kv, w, cls, bt, mml, iters=30):
    for _ in range(5):
        m.fp8_paged_mqa_logits_tl(q, kv, w, cls, bt, mml)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        m.fp8_paged_mqa_logits_tl(q, kv, w, cls, bt, mml)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", default="700,4000,25600,131072")
    ap.add_argument("--hip", action="store_true", help="test the HIP WMMA scorer instead of multi-row TileLang")
    args = ap.parse_args()
    mode = "hip" if args.hip else "1"
    ok = True
    for sl in [int(x) for x in args.ctx.split(",")]:
        q, kv, w, cls, bt = make_case(sl)
        mml = 524288
        m0, ref = run("0", q, kv, w, cls, bt, mml)
        t_ref = timeit(m0, q, kv, w, cls, bt, mml)
        m1, new = run(mode, q, kv, w, cls, bt, mml)
        t_new = timeit(m1, q, kv, w, cls, bt, mml)
        same_shape = ref.shape == new.shape
        eq = same_shape and torch.equal(ref, new)
        finite_eq = same_shape and torch.equal(torch.isfinite(ref), torch.isfinite(new))
        ok &= eq
        nz = (ref != new).sum().item() if same_shape else -1
        print(f"ctx={sl:>7}  rows={ref.shape[0]} width={ref.shape[1]}  "
              f"bit-exact={'YES' if eq else 'NO'} (finite-mask {'ok' if finite_eq else 'DIFF'}, "
              f"{nz} diff elems)  per-row {t_ref:8.1f} us  {mode:>8s} {t_new:8.1f} us  "
              f"x{t_ref / t_new:.2f}", flush=True)
    print("RESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
