// ds4_idx_score.hip — decode sparse-indexer scorer for gfx1151 (DS4_IDX_HIP).
//
//   logits[r, j] = sum_h relu(q[r,h] . k[j]) * w[r,h] * scale[j]     j < ke[r]
//                = -inf                                              j >= ke[r]
//
// Replaces, per decode step and indexer layer, the fused paged-cache gather
// (fp8 -> bf16 materialised copy of the WHOLE compressed context) plus one
// TileLang scorer launch PER ROW that streamed that copy again -- the term
// that grew linearly with context (~1.5 ms x 21 layers per step at 512K).
// This kernel reads the fp8 paged cache straight from the page table, one
// 64-token tile at a time, and scores every row of the request against it.
//
// Why it is fast (each of these was measured, not assumed):
//   * LDS footprint ~16.6 KB and 4-wave workgroups, so several share a WGP. The TileLang
//     kernel (and every larger-LDS variant tried) sits at one per WGP -- 20
//     concurrent workgroups on the whole GPU -- which, not memory bandwidth,
//     was the ceiling: grid size and tile order made no difference to it.
//     Only the K tile lives in LDS; each wave owns a 16-token block x ALL
//     64 heads, so the head reduction is register adds + xor-shuffles -- no
//     LDS exchange, no barriers beyond K staging. Q fragments come from L1
//     (a wave's Q block is 16 KB and stays resident across tiles).
//   * K rows are XOR-swizzled at 16-byte chunk granularity (TileLang's trick)
//     so the 16 fragment rows of a WMMA load spread over the 32 banks.
//   * rows OUTER (NRP per pass), tiles INNER: a persistent workgroup loads its Q
//     pointers once per pass and walks its share of tiles; the K re-reads
//     for the next pass come from the 32 MB Infinity Cache (a 512K context's
//     fp8 K is 16.7 MB). The next tile's page bytes are prefetched into
//     registers before the current tile computes.
//
// BIT-EXACT vs the TileLang scorer by construction (verified on random caches
// incl. NaN/subnormal/-0 fp8 bytes: host/bench/verify_indexer_multirow.py --hip):
//   * fp8 e4m3fn -> bf16 is value-exact; NaN bytes map to torch's 0x7FF0/0xFFF0.
//   * per-(token, head) score: the SAME raw __builtin_amdgcn_wmma_f32_16x16x16
//     _bf16_w32 intrinsic TileLang's gemm emits, fed the same way -- 16 k
//     values per lane in natural order, lanes 16..31 duplicating 0..15, 8
//     ascending k-steps into one accumulator (rocwmma's fragment packing feeds
//     it a different k order and drifts by 1 ulp -- do not use it here).
//   * epilogue (max(s,0) * w) * scale, then the 64-head sum in TileLang's
//     order: pairs (h, h+32) then (h, h+16) across waves, then xor-shuffles
//     8/4/2/1 within the 16-lane group.
//
// Build: hipcc -O3 --offload-arch=gfx1151 -shared -fPIC -o libds4idx.so ds4_idx_score.hip
// Measured (host/bench/verify_indexer_multirow.py --hip, vs the per-row TileLang
// path incl. its gather): 512K ctx 1534 -> 538 us, 100K 283 -> 121 us, 4K 108 -> 30 us.
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <math.h>

// NO FP CONTRACTION. hipcc defaults to fusing a*b+c into one FMA; the epilogue's
// (max(s,0)*w)*scale products are followed by the head-sum adds, and TileLang
// rounds the product before adding. With contraction on, the kernel drifts by
// 1 ulp in ~7% of scores (measured) and is no longer bit-exact. Pinned here so
// it cannot depend on the build line.
#pragma clang fp contract(off)

#define BN 64          // tokens per tile (== one 64-token page or slice)
#define HH 64          // indexer heads
#define DD 128         // index_head_dim
#define TPB 128        // 4 waves (wave32): wave w owns tokens [16w, 16w+16) x ALL 64 heads
#ifndef NRP
#define NRP 2          // query rows scored per K pass: 2 -> 190 VGPRs, 7 waves/SIMD (3 spills
                       // nothing but drops to 5 waves/SIMD; 6 spills 508 B/lane -- both measured slower)
#endif

typedef __attribute__((__vector_size__(16 * sizeof(short)))) short v16i16;
typedef __attribute__((__vector_size__(8 * sizeof(float)))) float v8f32;

__device__ __forceinline__ uint32_t fp8_to_bf16_bits(uint32_t b) {
    const uint32_t sign = (b & 0x80u) << 8;
    const uint32_t em = b & 0x7fu;
    if (em == 0x7fu) return 0x7FF0u | sign;
    float mag;
    if (em < 8u) mag = (float)em * 0x1p-9f;
    else mag = __uint_as_float((em << 20) + (120u << 23));
    return (__float_as_uint(mag) >> 16) | sign;
}

__device__ __forceinline__ uint2 cvt4(uint32_t v) {
    return make_uint2(fp8_to_bf16_bits(v & 0xffu) | (fp8_to_bf16_bits((v >> 8) & 0xffu) << 16),
                      fp8_to_bf16_bits((v >> 16) & 0xffu) | (fp8_to_bf16_bits(v >> 24) << 16));
}

// K_sh: 64 rows x 256 B; 16-byte chunk c of row t is stored at chunk c ^ (t & 15)
__device__ __forceinline__ int kswz(int row, int chunk) {
    return row * DD + ((chunk ^ (row & 15)) << 3);
}

__device__ __forceinline__ void stage_page(const uint4 p0, const uint4 p1, const uint4 p2, const uint4 p3,
                                           int ct, int ch0, uint16_t* K_sh) {
    // 64 fp8 bytes (tokens ct, chunks ch0..ch0+7) -> 8 swizzled 16-byte chunks
    const uint4 pp[4] = {p0, p1, p2, p3};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint2 e0 = cvt4(pp[i].x), e1 = cvt4(pp[i].y), e2 = cvt4(pp[i].z), e3 = cvt4(pp[i].w);
        *reinterpret_cast<uint4*>(K_sh + kswz(ct, ch0 + 2 * i)) = make_uint4(e0.x, e0.y, e1.x, e1.y);
        *reinterpret_cast<uint4*>(K_sh + kswz(ct, ch0 + 2 * i + 1)) = make_uint4(e2.x, e2.y, e3.x, e3.y);
    }
}

__global__ void __launch_bounds__(TPB)
idx_score_kernel(const uint8_t* __restrict__ flat, long row_bytes,
                 const int* __restrict__ bt, int npg, int bs,
                 const uint16_t* __restrict__ Q,   // [nr*HH][DD] bf16
                 const float* __restrict__ W,      // [nr][HH]
                 const int* __restrict__ KE,       // [nr]
                 float* __restrict__ out, long out_stride,
                 int nr, int ntiles) {
    __shared__ __align__(16) uint16_t K_sh[BN * DD];   // 16 KB (swizzled)
    __shared__ float S_sh[BN];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wave = tid >> 5;           // token block: tokens [wave*16, +16)
    const int l16 = lane & 15;
    const int lhi = lane >> 4;           // C-fragment row parity
    const int ct = tid >> 1, ch0 = (tid & 1) * 8;   // page copy: token ct, 64-byte half
    const int trow = wave * 16 + l16;    // this lane's A row

    for (int r0 = 0; r0 < nr; r0 += NRP) {
        const int nrp = (nr - r0 < NRP) ? (nr - r0) : NRP;
        const uint16_t* qrow[NRP][4];
        float wv[NRP][4];
        int ke[NRP];
#pragma unroll
        for (int j = 0; j < NRP; ++j) {
            const int r = r0 + (j < nrp ? j : 0);
            ke[j] = KE[r];
#pragma unroll
            for (int hb = 0; hb < 4; ++hb) {
                qrow[j][hb] = Q + ((long)r * HH + hb * 16 + l16) * DD;
                wv[j][hb] = W[r * HH + hb * 16 + l16];
            }
        }

        int tile = blockIdx.x;
        uint4 p0 = make_uint4(0, 0, 0, 0), p1 = p0, p2 = p0, p3 = p0;
        float psc = 0.0f;
        bool has = false;
        if (tile < ntiles) {
            const int pi = (tile * BN) / bs;
            has = pi < npg;
            if (has) {
                const uint8_t* base = flat + (long)bt[pi] * row_bytes;
                const int slot0 = tile * BN - pi * bs;
                const uint4* sp = reinterpret_cast<const uint4*>(base + (long)(slot0 + ct) * DD + ch0 * 8);
                p0 = sp[0]; p1 = sp[1]; p2 = sp[2]; p3 = sp[3];
                if (tid < BN) {
                    const uint8_t* s = base + (long)bs * DD + (long)(slot0 + tid) * 4;
                    psc = __uint_as_float((uint32_t)s[0] | ((uint32_t)s[1] << 8) | ((uint32_t)s[2] << 16) | ((uint32_t)s[3] << 24));
                }
            }
        }
        for (; tile < ntiles; tile += gridDim.x) {
            const int tok0 = tile * BN;
            const bool cur_has = has;
            __syncthreads();   // previous tile's K_sh reads are done
            if (cur_has) {
                stage_page(p0, p1, p2, p3, ct, ch0, K_sh);
                if (tid < BN) S_sh[tid] = psc;
            }
            const int ntile = tile + gridDim.x;
            has = false;
            if (ntile < ntiles) {
                const int pi = (ntile * BN) / bs;
                has = pi < npg;
                if (has) {
                    const uint8_t* base = flat + (long)bt[pi] * row_bytes;
                    const int slot0 = ntile * BN - pi * bs;
                    const uint4* sp = reinterpret_cast<const uint4*>(base + (long)(slot0 + ct) * DD + ch0 * 8);
                    p0 = sp[0]; p1 = sp[1]; p2 = sp[2]; p3 = sp[3];
                    if (tid < BN) {
                        const uint8_t* s = base + (long)bs * DD + (long)(slot0 + tid) * 4;
                        psc = __uint_as_float((uint32_t)s[0] | ((uint32_t)s[1] << 8) | ((uint32_t)s[2] << 16) | ((uint32_t)s[3] << 24));
                    }
                }
            }
            __syncthreads();   // K_sh / S_sh staged

            if (cur_has) {
                v8f32 c[NRP][4];
#pragma unroll
                for (int j = 0; j < NRP; ++j)
#pragma unroll
                    for (int hb = 0; hb < 4; ++hb) c[j][hb] = (v8f32){0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
                for (int ki = 0; ki < 8; ++ki) {
                    v16i16 a;
                    *reinterpret_cast<uint4*>(&a) = *reinterpret_cast<const uint4*>(K_sh + kswz(trow, 2 * ki));
                    *(reinterpret_cast<uint4*>(&a) + 1) = *reinterpret_cast<const uint4*>(K_sh + kswz(trow, 2 * ki + 1));
#pragma unroll
                    for (int j = 0; j < NRP; ++j)
#pragma unroll
                        for (int hb = 0; hb < 4; ++hb) {
                            v16i16 b;
                            *reinterpret_cast<uint4*>(&b) = *reinterpret_cast<const uint4*>(qrow[j][hb] + ki * 16);
                            *(reinterpret_cast<uint4*>(&b) + 1) = *reinterpret_cast<const uint4*>(qrow[j][hb] + ki * 16 + 8);
                            c[j][hb] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c[j][hb]);
                        }
                }
                // epilogue + head reduction in TileLang's order, all in registers:
                // element i <-> token wave*16 + 2i + lhi; head = hb*16 + l16
#pragma unroll
                for (int j = 0; j < NRP; ++j) {
                    if (j < nrp) {
                        float v[8];
#pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            const float sc = S_sh[wave * 16 + 2 * i + lhi];
                            const float s0 = (fmaxf(c[j][0][i], 0.0f) * wv[j][0]) * sc;
                            const float s1 = (fmaxf(c[j][1][i], 0.0f) * wv[j][1]) * sc;
                            const float s2 = (fmaxf(c[j][2][i], 0.0f) * wv[j][2]) * sc;
                            const float s3 = (fmaxf(c[j][3][i], 0.0f) * wv[j][3]) * sc;
                            const float a0 = s0 + s2;      // (h, h+32)
                            const float a1 = s1 + s3;      // (h+16, h+48)
                            float x = a0 + a1;             // (h, h+16)
                            x += __shfl_xor(x, 8, 32);
                            x += __shfl_xor(x, 4, 32);
                            x += __shfl_xor(x, 2, 32);
                            x += __shfl_xor(x, 1, 32);
                            v[i] = x;
                        }
                        if (l16 == 0) {
                            float* orow = out + (long)(r0 + j) * out_stride + tok0 + wave * 16;
                            const int kej = ke[j];
#pragma unroll
                            for (int i = 0; i < 8; ++i) {
                                const int t = 2 * i + lhi;
                                orow[t] = (tok0 + wave * 16 + t < kej) ? v[i] : -INFINITY;
                            }
                        }
                    }
                }
            } else {
                // beyond the page table: every column is >= ke for every row
                if (tid < BN) {
#pragma unroll
                    for (int j = 0; j < NRP; ++j)
                        if (j < nrp) out[(long)(r0 + j) * out_stride + tok0 + tid] = -INFINITY;
                }
            }
        }
        __syncthreads();
    }
}

extern "C" int ds4_idx_score(const void* flat, long row_bytes, const void* bt, int npg, int bs,
                             const void* Q, const void* W, const void* KE,
                             void* out, long out_stride, int nr, int ntiles,
                             int grid, void* stream) {
    if (bs % BN != 0 || ntiles <= 0 || nr <= 0) return -1;
    if (grid <= 0 || grid > ntiles) grid = ntiles;
    hipLaunchKernelGGL(idx_score_kernel, dim3(grid), dim3(TPB), 0, (hipStream_t)stream,
                       (const uint8_t*)flat, row_bytes, (const int*)bt, npg, bs,
                       (const uint16_t*)Q, (const float*)W, (const int*)KE,
                       (float*)out, out_stride, nr, ntiles);
    return (int)hipGetLastError();
}
