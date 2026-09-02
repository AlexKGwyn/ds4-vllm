// ds4_fp8_gemv_hip.cpp — hand-written small-M block-scaled fp8 GEMV, gfx1151.
//
//   C[M,N] = A[M,K](bf16) @ W[N,K](fp8 e4m3fn, 128x128 block scales fp32).T
//
// v12: the decisive finding is that this kernel is 100% MEMORY-bound -- with
// the dequant and every FMA deleted it runs the same speed (or slower). So the
// win is not in the math, it is in the DRAM access footprint. See the ksplit
// kernel below: a workgroup must walk a CONTIGUOUS span of memory, not touch
// 8-16 chunks a row-stride apart. That alone took wo from 103 to 212 GB/s
// (~92% of this box's ~226 GB/s practical peak) and made every dense decode
// shape beat bf16-hipblaslt, which reads twice the bytes at ~200 GB/s.
//
// Retained from v10 (all measured neutral-to-tiny once ksplit landed, because
// the arithmetic they optimize was already free):
//   #1 A loaded as uint4 (2 loads per m per 512B tile, not 8 dwords)
//   #3 K<=1024 "kfit" kernel: whole strip is 1-2 quads, all W front-loaded
//   #4 packed dequant: v_perm_b32 spreads fp8 bytes into half-word slots and
//      one shift/mask pair builds a bf16 pair equal to 2^-120 x the true fp8
//      value -- exactly, for normals AND denormals AND zero (fp8 mantissa
//      aligns to the bf16 mantissa top and the exponent bias offset is a
//      uniform 120, including the E=0 denormal case). The 2^120 is folded
//      into the per-block scale multiply, so there is no per-element compare
//      or select. With USE_DOT2 the pairs feed v_dot2_f32_bf16 directly.
//   #2 (USE_LDS) K>=2048: A staged in LDS per 2KB K-block, cooperative uint4
//      fill, amortized over the workgroup's 16 rows.
// Build knobs: -DKS_ROWS_4K -DKS_ROWS_SM -DNO_COMPUTE=1 (BW ceiling probe).
//
// wave32-native throughout: cross-half wave64 shuffles are broken on this
// toolchain (shfl_xor(_,32) self-adds even under -mwavefrontsize64).
// e4m3fn NaN (em==0x7f) not special-cased: weight checkpoints have no NaNs.

#include <hip/hip_runtime.h>
#include <stdint.h>

// USE_DOT2: v_dot2_f32_bf16 assembles and runs on gfx1151 but FLUSHES bf16
// denormal inputs, and the packed dequant maps every fp8 denormal to a bf16
// denormal -- measured rel-err 0.016 vs 0.0005. Off, and must stay off.
#ifndef USE_DOT2
#define USE_DOT2 0
#endif
// USE_LDS: staging A in LDS costs more (barriers) than the reuse saves; A is
// already L1-resident. Measured negative (wo 162->220us). Off.
#ifndef USE_LDS
#define USE_LDS 0
#endif
#ifndef NO_COMPUTE
#define NO_COMPUTE 0  // 1 = keep every load, drop dequant+FMA (BW ceiling probe)
#endif
#ifndef USE_PACKED
#define USE_PACKED 1  // 0 = v5 dequant (select on denormals, no 2^120 fold)
#endif
#if USE_PACKED
// The packed dequant leaves every product scaled by a uniform 2^-120. Undo it
// ONCE per output, after the reduction -- not by pre-scaling the block scale,
// which would overflow f32 for any scale > 2^8.
#define OFIX 0x1p120f
#else
#define OFIX 1.0f
#endif
#ifndef KS_ROWS_4K
#define KS_ROWS_4K 2  // rows per workgroup when one row needs all 8 waves
#endif
#ifndef KS_ROWS_SM
#define KS_ROWS_SM 8  // rows per workgroup when a row needs fewer than 8 waves
#endif
#ifndef USE_KSPLIT
#define USE_KSPLIT 1
#endif
#ifndef BIG_R
#define BIG_R 2   // rows per wave on the K>=2048 path
#endif
#ifndef BIG_PF
#define BIG_PF 1  // W quads prefetched ahead
#endif

typedef uint16_t bf16_t;
typedef unsigned int u32x4_t __attribute__((ext_vector_type(4)));
// W loads are NON-TEMPORAL (read-once weights stay out of L2/MALL): +2-4%
// across the decode shapes (wo 204 -> 212 GB/s), results bitwise identical.
__device__ __forceinline__ uint4 nt_load4(const uint4* p) { u32x4_t v = __builtin_nontemporal_load(reinterpret_cast<const u32x4_t*>(p)); return make_uint4(v.x, v.y, v.z, v.w); }


__device__ __forceinline__ float bf16_hi(uint32_t pair) {
    return __uint_as_float(pair & 0xffff0000u);
}
__device__ __forceinline__ float bf16_lo(uint32_t pair) {
    return __uint_as_float(pair << 16);
}

__device__ __forceinline__ bf16_t f32_to_bf16_rne(float f) {
    uint32_t u = __float_as_uint(f);
    uint32_t rounded = u + 0x7fffu + ((u >> 16) & 1u);
    return (bf16_t)(rounded >> 16);
}

// 4 packed fp8 -> 2 bf16-pair u32s, each half = 2^-120 x the true fp8 value.
__device__ __forceinline__ void fp8x4_to_bf16pk(uint32_t u, uint32_t* p) {
    const uint32_t t01 = __builtin_amdgcn_perm(0u, u, 0x0c010c00u);  // 0 b1 0 b0
    const uint32_t t23 = __builtin_amdgcn_perm(0u, u, 0x0c030c02u);  // 0 b3 0 b2
    p[0] = ((t01 << 4) & 0x07f007f0u) | ((t01 << 8) & 0x80008000u);
    p[1] = ((t23 << 4) & 0x07f007f0u) | ((t23 << 8) & 0x80008000u);
}

#if USE_DOT2
__device__ __forceinline__ float dot2_bf16(uint32_t w, uint32_t a, float c) {
    float r;
    asm("v_dot2_f32_bf16 %0, %1, %2, %3" : "=v"(r) : "v"(w), "v"(a), "v"(c));
    return r;
}
#endif

// v5 dequant: 4 packed fp8 e4m3fn -> 4 floats, exact, select on denormals.
__device__ __forceinline__ void fp8x4_to_f32(uint32_t u, float* out) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint32_t b = (u >> (8 * i)) & 0xffu;
        const uint32_t em = b & 0x7fu;
        const float mag = (em < 8u)
                              ? (float)em * 0x1p-9f
                              : __uint_as_float((em << 20) + (120u << 23));
        out[i] = __uint_as_float(__float_as_uint(mag) | ((b & 0x80u) << 24));
    }
}

// One 512B W quad (16 fp8 per lane) against 16 A elements held as 8 packed
// bf16 pairs. With USE_PACKED, s carries the 2^120 dequant compensation.
template <int M>
__device__ __forceinline__ void fma16(const uint4 w4, const uint32_t* apairs,
                                      float s, float* acc) {
#if USE_PACKED
    uint32_t wpk[8];
    fp8x4_to_bf16pk(w4.x, wpk + 0);
    fp8x4_to_bf16pk(w4.y, wpk + 2);
    fp8x4_to_bf16pk(w4.z, wpk + 4);
    fp8x4_to_bf16pk(w4.w, wpk + 6);
#else
    float w[16];
    fp8x4_to_f32(w4.x, w + 0);
    fp8x4_to_f32(w4.y, w + 4);
    fp8x4_to_f32(w4.z, w + 8);
    fp8x4_to_f32(w4.w, w + 12);
#endif
#if NO_COMPUTE
    const uint32_t wx = w4.x ^ w4.y ^ w4.z ^ w4.w;
#pragma unroll
    for (int m = 0; m < M; ++m)
        acc[m] += (float)(wx ^ apairs[m * 8]);
#else
#pragma unroll
    for (int m = 0; m < M; ++m) {
        float pm = 0.0f;
#pragma unroll
        for (int jp = 0; jp < 8; ++jp) {
            const uint32_t av = apairs[m * 8 + jp];
#if USE_PACKED
#if USE_DOT2
            pm = dot2_bf16(wpk[jp], av, pm);
#else
            pm = fmaf(bf16_lo(wpk[jp]), bf16_lo(av), pm);
            pm = fmaf(bf16_hi(wpk[jp]), bf16_hi(av), pm);
#endif
#else
            pm = fmaf(w[2 * jp + 0], bf16_lo(av), pm);
            pm = fmaf(w[2 * jp + 1], bf16_hi(av), pm);
#endif
        }
        acc[m] = fmaf(s, pm, acc[m]);
    }
#endif
}

__device__ __forceinline__ void load_a8(const bf16_t* src, uint32_t* dst) {
    const uint4* p = reinterpret_cast<const uint4*>(src);
    const uint4 a0 = p[0];
    const uint4 a1 = p[1];
    dst[0] = a0.x; dst[1] = a0.y; dst[2] = a0.z; dst[3] = a0.w;
    dst[4] = a1.x; dst[5] = a1.y; dst[6] = a1.z; dst[7] = a1.w;
}

// ------------------------------------------------- kfit  (K <= 1024) ------
// R rows per wave, W quads loaded per quad inside the K loop. The 2-rows-per-
// wave layout re-read the whole activation slice from L2 for 2 KB of weights
// (at M=6, K=1024: 96 KB of L2 traffic per 16 KB of DRAM weights per
// workgroup), which held the K=1024 shapes at 145-177 GB/s while wo (K=4096,
// ksplit) reached 208. R=4 amortises each A quad over 4 rows. Measured
// (bench_fp8_gemv_padded.py): 16384x1024 177 -> 198, 8192x1024 167 -> 189,
// 4096x1024 145 -> 155 GB/s. Loading all R x nq W quads up front spills
// (R=8: 272 B/lane scratch, 102 GB/s; R=4: 144 B, 137 GB/s) -- per-quad loads
// keep it at 161 VGPRs, no scratch. Per-row math is unchanged (same lane->k
// mapping, same fma order), so results are bitwise identical to before.
template <int M, int R>
__global__ void __launch_bounds__(256)
fp8_gemv_kfit_kernel(const bf16_t* __restrict__ A,
                     const uint32_t* __restrict__ W,
                     const float* __restrict__ WS,
                     bf16_t* __restrict__ C, int N, int K, int ldw4) {
    const int lane = threadIdx.x & 31;
    const int wave = threadIdx.x >> 5;
    const int r0 = (blockIdx.x * 8 + wave) * R;
    const int Kd4 = ldw4;
    const int Ksc = K >> 7;
    const int lane_b = lane * 16;
    const int kb_lane = lane_b >> 7;
    const int nq = K >> 9;  // 1 or 2

    float acc[R][M];
#pragma unroll
    for (int rr = 0; rr < R; ++rr)
#pragma unroll
        for (int m = 0; m < M; ++m) acc[rr][m] = 0.0f;

    for (int q = 0; q < nq; ++q) {
        const int kq = q << 9;
        uint4 wq[R];
#pragma unroll
        for (int rr = 0; rr < R; ++rr)
            wq[rr] = nt_load4(reinterpret_cast<const uint4*>(
                W + (r0 + rr) * Kd4 + ((kq + lane_b) >> 2)));
        uint32_t apairs[M * 8];
#pragma unroll
        for (int m = 0; m < M; ++m)
            load_a8(A + m * K + kq + lane_b, apairs + m * 8);
        const int sc_col = (kq >> 7) + kb_lane;
#pragma unroll
        for (int rr = 0; rr < R; ++rr)
            fma16<M>(wq[rr], apairs,
                     WS[((r0 + rr) >> 7) * Ksc + sc_col] , acc[rr]);
    }

#pragma unroll
    for (int rr = 0; rr < R; ++rr)
#pragma unroll
        for (int m = 0; m < M; ++m) {
            float v = acc[rr][m];
#pragma unroll
            for (int off = 16; off; off >>= 1) v += __shfl_xor(v, off, 32);
            if (lane == 0) C[m * N + r0 + rr] = f32_to_bf16_rne(v * OFIX);
        }
}

// -------------------------------------------------- big  (K >= 2048) ------
// R rows per wave (W streams in flight scale with R, and one A load feeds
// all R rows). PF quads of W prefetched ahead of the compute quad.
template <int M, int R, int PF>
__global__ void __launch_bounds__(256)
fp8_gemv_big_kernel(const bf16_t* __restrict__ A,
                    const uint32_t* __restrict__ W,
                    const float* __restrict__ WS,
                    bf16_t* __restrict__ C, int N, int K, int ldw4) {
    const int lane = threadIdx.x & 31;
    const int wave = threadIdx.x >> 5;
    const int r0 = (blockIdx.x * 8 + wave) * R;
    const int Kd4 = ldw4;
    const int Ksc = K >> 7;
    const int lane_b = lane * 16;
    const int kb_lane = lane_b >> 7;

    float acc[R][M];
#pragma unroll
    for (int rr = 0; rr < R; ++rr)
#pragma unroll
        for (int m = 0; m < M; ++m) acc[rr][m] = 0.0f;

    // rolling PF-deep W pipeline
    uint4 wbuf[PF][R];
#pragma unroll
    for (int p = 0; p < PF; ++p)
#pragma unroll
        for (int rr = 0; rr < R; ++rr)
            wbuf[p][rr] = nt_load4(reinterpret_cast<const uint4*>(W + (r0 + rr) * Kd4 + ((p * 512 + lane_b) >> 2)));

    const int nq = K >> 9;
    for (int q = 0; q < nq; ++q) {
        const int kq = q << 9;
        uint4 wcur[R];
#pragma unroll
        for (int rr = 0; rr < R; ++rr) wcur[rr] = wbuf[0][rr];
#pragma unroll
        for (int p = 0; p < PF - 1; ++p)
#pragma unroll
            for (int rr = 0; rr < R; ++rr) wbuf[p][rr] = wbuf[p + 1][rr];
        const int kn = kq + PF * 512;
        if (kn < K) {
#pragma unroll
            for (int rr = 0; rr < R; ++rr)
                wbuf[PF - 1][rr] = nt_load4(reinterpret_cast<const uint4*>(W + (r0 + rr) * Kd4 + ((kn + lane_b) >> 2)));
        }

        uint32_t apairs[M * 8];
#pragma unroll
        for (int m = 0; m < M; ++m)
            load_a8(A + m * K + kq + lane_b, apairs + m * 8);
        const int sc_col = (kq >> 7) + kb_lane;
#pragma unroll
        for (int rr = 0; rr < R; ++rr)
            fma16<M>(wcur[rr], apairs,
                     WS[((r0 + rr) >> 7) * Ksc + sc_col] , acc[rr]);
    }

#pragma unroll
    for (int rr = 0; rr < R; ++rr)
#pragma unroll
        for (int m = 0; m < M; ++m) {
            float v = acc[rr][m];
#pragma unroll
            for (int off = 16; off; off >>= 1) v += __shfl_xor(v, off, 32);
            if (lane == 0) C[m * N + r0 + rr] = f32_to_bf16_rne(v * OFIX);
        }
}

// ------------------------------------------------ ksplit (any K) ---------
// The wave-per-rows layout gives a workgroup an instantaneous footprint of
// 8-16 chunks one row-stride apart, which halves DRAM efficiency at K=4096
// (measured with compute removed: 79 GB/s vs 174 at K=1024, where rows are
// adjacent). Here a workgroup always covers a CONTIGUOUS span of memory:
// WPR = K/512 waves cover one row, RP = 8/WPR rows are in flight per pass,
// and passes step to adjacent rows. A depends only on the k-slice, so each
// wave loads its A slice once and reuses it for every row it touches.
template <int M, int ROWS, int WPR>
__global__ void __launch_bounds__(256)
fp8_gemv_ksplit_kernel(const bf16_t* __restrict__ A,
                       const uint32_t* __restrict__ W,
                       const float* __restrict__ WS,
                       bf16_t* __restrict__ C, int N, int K, int ldw4) {
    constexpr int RP0 = 8 / WPR;                    // rows a full wg covers
    constexpr int RP = (RP0 < ROWS) ? RP0 : ROWS;   // clamp: never exceed ROWS
    constexpr int PASSES = ROWS / RP;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wave = tid >> 5;
    const int krow = wave % WPR;         // which 512B slice of the row
    const int rsub = wave / WPR;         // which row of the pass
    const int r0 = blockIdx.x * ROWS;
    const int Kd4 = ldw4;
    const int Ksc = K >> 7;
    const int lane_b = krow * 512 + lane * 16;
    const int sc_col = lane_b >> 7;

    __shared__ float red[ROWS * M * WPR];

    uint32_t apairs[M * 8];
#pragma unroll
    for (int m = 0; m < M; ++m) load_a8(A + m * K + lane_b, apairs + m * 8);

    if (rsub >= RP) {                    // surplus waves idle (K < 4096, few rows)
        __syncthreads();
        return;
    }

    float acc[PASSES][M];
#pragma unroll
    for (int p = 0; p < PASSES; ++p)
#pragma unroll
        for (int m = 0; m < M; ++m) acc[p][m] = 0.0f;

    // one pass in flight ahead; the next rows are adjacent memory
    uint4 wnext = nt_load4(reinterpret_cast<const uint4*>(W + (r0 + rsub) * Kd4 + (lane_b >> 2)));
#pragma unroll
    for (int p = 0; p < PASSES; ++p) {
        const uint4 wcur = wnext;
        const int rr = p * RP + rsub;
        if (p + 1 < PASSES)
            wnext = nt_load4(reinterpret_cast<const uint4*>(W + (r0 + rr + RP) * Kd4 + (lane_b >> 2)));
        fma16<M>(wcur, apairs, WS[((r0 + rr) >> 7) * Ksc + sc_col], acc[p]);
    }

#pragma unroll
    for (int p = 0; p < PASSES; ++p) {
        const int rr = p * RP + rsub;
#pragma unroll
        for (int m = 0; m < M; ++m) {
            float v = acc[p][m];
#pragma unroll
            for (int off = 16; off; off >>= 1) v += __shfl_xor(v, off, 32);
            if (lane == 0) red[(rr * M + m) * WPR + krow] = v;
        }
    }
    __syncthreads();
    for (int i = tid; i < ROWS * M; i += 256) {
        const int rr = i / M, m = i - rr * M;
        float v = 0.0f;
#pragma unroll
        for (int w = 0; w < WPR; ++w) v += red[i * WPR + w];
        C[m * N + r0 + rr] = f32_to_bf16_rne(v * OFIX);
    }
}

__global__ void _probe_kernel(int* out) {
    int tid = threadIdx.x;
    float v = 1.0f;
    for (int off = 32; off; off >>= 1) v += __shfl_xor(v, off, 64);
    if (tid == 0) { out[0] = warpSize; out[1] = (int)v; }
    float u = (float)(tid & 63);
    for (int off = warpSize / 2; off; off >>= 1) u += __shfl_xor(u, off, warpSize);
    if (tid == 0) out[2] = (int)u;
}

extern "C" {

int ds4_probe(void* out, void* stream) {
    hipLaunchKernelGGL(_probe_kernel, dim3(1), dim3(64), 0, (hipStream_t)stream,
                       (int*)out);
    return (int)hipGetLastError();
}

// ldw: row stride of W in ELEMENTS. vLLM pads each weight row by 256 bytes
// (observed strides (K+256, 1)), so this is not K -- assuming it was is what
// made the earlier build fail eligible() and never run in the serve.
int ds4_fp8_gemv(const void* A, const void* W, const void* WS, void* C,
                 int M, int N, int K, int ldw, void* stream) {
    if (ldw <= 0) ldw = K;
    if (ldw & 15) return -4;          // rows must stay 16B-aligned for uint4
    const int ldw4 = ldw >> 2;
    if (M < 1 || M > 8 || (N & 15) || (K & 511)) return -1;
    if (K > 1024 && (K & 2047)) return -1;
    // wide shapes stream more W per wave already; narrow ones need R=4 to
    // amortize the A load and keep enough bytes in flight.
    const int R = (N >= 4096) ? 2 : 4;
    if (K > 1024 && (N % (8 * R))) return -1;
    // K==4096 fills exactly one 512B quad per wave across 8 waves
    const int wpr = (K >= 4096) ? 8 : (K >> 9);   // waves covering one row
    const int rp = 8 / wpr;                       // rows in flight per pass
    const int ks_rows = (wpr == 8) ? KS_ROWS_4K : KS_ROWS_SM;
    const bool ksplit = USE_KSPLIT && K >= 512 && K <= 4096 &&
                        (N % ks_rows) == 0;
    (void)rp;
#define KSPLIT_LAUNCH(m, wprv, rowsv)                                      \
    hipLaunchKernelGGL((fp8_gemv_ksplit_kernel<m, rowsv, wprv>),           \
                       dim3(N / (rowsv)),                                  \
                       block, 0, s, (const bf16_t*)A, (const uint32_t*)W,  \
                       (const float*)WS, (bf16_t*)C, N, K, ldw4);
#define KSPLIT_DISPATCH(m)                                                 \
    switch (wpr) {                                                         \
        case 8: KSPLIT_LAUNCH(m, 8, KS_ROWS_4K) break;                     \
        case 4: KSPLIT_LAUNCH(m, 4, KS_ROWS_SM) break;                     \
        case 2: KSPLIT_LAUNCH(m, 2, KS_ROWS_SM) break;                      \
        default: KSPLIT_LAUNCH(m, 1, KS_ROWS_SM) break;                    \
    }
    dim3 gk(N / 16), gb(N / (8 * R)), block(256);
    // kfit rows per wave: 4 while the grid still fills the GPU (>= 64 workgroups
    // of 8 waves), else 2 for the narrow shapes.
    const int kfit_r = (N % 32 == 0 && N / 32 >= 64) ? 4 : 2;
    hipStream_t s = (hipStream_t)stream;
#define CASE(m)                                                            \
    case m:                                                                \
        if (K <= 1024) {                                                   \
            if (kfit_r == 4)                                               \
                hipLaunchKernelGGL((fp8_gemv_kfit_kernel<m, 4>), dim3(N / 32), block, 0, \
                                   s, (const bf16_t*)A, (const uint32_t*)W, \
                                   (const float*)WS, (bf16_t*)C, N, K, ldw4); \
            else                                                           \
                hipLaunchKernelGGL((fp8_gemv_kfit_kernel<m, 2>), gk, block, 0, \
                                   s, (const bf16_t*)A, (const uint32_t*)W, \
                                   (const float*)WS, (bf16_t*)C, N, K, ldw4); \
        }                                                                  \
        else if (ksplit)                                                   \
            KSPLIT_DISPATCH(m)                                             \
        else if (R == 2)                                                   \
            hipLaunchKernelGGL((fp8_gemv_big_kernel<m, 2, 1>), gb, block,  \
                               0, s, (const bf16_t*)A, (const uint32_t*)W, \
                               (const float*)WS, (bf16_t*)C, N, K, ldw4);        \
        else                                                               \
            hipLaunchKernelGGL((fp8_gemv_big_kernel<m, 4, 1>), gb, block,  \
                               0, s, (const bf16_t*)A, (const uint32_t*)W, \
                               (const float*)WS, (bf16_t*)C, N, K, ldw4);        \
        break;
    switch (M) {
        CASE(1) CASE(2) CASE(3) CASE(4) CASE(5) CASE(6) CASE(7) CASE(8)
    }
#undef CASE
    return (int)hipGetLastError();
}
}
