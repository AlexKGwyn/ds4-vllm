// ds4_moe_mxfp4.cpp — hand-written MXFP4 MoE decode path for gfx1151.
//
// Replaces triton_kernel_fused_experts' matmul_ogs pair for small-M decode:
//     gemm1: ic[p, I]  = silu_clamp( x[t, K1] @ w1[e].T )     (fused activation)
//     gemm2: out[t, K1] += gamma_p * ( ic[p, I] @ w2[e].T )  (fused scatter)
// where p indexes the M*topk (token, slot) pairs and e = topk_ids[p].
//
// LAYOUT. _swizzle_mxfp4 takes the
// StridedLayout branch on ROCm, so nothing is swizzled -- it only hands triton a
// transpose_(-2,-1) VIEW. The memory really is
//     values [E, N, K/2] uint8 contiguous, row n at n*(K/2), 2 k per byte
//     scales [E, N, K/32] uint8 contiguous, row n at n*(K/32)
// so each output row's K is one contiguous run. triton reads it against the
// grain -- a (block_k=256, block_n=32) tile gathers 32 chunks of 128B that are
// K/2 apart, a scattered footprint that wastes most of each burst. Here a
// workgroup owns NR
// consecutive output rows and its waves split ONE row's K, so it marches
// contiguously; x is loaded once per wave and reused across every row.
//
// NT (tokens per entry) is a TEMPLATE parameter and fixed at 2. As a runtime
// bound the compiler must allocate for the worst case, which spilled and ran
// slower than triton. Entries carrying one token are padded with a dummy slot
// whose gamma is 0 and whose ic row is scratch, so no branch is needed.
//
// wave32-native: cross-half wave64 shuffles are broken on this toolchain.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#ifndef NR1
#define NR1 4        // ic COLUMNS per workgroup; it touches 2*NR1 weight rows
#endif
#ifndef NR2
#define NR2 32       // w2 output rows per workgroup (0.248 vs 0.344 at 8)
#endif
#ifndef NT
#define NT 3         // largest token count an entry carries (decode clusters at 1-3)
#endif
// per-lane weight load width in BYTES: 8 = global_load_dwordx2, 16 = dwordx4.
// WPR (waves tiling one row) falls out of it as Kb/(32*WB), so WB and WPR are
// one knob, not two: gemm1 K=4096 gives WPR 8 at WB=8 and 4 at WB=16; gemm2
// K=1024 gives 2 and 1. Costs 2*WB VGPRs of activation per token held.
//
// 8, not 16, and the reason is correctness rather than speed. WB sets how many
// k a lane accumulates before the cross-lane reduction, so widening it regroups
// the summation and the result is no longer bit-identical. Logits shift by a
// ULP or two, the sampled token sequence diverges, and MTP acceptance moves
// with it -- an unpredictable change in delivered throughput on top of an
// unchanged model. At WB=8 this kernel is bit-identical to the arithmetic
// decode below, so it changes nothing about what the model emits.
//
// These knobs are tuned against the dequant cost, not independently of it:
// re-measure them together if the decode changes.
#ifndef WB1
#define WB1 8
#endif
#ifndef WB2
#define WB2 8
#endif
// How many row-loads to keep in flight per lane. Each extra stage costs WB/4
// VGPRs and there are no spills at these sizes, so depth is nearly free --
// but deeper prefetch measured no better here, so it stays at 1.
#ifndef PF
#define PF 1
#endif
// NT must cover the real group sizes or the expert's weights get re-read once
// per chunk. A token picks an expert at most once, so a group is at most M
// slots; at decode the sizes cluster low. Too small an NT splits a group in
// two and re-reads that expert once per chunk, inflating the bytes read enough
// to cancel the advantage.

typedef uint16_t bf16_t;

__device__ __forceinline__ float bf16_lo(uint32_t p) { return __uint_as_float(p << 16); }
__device__ __forceinline__ float bf16_hi(uint32_t p) { return __uint_as_float(p & 0xffff0000u); }

__device__ __forceinline__ bf16_t f32_to_bf16_rne(float f) {
    uint32_t u = __float_as_uint(f);
    return (bf16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

// MXFP4 e2m1 nibble -> float. em: 0,.5,1,1.5,2,3,4,6.
//
// This decode is NOT free: it runs twice per weight byte, and the arithmetic
// form below costs about 13 ALU ops a nibble -- enough to hold the GEMM well
// short of what the same memory footprint sustains without it.
//
// ARITH is the straightforward version, kept as the readable reference for
// what the format means: for em>=2 the exponent is (em>>1)+126 with mantissa
// bit em&1, and em<2 needs a select.
__device__ __forceinline__ float fp4_to_f32_arith(uint32_t nib) {
    const uint32_t em = nib & 7u;
    const float mag = (em < 2u) ? (float)em * 0.5f
                                : __uint_as_float((((em >> 1) + 126u) << 23) |
                                                  ((em & 1u) << 22));
    return __uint_as_float(__float_as_uint(mag) | ((nib & 8u) << 28));
}

// FP16 is the same function in 3 ops. Placing the three magnitude bits at
// (nib & 7) << 9 puts e2m1's exponent in the fp16 exponent field and its
// mantissa bit at the top of the fp16 mantissa -- and e2m1's subnormal step
// (em = 0,1 -> 0, 0.5) then lands on a genuine fp16 SUBNORMAL, so the select
// the arithmetic version spends most of its ops on disappears into the fp16
// hardware. (nib & 8) << 12 carries the sign to bit 15.
//
// The result is 2^14 too small, by exactly the subnormal ladder we borrowed.
// The caller folds that back via FP4_POST_SCALE, which costs one multiply per
// 32-k block instead of one per element.
//
// This is EXACT, not an approximation: all eight e2m1 magnitudes
// {0,.5,1,1.5,2,3,4,6} are representable in fp16 and in f32, and 2^14 is a
// power of two, so the product is bit-identical to the arithmetic decode.
__device__ __forceinline__ float fp4_to_f32_fp16(uint32_t nib) {
    return __half2float(__ushort_as_half(
        (unsigned short)(((nib & 7u) << 9) | ((nib & 8u) << 12))));
}

#ifndef FP4_ARITH
#define FP4_ARITH 0      // 1 restores the original decode, for A/B
#endif

#if FP4_ARITH
#define FP4_POST_SCALE 1.0f
__device__ __forceinline__ float fp4_to_f32(uint32_t nib) {
    return fp4_to_f32_arith(nib);
}
#else
// 2^14: undoes the fp16 subnormal ladder fp4_to_f32_fp16 decodes through.
#define FP4_POST_SCALE 16384.0f
__device__ __forceinline__ float fp4_to_f32(uint32_t nib) {
    return fp4_to_f32_fp16(nib);
}
#endif

// e8m0 -> 2^(b-127): an fp32 whose exponent field is b.
__device__ __forceinline__ float e8m0_to_f32(uint32_t b) {
    return __uint_as_float(b << 23);
}

// WB packed weight bytes (2*WB k) against 2*WB bf16 activations held as WB u32
// pairs. ap[j] carries k = 2j (low half) and k = 2j+1 (high half).
//
// WB is the per-lane load width and the whole point of the knob: both GEMMs are
// weight-bandwidth bound, so what matters is bytes in flight per lane, not
// arithmetic. WB=8 issues global_load_dwordx2, WB=16 issues dwordx4 -- half as
// many requests, each twice as wide. WB=16 also lands each lane on exactly one
// mxfp4 e8m0 block (32 k), so the block scale is a per-lane constant instead of
// something two neighbouring lanes share.
template <int WB>
__device__ __forceinline__ float dotWB(const uint32_t* __restrict__ wp,
                                       const uint32_t* __restrict__ ap) {
    float s = 0.0f;
#pragma unroll
    for (int j = 0; j < WB; ++j) {
        const uint32_t wb = wp[j >> 2] >> (8 * (j & 3));
        s = fmaf(fp4_to_f32(wb & 0xfu), bf16_lo(ap[j]), s);        // k = 2j
        s = fmaf(fp4_to_f32((wb >> 4) & 0xfu), bf16_hi(ap[j]), s); // k = 2j+1
    }
    return s;
}

// the vector type that loads WB bytes in one instruction
template <int WB> struct WVec;
template <> struct WVec<8>  { using T = uint2; };
template <> struct WVec<16> { using T = uint4; };

// ------------------------------------------------------------ entries -------
// One workgroup. P = M*topk slots; groups them by expert into entries of up to
// NT slots. Entry slots beyond a group's size are padded (gamma 0, scratch ic
// row) so the GEMMs need no bounds branch. Inactive entries get eidx -1 and
// exit immediately, which lets the host launch a fixed grid of P entries and
// never sync to learn the real count.
template <typename IDT>
__global__ void __launch_bounds__(256)
moe_build_entries(const IDT* __restrict__ topk_ids,
                  const float* __restrict__ topk_w,
                  int* __restrict__ eidx, int* __restrict__ ent_n,
                  int* __restrict__ ent_tok, int* __restrict__ ent_slot,
                  float* __restrict__ ent_gam, int P, int topk) {
    const int p = threadIdx.x;
    if (p >= P) return;
    const int e = (int)topk_ids[p];

    int rank = 0;                       // this slot's index within its expert
    for (int q = 0; q < p; ++q) rank += ((int)topk_ids[q] == e);

    // the chunk this slot belongs to starts at the same-expert slot whose rank
    // is the NT-aligned floor of ours
    const int target = (rank / NT) * NT;
    int p0 = p, seen = 0;
    for (int q = 0; q < P; ++q) {
        if ((int)topk_ids[q] == e) {
            if (seen == target) { p0 = q; break; }
            ++seen;
        }
    }

    // entry ids are handed to chunk-starts in slot order
    int ent = 0;
    for (int q = 0; q < p0; ++q) {
        const int eq = (int)topk_ids[q];
        int rq = 0;
        for (int r = 0; r < q; ++r) rq += ((int)topk_ids[r] == eq);
        ent += (rq % NT == 0);
    }

    eidx[ent] = e;                      // every slot of the chunk writes the same
    // the chunk's real size, so each NT-specialised kernel can claim its own
    // entries without the host ever learning the bucket counts
    int total = 0;
    for (int q = 0; q < P; ++q) total += ((int)topk_ids[q] == e);
    int sz = total - target;
    ent_n[ent] = sz < NT ? sz : NT;
    const int i = rank % NT;
    ent_tok[ent * NT + i] = p / topk;
    ent_slot[ent * NT + i] = p;
    ent_gam[ent * NT + i] = topk_w[p];
}

// Pre-pads every entry slot: gamma 0 and the scratch ic row, so slots the build
// never fills contribute nothing and need no bounds branch in the GEMMs.
__global__ void moe_clear_entries(int* __restrict__ eidx, int* __restrict__ ent_n,
                                  int* __restrict__ ent_tok,
                                  int* __restrict__ ent_slot,
                                  float* __restrict__ ent_gam, int P) {
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i < P) { eidx[i] = -1; ent_n[i] = 0; }
    if (i < P * NT) {
        ent_tok[i] = 0;
        ent_slot[i] = P;                // scratch ic row
        ent_gam[i] = 0.0f;
    }
}

// -------------------------------------------------------------- gemm1 -------
// WPR waves cover one output row; RP = 8/WPR rows are in flight per pass.
template <int NR, int WPR, int NTT, int WB>
__global__ void __launch_bounds__(256)
moe_gemm1_kernel(const bf16_t* __restrict__ x, const uint8_t* __restrict__ V,
                 const uint8_t* __restrict__ S, const int* __restrict__ eidx,
                 const int* __restrict__ ent_n, const int* __restrict__ ent_tok,
                 const int* __restrict__ ent_slot,
                 bf16_t* __restrict__ ic, int N, int K, float alpha, float limit) {
    // DeepSeek-V4-Flash's MoE is SILU, not the gpt-oss swiglu: gate and up are
    // CONCATENATED halves of w1's N rows (gate = 0..I-1, up = I..2I-1) and the
    // activation is silu(gate)*up. So a workgroup owns NR output columns, which
    // means 2*NR weight rows in two contiguous runs I*Kb apart -- still two long
    // marches rather than the 32 scattered chunks triton reads.
    constexpr int RP = 8 / WPR;
    constexpr int NROWW = 2 * NR;           // weight rows this workgroup touches
    constexpr int PASSES = NROWW / RP;
    const int j = blockIdx.y;
    const int e = eidx[j];
    if (e < 0 || ent_n[j] != NTT) return;   // this entry belongs to another NT

    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int krow = wave % WPR, rsub = wave / WPR;
    const int Kb = K >> 1, Ks = K >> 5;
    const int I = N >> 1;                   // ic width = gate/up half width
    const size_t ebase = (size_t)e * (size_t)N * (size_t)Kb;
    const size_t sbase = (size_t)e * (size_t)N * (size_t)Ks;
    const int byte_off = krow * (Kb / WPR) + lane * WB;
    const int k_off = byte_off << 1;
    const int sc_off = k_off >> 5;

    __shared__ float red[NROWW * NTT * WPR];

    // WB weight bytes cover 2*WB k, i.e. WB u32 of bf16 activation = WB/4 uint4
    uint32_t ap[NTT][WB];
#pragma unroll
    for (int i = 0; i < NTT; ++i) {
        const uint4* p = reinterpret_cast<const uint4*>(
            x + (size_t)ent_tok[j * NT + i] * K + k_off);
#pragma unroll
        for (int v = 0; v < WB / 4; ++v) {
            const uint4 a = p[v];
            ap[i][4 * v + 0] = a.x; ap[i][4 * v + 1] = a.y;
            ap[i][4 * v + 2] = a.z; ap[i][4 * v + 3] = a.w;
        }
    }

    float acc[PASSES][NTT];
#pragma unroll
    for (int q = 0; q < PASSES; ++q)
#pragma unroll
        for (int i = 0; i < NTT; ++i) acc[q][i] = 0.0f;

    const int n0 = blockIdx.x * NR;
    // rr < NR -> gate row n0+rr ; rr >= NR -> up row n0+I+(rr-NR)
#define G1ROW(rr) ((rr) < NR ? (n0 + (rr)) : (n0 + I + (rr) - NR))
    using WT = typename WVec<WB>::T;
#define G1LOAD(rr) (*reinterpret_cast<const WT*>(                              \
    V + ebase + (size_t)G1ROW(rr) * Kb + byte_off))
    constexpr int PFD = PF < PASSES ? PF : PASSES;
    WT wbuf[PFD];
#pragma unroll
    for (int p = 0; p < PFD; ++p) wbuf[p] = G1LOAD(p * RP + rsub);
#pragma unroll
    for (int q = 0; q < PASSES; ++q) {
        const WT wcur = wbuf[q % PFD];
        const int rr = q * RP + rsub;
        const int n = G1ROW(rr);
        // reissue this slot for the pass PFD ahead before consuming wcur, so
        // PFD loads are outstanding across the dequant chain rather than one
        if (q + PFD < PASSES) wbuf[q % PFD] = G1LOAD(rr + PFD * RP);
        const float sc = e8m0_to_f32(S[sbase + (size_t)n * Ks + sc_off])
                         * FP4_POST_SCALE;
#pragma unroll
        for (int i = 0; i < NTT; ++i)
            acc[q][i] = fmaf(sc, dotWB<WB>(reinterpret_cast<const uint32_t*>(&wcur),
                                           ap[i]), acc[q][i]);
    }
#undef G1LOAD
#undef G1ROW

#pragma unroll
    for (int q = 0; q < PASSES; ++q) {
        const int rr = q * RP + rsub;
#pragma unroll
        for (int i = 0; i < NTT; ++i) {
            float v = acc[q][i];
#pragma unroll
            for (int off = 16; off; off >>= 1) v += __shfl_xor(v, off, 32);
            if (lane == 0) red[(rr * NTT + i) * WPR + krow] = v;
        }
    }
    __syncthreads();
    (void)alpha;                    // SILU takes no alpha (that is the OAI swiglu)
    // silu_and_mul over the CONCATENATED halves: this workgroup's local row c
    // is the gate and local row c+NR is the matching up.
    //
    // This must equal vllm/model_executor/layers/fused_moe/utils.py::
    // swiglu_limit_func, which is what UnfusedOAITritonExperts.activation()
    // dispatches to for MoEActivation.SILU with gemm1_clamp_limit set:
    //     if limit > 0: gate = min(gate, limit); up = clamp(up, -limit, limit)
    //     out = silu(gate) * up
    // DeepSeek-V4-Flash sets config.swiglu_limit = 10.0, so the clamp is LIVE --
    // it reaches the kernel as `limit` and must not be dropped. (limit <= 0 is
    // the no-clamp path, matching the same guard on the python side.)
    for (int idx = tid; idx < NR * NTT; idx += 256) {
        const int c = idx / NTT, i = idx - c * NTT;
        float g = 0.0f, u = 0.0f;
#pragma unroll
        for (int w = 0; w < WPR; ++w) {
            g += red[(c * NTT + i) * WPR + w];
            u += red[((c + NR) * NTT + i) * WPR + w];
        }
        if (limit > 0.0f) {
            g = fminf(g, limit);
            u = fminf(fmaxf(u, -limit), limit);
        }
        const float sg = g / (1.0f + __expf(-g));      // silu(gate)
        ic[(size_t)ent_slot[j * NT + i] * I + n0 + c] = f32_to_bf16_rne(sg * u);
    }
}

// -------------------------------------------------------------- gemm2 -------
template <int NR, int WPR, int NTT, int WB>
__global__ void __launch_bounds__(256)
moe_gemm2_kernel(const bf16_t* __restrict__ ic, const uint8_t* __restrict__ V,
                 const uint8_t* __restrict__ S, const int* __restrict__ eidx,
                 const int* __restrict__ ent_n, const int* __restrict__ ent_tok,
                 const int* __restrict__ ent_slot,
                 const float* __restrict__ ent_gam, float* __restrict__ out,
                 int N, int K) {
    constexpr int RP = 8 / WPR;
    constexpr int PASSES = NR / RP;
    const int j = blockIdx.y;
    const int e = eidx[j];
    if (e < 0 || ent_n[j] != NTT) return;   // this entry belongs to another NT

    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int krow = wave % WPR, rsub = wave / WPR;
    const int Kb = K >> 1, Ks = K >> 5;
    const size_t ebase = (size_t)e * (size_t)N * (size_t)Kb;
    const size_t sbase = (size_t)e * (size_t)N * (size_t)Ks;
    const int byte_off = krow * (Kb / WPR) + lane * WB;
    const int k_off = byte_off << 1;
    const int sc_off = k_off >> 5;

    __shared__ float red[NR * NTT * WPR];

    uint32_t ap[NTT][WB];
#pragma unroll
    for (int i = 0; i < NTT; ++i) {
        const uint4* p = reinterpret_cast<const uint4*>(
            ic + (size_t)ent_slot[j * NT + i] * K + k_off);
#pragma unroll
        for (int v = 0; v < WB / 4; ++v) {
            const uint4 a = p[v];
            ap[i][4 * v + 0] = a.x; ap[i][4 * v + 1] = a.y;
            ap[i][4 * v + 2] = a.z; ap[i][4 * v + 3] = a.w;
        }
    }

    float acc[PASSES][NTT];
#pragma unroll
    for (int q = 0; q < PASSES; ++q)
#pragma unroll
        for (int i = 0; i < NTT; ++i) acc[q][i] = 0.0f;

    const int n0 = blockIdx.x * NR;
    using WT = typename WVec<WB>::T;
#define G2LOAD(nn) (*reinterpret_cast<const WT*>(                              \
    V + ebase + (size_t)(nn) * Kb + byte_off))
    constexpr int PFD = PF < PASSES ? PF : PASSES;
    WT wbuf[PFD];
#pragma unroll
    for (int p = 0; p < PFD; ++p) wbuf[p] = G2LOAD(n0 + p * RP + rsub);
#pragma unroll
    for (int q = 0; q < PASSES; ++q) {
        const WT wcur = wbuf[q % PFD];
        const int n = n0 + q * RP + rsub;
        if (q + PFD < PASSES) wbuf[q % PFD] = G2LOAD(n + PFD * RP);
        const float sc = e8m0_to_f32(S[sbase + (size_t)n * Ks + sc_off])
                         * FP4_POST_SCALE;
#pragma unroll
        for (int i = 0; i < NTT; ++i)
            acc[q][i] = fmaf(sc, dotWB<WB>(reinterpret_cast<const uint32_t*>(&wcur),
                                           ap[i]), acc[q][i]);
    }
#undef G2LOAD

#pragma unroll
    for (int q = 0; q < PASSES; ++q) {
        const int rr = q * RP + rsub;
#pragma unroll
        for (int i = 0; i < NTT; ++i) {
            float v = acc[q][i];
#pragma unroll
            for (int off = 16; off; off >>= 1) v += __shfl_xor(v, off, 32);
            if (lane == 0) red[(rr * NTT + i) * WPR + krow] = v;
        }
    }
    __syncthreads();
    // scatter: each (slot, column) is produced by exactly ONE workgroup, so this
    // is a plain store into the slot's own row -- no contention, no atomics.
    //
    // It used to atomicAdd gamma*v straight into the TOKEN row. That is a
    // correctness bug, not a style point: a token's topk experts then summed in
    // workgroup-completion order, and f32 addition is not associative, so the
    // result changed run to run. One flipped bf16 ULP in a logit flips an
    // argmax, which changes the token, which collapses MTP acceptance -- the
    // reported symptom was nondeterministic temp-0 output with throughput
    // intermittently halving. The per-slot rows are summed in fixed slot order
    // by moe_reduce_slots below, which is also what the matmul_ogs path does
    // (scatter, then moe_sum over topk).
    //
    // Padded slots carry gamma 0 and all alias the scratch row P, so they are
    // skipped rather than racing each other over a row nobody reads.
    for (int idx = tid; idx < NR * NTT; idx += 256) {
        const int r = idx / NTT, i = idx - r * NTT;
        float v = 0.0f;
#pragma unroll
        for (int w = 0; w < WPR; ++w) v += red[(r * NTT + i) * WPR + w];
        const float gam = ent_gam[j * NT + i];
        if (gam != 0.0f)
            out[(size_t)ent_slot[j * NT + i] * N + n0 + r] = gam * v;
    }
}

// Sum a token's topk slot rows in ASCENDING SLOT ORDER and convert to bf16.
// Fixed order is the whole point: same inputs -> bitwise same output, every run.
__global__ void moe_reduce_slots(const float* __restrict__ part,
                                 bf16_t* __restrict__ dst, int M, int N,
                                 int topk) {
    const int idx = blockIdx.x * 256 + threadIdx.x;
    if (idx >= M * N) return;
    const int t = idx / N, n = idx - t * N;
    float s = 0.0f;
    for (int sl = 0; sl < topk; ++sl)
        s += part[(size_t)(t * topk + sl) * N + n];
    dst[idx] = f32_to_bf16_rne(s);
}

extern "C" {

int ds4_moe_build_entries(const void* topk_ids, const void* topk_w, void* eidx,
                          void* ent_n, void* ent_tok, void* ent_slot,
                          void* ent_gam, int P, int topk, int ids_is_64,
                          void* stream) {
    hipStream_t s = (hipStream_t)stream;
    hipLaunchKernelGGL(moe_clear_entries, dim3((P * NT + 255) / 256), dim3(256),
                       0, s, (int*)eidx, (int*)ent_n, (int*)ent_tok,
                       (int*)ent_slot, (float*)ent_gam, P);
    if (ids_is_64)
        hipLaunchKernelGGL((moe_build_entries<int64_t>), dim3(1), dim3(256), 0, s,
                           (const int64_t*)topk_ids, (const float*)topk_w,
                           (int*)eidx, (int*)ent_n, (int*)ent_tok,
                           (int*)ent_slot, (float*)ent_gam, P, topk);
    else
        hipLaunchKernelGGL((moe_build_entries<int>), dim3(1), dim3(256), 0, s,
                           (const int*)topk_ids, (const float*)topk_w,
                           (int*)eidx, (int*)ent_n, (int*)ent_tok,
                           (int*)ent_slot, (float*)ent_gam, P, topk);
    return (int)hipGetLastError();
}

int ds4_moe_gemm1(const void* x, const void* V, const void* S, const void* eidx,
                  const void* ent_n, const void* ent_tok, const void* ent_slot,
                  void* ic, int P, int N, int K, float alpha, float limit,
                  void* stream) {
    // N is w1's full row count (gate||up); a workgroup makes NR1 ic columns
    if ((N % (2 * NR1)) || (K & 511)) return -1;
    dim3 grid((N / 2) / NR1, P), block(256);
    // a wave covers 32*WB1 bytes of a row, so WPR waves tile the row exactly
    const int wpr = (K >> 1) / (32 * WB1);
    if (wpr != 8 && wpr != 4 && wpr != 2 && wpr != 1) return -2;
    hipStream_t s = (hipStream_t)stream;
    // one launch per token count: each kernel keeps only the registers its own
    // NTT needs, and blocks whose entry belongs to another NTT exit after two
    // loads. That is what lets the buckets exist without a host sync.
#define G1(w, nt)                                                             \
    hipLaunchKernelGGL((moe_gemm1_kernel<NR1, w, nt, WB1>), grid, block, 0, s,\
                       (const bf16_t*)x, (const uint8_t*)V, (const uint8_t*)S,\
                       (const int*)eidx, (const int*)ent_n,                   \
                       (const int*)ent_tok,                                   \
                       (const int*)ent_slot, (bf16_t*)ic, N, K, alpha, limit);
#define G1W(nt) \
    if (wpr == 8) { G1(8, nt) } else if (wpr == 4) { G1(4, nt) } \
    else if (wpr == 2) { G1(2, nt) } else { G1(1, nt) }
    G1W(1) G1W(2)
#if NT >= 3
    G1W(3)
#endif
#if NT >= 4
    G1W(4)
#endif
#undef G1W
#undef G1
    return (int)hipGetLastError();
}

int ds4_moe_gemm2(const void* ic, const void* V, const void* S, const void* eidx,
                  const void* ent_n, const void* ent_tok, const void* ent_slot,
                  const void* ent_gam, void* out, int P, int N, int K,
                  void* stream) {
    if (N % NR2 || (K & 511)) return -1;
    dim3 grid(N / NR2, P), block(256);
    const int wpr = (K >> 1) / (32 * WB2);
    if (wpr != 8 && wpr != 4 && wpr != 2 && wpr != 1) return -2;
    hipStream_t s = (hipStream_t)stream;
#define G2(w, nt)                                                             \
    hipLaunchKernelGGL((moe_gemm2_kernel<NR2, w, nt, WB2>), grid, block, 0, s,\
                       (const bf16_t*)ic, (const uint8_t*)V,                  \
                       (const uint8_t*)S, (const int*)eidx,                   \
                       (const int*)ent_n, (const int*)ent_tok,                \
                       (const int*)ent_slot,                                  \
                       (const float*)ent_gam, (float*)out, N, K);
#define G2W(nt) \
    if (wpr == 8) { G2(8, nt) } else if (wpr == 4) { G2(4, nt) } \
    else if (wpr == 2) { G2(2, nt) } else { G2(1, nt) }
    G2W(1) G2W(2)
#if NT >= 3
    G2W(3)
#endif
#if NT >= 4
    G2W(4)
#endif
#undef G2W
#undef G2
    return (int)hipGetLastError();
}

// src is the [P, N] per-slot partial buffer gemm2 wrote; dst is [M, N] bf16.
// Signature changed from (src, dst, n) when the atomic scatter was replaced:
// the reduction over a token's topk slots happens HERE, in slot order, so the
// whole path is deterministic. Callers must pass M and topk, and must size the
// partial buffer P = M*topk rows (plus the scratch row the padding aliases).
int ds4_moe_finish(const void* src, void* dst, int M, int N, int topk,
                   void* stream) {
    const int n = M * N;
    hipLaunchKernelGGL(moe_reduce_slots, dim3((n + 255) / 256), dim3(256), 0,
                       (hipStream_t)stream, (const float*)src, (bf16_t*)dst,
                       M, N, topk);
    return (int)hipGetLastError();
}
}
