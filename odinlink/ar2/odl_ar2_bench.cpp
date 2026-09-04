// odl_ar2_bench.cpp — standalone 2-process cross-box benchmark for odl_ar2.
//
// rank 0 on box1, rank 1 on box2 (rank1 listens on the TCP rendezvous port,
// rank0 connects to it — see odl_ar2.hip). Data goes over the OdinLink device
// given by argv; the TCP socket is only used as an init barrier.
//
//   usage: odl_ar2_bench <rank 0|1> <odl_dev_index> <rendezvous_ip> \
//                        [port=18541] [iters=2000] [warmup=200]
//
// Phases (both ranks execute the identical schedule; every all-reduce round
// blocks on the peer's message, so the ranks stay in lockstep):
//   1. correctness: bf16 @ 8 KiB and 48 KiB, fp32 @ 48 KiB, with
//      rank-dependent ramps; verifies dst == sum on both ranks.
//   2. timing: bf16 @ 8 KiB, 16 KiB, 48 KiB — warmup + iters rounds each,
//      per-op latency = host wall time around [enqueue + hipStreamSynchronize].
//
// Sizes: DeepSeek-V4-Flash decode all-reduces hidden states of 4096 bf16
// (8 KiB/token); the production decode collective is
// 48 KiB per op (6 tokens/step with the 3-stage MTP drafter — see the
// cuda_communicator hook comment in vllm-upstream.patch), so 48 KiB is the
// size to watch.
//
// Build (inside the vllm container):
//   hipcc -O2 --offload-arch=gfx1151 -o odl_ar2_bench odl_ar2.hip odl_ar2_bench.cpp \
//     -I$HOME/.cache/odinlink/lib/include -L$HOME/.cache/odinlink/build/lib \
//     -lodl_tb5 -lpthread

#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <vector>

extern "C" {
int odl2_init(int rank, int dev_index, const char *peer_ip, int port);
int odl2_enqueue(void *stream, void *dst, const void *src, uint32_t nbytes, int dtype);
int odl2_last_error(void);
uint32_t odl2_max_bytes(void);
uint64_t odl2_seq(void);
void odl2_shutdown(void);
}

#define HIP_CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); \
    exit(1); } } while (0)

/* bf16 <-> float, exact for the small integers used in the patterns */
static uint16_t f2bf(float f) { uint32_t u; memcpy(&u, &f, 4); return (uint16_t)(u >> 16); }
static float bf2f(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; }

static double now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

static double pct(std::vector<double> &v, double p) {
    size_t i = (size_t)(p / 100.0 * (v.size() - 1) + 0.5);
    return v[std::min(i, v.size() - 1)];
}

/* pattern: values exact in bf16 (< 256), sum depends on i and both ranks */
static float pat(int rank, uint32_t i) { return (float)((i % 97) + 1 + rank * 3); }
static float pat_sum(uint32_t i) { return pat(0, i) + pat(1, i); }

static int check_bf16(int rank, hipStream_t hs, void *d_src, void *d_dst, uint32_t nbytes) {
    uint32_t n = nbytes / 2;
    std::vector<uint16_t> h(n), out(n);
    for (uint32_t i = 0; i < n; i++) h[i] = f2bf(pat(rank, i));
    HIP_CHECK(hipMemcpy(d_src, h.data(), nbytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemsetAsync(d_dst, 0, nbytes, hs));
    if (odl2_enqueue(hs, d_dst, d_src, nbytes, /*bf16*/0)) { fprintf(stderr, "enqueue failed\n"); return -1; }
    HIP_CHECK(hipStreamSynchronize(hs));
    if (odl2_last_error()) { fprintf(stderr, "latched transport error\n"); return -1; }
    HIP_CHECK(hipMemcpy(out.data(), d_dst, nbytes, hipMemcpyDeviceToHost));
    int bad = 0;
    for (uint32_t i = 0; i < n; i++) {
        if (bf2f(out[i]) != pat_sum(i)) {
            if (bad < 5)
                fprintf(stderr, "  bf16 MISMATCH i=%u got=%f want=%f\n", i, bf2f(out[i]), pat_sum(i));
            bad++;
        }
    }
    printf("correctness bf16 %6u B: %s (%d/%u bad)\n", nbytes, bad ? "FAIL" : "PASS", bad, n);
    return bad ? -1 : 0;
}

static int check_fp32(int rank, hipStream_t hs, void *d_src, void *d_dst, uint32_t nbytes) {
    uint32_t n = nbytes / 4;
    std::vector<float> h(n), out(n);
    for (uint32_t i = 0; i < n; i++) h[i] = pat(rank, i) * 1.5f;
    HIP_CHECK(hipMemcpy(d_src, h.data(), nbytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemsetAsync(d_dst, 0, nbytes, hs));
    if (odl2_enqueue(hs, d_dst, d_src, nbytes, /*fp32*/2)) { fprintf(stderr, "enqueue failed\n"); return -1; }
    HIP_CHECK(hipStreamSynchronize(hs));
    if (odl2_last_error()) { fprintf(stderr, "latched transport error\n"); return -1; }
    HIP_CHECK(hipMemcpy(out.data(), d_dst, nbytes, hipMemcpyDeviceToHost));
    int bad = 0;
    for (uint32_t i = 0; i < n; i++) {
        if (out[i] != pat_sum(i) * 1.5f) {
            if (bad < 5)
                fprintf(stderr, "  fp32 MISMATCH i=%u got=%f want=%f\n", i, out[i], pat_sum(i) * 1.5f);
            bad++;
        }
    }
    printf("correctness fp32 %6u B: %s (%d/%u bad)\n", nbytes, bad ? "FAIL" : "PASS", bad, n);
    return bad ? -1 : 0;
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <rank 0|1> <odl_dev_index> <rendezvous_ip> [port] [iters] [warmup]\n", argv[0]);
        return 2;
    }
    int rank = atoi(argv[1]);
    int dev = atoi(argv[2]);
    const char *ip = argv[3];
    int port = argc > 4 ? atoi(argv[4]) : 18541;
    int iters = argc > 5 ? atoi(argv[5]) : 2000;
    int warmup = argc > 6 ? atoi(argv[6]) : 200;
    setvbuf(stdout, NULL, _IOLBF, 0);

    HIP_CHECK(hipSetDevice(0));
    hipStream_t hs;
    HIP_CHECK(hipStreamCreate(&hs));

    if (odl2_init(rank, dev, ip, port)) { fprintf(stderr, "odl2_init failed\n"); return 1; }

    uint32_t maxb = odl2_max_bytes();
    void *d_src, *d_dst;
    HIP_CHECK(hipMalloc(&d_src, maxb));
    HIP_CHECK(hipMalloc(&d_dst, maxb));

    /* ---- correctness first ---- */
    int cfail = 0;
    cfail |= check_bf16(rank, hs, d_src, d_dst, 8192);
    cfail |= check_bf16(rank, hs, d_src, d_dst, 49152);
    cfail |= check_fp32(rank, hs, d_src, d_dst, 49152);
    if (cfail) { fprintf(stderr, "CORRECTNESS FAILED — not timing a wrong all-reduce\n"); return 1; }

    /* ---- timing ---- */
    const uint32_t sizes[] = { 8192, 16384, 49152 };
    /* realistic decode payload in the send buffer (nonzero bf16) */
    {
        std::vector<uint16_t> h(maxb / 2);
        for (size_t i = 0; i < h.size(); i++) h[i] = f2bf(pat(rank, (uint32_t)i));
        HIP_CHECK(hipMemcpy(d_src, h.data(), maxb, hipMemcpyHostToDevice));
    }

    for (uint32_t nbytes : sizes) {
        std::vector<double> lat;
        lat.reserve(iters);
        for (int i = 0; i < warmup + iters; i++) {
            double t0 = now_us();
            if (odl2_enqueue(hs, d_dst, d_src, nbytes, /*bf16*/0)) {
                fprintf(stderr, "enqueue failed at iter %d\n", i); return 1;
            }
            HIP_CHECK(hipStreamSynchronize(hs));
            double t1 = now_us();
            if (i >= warmup) lat.push_back(t1 - t0);
        }
        if (odl2_last_error()) { fprintf(stderr, "latched transport error after %u B run\n", nbytes); return 1; }
        std::sort(lat.begin(), lat.end());
        double sum = 0;
        for (double v : lat) sum += v;
        printf("RESULT rank=%d dev=%d size=%uB iters=%d "
               "min=%.1f p50=%.1f p90=%.1f p99=%.1f max=%.1f mean=%.1f us\n",
               rank, dev, nbytes, iters,
               lat.front(), pct(lat, 50), pct(lat, 90), pct(lat, 99),
               lat.back(), sum / lat.size());
    }

    printf("done: %llu rounds total, no latched errors\n",
           (unsigned long long)odl2_seq());
    odl2_shutdown();
    return 0;
}
