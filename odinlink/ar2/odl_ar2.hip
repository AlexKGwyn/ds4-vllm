// odl_ar2.hip — 2-rank all-reduce over the OdinLink (odl_tb5) userspace API.
//
// Shape of the collective:
//
//   stream:  [D2H copy into send slot] -> [doorbell kernel] -> [wait+add kernel]
//   CPU:     progress thread polls the doorbell, moves bytes over OdinLink
//
// The GPU wait+add kernel spins on a per-slot flag in pinned memory, then adds
// the recv slot DIRECTLY (UMA zero-copy) — no H2D copy, no host wakeups.
// dst = src + peer(src), elementwise SUM; dtype 0=bf16 1=fp16 2=fp32
// (bf16/fp16 accumulate in fp32).
//
// Transport = OdinLink streams in LATENCY mode, replacing the RC QP RDMA
// writes:
//  * send: one ODL_TB5_IOCTL_STREAM_SEND per round. For <=64 KiB messages the
//    kernel takes the latency path: each 4024-byte fragment is copied straight
//    from our pinned send slot into a DMA pool frame and pushed to the NHI TX
//    ring inside the ioctl, which returns WITHOUT waiting for TX completion
//    (completion recycles pool slots via callback). A 48 KiB decode message is
//    13 frames submitted by a single syscall.
//  * recv: nonblocking ODL_TB5_IOCTL_STREAM_RECV (O_NONBLOCK on the fd) polled
//    by the progress thread; the kernel reassembles fragments (frag_idx-checked)
//    and the ioctl copies the complete message directly into the GPU-visible
//    pinned recv slot. The progress thread then release-stores the slot flag
//    that the wait+add kernel is spinning on.
//
// Why streams and not the legacy double-buffer SEND/RECV ioctls: the legacy
// path splits at exactly ODL_TB5_FRAME_SIZE (4096) per frame, and a frame of
// exactly ring-buffer size does not fit the RX slot once TB framing/CRC
// overhead is added — the NHI silently drops it (see odl_tb5_ioctl.h, the
// FRAME_TAIL_RESERVE comment). It also drain-waits the previous TX inside
// every submit and requires buffer-swap choreography that fights the stream
// demux. Streams fragment at 4024 (safe), submit without blocking, preserve
// order, and detect fragment loss. The dmabuf path blocks until TX complete
// and re-attaches the dmabuf per call — worse for latency.
//
// Ordering comes from the stream itself rather than a flag-after-data write:
// a stream delivers complete messages in order, so message k
// on the stream IS round k, and the flag store happens locally after the
// message body has been copied into the slot.
//
// Slot-reuse safety: the peer only sends round
// N+2 after its wait_add(N+1) completed, which requires our N+1 arrived, which
// (stream order on our side) requires our wait_add(N) finished reading recv
// slot N — so an arriving message never overwrites a slot the GPU still reads.
// Send slots are free for reuse as soon as stream_send returns (the ioctl
// copies the payload into kernel DMA frames synchronously).
//
// Rendezvous: a TCP barrier in which rank1 LISTENS and rank0 CONNECTS
// (rank0 may live inside a podman container without inbound port mapping;
// outbound works everywhere).
//
// Build (inside the vllm container, which has hipcc):
//   hipcc -O2 --offload-arch=gfx1151 -shared -fPIC -o libodl_ar2.so odl_ar2.hip \
//     -I$HOME/.cache/odinlink/lib/include -L$HOME/.cache/odinlink/build/lib \
//     -lodl_tb5 -lpthread
//
// Progress thread is adaptive: spins hot within 5ms of activity (decode
// bursts), sleeps 200us otherwise (the rocm-idle-spin policy).

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <odl_tb5/odl_tb5.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdint.h>
#include <time.h>

#define ODL2_MAX_BYTES (1u << 20)
#define ODL2_NSLOTS 2 /* both send and recv are double-buffered */
#define ODL2_DB_SEQ_SHIFT 24
#define ODL2_DB_NBYTES_MASK ((1u << ODL2_DB_SEQ_SHIFT) - 1)
/* fixed stream ids, away from ids other OdinLink users pick (1..N) */
#define ODL2_SID_RANK0 11
#define ODL2_SID_RANK1 12
#define ODL2_PING_MAGIC 0x4f444c32u /* "ODL2" */

/* ------------------------------------------------------------------ */
/* transport state                                                     */
/* ------------------------------------------------------------------ */
static odl_tb5_t H;
static uint8_t my_sid, peer_sid;
static char *send_buf, *recv_buf;          /* host pointers (pinned)   */
static char *send_buf_dev, *recv_buf_dev;  /* device views of the same */
static volatile uint64_t *recv_flag;       /* host view, after recv slots */
static uint64_t *recv_flag_dev;            /* device view                 */
static uint64_t host_seq;                  /* caller-side round counter */
static int inited;

/* doorbell + error flags live in one small pinned block */
struct odl2_shared {
    uint64_t doorbell;      /* (seq << 24) | nbytes, GPU release-store  */
    int kernel_err;         /* wait+add timeout latch (GPU writes)      */
    int thread_err;         /* progress-thread error latch              */
};
static struct odl2_shared *shr;       /* host view   */
static struct odl2_shared *shr_dev;   /* device view */

static pthread_t progress_tid;
static volatile int progress_exit;

extern "C" {
char *odl2_send_ptr(void) { return send_buf; }
char *odl2_recv_ptr(void) { return recv_buf; }
uint32_t odl2_max_bytes(void) { return ODL2_MAX_BYTES; }
uint32_t odl2_nslots(void) { return ODL2_NSLOTS; }
int odl2_last_error(void) {
    int e = shr ? (shr->kernel_err | shr->thread_err) : 0;
    if (shr) { shr->kernel_err = 0; shr->thread_err = 0; }
    return e;
}
}

/* ------------------------------------------------------------------ */
/* TCP rendezvous / barrier (rank1 listens, rank0 connects)            */
/* ------------------------------------------------------------------ */
struct xchg { uint32_t magic; uint32_t rank; uint32_t max_bytes; uint32_t nslots; };

static int rdv_fd = -1;

static int tcp_connect_rdv(int rank, const char *peer_ip, int port) {
    int fd = -1;
    if (rank == 1) {
        int lfd = socket(AF_INET, SOCK_STREAM, 0);
        int one = 1;
        setsockopt(lfd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
        struct sockaddr_in a = {};
        a.sin_family = AF_INET; a.sin_port = htons(port);
        a.sin_addr.s_addr = INADDR_ANY;
        if (bind(lfd, (struct sockaddr*)&a, sizeof a) || listen(lfd, 1)) { perror("odl_ar2: bind/listen"); return -1; }
        fd = accept(lfd, NULL, NULL);
        close(lfd);
        if (fd < 0) { perror("odl_ar2: accept"); return -1; }
    } else {
        for (int tries = 0; tries < 600; tries++) {
            fd = socket(AF_INET, SOCK_STREAM, 0);
            struct sockaddr_in a = {};
            a.sin_family = AF_INET; a.sin_port = htons(port);
            inet_pton(AF_INET, peer_ip, &a.sin_addr);
            if (connect(fd, (struct sockaddr*)&a, sizeof a) == 0) break;
            close(fd); fd = -1;
            usleep(100000);
        }
        if (fd < 0) { fprintf(stderr, "odl_ar2: connect failed\n"); return -1; }
    }
    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
    rdv_fd = fd;
    return 0;
}

/* full-duplex exchange of one struct over the rendezvous socket = barrier */
static int tcp_barrier(struct xchg *mine, struct xchg *theirs) {
    if (write(rdv_fd, mine, sizeof *mine) != (ssize_t)sizeof *mine) { perror("odl_ar2: barrier write"); return -1; }
    size_t got = 0;
    while (got < sizeof *theirs) {
        ssize_t r = read(rdv_fd, (char*)theirs + got, sizeof *theirs - got);
        if (r <= 0) { perror("odl_ar2: barrier read"); return -1; }
        got += r;
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* progress thread: doorbell -> stream_send; poll stream_recv          */
/* ------------------------------------------------------------------ */
static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + ts.tv_nsec;
}

/* one round's TX: single ioctl, kernel fragments + submits, no TX wait */
static int post_round(uint64_t s, uint32_t nbytes) {
    uint32_t slot = (uint32_t)(s % ODL2_NSLOTS);
    const char *src = send_buf + (uint64_t)slot * ODL2_MAX_BYTES;
    uint64_t t0 = now_ns();
    for (;;) {
        int ret = odl_tb5_stream_send(H, my_sid, peer_sid, src, nbytes);
        if (ret == 0) return 0;
        if (ret != -EAGAIN) { /* EAGAIN = frame pool low (O_NONBLOCK) */
            fprintf(stderr, "odl_ar2: stream_send failed: %d\n", ret);
            return -1;
        }
        if (now_ns() - t0 > 2000000000ull) {
            fprintf(stderr, "odl_ar2: stream_send starved (frame pool)\n");
            return -1;
        }
    }
}

/* nonblocking poll for the next in-order message; returns 1 if a round
 * landed, 0 if none pending, -1 on error */
static int poll_recv(uint64_t next_seq) {
    uint32_t slot = (uint32_t)(next_seq % ODL2_NSLOTS);
    char *dst = recv_buf + (uint64_t)slot * ODL2_MAX_BYTES;
    uint8_t src_id = 0;
    uint32_t actual = 0;
    int ret = odl_tb5_stream_recv(H, my_sid, dst, ODL2_MAX_BYTES, &src_id, &actual);
    if (ret == -EAGAIN) return 0;
    if (ret < 0) {
        fprintf(stderr, "odl_ar2: stream_recv failed: %d\n", ret);
        return -1;
    }
    if (src_id != peer_sid) {
        fprintf(stderr, "odl_ar2: unexpected src stream %u\n", src_id);
        return -1;
    }
    /* publish: data is in the slot before the flag store (release) */
    __atomic_store_n((uint64_t *)&recv_flag[slot], next_seq, __ATOMIC_RELEASE);
    return 1;
}

static void *progress_main(void *unused) {
    (void)unused;
    uint64_t posted = 0;
    uint64_t rcvd = 0;
    uint64_t last_act = now_ns();

    while (!progress_exit) {
        uint64_t db = __atomic_load_n(&shr->doorbell, __ATOMIC_ACQUIRE);
        uint64_t s = db >> ODL2_DB_SEQ_SHIFT;
        if (s > posted) {
            /* replay each seq in order with the latest nbytes (rounds are
             * same-shape within a decode step; a mismatch only over-sends
             * within the slot, which is harmless) */
            uint32_t nbytes = (uint32_t)(db & ODL2_DB_NBYTES_MASK);
            while (posted < s) {
                posted++;
                if (post_round(posted, nbytes)) { shr->thread_err = 1; break; }
            }
            last_act = now_ns();
        }
        int r = poll_recv(rcvd + 1);
        if (r > 0) { rcvd++; last_act = now_ns(); }
        else if (r < 0) { shr->thread_err = 1; }
        if (now_ns() - last_act > 5000000ull) /* 5ms quiet -> back off */
            usleep(200);
    }
    return NULL;
}

/* ------------------------------------------------------------------ */
/* GPU kernels                                                        */
/* ------------------------------------------------------------------ */
__global__ void odl2_doorbell_kernel(uint64_t *db, uint64_t val) {
    if (threadIdx.x == 0)
        __hip_atomic_store(db, val, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}

/* One block: thread 0 spins on the peer flag, then all threads add the
 * pinned recv slot into dst. Bail after ~2e9 polls (~ seconds) so a dead
 * peer degrades to an error, not a wedged GPU. */
template <typename T, typename ACC>
__global__ void odl2_wait_add_kernel(T *__restrict__ dst,
                                     const T *__restrict__ src,
                                     const T *__restrict__ recv,
                                     uint32_t n,
                                     uint64_t *flag, uint64_t seq,
                                     int *err) {
    if (threadIdx.x == 0) {
        long iters = 0;
        for (;;) {
            uint64_t v = __hip_atomic_load(flag, __ATOMIC_ACQUIRE,
                                           __HIP_MEMORY_SCOPE_SYSTEM);
            if (v >= seq) break;
            if (++iters > (2L << 30)) { *err = 1; break; }
            __builtin_amdgcn_s_sleep(8);
        }
    }
    __syncthreads();
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        dst[i] = (T)((ACC)src[i] + (ACC)recv[i]);
    }
}

/* ------------------------------------------------------------------ */
/* public API                                                          */
/* ------------------------------------------------------------------ */
extern "C" int odl2_init(int rank, int dev_index, const char *peer_ip, int port) {
    if (inited) return 0;
    if (rank != 0 && rank != 1) { fprintf(stderr, "odl_ar2: rank must be 0/1\n"); return -1; }

    int ret = odl_tb5_open(&H, dev_index);
    if (ret < 0) { fprintf(stderr, "odl_ar2: odl_tb5_open(%d) failed: %d\n", dev_index, ret); return -1; }
    ret = odl_tb5_wait_peer(H, 15000);
    if (ret < 0) { fprintf(stderr, "odl_ar2: wait_peer failed: %d (link not ready?)\n", ret); return -1; }

    uint8_t want = rank == 0 ? ODL2_SID_RANK0 : ODL2_SID_RANK1;
    peer_sid = rank == 0 ? ODL2_SID_RANK1 : ODL2_SID_RANK0;
    ret = odl_tb5_stream_open(H, want, &my_sid);
    if (ret < 0 || my_sid != want) { fprintf(stderr, "odl_ar2: stream_open(%u) failed: %d\n", want, ret); return -1; }

    /* nonblocking fd: stream_recv returns -EAGAIN when no complete message,
     * so the progress thread can hot-poll and skip the kernel wake path */
    int fd = odl_tb5_get_fd(H);
    if (fcntl(fd, F_SETFL, fcntl(fd, F_GETFL, 0) | O_NONBLOCK) < 0) {
        perror("odl_ar2: fcntl O_NONBLOCK"); return -1;
    }

    /* Pinned, GPU-visible staging */
    if (hipHostMalloc((void**)&send_buf, ODL2_NSLOTS * ODL2_MAX_BYTES, hipHostMallocDefault) != hipSuccess)
        { fprintf(stderr, "odl_ar2: hipHostMalloc send failed\n"); return -1; }
    if (hipHostMalloc((void**)&recv_buf, ODL2_NSLOTS * ODL2_MAX_BYTES + 4096, hipHostMallocDefault) != hipSuccess)
        { fprintf(stderr, "odl_ar2: hipHostMalloc recv failed\n"); return -1; }
    if (hipHostMalloc((void**)&shr, sizeof *shr, hipHostMallocDefault) != hipSuccess)
        { fprintf(stderr, "odl_ar2: hipHostMalloc shr failed\n"); return -1; }
    memset(send_buf, 0, ODL2_NSLOTS * ODL2_MAX_BYTES);
    memset(recv_buf, 0, ODL2_NSLOTS * ODL2_MAX_BYTES + 4096);
    memset(shr, 0, sizeof *shr);
    recv_flag = (volatile uint64_t *)(recv_buf + ODL2_NSLOTS * ODL2_MAX_BYTES);

    if (hipHostGetDevicePointer((void**)&send_buf_dev, send_buf, 0) != hipSuccess ||
        hipHostGetDevicePointer((void**)&recv_buf_dev, recv_buf, 0) != hipSuccess ||
        hipHostGetDevicePointer((void**)&shr_dev, (void*)shr, 0) != hipSuccess)
        { fprintf(stderr, "odl_ar2: hipHostGetDevicePointer failed\n"); return -1; }
    recv_flag_dev = (uint64_t *)(recv_buf_dev + ODL2_NSLOTS * ODL2_MAX_BYTES);

    /* barrier 1: both streams are open before anyone sends */
    struct xchg mine = { ODL2_PING_MAGIC, (uint32_t)rank, ODL2_MAX_BYTES, ODL2_NSLOTS }, theirs;
    if (tcp_connect_rdv(rank, peer_ip, port)) return -1;
    if (tcp_barrier(&mine, &theirs)) return -1;
    if (theirs.magic != ODL2_PING_MAGIC || theirs.rank != (uint32_t)(1 - rank) ||
        theirs.max_bytes != ODL2_MAX_BYTES || theirs.nslots != ODL2_NSLOTS) {
        fprintf(stderr, "odl_ar2: rendezvous mismatch\n"); return -1;
    }

    /* stream self-test: one small message each way over the odl link */
    uint64_t ping = ((uint64_t)ODL2_PING_MAGIC << 32) | (uint32_t)rank;
    if (odl_tb5_stream_send(H, my_sid, peer_sid, &ping, sizeof ping) != 0) {
        fprintf(stderr, "odl_ar2: self-test send failed\n"); return -1;
    }
    uint64_t pong = 0; uint8_t sidr = 0; uint32_t alen = 0;
    uint64_t t0 = now_ns();
    for (;;) {
        int r = odl_tb5_stream_recv(H, my_sid, &pong, sizeof pong, &sidr, &alen);
        if (r == 0) break;
        if (r != -EAGAIN) { fprintf(stderr, "odl_ar2: self-test recv failed: %d\n", r); return -1; }
        if (now_ns() - t0 > 5000000000ull) {
            fprintf(stderr, "odl_ar2: self-test timeout (no odl traffic from peer)\n"); return -1;
        }
    }
    if (alen != sizeof pong || sidr != peer_sid ||
        pong != (((uint64_t)ODL2_PING_MAGIC << 32) | (uint32_t)(1 - rank))) {
        fprintf(stderr, "odl_ar2: self-test payload mismatch\n"); return -1;
    }

    /* barrier 2: both self-tests done, stream queues empty -> data phase */
    if (tcp_barrier(&mine, &theirs)) return -1;
    close(rdv_fd); rdv_fd = -1;

    host_seq = 0;
    progress_exit = 0;
    if (pthread_create(&progress_tid, NULL, progress_main, NULL)) {
        fprintf(stderr, "odl_ar2: progress thread failed\n"); return -1;
    }
    inited = 1;
    fprintf(stderr, "odl_ar2: rank%d ready (dev=%d sid=%u peer_sid=%u)\n",
            rank, dev_index, my_sid, peer_sid);
    return 0;
}

/* Enqueue one all-reduce on `stream`: dst = src + peer_data, fully stream-
 * ordered. dtype: 0=bf16 1=fp16 2=fp32.
 *
 * Order on the stream: [copy src -> send slot] -> [doorbell] -> [wait+add].
 * The doorbell's system-scope release store publishes the staged send data to
 * the progress thread (acquire-load), so stream_send's copy_from_user only
 * reads the send slot after the copy has landed in host memory. */
extern "C" int odl2_enqueue(void *stream, void *dst, const void *src,
                            uint32_t nbytes, int dtype) {
    if (!inited || nbytes > ODL2_MAX_BYTES) return -1;
    host_seq++;
    uint64_t s = host_seq;
    uint32_t slot = (uint32_t)(s % ODL2_NSLOTS);
    hipStream_t hs = (hipStream_t)stream;

    char *send_slot = send_buf_dev + (uint64_t)slot * ODL2_MAX_BYTES;
    if (hipMemcpyAsync(send_slot, src, nbytes, hipMemcpyDeviceToDevice, hs) != hipSuccess)
        return -1;

    uint64_t db = (s << ODL2_DB_SEQ_SHIFT) | nbytes;
    hipLaunchKernelGGL(odl2_doorbell_kernel, dim3(1), dim3(1), 0, hs,
                       &shr_dev->doorbell, db);

    const char *recv_slot = recv_buf_dev + (uint64_t)slot * ODL2_MAX_BYTES;
    uint64_t *flag = recv_flag_dev + slot;
    int *err = &shr_dev->kernel_err;

    switch (dtype) {
    case 0: {
        uint32_t n = nbytes / 2;
        hipLaunchKernelGGL((odl2_wait_add_kernel<__hip_bfloat16, float>),
                           dim3(1), dim3(1024), 0, hs,
                           (__hip_bfloat16*)dst, (const __hip_bfloat16*)src,
                           (const __hip_bfloat16*)recv_slot, n, flag, s, err);
        break;
    }
    case 1: {
        uint32_t n = nbytes / 2;
        hipLaunchKernelGGL((odl2_wait_add_kernel<__half, float>),
                           dim3(1), dim3(1024), 0, hs,
                           (__half*)dst, (const __half*)src,
                           (const __half*)recv_slot, n, flag, s, err);
        break;
    }
    case 2: {
        uint32_t n = nbytes / 4;
        hipLaunchKernelGGL((odl2_wait_add_kernel<float, float>),
                           dim3(1), dim3(1024), 0, hs,
                           (float*)dst, (const float*)src,
                           (const float*)recv_slot, n, flag, s, err);
        break;
    }
    default:
        return -1;
    }
    return (hipGetLastError() == hipSuccess) ? 0 : -1;
}

extern "C" uint64_t odl2_seq(void) { return host_seq; }

/* Tear down the progress thread and close the odl device. Safe to call once
 * all enqueued work has completed (e.g. after hipStreamSynchronize). */
extern "C" void odl2_shutdown(void) {
    if (!inited) return;
    progress_exit = 1;
    pthread_join(progress_tid, NULL);
    odl_tb5_stream_close(H, my_sid);
    odl_tb5_close(H);
    inited = 0;
}
