// ib_ar2.hip — 2-rank RDMA all-reduce over native InfiniBand (ConnectX-3).
//
// Port of tbv_ar2.hip (the USB4/RoCE ibverbs variant) to the mlx4 HCA:
// device selection by name prefix (DS4_IB_AR2_HCA, default mlx4), LID-based
// RC addressing (no GRH -- native IB with opensm-assigned LIDs), and
// path_mtu clamped to the port's active MTU (FDR link runs 2048, and mlx4
// rejects a 4096 request). Slots stay pinned-host: mlx4 has no dmabuf MR,
// and on unified memory the GPU reads pinned-host slots at DRAM speed
// anyway. Everything below the QP setup is byte-identical to tbv_ar2:
//
//   stream:  ONE fused kernel: [copy into send slot + doorbell + wait + add]
//   CPU:     progress thread polls the doorbell, posts the RDMA writes
//
// (tbv_ar2 enqueues three stream ops per round -- hipMemcpyAsync staging,
// doorbell kernel, wait+add kernel. On UMA the copy engine buys nothing, so
// this port folds all three into one launch; the in-kernel
// __threadfence_system + barrier before the doorbell store replaces the
// stream-ordering the separate launches provided.)
//
// The GPU wait+add kernel spins on the peer's flag (QP-ordered after its
// data) landing in pinned memory, then adds the recv slot DIRECTLY (UMA
// zero-copy) — no H2D copy, no separate add launch, no host wakeups.
//
// Send staging is double-buffered: the host returns before the NIC reads the
// buffer, so round N+1 must land in the other slot. Round N+2 reusing N's
// slot is safe: its copy is stream-ordered after wait_add(N+1), which only
// completes once the peer received round N+1, implying our round-N send
// finished long before (RC QP ordering).
//
// Build: hipcc -O2 --offload-arch=gfx1151 -shared -fPIC \
//          -o libib_ar2.so ib_ar2.hip -libverbs -lpthread
//
// Progress thread is adaptive: spins hot within 5ms of activity (decode
// bursts), sleeps 200us otherwise — no idle core burn (see rocm-idle-spin
// saga for why this matters on these boxes).

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <infiniband/verbs.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdint.h>
#include <time.h>

#define IBAR2_MAX_BYTES (1u << 20)
#define IBAR2_NSLOTS 2 /* both send and recv are double-buffered */
#define IBAR2_DB_SEQ_SHIFT 24
#define IBAR2_DB_NBYTES_MASK ((1u << IBAR2_DB_SEQ_SHIFT) - 1)

/* ------------------------------------------------------------------ */
/* transport state (mirrors v1)                                        */
/* ------------------------------------------------------------------ */
static struct ibv_context *ctx;
static struct ibv_pd *pd;
static struct ibv_cq *cq;
static struct ibv_qp *qp;
static struct ibv_mr *mr_send, *mr_recv;
static char *send_buf, *recv_buf;          /* host pointers (pinned)   */
static char *send_buf_dev, *recv_buf_dev;  /* device views of the same */
static volatile uint64_t *recv_flag;       /* host view, in recv region */
static uint64_t *recv_flag_dev;            /* device view               */
static uint64_t *send_flag;                /* scratch for the flag WR   */
static uint32_t peer_rkey_data, peer_rkey_flag;
static uint64_t peer_addr_data, peer_addr_flag;
static uint64_t host_seq;                  /* python-side round counter */
static int inited;

/* doorbell + error flags live in one small pinned block */
struct ibar2_shared {
    uint64_t doorbell;      /* (seq << 24) | nbytes, GPU release-store  */
    int kernel_err;         /* wait+add timeout latch (GPU writes)      */
    int thread_err;         /* progress-thread error latch              */
};
static struct ibar2_shared *shr;       /* host view   */
static struct ibar2_shared *shr_dev;   /* device view */

static pthread_t progress_tid;
static volatile int progress_exit;
static uint8_t ib_port;                    /* HCA port (DS4_IB_AR2_IB_PORT) */
static enum ibv_mtu path_mtu;              /* clamped to the port's active MTU */

extern "C" {
char *ibar2_send_ptr(void) { return send_buf; }
char *ibar2_recv_ptr(void) { return recv_buf; }
uint32_t ibar2_max_bytes(void) { return IBAR2_MAX_BYTES; }
uint32_t ibar2_nslots(void) { return IBAR2_NSLOTS; }
int ibar2_last_error(void) {
    int e = shr ? (shr->kernel_err | shr->thread_err) : 0;
    if (shr) { shr->kernel_err = 0; shr->thread_err = 0; }
    return e;
}
}

/* ------------------------------------------------------------------ */
/* TCP rendezvous (identical to v1)                                    */
/* ------------------------------------------------------------------ */
struct xchg { uint32_t qpn; uint32_t rkey_data; uint32_t rkey_flag;
              uint64_t addr_data; uint64_t addr_flag; uint16_t lid; };

static int tcp_rendezvous(int rank, const char *peer_ip, int port,
                          struct xchg *mine, struct xchg *theirs) {
    int fd = -1;
    if (rank == 0) {
        int lfd = socket(AF_INET, SOCK_STREAM, 0);
        int one = 1;
        setsockopt(lfd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
        struct sockaddr_in a = {0};
        a.sin_family = AF_INET; a.sin_port = htons(port);
        a.sin_addr.s_addr = INADDR_ANY;
        if (bind(lfd, (struct sockaddr*)&a, sizeof a) || listen(lfd, 1)) { perror("bind/listen"); return -1; }
        fd = accept(lfd, NULL, NULL);
        close(lfd);
    } else {
        for (int tries = 0; tries < 600; tries++) {
            fd = socket(AF_INET, SOCK_STREAM, 0);
            struct sockaddr_in a = {0};
            a.sin_family = AF_INET; a.sin_port = htons(port);
            inet_pton(AF_INET, peer_ip, &a.sin_addr);
            if (connect(fd, (struct sockaddr*)&a, sizeof a) == 0) break;
            close(fd); fd = -1;
            usleep(100000);
        }
        if (fd < 0) { fprintf(stderr, "ib_ar2: connect failed\n"); return -1; }
    }
    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
    if (write(fd, mine, sizeof *mine) != sizeof *mine) { perror("write"); return -1; }
    size_t got = 0;
    while (got < sizeof *theirs) {
        ssize_t r = read(fd, (char*)theirs + got, sizeof *theirs - got);
        if (r <= 0) { perror("read"); return -1; }
        got += r;
    }
    close(fd);
    return 0;
}

/* ------------------------------------------------------------------ */
/* progress thread: doorbell -> post RDMA writes; drains CQ            */
/* ------------------------------------------------------------------ */
static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + ts.tv_nsec;
}

static int post_round(uint64_t s, uint32_t nbytes) {
    uint32_t slot = (uint32_t)(s % IBAR2_NSLOTS);

    struct ibv_sge sge_d = { (uint64_t)(send_buf + (uint64_t)slot * IBAR2_MAX_BYTES),
                             nbytes, mr_send->lkey };
    struct ibv_send_wr wr_d = {}, *bad;
    wr_d.wr_id = 1;
    wr_d.sg_list = &sge_d; wr_d.num_sge = 1;
    wr_d.opcode = IBV_WR_RDMA_WRITE;
    wr_d.wr.rdma.remote_addr = peer_addr_data + (uint64_t)slot * IBAR2_MAX_BYTES;
    wr_d.wr.rdma.rkey = peer_rkey_data;

    *send_flag = s;
    struct ibv_sge sge_f = { (uint64_t)send_flag, 8, mr_recv->lkey };
    struct ibv_send_wr wr_f = {};
    wr_f.wr_id = 2;
    wr_f.sg_list = &sge_f; wr_f.num_sge = 1;
    wr_f.opcode = IBV_WR_RDMA_WRITE;
    wr_f.send_flags = IBV_SEND_SIGNALED;
    wr_f.wr.rdma.remote_addr = peer_addr_flag + (uint64_t)slot * 8;
    wr_f.wr.rdma.rkey = peer_rkey_flag;
    wr_d.next = &wr_f;

    if (ibv_post_send(qp, &wr_d, &bad)) {
        fprintf(stderr, "ib_ar2: post_send failed\n");
        return -1;
    }
    return 0;
}

static void *progress_main(void *unused) {
    (void)unused;
    uint64_t posted = 0;
    uint64_t last_act = now_ns();
    struct ibv_wc wc;

    while (!progress_exit) {
        uint64_t db = __atomic_load_n(&shr->doorbell, __ATOMIC_ACQUIRE);
        uint64_t s = db >> IBAR2_DB_SEQ_SHIFT;
        if (s > posted) {
            /* posts may lag multiple doorbells only if the GPU raced ahead
             * two rounds; replay each seq in order with the latest nbytes
             * (rounds are same-shape within a decode step; a mismatch only
             * over-sends within the slot, which is harmless). */
            uint32_t nbytes = (uint32_t)(db & IBAR2_DB_NBYTES_MASK);
            while (posted < s) {
                posted++;
                if (post_round(posted, nbytes)) { shr->thread_err = 1; break; }
            }
            last_act = now_ns();
        }
        int n = ibv_poll_cq(cq, 1, &wc);
        if (n < 0) { shr->thread_err = 1; }
        else if (n == 1) {
            if (wc.status != IBV_WC_SUCCESS) {
                fprintf(stderr, "ib_ar2: wc status %d\n", wc.status);
                shr->thread_err = 1;
            }
            last_act = now_ns();
        }
        if (now_ns() - last_act > 5000000ull) /* 5ms quiet -> back off */
            usleep(200);
    }
    return NULL;
}

/* ------------------------------------------------------------------ */
/* GPU kernels                                                         */
/* ------------------------------------------------------------------ */
/* One block, ONE launch per all-reduce (vs tbv_ar2's memcpy + doorbell +
 * wait_add = three stream ops, two of them kernel launches and one an SDMA
 * queue transition): all threads copy src into the pinned send slot (UMA --
 * a kernel writes host memory at DRAM speed, no copy engine needed), fence
 * system-wide, thread 0 rings the doorbell and spins on the peer flag, then
 * all threads add the recv slot into dst. Bail after ~2e9 polls (~ seconds)
 * so a dead peer degrades to an error, not a wedged GPU. */
template <typename T, typename ACC>
__global__ void ibar2_fused_kernel(T *__restrict__ dst,
                                   const T *__restrict__ src,
                                   T *__restrict__ send,
                                   const T *__restrict__ recv,
                                   uint32_t n,
                                   uint64_t *db, uint64_t dbval,
                                   uint64_t *flag, uint64_t seq,
                                   int *err) {
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x)
        send[i] = src[i];
    /* Publish every thread's slot writes system-wide BEFORE the doorbell:
     * the progress thread acquire-loads the doorbell and the NIC then reads
     * the slot, so the fence+barrier pair is what makes the release store
     * cover writes from all threads, not just thread 0's. */
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        __hip_atomic_store(db, dbval, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
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
extern "C" int ibar2_init(int rank, const char *peer_ip, int port) {
    if (inited) return 0;
    int num;
    const char *hca = getenv("DS4_IB_AR2_HCA");
    if (!hca || !*hca) hca = "mlx4";
    const char *portenv = getenv("DS4_IB_AR2_IB_PORT");
    ib_port = (uint8_t)(portenv ? atoi(portenv) : 1);
    struct ibv_device **devs = ibv_get_device_list(&num);
    if (!devs) { fprintf(stderr, "ib_ar2: no verbs devices\n"); return -1; }
    struct ibv_device *dev = NULL;
    for (int i = 0; i < num; i++)
        if (!strncmp(ibv_get_device_name(devs[i]), hca, strlen(hca))) { dev = devs[i]; break; }
    if (!dev) { fprintf(stderr, "ib_ar2: no %s* device\n", hca); ibv_free_device_list(devs); return -1; }
    ctx = ibv_open_device(dev);
    ibv_free_device_list(devs);
    if (!ctx) { fprintf(stderr, "ib_ar2: open_device failed\n"); return -1; }

    pd = ibv_alloc_pd(ctx);
    cq = ibv_create_cq(ctx, 64, NULL, NULL, 0);
    if (!pd || !cq) { fprintf(stderr, "ib_ar2: pd/cq failed\n"); return -1; }

    /* Pinned, GPU-visible staging. hipHostMalloc memory is regular user VA
     * (GTT-backed) so ibv_reg_mr's get_user_pages pins it like any malloc. */
    if (hipHostMalloc((void**)&send_buf, IBAR2_NSLOTS * IBAR2_MAX_BYTES, hipHostMallocDefault) != hipSuccess)
        { fprintf(stderr, "ib_ar2: hipHostMalloc send failed\n"); return -1; }
    if (hipHostMalloc((void**)&recv_buf, IBAR2_NSLOTS * IBAR2_MAX_BYTES + 4096, hipHostMallocDefault) != hipSuccess)
        { fprintf(stderr, "ib_ar2: hipHostMalloc recv failed\n"); return -1; }
    if (hipHostMalloc((void**)&shr, sizeof *shr, hipHostMallocDefault) != hipSuccess)
        { fprintf(stderr, "ib_ar2: hipHostMalloc shr failed\n"); return -1; }
    memset(send_buf, 0, IBAR2_NSLOTS * IBAR2_MAX_BYTES);
    memset(recv_buf, 0, IBAR2_NSLOTS * IBAR2_MAX_BYTES + 4096);
    memset(shr, 0, sizeof *shr);
    recv_flag = (volatile uint64_t *)(recv_buf + IBAR2_NSLOTS * IBAR2_MAX_BYTES);
    send_flag = (uint64_t *)(recv_buf + IBAR2_NSLOTS * IBAR2_MAX_BYTES + IBAR2_NSLOTS * 8);

    if (hipHostGetDevicePointer((void**)&send_buf_dev, send_buf, 0) != hipSuccess ||
        hipHostGetDevicePointer((void**)&recv_buf_dev, recv_buf, 0) != hipSuccess ||
        hipHostGetDevicePointer((void**)&shr_dev, (void*)shr, 0) != hipSuccess)
        { fprintf(stderr, "ib_ar2: hipHostGetDevicePointer failed\n"); return -1; }
    recv_flag_dev = (uint64_t *)(recv_buf_dev + IBAR2_NSLOTS * IBAR2_MAX_BYTES);

    mr_send = ibv_reg_mr(pd, send_buf, IBAR2_NSLOTS * IBAR2_MAX_BYTES,
                         IBV_ACCESS_LOCAL_WRITE);
    mr_recv = ibv_reg_mr(pd, recv_buf, IBAR2_NSLOTS * IBAR2_MAX_BYTES + 4096,
                         IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    if (!mr_send || !mr_recv) { fprintf(stderr, "ib_ar2: reg_mr failed (hipHostMalloc mem?)\n"); return -1; }

    struct ibv_qp_init_attr qia = {};
    qia.send_cq = cq; qia.recv_cq = cq;
    qia.qp_type = IBV_QPT_RC;
    qia.cap.max_send_wr = 32; qia.cap.max_recv_wr = 4;
    qia.cap.max_send_sge = 1; qia.cap.max_recv_sge = 1;
    qp = ibv_create_qp(pd, &qia);
    if (!qp) { fprintf(stderr, "ib_ar2: create_qp failed\n"); return -1; }

    struct ibv_port_attr pattr;
    if (ibv_query_port(ctx, ib_port, &pattr)) { fprintf(stderr, "ib_ar2: query_port failed\n"); return -1; }
    if (pattr.state != IBV_PORT_ACTIVE) {
        fprintf(stderr, "ib_ar2: port %u not ACTIVE (state %d) -- opensm running?\n", ib_port, pattr.state);
        return -1;
    }
    path_mtu = pattr.active_mtu;

    struct xchg mine = { qp->qp_num, mr_recv->rkey, mr_recv->rkey,
                         (uint64_t)recv_buf, (uint64_t)recv_flag, pattr.lid }, theirs;
    if (tcp_rendezvous(rank, peer_ip, port, &mine, &theirs)) return -1;
    peer_rkey_data = theirs.rkey_data; peer_rkey_flag = theirs.rkey_flag;
    peer_addr_data = theirs.addr_data; peer_addr_flag = theirs.addr_flag;

    struct ibv_qp_attr at = {};
    at.qp_state = IBV_QPS_INIT;
    at.pkey_index = 0; at.port_num = ib_port;
    at.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE;
    if (ibv_modify_qp(qp, &at, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT |
                      IBV_QP_ACCESS_FLAGS)) { fprintf(stderr, "ib_ar2: ->INIT failed\n"); return -1; }

    memset(&at, 0, sizeof at);
    at.qp_state = IBV_QPS_RTR;
    at.path_mtu = path_mtu;
    at.dest_qp_num = theirs.qpn;
    at.rq_psn = 0;
    at.max_dest_rd_atomic = 1;
    at.min_rnr_timer = 12;
    /* Native IB: plain LID addressing, no GRH. */
    at.ah_attr.is_global = 0;
    at.ah_attr.dlid = theirs.lid;
    at.ah_attr.sl = 0;
    at.ah_attr.src_path_bits = 0;
    at.ah_attr.port_num = ib_port;
    if (ibv_modify_qp(qp, &at, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU |
                      IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                      IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER)) {
        fprintf(stderr, "ib_ar2: ->RTR failed\n"); return -1;
    }

    memset(&at, 0, sizeof at);
    at.qp_state = IBV_QPS_RTS;
    at.sq_psn = 0;
    at.timeout = 14; at.retry_cnt = 7; at.rnr_retry = 7;
    at.max_rd_atomic = 1;
    if (ibv_modify_qp(qp, &at, IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                      IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC)) {
        fprintf(stderr, "ib_ar2: ->RTS failed\n"); return -1;
    }

    host_seq = 0;
    progress_exit = 0;
    if (pthread_create(&progress_tid, NULL, progress_main, NULL)) {
        fprintf(stderr, "ib_ar2: progress thread failed\n"); return -1;
    }
    inited = 1;
    fprintf(stderr, "ib_ar2: rank%d ready (qpn=%u peer_qpn=%u lid=%u peer_lid=%u mtu=%d)\n",
            rank, qp->qp_num, theirs.qpn, mine.lid, theirs.lid, 128 << path_mtu);
    return 0;
}

/* Enqueue one all-reduce on `stream`: dst = src + peer_data, fully stream-
 * ordered. dtype: 0=bf16 1=fp16 2=fp32.
 *
 * ONE fused kernel does [copy src -> send slot] -> [doorbell] -> [wait+add];
 * the in-kernel fence+barrier before the doorbell's system-scope release
 * store publishes the staged send data to the progress thread (which
 * acquire-loads it), so the NIC only reads the send slot after the copy has
 * landed in host memory. */
extern "C" int ibar2_enqueue(void *stream, void *dst, const void *src,
                            uint32_t nbytes, int dtype) {
    if (!inited || nbytes > IBAR2_MAX_BYTES) return -1;
    host_seq++;
    uint64_t s = host_seq;
    uint32_t slot = (uint32_t)(s % IBAR2_NSLOTS);
    hipStream_t hs = (hipStream_t)stream;

    char *send_slot = send_buf_dev + (uint64_t)slot * IBAR2_MAX_BYTES;
    const char *recv_slot = recv_buf_dev + (uint64_t)slot * IBAR2_MAX_BYTES;
    uint64_t *db = &shr_dev->doorbell;
    uint64_t dbval = (s << IBAR2_DB_SEQ_SHIFT) | nbytes;
    uint64_t *flag = recv_flag_dev + slot;
    int *err = &shr_dev->kernel_err;

    switch (dtype) {
    case 0: {
        uint32_t n = nbytes / 2;
        hipLaunchKernelGGL((ibar2_fused_kernel<__hip_bfloat16, float>),
                           dim3(1), dim3(1024), 0, hs,
                           (__hip_bfloat16*)dst, (const __hip_bfloat16*)src,
                           (__hip_bfloat16*)send_slot,
                           (const __hip_bfloat16*)recv_slot, n, db, dbval, flag, s, err);
        break;
    }
    case 1: {
        uint32_t n = nbytes / 2;
        hipLaunchKernelGGL((ibar2_fused_kernel<__half, float>),
                           dim3(1), dim3(1024), 0, hs,
                           (__half*)dst, (const __half*)src,
                           (__half*)send_slot,
                           (const __half*)recv_slot, n, db, dbval, flag, s, err);
        break;
    }
    case 2: {
        uint32_t n = nbytes / 4;
        hipLaunchKernelGGL((ibar2_fused_kernel<float, float>),
                           dim3(1), dim3(1024), 0, hs,
                           (float*)dst, (const float*)src,
                           (float*)send_slot,
                           (const float*)recv_slot, n, db, dbval, flag, s, err);
        break;
    }
    default:
        return -1;
    }
    return (hipGetLastError() == hipSuccess) ? 0 : -1;
}

extern "C" uint64_t ibar2_seq(void) { return host_seq; }
