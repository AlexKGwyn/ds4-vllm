/* odl_pingpong.c — RTT latency benchmark over the OdinLink stream layer.
 *
 * Framing mirrors odl_stress.c: each message = [4-byte size header] +
 * payload in <=ODL_CHUNK single-frame sends on one stream.
 *
 *   server: ./odl_pingpong server <dev> <msg_bytes> <iters> [warmup]
 *           echoes every received message back; exits after warmup+iters echoes.
 *   client: ./odl_pingpong client <dev> <msg_bytes> <iters> [warmup]
 *           sends message, blocks for echo, records RTT (CLOCK_MONOTONIC).
 *           Warmup iterations are excluded from stats.
 *
 * Server recv stream id = 1, client recv stream id = 2.
 * Prints min/p50/p90/p99/max RTT in microseconds and writes raw RTTs to
 * odl_pingpong_dev<dev>_<msg>B.txt in the current directory.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <time.h>
#include <odl_tb5/odl_tb5.h>

#define ODL_CHUNK 4000
#define SRV_ID 1
#define CLI_ID 2

static odl_tb5_t H;

static double now_us(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

/* send one framed message: 4-byte size header + chunked payload */
static int msg_send(uint8_t sid, uint8_t dst, const uint8_t *buf, uint32_t sz)
{
	uint32_t hdr = sz;
	int ret = odl_tb5_stream_send(H, sid, dst, &hdr, sizeof(hdr));
	uint32_t off = 0;
	while (ret >= 0 && off < sz) {
		uint32_t n = sz - off; if (n > ODL_CHUNK) n = ODL_CHUNK;
		ret = odl_tb5_stream_send(H, sid, dst, buf + off, n);
		off += n;
	}
	return ret < 0 ? ret : 0;
}

/* recv one framed message into buf (cap bytes); returns size or <0 */
static int msg_recv(uint8_t sid, uint8_t *buf, uint32_t cap)
{
	uint32_t total = 0, actual = 0;
	uint8_t src = 0;
	int ret = odl_tb5_stream_recv(H, sid, &total, sizeof(total), &src, &actual);
	if (ret < 0) return ret;
	if (actual != sizeof(total)) { fprintf(stderr, "short header (%u)\n", actual); return -1000; }
	if (total > cap) { fprintf(stderr, "oversize msg %u > cap %u\n", total, cap); return -1001; }
	uint32_t off = 0;
	while (off < total) {
		uint32_t n = total - off; if (n > ODL_CHUNK) n = ODL_CHUNK;
		ret = odl_tb5_stream_recv(H, sid, buf + off, n, &src, &actual);
		if (ret < 0) return ret;
		if (actual == 0) { fprintf(stderr, "zero-length chunk at %u/%u\n", off, total); return -1002; }
		off += actual;
	}
	return (int)total;
}

static int cmp_d(const void *a, const void *b)
{
	double x = *(const double *)a, y = *(const double *)b;
	return (x > y) - (x < y);
}

static double pct(const double *v, int n, double p)
{
	int i = (int)(p / 100.0 * (n - 1) + 0.5);
	if (i < 0) i = 0;
	if (i >= n) i = n - 1;
	return v[i];
}

int main(int argc, char **argv)
{
	if (argc < 5) {
		fprintf(stderr, "usage: %s server|client <dev> <msg_bytes> <iters> [warmup]\n", argv[0]);
		return 2;
	}
	int server = !strcmp(argv[1], "server");
	int dev = atoi(argv[2]);
	uint32_t msg = (uint32_t)strtoul(argv[3], NULL, 10);
	int iters = atoi(argv[4]);
	int warmup = argc > 5 ? atoi(argv[5]) : 50;
	int total = iters + warmup;
	if (msg < 4) msg = 4;
	setvbuf(stdout, NULL, _IOLBF, 0);

	if (odl_tb5_open(&H, dev) < 0) { perror("odl_tb5_open"); return 1; }
	if (odl_tb5_wait_peer(H, 15000) < 0) { fprintf(stderr, "wait_peer timed out\n"); return 1; }

	uint8_t sid;
	if (odl_tb5_stream_open(H, server ? SRV_ID : CLI_ID, &sid) < 0) {
		fprintf(stderr, "stream_open failed\n"); return 1;
	}
	printf("%s: dev=%d msg=%u iters=%d warmup=%d stream=%u\n",
	       argv[1], dev, msg, iters, warmup, sid);

	uint8_t *buf = malloc(msg);
	if (!buf) { fprintf(stderr, "malloc failed\n"); return 1; }

	if (server) {
		/* echo loop */
		for (int i = 0; i < total; i++) {
			int n = msg_recv(sid, buf, msg);
			if (n < 0) { fprintf(stderr, "server recv fail i=%d ret=%d\n", i, n); return 1; }
			if (msg_send(sid, CLI_ID, buf, (uint32_t)n) < 0) {
				fprintf(stderr, "server echo fail i=%d\n", i); return 1;
			}
		}
		printf("server: echoed %d messages, done\n", total);
		odl_tb5_close(H);
		return 0;
	}

	/* client */
	sleep(2); /* let server post its recv stream first */
	for (uint32_t j = 0; j < msg; j++) buf[j] = (uint8_t)(j * 7 + 13);

	double *rtt = malloc(sizeof(double) * iters);
	for (int i = 0; i < total; i++) {
		uint32_t seq = (uint32_t)i;
		memcpy(buf, &seq, 4);
		double t0 = now_us();
		if (msg_send(sid, SRV_ID, buf, msg) < 0) {
			fprintf(stderr, "client send fail i=%d\n", i); return 1;
		}
		int n = msg_recv(sid, buf, msg);
		double t1 = now_us();
		if (n < 0) { fprintf(stderr, "client recv fail i=%d ret=%d\n", i, n); return 1; }
		if ((uint32_t)n != msg) {
			fprintf(stderr, "echo size mismatch i=%d got=%d want=%u\n", i, n, msg); return 1;
		}
		uint32_t rseq; memcpy(&rseq, buf, 4);
		if (rseq != seq) {
			fprintf(stderr, "echo seq mismatch i=%d got=%u\n", i, rseq); return 1;
		}
		if (i >= warmup) rtt[i - warmup] = t1 - t0;
	}

	char fn[128];
	snprintf(fn, sizeof(fn), "odl_pingpong_dev%d_%uB.txt", dev, msg);
	FILE *f = fopen(fn, "w");
	if (f) {
		for (int i = 0; i < iters; i++) fprintf(f, "%.3f\n", rtt[i]);
		fclose(f);
	}

	qsort(rtt, iters, sizeof(double), cmp_d);
	double sum = 0;
	for (int i = 0; i < iters; i++) sum += rtt[i];
	printf("RESULT dev=%d msg=%uB iters=%d "
	       "min=%.2f p50=%.2f p90=%.2f p99=%.2f max=%.2f mean=%.2f us\n",
	       dev, msg, iters,
	       rtt[0], pct(rtt, iters, 50), pct(rtt, iters, 90),
	       pct(rtt, iters, 99), rtt[iters - 1], sum / iters);

	odl_tb5_close(H);
	free(rtt);
	free(buf);
	return 0;
}
