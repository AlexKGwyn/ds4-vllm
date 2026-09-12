# DeepSeek-V4-Flash on vLLM — gfx1151 2-box rebuild kit

A reproducible rebuild of the hand-patched vLLM engine that serves
**DeepSeek-V4-Flash** across **two AMD Strix Halo (gfx1151) boxes**, tensor-parallel
(TP=2), with the inter-GPU all-reduce carried over a **Thunderbolt-4 / USB4**
link (the OdinLink driver), or optionally over a native **InfiniBand** link
(ConnectX-3). It contains everything needed to reconstruct the *software* from a public
base image plus the host-side scripts that launch and drive it.

This code and stack was pretty much entirely put together by AI, I probably can not help too much outside of prompting my agent.
PRs are welcome for performance improvements.

## Performance

Single-stream, TP=2 across the two boxes over Thunderbolt RDMA, with DSpark
MTP speculative decoding enabled. Decode
speed depends on how often the drafter's parallel tokens are accepted, so
prose and code generate at different rates. Measured 2026-09-11 on the
reference rig (2× Ryzen AI Max+ 395 / Radeon 8060S, 128 GB UMA each) with the
current defaults (async scheduling, the HIP indexer scorer, disk KV on):
fresh uncached prompts, temperature 0, thinking disabled, 300-token
generations, decode timed from the first token.

| context | prefill tok/s | decode — prose | decode — code |
|---|---|---|---|
| 512 | ~345 | 21 | 33 |
| 10k | ~370 | 21 | 35 |
| 50k | ~330 | 18 | 34 |
| 100k | ~295 | 18 | 27 |

Prefill is content-agnostic (prose and code measured within a few percent).
Decode step time is nearly flat with depth (~110 ms at 512 tokens, ~125 ms at
100k); what moves tok/s is MTP acceptance, which is why code (~4 tokens per
step) outruns prose (~2.3). A fresh bringup warms its own kernels/caches
automatically (`warmup_ctx`), so these rates hold from the first real
request. The numbers were taken on the Vision-Exp checkpoint with text-only
requests, which take the same engine path as the text checkpoint.

---

## TL;DR — minimal bring-up

Hardware: **2× AMD Strix Halo (gfx1151)** boxes (~128 GB unified memory each), a
**Thunderbolt-4/USB4 cable** between them, **Secure Boot disabled**. Then, in
order:

```bash
# 0. model weights, ~150 GB, on BOTH boxes (https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
hf download deepseek-ai/DeepSeek-V4-Flash-0731
#    image input? download deepseek-ai/DeepSeek-V4-Flash-Vision-Exp instead -- see "Vision" below

# 1. Thunderbolt fabric, on BOTH boxes (stock kernel; ONE cable between them)
odinlink/build-odinlink.sh && sudo odinlink/install-odinlink.sh
#    migrating from the old tbv stack? sudo odinlink/uninstall-tbv.sh --apply first

# 2. the patched vLLM image (box1; copy to box2 with podman save | podman load)
container/build.sh                                  # then the ROCm 7.14 layer (§1b) and the distrobox (§2)

# 3. site config + host scripts
#    edit host/ds4-config.yaml (IPs, transport odl|tcp|ib, memory) and deploy per §3

# 4. launch (box1) — full 2-box bringup, OpenAI API on :1234 when done
host/ds4-serve.sh start
```

Full ordered runbook with the gates and gotchas: [`AGENTS.md`](AGENTS.md).

---

> **Read this first — what "rebuildable" means here.** The **container rebuilds
> deterministically on any machine** with podman (`container/` below). *Serving*
> the model, however, needs the matching rig: 2× gfx1151 boxes, a working
> Thunderbolt fabric, ROCm 7, and the model weights. This is a
> hardware-specific research build, not a general-purpose vLLM package. See
> **Prerequisites**.
>
> **Setting it up? Follow [`AGENTS.md`](AGENTS.md)** — the ordered end-to-end
> runbook (fabric → container → serve) written for a person or agent doing the
> bring-up on a fresh pair of boxes.

---

## License & attribution

Original work here is **Apache-2.0** ([LICENSE](LICENSE)). This project
builds on **vLLM** (Apache-2.0) and **Geramy/OdinLink-Five** (GPL-2.0) — their
sources are fetched at pinned revisions at build time rather than
redistributed; the patches shipped here are derivative works licensed like
the code they modify. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
for the full component/license table and upstream references.

---

## Layout

```
ds4vllm-public/
├── README.md                     ← this file
├── AGENTS.md                     ← ordered bring-up runbook (fabric → container → serve)
├── container/                    ← rebuild the patched vLLM engine
│   ├── Dockerfile                ← FROM kyuz0 gfx1151 base + patch-set + in-image native builds
│   ├── Dockerfile.rocm714        ← second layer: stable therock-7.14 ROCm/torch wheels (§1b)
│   ├── build.sh                  ← podman build helper (runs the packaging tests first)
│   ├── verify-patches.sh         ← prove patches/ really is base → rootfs
│   ├── native/                   ← HIP sources built in-image: MXFP4 MoE, fp8 GEMV, indexer scorer, ib_ar2
│   ├── rootfs/                   ← the NEW files at their real paths (modified files ship as the patch)
│   └── patches/                  ← vllm-upstream.patch (base → patched) + MANIFEST.md
├── odinlink/                     ← the Thunderbolt fabric (OdinLink driver, transport: odl)
│   ├── build-odinlink.sh         ← fetch pinned OdinLink-Five + build odl_tb5.ko (host)
│   ├── install-odinlink.sh, odl-swap.sh, systemd/ ← install + MOK-sign + boot unit
│   ├── uninstall-tbv.sh          ← remove a previous tbv/ibverbs deployment
│   ├── nhi-throttle-mod/         ← optional NHI IRQ-throttle module (latency)
│   ├── odinlink-local.patch, ar2/ ← our diff on the pin; odl_ar2 decode all-reduce
├── host/                         ← host-side orchestration (run outside the container)
│   ├── ds4-config.yaml, ds4-config ← site config (IPs, transport, disk KV) + loader
│   ├── ds4-ib-config.yaml        ← the same site config on the InfiniBand transport (example)
│   ├── ds4-serve.sh              ← start | stop | restart | status | logs (the entry point)
│   ├── ds4-cluster-restart.sh    ← full validated bringup (what `start` runs)
│   ├── ds4-cluster-down.sh       ← full teardown (what `stop` runs)
│   ├── ds4-vllm-manual-serve.sh  ← the vllm serve invocation + all serving flags
│   ├── ds4-vllm-warmup.py        ← post-start JIT/prefill-cache warmer (warmup_ctx)
│   ├── ds4-cluster-env*.sh       ← canonical env + DS4_* tuning knobs (odl/tcp/ib variants)
│   ├── ds4-rccl-bench.{sh,py}    ← RCCL all-reduce latency probe for a transport
│   ├── bench/                    ← ib_ar2 and indexer-scorer verification probes
│   └── container-heal.sh         ← reconcile/start a wedged podman container
├── tests/                        ← packaging + kernel/vision unit tests (torch-dependent ones skip on a bare host)
├── examples/systemd/             ← OPTIONAL unit wrapping ds4-serve.sh (not installed)
```

---

## Prerequisites (to actually serve)

- **2× AMD Strix Halo / gfx1151** boxes, ~128 GB unified memory each. box1 is the
  ray head; box2 joins as a worker.
- **ROCm 7** (provided inside the container via the kyuz0 base — you do not
  install it on the host).
- **podman** + **distrobox** on both hosts (rootless is fine; the live setup uses it).
- **Exactly one Thunderbolt-4 / USB4 cable** between the boxes, carrying both
  the OdinLink RDMA fabric and the `thunderbolt0` IP interface the cluster
  bootstraps over. A second cable puts the peer at the same route on both
  links and breaks peer demultiplexing — see [`odinlink/README.md`](odinlink/README.md).
  The driver is built from [`odinlink/`](odinlink/) against a **stock kernel**
  (no patched thunderbolt core, no blacklist, no kernel arguments) and is
  vermagic-locked, so rebuild it after a kernel update. **Secure Boot must be
  disabled** unless the box has an enrolled MOK — the load path signs the
  module with it when one is present. Without the fabric the stack still runs
  on `transport: tcp` (much slower decode).
- **Previously ran the old `tbv` ibverbs stack?** Run
  `sudo odinlink/uninstall-tbv.sh --apply` and reboot before building. It
  removes the `blacklist=thunderbolt` kernel arguments, without which no
  thunderbolt driver loads at all.
- **Optional: an InfiniBand link.** `transport: ib` moves the TP collectives
  onto RCCL-over-IB plus the `ib_ar2` decode all-reduce on a ConnectX-3 (mlx4)
  fabric between the boxes. The Thunderbolt cable is still required for the
  control plane. See **InfiniBand transport** below.
- **Model weights**: [`deepseek-ai/DeepSeek-V4-Flash-0731`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
  (~150 GB checkpoint). Not included — `hf download deepseek-ai/DeepSeek-V4-Flash-0731`
  on both boxes (the `model:` key in `ds4-config.yaml` takes an HF id or a local
  path). Served as `deepseek-v4-flash`. For image input use
  [`deepseek-ai/DeepSeek-V4-Flash-Vision-Exp`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp)
  instead — see **Vision** below.

---

## 1. Rebuild the container

```bash
cd container
./build.sh                       # -> ds4-vllm-patched:local
```

This does `FROM docker.io/kyuz0/vllm-therock-gfx1151@<pinned-digest>`, applies
`container/patches/vllm-upstream.patch` to the base's own vLLM sources,
overlays the new files from `container/rootfs/`
(see [`container/patches/MANIFEST.md`](container/patches/MANIFEST.md)), builds
the OdinLink userspace (the RCCL net plugin and the `odl_ar2` all-reduce, both
fetched at a pinned revision), compiles the four HIP libraries in
`container/native/` (the MXFP4 MoE decode pair, the fp8 GEMV, the indexer
scorer, and `ib_ar2`; `rootfs/` carries only their Python wrappers), installs
`rdma-core` for the optional InfiniBand transport, and rebuilds ROCr with the
idle-wait fix. The base is ~35 GB and is pulled on first build; network is
needed on the first build for the pinned source fetches.

**Base image drift.** The Dockerfile pins the base by **digest** so the rebuild
matches the engine the patches were developed against (vLLM commit `470229c`). If
that digest is ever unpullable, replace it with `:latest` — but be aware kyuz0's
`latest` moves, and a newer base could carry a different vLLM whose files the
patches assume. Prefer the pinned digest.

**Keeping the patch honest.** Two checks:

```bash
python3 -m unittest discover -s tests -v   # manifest vs rootfs vs patch vs Dockerfile
container/verify-patches.sh                # patch applies cleanly to the pinned base
```

`build.sh` runs the first automatically; the second needs the base image
locally. To regenerate the patch after editing engine files, point
`DS4_PATCH_SRC` at a tree holding the desired files and run
`container/verify-patches.sh --write`.

### 1b. The stable ROCm 7.14 layer

The base ships an alpha nightly ROCm/torch/triton set. The piecewise-graph and
decode-kernel patch set was validated on the stable therock-7.14 release
wheels, so build a second layer on top of the image from §1 and serve from
that one:

```bash
podman build -t ds4-vllm-rocm714:local \
  --build-arg DS4_PREV=ds4-vllm-patched:local \
  -f container/Dockerfile.rocm714 .
```

It swaps the venv's ROCm, torch, torchvision, torchaudio and triton wheels for
the 7.14.0 release set from `repo.amd.com`, keeps the patched vLLM wheel
(same torch 2.13 line), re-applies the `rootfs/` overlay, and runs an import
smoke check. Note the stock stable ROCr replaces the idle-wait-fixed ROCr the
first stage built. Network is needed for the wheel pulls.

## 2. Create the serving distrobox

The cluster scripts `podman exec` into a container named per
`ds4-config.yaml` (default **`vllm`**), so create one from the image you just
built (on **both** boxes; use `ds4-vllm-patched:local` only if you skipped
§1b):

```bash
distrobox create --name vllm --image ds4-vllm-rocm714:local --additional-flags \
  '--privileged --ipc host --pid host \
   --device /dev/kfd --device /dev/dri --device /dev/infiniband \
   --group-add video --group-add render --security-opt seccomp=unconfined'
distrobox enter vllm -- vllm --version
```

**Do not pass `--network host` in `--additional-flags`.** distrobox already
selects host networking, and passing it again fails with
`cannot set multiple networks without bridge network mode`. The resulting
container still gets `net=host`; verify with
`podman inspect vllm --format '{{.HostConfig.NetworkMode}}'`.

## 3. Run

Site specifics live in **`host/ds4-config.yaml`** — deploy it (edited for your
site) as `~/ds4-config.yaml` on box1 next to the scripts:

```yaml
model: deepseek-ai/DeepSeek-V4-Flash-0731   # HF id or local path; weights on BOTH boxes (Vision-Exp for image input, see "Vision")
transport: odl         # odl | tcp | ib — which ds4-cluster-env.<transport>.sh the cluster sources
head_ip: 192.168.100.1
worker_ip: 192.168.100.2
container: vllm        # podman container name (same on both boxes)
control_iface_head: thunderbolt0   # NCCL/GLOO bootstrap-socket interface per box
control_iface_worker: thunderbolt0
odl_rank1_ip: 192.168.100.2        # odl_ar2 rendezvous address (usually = worker_ip)
api_port: 1234
disk_kv: true          # NVMe prefix-KV tier (fs_lru); prefixes survive restarts
disk_kv_gib: 30        # per-NODE disk cap; check df on BOTH boxes before raising
max_ctx: 524288        # --max-model-len (512K, the validated profile)
kv_pin_gib: 6          # pinned GPU KV pool; sized against MemAvailable, not grown by max_ctx
gpu_mem_util: 0.83     # vLLM --gpu-memory-utilization; IGNORED while the KV pin above is set
warmup_ctx: 2048       # post-start warmup prefill size; 0 disables
```

`host/ds4-config` (stdlib Python, no pyyaml) turns it into `DS4_*` exports for
the scripts. `transport: odl` is the fabric; `transport: tcp` runs the same
cluster over plain sockets (correctness / fallback profile, with the `odl_ar2`
decode all-reduce disabled) and is useful for isolating a bring-up problem to
the fabric or the model path. `transport: ib` runs the collectives over a
native InfiniBand link instead of Thunderbolt; `host/ds4-ib-config.yaml` is
that profile, with a `rdma_hca:` pin for the HCA port (deploy it *as*
`~/ds4-config.yaml` — the scripts read only that file).

Deploy (paths are `$HOME`-relative, same layout on both boxes):

- **box1**: `host/ds4-config{,.yaml}`, `ds4-serve.sh`,
  `ds4-cluster-restart.sh`, `ds4-cluster-down.sh`, `ds4-vllm-manual-serve.sh`,
  `ds4-vllm-warmup.py`, `container-heal.sh`, and every `ds4-cluster-env*.sh`.
- **box2**: `ds4-cluster-env*.sh` and `container-heal.sh` only — box2 is driven
  over ssh (key auth box1→box2 required).

Then `./ds4-serve.sh start` brings up the whole 2-box cluster (teardown →
container heal → ray on both boxes → `vllm serve` → API/all-reduce verify);
`stop` tears it down and `status` reports on it. There is no systemd here on
purpose — `examples/systemd/ds4-vllm.service` wraps the same script if you
want a unit. The env files must stay **identical on both
boxes** — the two TP ranks silently diverge otherwise. 

---

## Vision (DeepSeek-V4-Flash-Vision-Exp)

The same image and scripts serve the multimodal
[`deepseek-ai/DeepSeek-V4-Flash-Vision-Exp`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp)
checkpoint — the text model plus a BF16 vision tower and aligner — taking
images through the standard OpenAI chat `image_url` content parts. Everything
else is as for the text checkpoint: TP=2 over the fabric, 512K context, MTP-5
speculative decoding, fp8 KV, the disk KV tier, the `deepseek-v4-flash` served
name. This port was built against checkpoint revision `e46e16b`.

```bash
hf download deepseek-ai/DeepSeek-V4-Flash-Vision-Exp    # on BOTH boxes
```

Then point `model:` in `~/ds4-config.yaml` on box1 at it, and restart:

```yaml
model: deepseek-ai/DeepSeek-V4-Flash-Vision-Exp   # or a local copy of it
```

```bash
./ds4-serve.sh restart
curl -s http://127.0.0.1:1234/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "deepseek-v4-flash",
  "messages": [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "https://upload.wikimedia.org/wikipedia/commons/3/3f/JPEG_example_flower.jpg"}},
    {"type": "text", "text": "What is in this picture?"}]}]}'
```

There is no switch to flip: the launcher reads the checkpoint's `config.json`
and, when it declares a vision tower (`vision_n_layers`), passes
`--hf-overrides '{"architectures":["DeepseekV4VForConditionalGeneration"],"num_nextn_predict_layers":1}'`.
That is needed because the checkpoint's own config still declares the text
architecture, and declares its (identical) three-stage DSpark drafter as three
predictor layers where the text checkpoint declares one. The serve log says
`vision checkpoint (vision_n_layers=32): serving the multimodal wrapper` when
it took effect. To go back, point `model:` at the text checkpoint and restart.

What the engine does differently for this checkpoint (rows in
`container/patches/MANIFEST.md`):

- **Image preprocessing + tower** (`mm_preprocess.py`, `vision.py`,
  `vision_model.py`): the reference resize/grid/C4 token layout, the BF16 ViT
  and aligner, and the five out-of-vocabulary image sentinel ids the language
  model sees for expert routing and attention.
- **Expert routing**: image positions use the checkpoint's visual expert
  selection bias; text positions keep the text bias and the fused gfx1151
  selector.
- **Span-bidirectional attention**: inside each image span attention is
  bidirectional, as in the reference model; text tokens take the unchanged
  causal path. `DS4_VISION_SPAN_ATTN=0` in `ds4-cluster-env.sh` (both boxes)
  keeps image spans causal instead. An image span cut by a chunked-prefill
  boundary is attended causally for the cut fragment.
- **MTP**: the DSpark drafter runs unchanged (one predictor, three internal
  stages). Its drafts over image spans are merely lower quality — the target
  model verifies every drafted token, so outputs are unaffected.

Images cost prompt tokens (a few hundred per image at the default grid) and
count against `max_ctx` like text. Text-only requests to the vision
checkpoint take the same path as the text checkpoint, and the performance
table above was measured on it.

---

## OdinLink transport

`transport: odl` carries the inter-box fabric on the
[OdinLink](https://github.com/Geramy/OdinLink-Five) Thunderbolt driver
(`odl_tb5`): RCCL runs over the OdinLink net plugin and the decode all-reduce
over `odl_ar2`, both built into the image — the kernel driver is the only host
build, and it builds against a **stock kernel**. Requires **exactly one**
Thunderbolt cable between the boxes.

Bring-up, on **both** boxes:

```bash
odinlink/build-odinlink.sh            # driver + lib, stock headers, no sudo
sudo odinlink/install-odinlink.sh     # stage /usr/local + enable odinlink.service
sudo systemctl start odinlink.service # or reboot; the driver then loads at boot
```

The load path signs `odl_tb5.ko` with the host's enrolled MOK when one is
present, so it can load under Secure Boot (see **Prerequisites**). Once
`odl-state` reports `state=ready` on both boxes, start the cluster normally.

The driver is otherwise stock: the local patch on the pinned upstream is five
bug fixes and nothing else, and the tuning/topology module parameters this
repo used to set have been dropped. Two of the fixes matter on a
unified-memory box that runs close to full: upstream could deliver a
reassembly buffer with a hole in it when an atomic allocation failed (a short
collective that looked valid, presenting as the model answering questions
nobody asked), and the RX assembly buffers are now preallocated per stream
(`rx_asm_max`, `rx_asm_pool`) so that allocation no longer happens in atomic
context at all. Details, constraints and the patch inventory:
[`odinlink/`](odinlink/).

---

## InfiniBand transport (optional)

`transport: ib` runs the TP collectives over RCCL's built-in IB net on a
ConnectX-3 (mlx4) fabric cabled back-to-back between the boxes, with the
`ib_ar2` fused all-reduce carrying every decode-sized collective and RCCL
only the prefill-sized ones. The control plane (ray, the NCCL/GLOO
rendezvous, `thunderbolt0` with the 192.168.100.x addresses) stays on the
Thunderbolt cable, so that link is still required; only the verbs data path
moves. What it needs beyond the OdinLink setup:

- `rdma-core` is already in the image; the distrobox command in §2 passes
  `/dev/infiniband` through.
- A subnet manager: the reference rig runs `opensm` in a podman container on
  box1. `rdma link` must show the pinned port `ACTIVE` on both boxes before
  bring-up, and `ds4-cluster-restart.sh` checks that on this transport.
- `host/ds4-ib-config.yaml` deployed as `~/ds4-config.yaml`, with `rdma_hca:`
  naming the HCA and cabled port (`mlx4_0:1` on the rig). The `ib` env pins
  `NCCL_IB_GID_INDEX=0`: index 0 is the native-IB GID, and the RoCE index the
  base env uses does not exist on an IB link.

`host/ds4-rccl-bench.sh` and `host/bench/ib_ar2_test.py` are the latency
probes for comparing a transport before serving on it. On the reference rig's
x4-limited slots the link ceiling is ~28 Gb/s, and `ib_ar2` measures ~48 µs
per 48 KiB decode all-reduce; `DS4_IB_AR2=0` falls back to RCCL for every
collective.

---

## What was patched, and why

Full table in [`container/patches/MANIFEST.md`](container/patches/MANIFEST.md).
The themes:

- **DeepSeek-V4 model on gfx1151** — the AMD/ROCm DSpark model path, MLA
  attention, fp8 (UE8M0) KV-latent compress/quant, and the MTP drafter.
- **Mid-context retrieval** — the sparse indexer runs the *official* QAT graph
  (Hadamard128 + FP4 sim) before top-512 scoring (`DS4_IDX_OFFICIAL`), which the
  stock FP8 indexer skipped; plus a ROCm sparse-MLA attention rewrite. At
  decode the scorer is a hand-HIP WMMA kernel (`DS4_IDX_HIP`, default on) that
  reads the fp8 paged cache directly — no gather, no bf16 copy, one launch per
  request instead of one per speculative row — and is bit-exact against the
  TileLang path it replaces. Per layer at 512K context: 1534 → 542 µs.
- **Hand-written decode kernels** — an MXFP4 MoE decode path (gemm1 + fused
  SILU/clamp + gemm2 with fused scatter, one contiguous march per workgroup
  instead of the against-the-grain tile the stock path reads) and a dense fp8
  GEMV that replaces a bf16 path reading twice the bytes. Both are ctypes
  wrappers over libraries built in-image from `container/native/`, and both fall
  back to the stock path on any layout they do not recognise, so a missing
  library costs speed and never correctness. Weight loads in both are
  non-temporal (read-once bytes stay out of L2/MALL: the MoE pair 180 → 193 GB/s
  against a measured 241 GB/s ceiling), and the K≤1024 GEMV scores four rows
  per wave. The decode layer also folds its two RMSNorms into the mhc kernels'
  output write (the kernel support existed; it was never wired), and the
  single-kernel ragged-build path carries a `tl.assume` guard against a
  triton 3.7 guarded-loop miscompile.
- **Reasoning effort levels** — `low` / `high` / `max` / `none` all render.
  The encoder in the base image emitted a preamble only for `max` and silently
  ignored `high`, so a server configured for high reasoning got no preamble and
  no error; upstream's table is backported so the setting means something. An
  absent `reasoning_effort` now defaults to `high`.
- **Async scheduling** — vLLM's async scheduler was auto-disabled by the
  unpadded drafter batches the speculative config asked for. The launcher now
  runs padded DSpark batches with `--async-scheduling` (`DS4_ASYNC_SCHED=1`,
  default; `0` restores the sync configuration). Two drafter changes make
  padded batches correct: drafting at the proposer's accepted-row anchors, and
  routing rejected rows' KV to a trash slot — both fail silently as collapsed
  acceptance otherwise. Acceptance is unchanged and output stays deterministic.
- **MoE / GEMM tuning** — decode-scoped MXFP4 `matmul_ogs` knobs
  (`DS4_MOE_BN/NW/NS/BK/WPE`, the `block_k` bandwidth lever), a tuned gfx1151
  A8W8 GEMM config, and a `DS4_W8A8_BF16` fast bf16 path.
- **Fabric all-reduce and control plane** — `odl_ar2` hooked into vLLM's
  communicator, replacing RCCL for the decode TP all-reduce on the Thunderbolt
  link; `ib_ar2`, the same design ported to a mlx4 InfiniBand HCA as one fused
  kernel per round (~48 µs for a 48 KiB all-reduce vs ~59 µs RCCL-over-IB,
  bit-exact); and `odl_mq`, which moves the EngineCore→worker broadcast and the
  worker response queue off zmq-over-TCP onto OdinLink streams, so decode no
  longer inherits a TCP round-trip per step. Any init failure falls back to the
  stock zmq path. `NCCL_PROTO` is no longer pinned to LL on either transport:
  RCCL now only sees prefill-sized ops, where LL halves the link.
- **Disk KV cache** — an `fs_lru` secondary tier gives the KV offloader a byte
  cap with LRU eviction, which the stock `fs` tier has no mechanism for, so it
  can point at a filesystem shared with everything else. It is *distributed*:
  the scheduler only decides what to store, evict and promote, and ships the
  byte movement to every TP rank, which reads and writes its own slice under
  its own node-local directory. The stock scheduler-side design assumes every
  rank shares the scheduler's staging mmap, which is false across two boxes
  and restored silently wrong KV on the remote rank. Alongside it, the
  offloading scheduler now bounds each store batch: stock asks for every
  un-offloaded block at once and does not advance its cursor when the tier
  refuses, so a single refusal ratchets the ask past the tier and stores stop
  for the rest of the request. Bounding it lets a few hundred MiB of staging
  carry a full-length prefill, and the staging region itself lives on the NVMe
  as reclaimable page cache rather than in `/dev/shm`.

---

## Provenance & notes

- Base: `docker.io/kyuz0/vllm-therock-gfx1151@sha256:25fd294f…`, vLLM commit `470229c`.
- The original container also had `py-spy` pip-installed (a profiler) and some
  incidental OS packages under `/usr/lib/python3.14`; neither affects serving and
  both are intentionally omitted. Re-add `py-spy` with `pip install py-spy` inside
  the container if you want it.

## Special Thanks
@Jawnnypoo for helping get the initial version of this repository running.
