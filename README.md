# DeepSeek-V4-Flash on vLLM — gfx1151 2-box rebuild kit

A reproducible rebuild of the hand-patched vLLM engine that serves
**DeepSeek-V4-Flash** across **two AMD Strix Halo (gfx1151) boxes**, tensor-parallel
(TP=2), with the inter-GPU all-reduce carried over a **Thunderbolt-4 / USB4**
link (the OdinLink driver). It contains everything needed to reconstruct the *software* from a public
base image plus the host-side scripts that launch and drive it.

This code and stack was pretty much entirely put together by AI, I probably can not help too much outside of prompting my agent.
PRs are welcome for performance improvements.

## Performance

Single-stream, TP=2 across the two boxes over the Thunderbolt fabric, with the
**DFlash parallel drafter** (DSpark MTP speculative decoding) enabled. Decode
speed depends on how often the drafter's parallel tokens are accepted, so
prose and code generate at different rates. Measured on the reference rig
(2× Ryzen AI Max+ 395 / Radeon 8060S, 128 GB UMA each): fresh uncached
prompts, temperature 0, thinking disabled, 300-token generations.

| context | prefill tok/s | decode — prose | decode — code |
|---|---|---|---|
| 512 | ~300 | 23 | 32 |
| 10k | ~270 | 23 | 27 |
| 50k | ~239 | 19 | 27 |
| 100k | ~191 | 19 | 22 |

Prefill is content-agnostic (prose and code measured within a few percent).
A fresh bringup warms its own kernels/caches automatically
(`warmup_ctx`), so these rates hold from the first real request.

---

## TL;DR — minimal bring-up

Hardware: **2× AMD Strix Halo (gfx1151)** boxes (~128 GB unified memory each), a
**Thunderbolt-4/USB4 cable** between them, **Secure Boot disabled**. Then, in
order:

```bash
# 0. model weights, ~150 GB, on BOTH boxes (https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
hf download deepseek-ai/DeepSeek-V4-Flash-0731

# 1. Thunderbolt fabric, on BOTH boxes (stock kernel; ONE cable between them)
odinlink/build-odinlink.sh && sudo odinlink/install-odinlink.sh
#    migrating from the old tbv stack? sudo odinlink/uninstall-tbv.sh --apply first

# 2. the patched vLLM image (box1; copy to box2 with podman save | podman load)
container/build.sh                                  # then create the distrobox — §2 below

# 3. site config + host scripts
#    edit host/ds4-config.yaml (IPs, transport, memory) and deploy per §3

# 4. launch (box1) — full 2-box bringup, OpenAI API on :1234 when done
systemctl --user start ds4-vllm
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
├── container/                    ← rebuild the patched vLLM engine (route 2)
│   ├── Dockerfile                ← FROM kyuz0 gfx1151 base + COPY the patch-set
│   ├── build.sh                  ← podman build helper (runs the packaging tests first)
│   ├── verify-patches.sh         ← prove patches/ really is base → rootfs
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
│   ├── ds4-cluster-restart.sh    ← full validated bringup (ExecStart of ds4-vllm.service)
│   ├── ds4-cluster-down.sh       ← full teardown (ExecStop/StopPost)
│   ├── ds4-vllm-manual-serve.sh  ← the vllm serve invocation + all serving flags
│   ├── ds4-vllm-warmup.py        ← post-start JIT/prefill-cache warmer (warmup_ctx)
│   ├── ds4-cluster-env*.sh       ← canonical env + DS4_* tuning knobs (odl/tcp variants)
│   ├── container-heal.sh         ← reconcile/start a wedged podman container
│   └── systemd/                  ← ds4-vllm.service
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
- **Model weights**: [`deepseek-ai/DeepSeek-V4-Flash-0731`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
  (~150 GB checkpoint). Not included — `hf download deepseek-ai/DeepSeek-V4-Flash-0731`
  on both boxes (the `model:` key in `ds4-config.yaml` takes an HF id or a local
  path). Served as `deepseek-v4-flash`.

---

## 1. Rebuild the container

```bash
cd container
./build.sh                       # -> ds4-vllm-patched:local
```

This does `FROM docker.io/kyuz0/vllm-therock-gfx1151@<pinned-digest>`, applies
`container/patches/vllm-upstream.patch` to the base's own vLLM sources (36
files), overlays the 15 new files from `container/rootfs/`
(see [`container/patches/MANIFEST.md`](container/patches/MANIFEST.md)), builds
the OdinLink userspace (the RCCL net plugin and the `odl_ar2` all-reduce, both
fetched at a pinned revision), and rebuilds ROCr with the idle-wait fix. The
base is ~35 GB and is pulled on first build; network is needed on the first
build for the pinned source fetches.

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

## 2. Create the serving distrobox

The cluster scripts `podman exec` into a container named per
`ds4-config.yaml` (default **`vllm`**), so create one from the image you just
built (on **both** boxes):

```bash
distrobox create --name vllm --image ds4-vllm-patched:local --additional-flags \
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
model: deepseek-ai/DeepSeek-V4-Flash-0731   # HF id or local path; weights on BOTH boxes
transport: odl         # odl | tcp — which ds4-cluster-env.<transport>.sh the cluster sources
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
the fabric or the model path.

Deploy (paths are `$HOME`-relative, same layout on both boxes):

- **box1**: `host/ds4-config{,.yaml}`, `ds4-cluster-restart.sh`,
  `ds4-cluster-down.sh`, `ds4-vllm-manual-serve.sh`, `ds4-vllm-warmup.py`,
  `container-heal.sh`, all four `ds4-cluster-env*.sh`, and
  `host/systemd/ds4-vllm.service` into `~/.config/systemd/user/`.
- **box2**: `ds4-cluster-env*.sh` and `container-heal.sh` only — box2 is driven
  over ssh (key auth box1→box2 required).

Then `systemctl --user start ds4-vllm` brings up the whole 2-box cluster
(teardown → container heal → ray on both boxes → `vllm serve` → API/RDMA
verify); `stop` tears it down. The env files must stay **identical on both
boxes** — the two TP ranks silently diverge otherwise. 

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
Details, constraints and the local patch inventory: [`odinlink/`](odinlink/).

---

## What was patched, and why

Full table in [`container/patches/MANIFEST.md`](container/patches/MANIFEST.md).
The themes:

- **DeepSeek-V4 model on gfx1151** — the AMD/ROCm DSpark model path, MLA
  attention, fp8 (UE8M0) KV-latent compress/quant, and the MTP drafter.
- **Mid-context retrieval** — the sparse indexer runs the *official* QAT graph
  (Hadamard128 + FP4 sim) before top-512 scoring (`DS4_IDX_OFFICIAL`), which the
  stock FP8 indexer skipped; plus a ROCm sparse-MLA attention rewrite.
- **Hand-written decode kernels** — an MXFP4 MoE decode path (gemm1 + fused
  SILU/clamp + gemm2 with fused scatter, one contiguous march per workgroup
  instead of the against-the-grain tile the stock path reads) and a dense fp8
  GEMV that replaces a bf16 path reading twice the bytes. Both are ctypes
  wrappers over libraries built in-image from `container/native/`, and both fall
  back to the stock path on any layout they do not recognise, so a missing
  library costs speed and never correctness.
- **Reasoning effort levels** — `low` / `high` / `max` / `none` all render.
  The encoder in the base image emitted a preamble only for `max` and silently
  ignored `high`, so a server configured for high reasoning got no preamble and
  no error; upstream's table is backported so the setting means something.
- **MoE / GEMM tuning** — decode-scoped MXFP4 `matmul_ogs` knobs
  (`DS4_MOE_BN/NW/NS/BK/WPE`, the `block_k` bandwidth lever), a tuned gfx1151
  A8W8 GEMM config, and a `DS4_W8A8_BF16` fast bf16 path.
- **Thunderbolt all-reduce** — `odl_ar2` hooked into vLLM's communicator,
  replacing RCCL for the decode TP all-reduce on the Thunderbolt link.
- **Disk KV cache** an `fs_lru` secondary
  tier gives the KV offloader a byte cap with LRU eviction, which the stock `fs`
  tier has no mechanism for, so it can point at a filesystem shared with
  everything else. Alongside it, the offloading scheduler now bounds each store
  batch: stock asks for every un-offloaded block at once and does not advance its
  cursor when the tier refuses, so a single refusal ratchets the ask past the
  tier and stores stop for the rest of the request. Bounding it lets a few
  hundred MiB of staging carry a full-length prefill.

---

## Provenance & notes

- Base: `docker.io/kyuz0/vllm-therock-gfx1151@sha256:25fd294f…`, vLLM commit `470229c`.
- The original container also had `py-spy` pip-installed (a profiler) and some
  incidental OS packages under `/usr/lib/python3.14`; neither affects serving and
  both are intentionally omitted. Re-add `py-spy` with `pip install py-spy` inside
  the container if you want it.

## Special Thanks
@Jawnnypoo for helping get the initial version of this repository running.
