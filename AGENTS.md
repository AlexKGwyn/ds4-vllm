# AGENTS.md — bring up DeepSeek-V4-Flash on the 2-box gfx1151 vLLM cluster

You are an agent setting this up on a fresh pair of machines. This file is the
runbook: build the pieces, wire them together, get the model serving, and verify
it. Read it top to bottom **before** running anything — several steps are
hard-to-reverse and order matters.

## 0. What you are building

DeepSeek-V4-Flash served by a **patched vLLM**, tensor-parallel across **two AMD
Strix Halo (gfx1151) boxes**, with the inter-GPU all-reduce carried over a
**Thunderbolt-4 / USB4 RoCE-RDMA** link.

```
        ┌────────────── box1 (ray HEAD, gfx1151) ──────────────┐
        │  distrobox "vllm"  ──►  vllm serve  TP rank 0         │
        │  ds4-vllm.service → ds4-cluster-restart.sh            │
        └───────────────┬───────────────────────────────────────┘
                        │  Thunderbolt-4 cable
                        │  OdinLink RDMA  = /dev/odl_tb5_*
                        │  IP link thunderbolt0 = 192.168.100.1/.2
        ┌───────────────┴───────────────────────────────────────┐
        │  distrobox "vllm"  ──►  ray worker  TP rank 1          │
        └────────────── box2 (ray WORKER, gfx1151) ─────────────┘
```

Three independent layers, build/verify them in this order:

1. **OdinLink fabric** (`odinlink/`) — the Thunderbolt interconnect.
   Foundational and the riskiest; do it first. *(The cluster will also run
   without it on a slow TCP fallback — see §1.2 to de-risk by validating vLLM
   first, then adding the fabric.)*
2. **vLLM engine** (`container/`) — rebuild the patched image, one distrobox per box.
3. **Host orchestration** (`host/`) — the launch scripts, env, model weights.

## 0.1 Prerequisites (verify these exist; do NOT try to synthesize them)

- **2× AMD Strix Halo / gfx1151**, ~128 GB unified memory each, on the same LAN.
- A **Thunderbolt-4 / USB4 cable** physically connecting the two boxes.
- Linux with **kernel headers/devel** for the running kernel on each box,
  `podman`, `distrobox`, `git`, build toolchain. A stock kernel is fine —
  nothing here needs a patched thunderbolt core.
- The model weights **`deepseek-ai/DeepSeek-V4-Flash-0731`** (~150 GB) downloaded
  on **both** boxes (`hf download deepseek-ai/DeepSeek-V4-Flash-0731`).
- Root/sudo on both boxes (kernel modules, systemd units).
- **Secure Boot disabled on both boxes**, unless the box has an enrolled MOK:
  `odl_tb5.ko` is built locally, and `odl-swap.sh` signs it with the enrolled
  key when one is present. With no key and Secure Boot on, the kernel refuses
  the `insmod` in §1.

Pick roles now and keep them consistent everywhere: **box1 = ray head**, IP on
Thunderbolt `192.168.100.1`; **box2 = worker**, `192.168.100.2`. Site values
(IPs, container name, transport, HCA pin, disk KV) live in
`host/ds4-config.yaml`, deployed as `~/ds4-config.yaml` on box1 (see §3);
paths in the scripts are `$HOME`-relative.

---

## 0.2 Recall / context integrity — fixed, gated

Long-context recall on this stack is correct for the deployed profile. What
keeps it correct:

- `DS4_IDX_OFFICIAL=1` in `host/ds4-cluster-env.sh` -- the sparse indexer's
  official Hadamard128 + FP4 QAT scoring graph. Must be engaged on BOTH TP
  ranks (it is exported from the shared env, so keep the env identical).
- The `deepseek_v4_encoding.py` patch -- the chat encoder no longer strips
  prior assistant reasoning on tool conversations.

Any change to the indexer, MTP, kernels, or tuning knobs must re-pass
needle/recall probes at your target context depth before it ships (see §5).

---

## 1. OdinLink — the Thunderbolt fabric (do this first)

Full detail in [`odinlink/README.md`](odinlink/README.md); this is the ordered
action list. Run every step on **both** boxes. The kernel driver is
vermagic-locked to the running kernel, so rebuild it after a kernel update.

Requires **exactly one** Thunderbolt cable between the boxes — see "Why this
builds on a stock kernel" in the OdinLink README for why that is a rule and not
a preference.

### 1.0 Migrating from the old tbv stack (skip on a fresh box)

Earlier revisions of this repo ran the fabric on a patched thunderbolt core
plus `thunderbolt_ibverbs`, with the stock driver blacklisted. None of that is
needed now. If a box was ever set up that way:

```bash
sudo odinlink/uninstall-tbv.sh           # dry run — prints what it would change
sudo odinlink/uninstall-tbv.sh --apply   # then REBOOT this box
```

This matters most for the `blacklist=thunderbolt` kernel arguments: with tbv's
modules gone and the blacklist still in place, **no** thunderbolt driver loads
at all and the boxes have no link — a failure that shows up a long way from its
cause.

### 1.1 Build and install (per box, per kernel)

```bash
odinlink/build-odinlink.sh             # odl_tb5.ko + libodl_tb5, no sudo
sudo odinlink/install-odinlink.sh      # stage /usr/local + enable odinlink.service
sudo systemctl start odinlink.service  # or reboot
```

`build-odinlink.sh` fetches OdinLink at its pin, applies
`odinlink/odinlink-local.patch`, and builds against
`/lib/modules/$(uname -r)/build` — stock headers, no out-of-tree thunderbolt
core, no blacklist, no kernel arguments. Secure Boot must be off unless the box
has an enrolled MOK, in which case `odl-swap.sh` signs the module with it.

The RCCL net plugin and the `odl_ar2` decode all-reduce ship inside the §2
image; the kernel driver is the only host build.

Load the driver **before** connecting the cable, and start both boxes promptly
together.

### 1.2 Verify the link (gate)

```bash
odl-state                      # -> state=ready on BOTH boxes
ip addr show thunderbolt0      # -> 192.168.100.x (or your configured pair)
```

`thunderbolt0` comes from mainline `thunderbolt_net` and carries the bootstrap
sockets, Ray and ssh. If it has no address the cluster cannot come up whatever
OdinLink reports, so fix that first.

**Optional latency tweak:** `odinlink/nhi-throttle-mod/` lowers the NHI
interrupt moderation the mainline driver hardcodes:
`make -C odinlink/nhi-throttle-mod && sudo insmod odinlink/nhi-throttle-mod/nhi_throttle.ko ns=8000`.

**De-risk option:** the fabric is a performance layer, not a correctness gate —
set `transport: tcp` in `~/ds4-config.yaml` to run the same cluster over
sockets (much slower decode). If you want to validate the model path first,
skip to §2–§4 on TCP and come back.

## 2. Build the vLLM engine

See [`container/`](container/). On **each** box:

```bash
cd container && ./build.sh                # -> ds4-vllm-patched:local  (base ~35 GB pulled once)
```
This is `FROM kyuz0/vllm-therock-gfx1151@<pinned digest>` + the DS4 patch-set
(35 modified files as `patches/vllm-upstream.patch`, 17 new — see
`container/patches/MANIFEST.md`). Then create the serving container, named per
`container:` in `ds4-config.yaml` (default **`vllm`**):

```bash
distrobox create --name vllm --image ds4-vllm-patched:local --additional-flags \
  '--privileged --ipc host --pid host \
   --device /dev/kfd --device /dev/dri --device /dev/infiniband \
   --group-add video --group-add render --security-opt seccomp=unconfined'
distrobox enter vllm -- vllm --version          # gate: prints a version
distrobox enter vllm -- ibv_devices             # gate: lists usb4_rdma0 (if §1 done)
```

## 3. Host orchestration + config

Deploy the `host/` files per README §3 and set the site values in
`~/ds4-config.yaml` on box1 (head/worker IPs, container name, `transport:
rdma|tcp|odl`, RDMA HCA pin, disk KV). Two rules that bite:

- `ds4-cluster-env*.sh` **must be byte-identical on both boxes** — the two TP
  ranks silently diverge otherwise. Copy the same files to both.
- Box1 needs passwordless ssh to the worker IP: the cluster scripts drive
  box2's container over ssh.

## 4. Start serving

```bash
systemctl --user start ds4-vllm      # box1; ~5 min warm
```

`ds4-cluster-restart.sh` (the unit's ExecStart) does the whole sequence:
teardown + stranded-process reap, container heal on both boxes, ray head +
box2 worker (2 GPUs gate), then `vllm serve` via `ds4-vllm-manual-serve.sh`
as the transient `ds4-vllm-manual` unit (MTP speculative decode,
`deepseek_v4` tokenizer/reasoning/tool parsers, fp8 KV, eager, disk KV
tier), verifies the API and the RDMA all-reduce, and dispatches the
warmup (`ds4-vllm-warmup.py`, `warmup_ctx` in the yaml) before reporting
success. `systemctl --user stop ds4-vllm` tears everything down.
