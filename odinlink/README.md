# OdinLink transport

Runs the DS4 cluster's inter-box fabric on the OdinLink Thunderbolt RDMA
driver ([Geramy/OdinLink-Five](https://github.com/Geramy/OdinLink-Five),
fetched at a pinned revision — nothing vendored). This is the cluster fabric,
selected with `transport: odl` in `ds4-config.yaml`.

## Why this builds on a stock kernel

Nothing here needs a patched kernel: no out-of-tree thunderbolt core, no
driver blacklist, no kernel arguments. `odinlink-local.patch` deliberately
registers only the `tb_protocol_handler` fields mainline defines — `.uuid` and
`.callback` — so `odl_tb5.ko` compiles against the stock headers in
`/lib/modules/$(uname -r)/build`.

That costs one thing, and it is worth knowing before adding a second cable.
Mainline hands protocol callbacks the packet but not its source XDomain, so
peers are demultiplexed by route. Routes are domain-local and two
point-to-point links put the peer at the *same* route, which is why the
single-cable rule below is a rule and not a preference. Distinguishing them
needs a `.callback_xd` field the mainline struct does not have, i.e. a patched
thunderbolt core.

Earlier revisions of this repo did carry that patched core (the `tbv` stack).
If you set a box up that way, run [`uninstall-tbv.sh`](uninstall-tbv.sh)
before building — in particular it removes the `blacklist=thunderbolt` kernel
arguments, which otherwise leave the box with no thunderbolt driver at all.

## Pieces

- `build-odinlink.sh` — fetches OdinLink at the pin, applies
  `odinlink-local.patch`, builds the kernel driver against the stock kernel
  headers, and builds the userspace `libodl_tb5` the readiness helper links
  against. No sudo. The RCCL net plugin and the `odl_ar2` all-reduce library
  are built INTO the serving image by `container/Dockerfile` (odinlink-build
  stage) — the kernel driver is the only per-kernel host build.
- `install-odinlink.sh` — run with sudo after the build: stages the driver,
  library and udev rule under `/usr/local`, compiles the `odl-state`
  readiness helper, installs `odl-swap.sh` and enables `odinlink.service`
  (loads the driver at boot and gates on the cross-host handshake reaching
  READY).
- `odl-swap.sh [ring] [e2e] [busy_poll_us]` — the actual load
  path, also usable by hand on a live box. Installs the udev rule, **signs
  `odl_tb5.ko` with the host's enrolled MOK when one is present**
  (`/var/lib/shim-signed/mok/`, the key DKMS enrolls; override with
  `ODL_MOK_PRIV`/`ODL_MOK_DER`), unloads any stale `thunderbolt_ibverbs`,
  loads `odl_tb5`, and waits for READY. With no
  enrolled key the signing step is a no-op and Secure Boot must be disabled.
- `odinlink-local.patch` — three bug fixes on the pinned upstream, and
  nothing else. Connection restart after a failed DMA verify (without it the
  link parks in CONNECTED and wedges until a module reload);
  verify-survives-peer-relogin (both ends restarting otherwise phase-lock and
  never converge); and the RCCL plugin's logger-ABI segfault, where a 2-arg
  call into a 6-arg `ncclDebugLogger_t` made RCCL dereference the device count
  as a filename. No tuning and no topology workarounds: every module parameter
  except ring size is left at the driver default.
- `ar2/` — `odl_ar2`: the decode all-reduce carried over OdinLink
  streams (HIP + ctypes wrapper). Wired into the engine by `DS4_ODL_AR2=1`
  (branch carried in `container/patches/vllm-upstream.patch`, wrapper in
  `container/rootfs/.../odl_ar2.py`, library built in-image). The `odl`
  transport profile sets it; the `tcp` fallback profile does not.
- `odl_state.c`, `odl_pingpong.c` — link state probe (installed as
  `/usr/local/bin/odl-state`) and RTT benchmark.
- `71-odl-tb5.rules` — device-node permissions + keeps NHI runtime PM on so
  XDomain hotplug events are not missed.

## Bring-up (both boxes)

```bash
sudo odinlink/uninstall-tbv.sh --apply  # ONLY if this box ran the old tbv stack; then reboot
odinlink/build-odinlink.sh             # driver + lib, against the stock kernel headers
sudo odinlink/install-odinlink.sh      # stage /usr/local + enable odinlink.service
sudo systemctl start odinlink.service  # or reboot; start both boxes promptly together
```

Reliable link order: load the driver first (the service does this at boot),
then connect the cable. Once `odl-state` reports `state=ready` on both boxes,
bring the cluster up normally (`transport: odl` is the default in
`ds4-config.yaml`).

## Operating constraints

- **Single cable only**: with two cables the identical-route XDomains cross
  DMA paths — see "Why this builds on a stock kernel" above. tbnet coexists on
  the same XDomain and carries the control plane (`thunderbolt0`).
- `ring_size=1024` is the ceiling on these boxes (no CMA pool; the 4096
  ring's 16 MB contiguous alloc fails). `cma=256M` on the kernel cmdline
  would lift it. `odl-swap.sh` defaults to 1024 for that reason.
- The driver is vermagic-locked to the running kernel: rebuild it after a
  kernel update.
- Measure decode with non-streamed `usage.completion_tokens`: SSE chunks
  pack several MTP tokens each, so chunk counting understates decode.
