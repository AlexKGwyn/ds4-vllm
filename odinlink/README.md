# OdinLink transport

Runs the DS4 cluster's inter-box fabric on the OdinLink Thunderbolt RDMA
driver ([Geramy/OdinLink-Five](https://github.com/Geramy/OdinLink-Five),
fetched at a pinned revision — nothing vendored) instead of the tbv ibverbs
stack. An alternative transport, selected with `transport: odl` in
`ds4-config.yaml`; the tbv stack remains in `tbv/` and is restored by setting
`transport: rdma`, re-enabling `tbv-roce.service`, and rebooting.

The tbv base layer is still a prerequisite: OdinLink's driver builds against
the tbv-patched thunderbolt headers and runs on top of the patched
`thunderbolt` core that `tbv-thunderbolt-patched.service` loads at boot. Do
[`AGENTS.md`](../AGENTS.md) §1.1 first. Only `thunderbolt_ibverbs` is
replaced — `odinlink.service` carries `Conflicts=tbv-roce.service`, so the
two RDMA bring-ups are mutually exclusive by construction.

## Pieces

- `build-odinlink.sh` — fetches OdinLink at the pin, applies
  `odinlink-local.patch`, builds the kernel driver against the tbv-patched
  thunderbolt header (mandatory: stock headers ABI-mismatch the running
  thunderbolt module and oops on attach) and against the running module's
  exported symbol CRCs, and builds the userspace `libodl_tb5` the readiness
  helper links against. No sudo. The RCCL net plugin and the `odl_ar2`
  all-reduce library are built INTO the serving image by
  `container/Dockerfile` (odinlink-build stage) — the kernel driver is the
  only per-kernel host build, as with the tbv modules.
- `install-odinlink.sh` — run with sudo after the build: stages the driver,
  library and udev rule under `/usr/local`, compiles the `odl-state`
  readiness helper, installs `odl-swap.sh` and enables `odinlink.service`
  (loads the driver at boot and gates on the cross-host handshake reaching
  READY).
- `odl-swap.sh [ring] [e2e] [busy_poll_us] [rx_poll_ns]` — the actual load
  path (also usable manually to swap a live box from tbv ibverbs to
  OdinLink). Installs the udev rule, **signs `odl_tb5.ko` with the host's
  enrolled MOK when one is present** (`/var/lib/shim-signed/mok/`, the key
  DKMS enrolls; override with `ODL_MOK_PRIV`/`ODL_MOK_DER`), unloads
  `thunderbolt_ibverbs`, loads `odl_tb5`, and waits for READY. With no
  enrolled key the signing step is a no-op and Secure Boot must be disabled,
  exactly as for the tbv modules (README "Prerequisites").
- `odinlink-local.patch` — our driver/plugin fixes on the pinned upstream:
  XDomain-aware protocol demux (`callback_xd`), connection restart after
  failed DMA verify, verify-survives-peer-relogin, the plugin's logger-ABI
  segfault fix, and the runtime-tunable `rx_poll_ns` ring-poll cadence
  (writable via `/sys/module/odl_tb5/parameters/rx_poll_ns`).
- `ar2/` — `odl_ar2`: the tbv_ar2-class decode all-reduce ported to OdinLink
  streams (HIP + ctypes wrapper). Wired into the engine by `DS4_ODL_AR2=1`
  (branch carried in `container/patches/vllm-upstream.patch`, wrapper in
  `container/rootfs/.../odl_ar2.py`, library built in-image). Inert unless
  the env sets `DS4_ODL_AR2=1` — the default rdma/tcp profiles are
  unaffected.
- `odl_state.c`, `odl_pingpong.c` — link state probe (installed as
  `/usr/local/bin/odl-state`) and RTT benchmark.
- `71-odl-tb5.rules` — device-node permissions + keeps NHI runtime PM on so
  XDomain hotplug events are not missed.

## Bring-up (both boxes)

```bash
odinlink/build-odinlink.sh            # driver + lib (needs tbv/build-modules.sh done once)
sudo odinlink/install-odinlink.sh     # stage /usr/local + enable odinlink.service
sudo systemctl disable --now tbv-roce.service   # odl and tbv-roce are exclusive
sudo systemctl start odinlink.service           # or reboot; start both boxes promptly together
```

Reliable link order: load the driver first (the service does this at boot),
then connect the cable. Once `odl-state` reports `state=ready` on both boxes,
set `transport: odl` in `~/ds4-config.yaml` and bring the cluster up normally
(`systemctl --user start ds4-vllm`).

## Operating constraints

- **Single cable only**: with two cables the identical-route XDomains cross
  DMA paths (unsolved upstream). tbnet coexists on the same XDomain and
  carries the control plane (`thunderbolt0`, 192.168.100.x).
- `ring_size=1024` is the ceiling on these boxes (no CMA pool; the 4096
  ring's 16 MB contiguous alloc fails). `cma=256M` on the kernel cmdline
  would lift it. The systemd unit loads with `1024 1`.
- The driver is vermagic- and symbol-CRC-locked to the running (tbv-patched)
  kernel/thunderbolt pair: rebuild after any kernel or tbv rebuild.
- Measure decode with non-streamed `usage.completion_tokens`: SSE chunks
  pack several MTP tokens each, so chunk counting understates decode.
