#!/usr/bin/env bash
# Build the OdinLink host pieces for this box: the odl_tb5 kernel driver and
# the userspace libodl_tb5 (which install-odinlink.sh links the odl-state
# readiness helper against). Third-party sources are fetched at pinned
# revisions, not vendored.
#
# The RCCL net plugin and the odl_ar2 all-reduce library are NOT built here --
# they need ROCm and are built INTO the serving image by container/Dockerfile
# (odinlink-build stage). The kernel driver is the only per-kernel host build.
#
# Layout produced under $WORK (default ~/.cache/odinlink):
#   driver/odl_tb5.ko       kernel driver (install with ../install-odinlink.sh)
#   build/lib/libodl_tb5.so userspace lib (for the odl-state readiness gate)
#
# Builds against the STOCK kernel headers -- no patched thunderbolt core, no
# out-of-tree module set. odinlink-local.patch deliberately registers only the
# tb_protocol_handler fields mainline defines (.uuid/.callback); see
# README.md "Why this builds on a stock kernel".
set -euo pipefail

ODL_REPO=https://github.com/Geramy/OdinLink-Five
ODL_PIN=4534f58
WORK=${WORK:-$HOME/.cache/odinlink}
HERE=$(cd "$(dirname "$0")" && pwd)

if [ ! -d "$WORK/.git" ]; then
    git clone "$ODL_REPO" "$WORK"
fi
cd "$WORK"
git fetch -q origin
git checkout -q "$ODL_PIN"
git reset --hard -q "$ODL_PIN"
git clean -qfdx
git apply --check "$HERE/odinlink-local.patch"
git apply "$HERE/odinlink-local.patch"

make -C driver
echo "driver: $(modinfo -F vermagic driver/odl_tb5.ko)"

# Userspace libodl_tb5: five plain-C files over the driver's uapi -- no ROCm,
# no cmake configure of the whole tree needed on the host. Same soname/version
# scheme as the project's cmake build so the odl-state link and the in-image
# artifacts stay interchangeable.
mkdir -p build/lib
gcc -O2 -fPIC -shared -Wl,-soname,libodl_tb5.so.0 \
    -I lib/include -I driver/uapi \
    lib/src/odl_tb5_dev.c lib/src/odl_tb5_xfer.c lib/src/odl_tb5_peer.c \
    lib/src/odl_tb5_completion.c lib/src/odl_tb5_stream.c \
    -o build/lib/libodl_tb5.so.0.1.0
ln -sf libodl_tb5.so.0.1.0 build/lib/libodl_tb5.so.0
ln -sf libodl_tb5.so.0 build/lib/libodl_tb5.so
echo "lib: build/lib/libodl_tb5.so"

# Keep the udev rule next to the driver so install-odinlink.sh / odl-swap.sh
# find both in one place.
cp "$HERE/71-odl-tb5.rules" driver/71-odl-tb5.rules

echo "done. Install driver + service: sudo $HERE/install-odinlink.sh"
