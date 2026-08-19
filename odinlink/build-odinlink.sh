#!/usr/bin/env bash
# Build the OdinLink host pieces for this box: the odl_tb5 kernel driver and
# the userspace libodl_tb5 (which install-odinlink.sh links the odl-state
# readiness helper against). Third-party sources are fetched at pinned
# revisions, not vendored (same pattern as tbv/build-modules.sh).
#
# The RCCL net plugin and the odl_ar2 all-reduce library are NOT built here --
# they need ROCm and are built INTO the serving image by container/Dockerfile
# (odinlink-build stage). The kernel driver is the only per-kernel host build,
# exactly as with the tbv modules.
#
# Layout produced under $WORK (default ~/.cache/odinlink):
#   driver/odl_tb5.ko       kernel driver (install with ../install-odinlink.sh)
#   build/lib/libodl_tb5.so userspace lib (for the odl-state readiness gate)
#
# The driver MUST be built against the tbv-patched thunderbolt headers, not
# the stock kernel's: the running thunderbolt.ko is the westeri/tbv module and
# its struct tb_nhi/tb_protocol_handler layouts differ. Building against stock
# headers produces a module that oopses inside dma_alloc_attrs on attach.
set -euo pipefail

ODL_REPO=https://github.com/Geramy/OdinLink-Five
ODL_PIN=4534f58
WESTERI_TREE=${WESTERI_TREE:-$HOME/.cache/tbv-build/westeri-thunderbolt}
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

# ABI-matching thunderbolt header from the tbv build tree (build tbv first if
# this is missing: tbv/build-modules.sh).
[ -f "$WESTERI_TREE/include/linux/thunderbolt.h" ] || {
    echo "!! $WESTERI_TREE/include/linux/thunderbolt.h missing -- run tbv/build-modules.sh first"; exit 1; }
cp "$WESTERI_TREE/include/linux/thunderbolt.h" driver/thunderbolt-tbv.h

# CONFIG_MODVERSIONS hashes the custom running thunderbolt module's exported
# ABI. Build against those exact CRCs rather than the stock kernel's symbols;
# otherwise the signed module still fails to load with "disagrees about
# version of symbol" on every Thunderbolt API it consumes.
RUNNING_TB="$(modinfo -n thunderbolt)"
RUNNING_TB_SYMVERS="$WORK/driver/Module.symvers.running-thunderbolt"
modprobe --show-exports "$RUNNING_TB" | while IFS= read -r line; do
    # kmod prints the separator as the two characters "\\t" on these hosts.
    line="$(printf '%b' "$line")"
    read -r crc symbol <<< "$line"
    printf '%s\t%s\tthunderbolt\tEXPORT_SYMBOL\t\n' "$crc" "$symbol"
done > "$RUNNING_TB_SYMVERS"
[ -s "$RUNNING_TB_SYMVERS" ] || {
    echo "!! could not extract exported symbol CRCs from $RUNNING_TB"; exit 1; }

KBUILD_EXTRA_SYMBOLS="$RUNNING_TB_SYMVERS" make -C driver
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
