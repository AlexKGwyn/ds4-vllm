#!/usr/bin/env bash
# Install the already-built OdinLink driver and userspace readiness helper.
set -euo pipefail
[ "$(id -u)" = 0 ] || { echo "run with sudo" >&2; exit 1; }

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
HOME_DIR=${ODL_HOME:-}
if [ -z "$HOME_DIR" ] && [ -n "${SUDO_USER:-}" ] && [ "$SUDO_USER" != root ]; then
    HOME_DIR=$(eval echo "~$SUDO_USER")
fi
[ -n "$HOME_DIR" ] || { echo "set ODL_HOME when installing outside sudo" >&2; exit 1; }

DRIVER_DIR=${ODL_DRIVER_DIR:-$HOME_DIR/.cache/odinlink/driver}
LIB_BUILD_DIR=${ODL_LIB_BUILD_DIR:-$HOME_DIR/.cache/odinlink/build/lib}
LIB_INCLUDE_DIR=${ODL_LIB_INCLUDE_DIR:-$HOME_DIR/.cache/odinlink/lib/include}

[ -f "$DRIVER_DIR/odl_tb5.ko" ] || { echo "missing built odl_tb5.ko" >&2; exit 1; }
[ -f "$DRIVER_DIR/71-odl-tb5.rules" ] || { echo "missing OdinLink udev rule" >&2; exit 1; }
[ -f "$LIB_BUILD_DIR/libodl_tb5.so" ] || { echo "missing built libodl_tb5.so" >&2; exit 1; }

install -d -m 0755 /usr/local/lib/odinlink /usr/local/share/odinlink
install -m 0644 "$DRIVER_DIR/odl_tb5.ko" /usr/local/lib/odinlink/odl_tb5.ko
install -m 0644 "$DRIVER_DIR/71-odl-tb5.rules" /usr/local/share/odinlink/71-odl-tb5.rules
cp -a "$LIB_BUILD_DIR"/libodl_tb5.so* /usr/local/lib/odinlink/

gcc -O2 -I "$LIB_INCLUDE_DIR" \
    "$SCRIPT_DIR/odl_state.c" \
    -L /usr/local/lib/odinlink -Wl,-rpath,/usr/local/lib/odinlink \
    -lodl_tb5 -o /usr/local/bin/odl-state
install -m 0755 "$SCRIPT_DIR/odl-swap.sh" /usr/local/sbin/odl-swap.sh
install -m 0644 "$SCRIPT_DIR/systemd/odinlink.service" /etc/systemd/system/odinlink.service
systemctl daemon-reload
systemctl enable odinlink.service

echo "installed persistent OdinLink service; start it on both peers together"
