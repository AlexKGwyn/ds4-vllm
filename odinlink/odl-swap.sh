#!/usr/bin/env bash
# Swap thunderbolt_ibverbs -> odl_tb5 (real link mode) on THIS box.
# Run with sudo on box1, then on box2 (any order, promptly after each other).
# Usage: odl-swap.sh [ring=4096] [e2e=1] [busy_poll_us=0] [rx_poll_ns=3000]
set -e
[ "$(id -u)" = 0 ] || { echo "run with sudo"; exit 1; }
HOME_DIR=${ODL_HOME:-}
if [ -z "$HOME_DIR" ] && [ -n "${SUDO_USER:-}" ] && [ "$SUDO_USER" != root ]; then
    HOME_DIR=$(eval echo "~$SUDO_USER")
fi
KO=${ODL_KO:-${HOME_DIR:+$HOME_DIR/.cache/odinlink/driver/odl_tb5.ko}}
RULE=${ODL_UDEV_RULE:-${HOME_DIR:+$HOME_DIR/.cache/odinlink/driver/71-odl-tb5.rules}}
RING=${1:-4096}
E2E=${2:-1}
BUSY=${3:-0}
RXPOLL=${4:-3000}
BIND_ANY=${ODL_BIND_ANY:-0}
MAX_DEVICES=${ODL_MAX_DEVICES:-1}
[ -n "$KO" ] && [ -f "$KO" ] || { echo "no OdinLink module; set ODL_KO or ODL_HOME"; exit 1; }
[ -n "$RULE" ] && [ -f "$RULE" ] || { echo "no OdinLink udev rule; set ODL_UDEV_RULE or ODL_HOME"; exit 1; }

# Keep an established peer link intact when systemd adopts an already-loaded
# driver or receives an idempotent start request.
if [ -e /sys/module/odl_tb5 ] && [ -n "${ODL_STATE:-}" ] && [ -x "$ODL_STATE" ]; then
    state_output=$($ODL_STATE 2>&1 || true)
    if printf '%s\n' "$state_output" | grep -q 'state=ready'; then
        echo "OdinLink already READY: $state_output"
        exit 0
    fi
fi

install -m644 "$RULE" /etc/udev/rules.d/
udevadm control --reload

# Secure Boot rejects an unsigned out-of-tree module. Reuse the host's
# already-enrolled DKMS/MOK key when present; each host signs its own artifact.
# See "Secure Boot" in the repo README -- with no enrolled key this is a no-op
# and Secure Boot must be disabled.
SIGN_FILE="/lib/modules/$(uname -r)/build/scripts/sign-file"
MOK_PRIV=${ODL_MOK_PRIV:-/var/lib/shim-signed/mok/MOK.priv}
MOK_DER=${ODL_MOK_DER:-/var/lib/shim-signed/mok/MOK.der}
if [ -x "$SIGN_FILE" ] && [ -r "$MOK_PRIV" ] && [ -r "$MOK_DER" ]; then
    "$SIGN_FILE" sha256 "$MOK_PRIV" "$MOK_DER" "$KO"
    echo "signed odl_tb5 with enrolled host MOK: $(modinfo -F signer "$KO")"
fi

rmmod odl_tb5 2>/dev/null && echo "removed old odl_tb5 (loopback)" || true
rmmod thunderbolt_ibverbs 2>/dev/null && echo "removed thunderbolt_ibverbs" || true
insmod "$KO" odl_ring_size=$RING e2e=$E2E odl_busy_poll_us=$BUSY rx_poll_ns=$RXPOLL bind_any=$BIND_ANY max_devices=$MAX_DEVICES
echo "loaded odl_tb5 (real link, odl_ring_size=$RING e2e=$E2E odl_busy_poll_us=$BUSY rx_poll_ns=$RXPOLL bind_any=$BIND_ANY max_devices=$MAX_DEVICES)"

# A device node only proves probe succeeded. The RCCL plugin requires the
# cross-host DMA handshake to reach READY, so a persistent service must not
# report success before this gate passes.
if [ -n "${ODL_STATE:-}" ]; then
    [ -x "$ODL_STATE" ] || { echo "no executable ODL_STATE helper: $ODL_STATE"; exit 1; }
    ready=0
    for _ in $(seq 1 90); do
        state_output=$($ODL_STATE 2>&1 || true)
        if printf '%s\n' "$state_output" | grep -q 'state=ready'; then
            ready=1
            break
        fi
        sleep 1
    done
    if [ "$ready" != 1 ]; then
        echo "OdinLink did not reach READY: $state_output" >&2
        rmmod odl_tb5 2>/dev/null || true
        exit 1
    fi
    echo "OdinLink READY: $state_output"
fi
ls -l /dev/odl_tb5* 2>/dev/null || echo "no device node yet (peer not up?)"
journalctl -k --since -1min --no-pager | grep -i odl | tail -5
