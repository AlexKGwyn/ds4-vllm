#!/usr/bin/env bash
# uninstall-tbv.sh — remove a previous tbv (thunderbolt-ibverbs) deployment.
#
# Earlier revisions of this repo ran the fabric on tbv: a patched thunderbolt
# core plus the thunderbolt_ibverbs RoCE driver, with the stock thunderbolt
# driver blacklisted so the patched one could load in its place. OdinLink needs
# none of that — it builds against the stock kernel headers and runs on the
# stock thunderbolt core — so a box that was set up for tbv must be returned to
# the stock stack before it will work.
#
# The important line here is the kernel-argument removal. tbv adds
# rd.driver.blacklist=thunderbolt modprobe.blacklist=thunderbolt so that its own
# core wins. Leaving those in place after removing tbv's modules means NO
# thunderbolt driver loads at all: no thunderbolt0, no link between the boxes,
# and a bring-up that fails a long way from the cause.
#
# Dry-run by default; it prints exactly what it would touch and changes nothing.
# Re-run with --apply to perform the removal. Idempotent either way — safe to
# run on a box that never had tbv.
#
#   sudo odinlink/uninstall-tbv.sh            # show what would change
#   sudo odinlink/uninstall-tbv.sh --apply    # do it, then REBOOT both boxes
set -uo pipefail

APPLY=0
[ "${1:-}" = "--apply" ] && APPLY=1

if [ "$(id -u)" -ne 0 ]; then
    echo "!! run as root (sudo $0 ${1:-})" >&2
    exit 1
fi

changed=0
say() { printf '  %s\n' "$*"; }
act() {   # act <description> <command...>
    local what="$1"; shift
    changed=1
    if [ "$APPLY" = 1 ]; then
        say "$what"
        "$@" || say "   (failed, continuing: $*)"
    else
        say "would $what"
    fi
}

echo "== systemd units =="
for u in tbv-roce.service tbv-thunderbolt-patched.service; do
    if systemctl list-unit-files 2>/dev/null | grep -q "^$u"; then
        act "disable + stop $u" systemctl disable --now "$u"
    fi
    for f in "/etc/systemd/system/$u" "/etc/systemd/system/$u.d"; do
        [ -e "$f" ] && act "remove $f" rm -rf "$f"
    done
done

echo "== module blacklist =="
[ -f /etc/modprobe.d/00-tbv-thunderbolt.conf ] &&
    act "remove /etc/modprobe.d/00-tbv-thunderbolt.conf" \
        rm -f /etc/modprobe.d/00-tbv-thunderbolt.conf

echo "== kernel arguments (the one that strands thunderbolt) =="
# Read them back rather than trusting that install added them.
# Two mechanisms in the wild: grubby on a package-based Fedora, rpm-ostree
# kargs on an image-based one (Bazzite and friends, where grubby is absent).
# Check both, and check the live cmdline too so a box that had the args added
# by hand is still detected.
tb_karg_set=0
grep -qs "blacklist=thunderbolt" /proc/cmdline && tb_karg_set=1
command -v rpm-ostree >/dev/null &&
    rpm-ostree kargs 2>/dev/null | grep -q "blacklist=thunderbolt" && tb_karg_set=1
command -v grubby >/dev/null &&
    grubby --info=ALL 2>/dev/null | grep -q "blacklist=thunderbolt" && tb_karg_set=1

if [ "$tb_karg_set" = 1 ]; then
    if command -v rpm-ostree >/dev/null; then
        act "drop the thunderbolt blacklist kargs (rpm-ostree, stages a new deployment)" \
            rpm-ostree kargs --delete=rd.driver.blacklist=thunderbolt \
                             --delete=modprobe.blacklist=thunderbolt
    elif command -v grubby >/dev/null; then
        act "drop rd.driver.blacklist/modprobe.blacklist=thunderbolt from ALL kernels" \
            grubby --update-kernel=ALL \
                   --remove-args="rd.driver.blacklist=thunderbolt modprobe.blacklist=thunderbolt"
    else
        say "!! neither rpm-ostree nor grubby found — remove these from the"
        say "   bootloader arguments by hand, or thunderbolt will never load:"
        say "     rd.driver.blacklist=thunderbolt modprobe.blacklist=thunderbolt"
        changed=1
    fi
fi

echo "== udev / NetworkManager / limits =="
for f in /etc/udev/rules.d/60-rdma-persistent-naming.rules \
         /etc/NetworkManager/conf.d/99-tbv-zc-second-link.conf \
         /etc/security/limits.d/99-rdma-memlock.conf \
         /etc/systemd/system.conf.d/90-memlock.conf \
         /etc/systemd/user.conf.d/90-memlock.conf; do
    [ -f "$f" ] && act "remove $f" rm -f "$f"
done

echo "== staged modules and helpers =="
[ -d /var/lib/tbv ] && act "remove /var/lib/tbv" rm -rf /var/lib/tbv

echo "== loaded modules =="
if lsmod | grep -q '^thunderbolt_ibverbs'; then
    act "unload thunderbolt_ibverbs" rmmod thunderbolt_ibverbs
fi

if [ "$APPLY" = 1 ]; then
    systemctl daemon-reload 2>/dev/null || true
    depmod -a 2>/dev/null || true
fi

echo
if [ "$changed" = 0 ]; then
    echo ">> nothing to do — no tbv deployment found on this box."
elif [ "$APPLY" = 1 ]; then
    echo ">> tbv removed. REBOOT this box so the stock thunderbolt core loads,"
    echo "   then continue with odinlink/build-odinlink.sh (see odinlink/README.md)."
    echo "   Do both boxes before bringing the cluster up."
else
    echo ">> dry run: nothing was changed. Re-run with --apply to remove."
fi
