#!/usr/bin/env bash
# Check Docker Hub mirrors with `skopeo inspect`.
# The image comes from the first argument or `PROBE_DEFAULT_IMG`.

set -u

IMG="${1:-${PROBE_DEFAULT_IMG:-}}"
if [[ -z "$IMG" ]]; then
    echo "ERROR: no image given. Pass it as the first argument, e.g." >&2
    echo "  bash probe_mirrors.sh swebench/swesmith.x86_64.paramiko_1776_paramiko.23f92003" >&2
    exit 1
fi

# Candidate mirrors (April 2026, trimmed to non-whitelist hosts).
MIRRORS=(
    docker.xuanyuan.me
    docker.1ms.run
    docker.1panel.live
    hub.rat.dev
    dockerproxy.net
    proxy.vvvv.ee
    docker.xiaogenban1993.com
    docker.yomansunter.com
    lispy.org
    registry.cyou
)

source ${PSRL_WORKSPACE}/env/psrl.sh >/dev/null 2>&1 || true

echo "Probing ${#MIRRORS[@]} mirrors against image:"
echo "  $IMG"
echo
printf '%-32s  %-8s  %s\n' MIRROR RESULT MESSAGE
printf '%-32s  %-8s  %s\n' "--------------------------------" "--------" "----"

for m in "${MIRRORS[@]}"; do
    out=$(timeout 25 skopeo inspect "docker://$m/$IMG" 2>&1 >/dev/null)
    rc=$?
    if [[ $rc -eq 0 ]]; then
        printf '%-32s  %-8s  %s\n' "$m" "OK" "manifest reachable"
    elif [[ $rc -eq 124 ]]; then
        printf '%-32s  %-8s  %s\n' "$m" "TIMEOUT" "> 25s"
    else
        # Trim noisy error for a one-line summary.
        msg=$(echo "$out" | tr '\n' ' ' | sed -E 's/.*level=fatal msg="([^"]{0,120}).*/\1/' | cut -c1-120)
        printf '%-32s  %-8s  %s\n' "$m" "FAIL" "$msg"
    fi
done
