#!/usr/bin/env bash
# Usage: eval/time_modes.sh <n> [modes...]
# Times the fawkes CLI as a user runs it (default batch size, automatic target) on 1 and on <n> LFW
# photos of one person, for each mode; the difference divided by n - 1 is the time per face without
# model loading and target selection. LFW must already be in ~/scikit_learn_data (eval/harness.py).
set -euo pipefail
cd "$(dirname "$0")/.."
n="${1:?number of photos}"; shift
modes=("$@"); [ ${#modes[@]} -gt 0 ] || modes=(low subtle mid high)
lfw="${LFW_DIR:-$HOME/scikit_learn_data/lfw_home/lfw_funneled}"
person="${TIME_PERSON:-George_W_Bush}"
work="$(mktemp -d)"
# the CLI remembers each person's target; keep the cluster's memory as it was
assign=fawkes/model/target_pool/assignments.json
[ -f "$assign" ] && cp "$assign" "$work/assignments.json"
restore() { if [ -f "$work/assignments.json" ]; then cp "$work/assignments.json" "$assign"; else rm -f "$assign"; fi; rm -rf "$work"; }
trap restore EXIT
# one untimed run first: the first start on a node pays for the uv environment check and file caches
mkdir -p "$work/warmup"
find "$lfw/$person" -name '*.jpg' | sort | awk 'NR == 1' | xargs -I{} cp {} "$work/warmup/"
uv run fawkes -d "$work/warmup" --mode low >"$work/warmup.log" 2>&1
echo "device:$(uv run python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")'), threads ${OMP_NUM_THREADS:-default}"
for m in "${modes[@]}"; do
  for k in 1 "$n"; do
    d="$work/$m-$k"; mkdir -p "$d"
    find "$lfw/$person" -name '*.jpg' | sort | awk -v k="$k" 'NR <= k' | xargs -I{} cp {} "$d/"
    rm -f "$assign"
    t0=$(date +%s.%N)
    uv run fawkes -d "$d" --mode "$m" >"$d.log" 2>&1
    t1=$(date +%s.%N)
    awk -v m="$m" -v k="$k" -v a="$t0" -v b="$t1" 'BEGIN {printf "%s %s %.1f\n", m, k, b - a}' | tee -a "$work/times"
  done
done
awk -v n="$n" '{t[$1, $2] = $3; m[$1] = 1} END {for (k in m) printf "%s: %.1f s per face (%.1f s for 1 photo, %.1f s for %d)\n", k, (t[k, n] - t[k, 1]) / (n - 1), t[k, 1], t[k, n], n}' "$work/times"
for m in "${modes[@]}"; do echo "$m, $n photos, the CLI's own count: $(grep -h 'protection cost' "$work/$m-$n.log" | tail -1)"; done
