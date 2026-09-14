#!/usr/bin/env bash
# Usage: eval/run_mode.sh <mode> [extra harness args...]
# Runs the LFW harness for the v2 cloaker in <mode> against the four held-out evaluators, then
# re-evaluates the same cloaks after JPEG-75 re-encoding. Each mode uses its own workdir. This is
# how the "Final modes" tables in RESULTS.md were produced.
set -euo pipefail
cd "$(dirname "$0")/.."
mode="${1:?mode}"; shift
work="eval/work/${mode}"
ev=(--evaluator buffalo_l --evaluator antelopev2 --evaluator adaface_vit_b --evaluator lvface_t)
echo "=== v2 ${mode} $(date)"
uv run python eval/harness.py --cloaker v2 --mode "${mode}" --batch-size 16 --workdir "${work}" "${ev[@]}" "$@"
echo "=== v2 ${mode} jpeg75 $(date)"
uv run python eval/harness.py --cloaker v2 --mode "${mode}" --batch-size 16 --workdir "${work}" --jpeg 75 --skip-cloak "${ev[@]}" "$@"
echo "=== done $(date)"
