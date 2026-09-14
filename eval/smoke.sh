#!/usr/bin/env bash
# Smoke test: test suite, model bench, then one mid-mode cloak of a few LFW photos with timing.
set -euo pipefail
cd "$(dirname "$0")/.."
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
uv run python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
uv run pytest -q -x
uv run python -m fawkes.models bench --batch 8 || true
rm -rf out/smoke && mkdir -p out/smoke/imgs out/smoke/target
uv run python - <<'PY'
# whole LFW photos (250x250 funneled JPEGs) as the harness uses them: two people to cloak, one target
import importlib.util
spec = importlib.util.spec_from_file_location("harness", "eval/harness.py")
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)
counts = h.load_lfw()
for name in ["George_W_Bush", "Colin_Powell"]:
    for j, src in enumerate(counts[name][:3]):
        h.write_png(src, f"out/smoke/imgs/{name}_{j}.png")
for j, src in enumerate(counts["Tiger_Woods"][:5]):
    h.write_png(src, f"out/smoke/target/{j}.png")
PY
uv run python -m fawkes -d out/smoke/imgs -t out/smoke/target -m mid --debug --batch-size 8
ls -la out/smoke/imgs
