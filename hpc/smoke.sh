#!/usr/bin/env bash
#SBATCH --job-name=fawkes-smoke
#SBATCH --time=00:25:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --output=out/%x_%j.out
# Smoke test on a GPU node: test suite, then one mid-mode cloak of the sample photo with timing.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PATH="$HOME/.local/bin:$HOME/bin:$PATH"
export UV_CACHE_DIR=/scratch/work/zalessi1/.uv-cache
export HF_HUB_OFFLINE=1
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
uv run python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
uv run pytest -q -x
uv run python -m fawkes.models bench --batch 8 || true
rm -rf out/smoke && mkdir -p out/smoke/imgs out/smoke/target
cp tests/data/aligned_112.png out/smoke/target/t.png
cp eval/work/full/_target/*.png out/smoke/target/ 2>/dev/null || true
uv run python - <<'PY'
# a photo to cloak: the LFW fixture identity is the target, so cloak a different LFW person
from sklearn.datasets import fetch_lfw_people
from PIL import Image
import numpy as np
d = fetch_lfw_people(min_faces_per_person=20, color=True, resize=1.0, funneled=True)
names = list(d.target_names)
for i, name in enumerate(["George_W_Bush", "Colin_Powell"]):
    idx = np.where(d.target == names.index(name))[0][:3]
    for j, k in enumerate(idx):
        Image.fromarray((d.images[k] * 255).astype(np.uint8)).save(f"out/smoke/imgs/{name}_{j}.png")
PY
uv run python -m fawkes -d out/smoke/imgs -t out/smoke/target -m mid --debug --batch-size 8
ls -la out/smoke/imgs
