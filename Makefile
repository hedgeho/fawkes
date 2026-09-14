# Fawkes shortcuts: run the cloaking CLI locally or on the Triton cluster (Slurm, via `hpc run`).
#
#   make cloak     IMGS=./imgs TARGET=./target MODE=mid          # this machine
#   make hpc-cloak IMGS=./imgs TARGET=./target MODE=high         # on the cluster clone, GPU node
#   make hpc-models                                              # prefetch weights on the login node
#
# Extra CLI options go in ARGS, e.g. ARGS="--batch-size 16 --debug" (long names; see FAWKES below). `make help` lists the knobs.

IMGS   ?= ./imgs
TARGET ?= ./target
MODE   ?= mid
ARGS   ?=

# Slurm resources for hpc-cloak. `hpc run` only defaults --cpus-per-task=1 and --time=0:25:00, and
# a bare --gpus=1 lands on a V100 partition (compute capability 7.0): the cu128 torch wheels ship no
# Volta kernels, so the run fails. Pin the Ampere/Hopper partitions the harness jobs use instead.
HPC_PARTITION ?= gpu-a100-80g,gpu-h100-80g-short
HPC_TIME      ?= 1:00:00
HPC_MEM       ?= 32G
HPC_CPUS      ?= 6
HPC_OPTS      ?=

# Long options only: `hpc run` strips any `-m <text>` (its --comment shorthand) from the whole command line.
FAWKES = uv run fawkes --directory $(IMGS) --target-dir $(TARGET) --mode $(MODE) $(ARGS)

.PHONY: help cloak hpc-cloak hpc-models

help:
	@echo "Targets:"
	@echo "  cloak       run the CLI here:            $(FAWKES)"
	@echo "  hpc-cloak   same inside srun on a GPU node (hpc run --partition=$(HPC_PARTITION) --gpus=1"
	@echo "              --mem=$(HPC_MEM) --cpus-per-task=$(HPC_CPUS) --time=$(HPC_TIME) $(HPC_OPTS))"
	@echo "  hpc-models  download the detector and surrogate weights on the login node (compute nodes are offline)"
	@echo
	@echo "Variables: IMGS TARGET MODE ARGS  |  HPC_PARTITION HPC_TIME HPC_MEM HPC_CPUS HPC_OPTS"

cloak:
	$(FAWKES)

# Compute nodes have no internet (HF_HUB_OFFLINE), the home quota is small (uv cache on scratch),
# and Slurm cpusets break onnxruntime's thread defaults (OMP_NUM_THREADS). srun exports the
# environment to the job, so these apply inside the allocation.
hpc-cloak:
	HF_HUB_OFFLINE=1 UV_CACHE_DIR=/scratch/work/$$USER/.uv-cache OMP_NUM_THREADS=$(HPC_CPUS) \
	hpc run --job-name=fawkes-cloak --partition=$(HPC_PARTITION) --gpus=1 \
	    --mem=$(HPC_MEM) --cpus-per-task=$(HPC_CPUS) --time=$(HPC_TIME) $(HPC_OPTS) \
	    $(FAWKES)

hpc-models:
	UV_CACHE_DIR=/scratch/work/$$USER/.uv-cache uv run python -m fawkes.models download
