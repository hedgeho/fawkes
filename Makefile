# Fawkes shortcuts.
#
#   make cloak  IMGS=./imgs TARGET=./target MODE=mid   # run the CLI (GPU when available; see README)
#   make models                                        # prefetch the detector and surrogate weights
#
# Extra CLI options go in ARGS, e.g. ARGS="--batch-size 16 --debug". `make help` lists the knobs.

IMGS   ?= ./imgs
TARGET ?= ./target
MODE   ?= mid
ARGS   ?=

FAWKES = uv run fawkes --directory $(IMGS) --target-dir $(TARGET) --mode $(MODE) $(ARGS)

.PHONY: help cloak models

# Double-colon so Makefile.local can append its own `help::` lines.
help::
	@echo "Targets:"
	@echo "  cloak    run the CLI:  $(FAWKES)"
	@echo "  models   download the detector and surrogate weights"
	@echo
	@echo "Variables: IMGS TARGET MODE ARGS"

cloak:
	$(FAWKES)

models:
	uv run python -m fawkes.models download

# Site-specific targets (e.g. a Slurm cluster) live in Makefile.local, which is gitignored.
-include Makefile.local
