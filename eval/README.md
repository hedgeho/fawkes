# Evaluation harness

`eval/harness.py` measures how well Fawkes cloaks protect against face recognisers the cloaker
has **never seen**. It uses the public LFW dataset, so the numbers need no personal photos and can
be reproduced by anyone.

## Running

```bash
uv run python eval/harness.py --cloaker none   --smoke                 # sanity check, ~1 min
uv run python eval/harness.py --cloaker legacy --mode low --smoke      # TF Fawkes, a few minutes
uv run python eval/harness.py --cloaker legacy --mode mid              # full baseline, 15-30 min on 8 CPU cores
uv run python eval/harness.py --cloaker legacy --mode mid --jpeg 75    # same, cloaks re-encoded as JPEG
uv run python eval/harness.py --cloaker v2 --mode mid --batch-size 16 # torch pipeline, one target per identity
uv run python eval/harness.py --cloaker v2 --mode mid --cloak-arg steps=120 --cloak-arg self_weight=1 --tag long
```

Flags:

| flag | default | meaning |
|---|---|---|
| `--cloaker {none,legacy,v2}` | `none` | `none` copies the clean photo (protection should be ~0), `legacy` is the TF Fawkes, `v2` the torch pipeline |
| `--mode` | `mid` | passed to the cloaker |
| `--batch-size` | 1 | faces the v2 cloaker optimises together (16 on a GPU) |
| `--cloak-arg NAME=VALUE` | none | repeatable; overrides a `CloakParams` field of the v2 cloaker (`steps`, `eps`, `dssim_budget`, `self_weight`, `laggard`, `stop_cos`, `models=a,b`, the transformer switches `pna`, `tgr`, `sgm`, `token_mask`, `patchout`, `delta_sigma`, `grad_norm`, ...) |
| `--shared-target` | off | all protected identities mimic the same target (default: a different target per identity, as with independent users) |
| `--tag` | none | label added to the results file name and the report header |
| `--jpeg Q` | off | re-encode every cloaked photo as JPEG quality Q before the adversary sees it (a social-network upload) |
| `--evaluator KEY` | `buffalo_l`, `antelopev2` | repeatable; built-in keys are insightface packs, any other key (`adaface_vit_b`, `lvface_t`) goes through `fawkes.models.load_evaluator` |
| `--seed` | 0 | selects identities and the train/test split; everything is deterministic given the seed |
| `--n-protected`, `--n-clean` | 10, 10 | identities that get cloaked / stay clean |
| `--train-per-id`, `--test-per-id` | 10, 5 | photos per identity |
| `--smoke` | off | 2 + 2 identities, 4 train + 2 test photos |
| `--workdir` | `eval/work/` | photos, cloaks and the embedding cache |
| `--skip-cloak` | off | evaluate the `*_cloaked.png` files already in the workdir |

The first run downloads LFW (about 230 MB, cached by scikit-learn in `~/scikit_learn_data`) and the
insightface packs (`~/.insightface/models`). insightface extracts `antelopev2.zip` into a nested
`antelopev2/antelopev2/` folder and then fails to find it; move the `.onnx` files one level up.

Results go to `eval/results/<timestamp>_<cloaker>_<mode>[_tag][_jpegQ][_smoke].json` together with
all arguments and the identity split, and a markdown table is printed. `hpc/sweep.sh` runs a list of
configurations on a Slurm GPU node.

## Protocol

1. From the LFW identities with at least 20 photos, pick `n_protected + n_clean` identities with
   the seed, plus one **target** identity per protected identity (or a single one with
   `--shared-target`); the protected and clean sets do not depend on that choice. Each identity's
   photos are split into train and test. Targets are written to `<workdir>/_targets/<name>/` and
   the v2 cloaker's cached target embedding lives there. The photos are the 250x250 funneled JPEGs
   (whole photos, not the 125x94 benchmark crops), materialised as PNGs under
   `<workdir>/<identity>/<train|test>/<n>.png`.
2. Only the **train photos of the protected identities** are cloaked; the cloaker writes
   `<n>_cloaked.png` next to each input. A photo the cloaker gives no output for (no face detected)
   is evaluated uncloaked and counted in the `uncloaked` column.
3. Each evaluator is the adversary's *own* pipeline: insightface SCRFD detection, 5-landmark
   similarity alignment to 112x112 and the pack's ArcFace model, largest face per photo. Photos in
   which no face is detected are dropped and counted. Embeddings are cached in
   `<workdir>/embeddings.sqlite` keyed by (evaluator, sha256 of the file), so re-running with another
   cloaker only embeds the files that changed.

## What the numbers mean

**protection rate** = 1 − top-1 accuracy of a logistic-regression probe (sklearn, C=1) that was
trained on the train embeddings (cloaked for protected identities, clean for the clean ones) and is
asked to recognise the *clean test photos of the protected identities*. This is what a low-effort
adversary does: scrape the (cloaked) photos, embed them with an off-the-shelf model, fit a
classifier, and run it on new (uncloaked) sightings. 0 = the cloak did nothing, 1 = the adversary
never recognises a protected person. With `--cloaker none` it must be about 0.

**clean-id acc** = the same probe's accuracy on the clean identities' test photos. It should stay
near 1; if it drops, the probe or the data is broken, not the cloak.

**cos cloaked→centroid** = mean cosine similarity between each cloaked train photo and the
centroid of the same identity's clean test embeddings, next to the same number for the clean train
photos (**cos clean→centroid**). The gap shows how far the cloak moved the identity in the
adversary's feature space. **frac cloaked < 0.3** is the fraction of cloaked photos that a 1:1
verifier with the conventional 0.3 cosine threshold (`VERIFICATION_THRESHOLD`) would *not* match to
their own identity.

**PSNR / DSSIM photo / DSSIM face box** compare the cloaked PNG with the clean one (before any
JPEG re-encoding). DSSIM = (1 − SSIM)/2. "face box" restricts DSSIM to the bounding box of the
pixels the cloaker changed, which is the face crop for Fawkes and is the region its per-mode DSSIM
budget (0.004 / 0.012 / 0.017) refers to. **s/photo** is wall-clock seconds per cloaked photo on this
machine.

## Caveat: evaluators and surrogates must be disjoint

The cloak is optimised against a *surrogate* ensemble. Radiya-Dixit and Tramer (ICLR 2022) showed
that measuring protection against the surrogate itself is meaningless; only transfer to models the
cloaker never saw counts. The built-in evaluators (`buffalo_l/w600k_r50`, `antelopev2/glintr100`)
must therefore never be added to the surrogate ensemble in `fawkes/models.py`, and any custom
`--evaluator` must likewise be a model the cloaker does not train against. Also note the harness
only measures transfer to today's models: no cloak can defend against a recogniser trained
adaptively on cloaked photos after they are public.

## Custom evaluators

`--evaluator KEY` for a key that is not built in calls `fawkes.models.load_evaluator(KEY)`. The
returned object gets a list of aligned 112x112 RGB uint8 crops (insightface detection and alignment
are still used) through `.embed(crops)` if present, otherwise by calling it, and must return one
512-d vector per crop; the harness L2-normalises them.

## Tests

`uv run pytest tests/test_harness.py` covers the split determinism, the probe metric on synthetic
embeddings, the verification metric, JPEG re-encoding and the image-quality metrics, without any
downloads.
