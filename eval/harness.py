#!/usr/bin/env python
"""LFW evaluation harness: how well do Fawkes cloaks transfer to held-out face recognisers?

Protocol (see eval/README.md):
  1. pick protected, clean and (for v2) target identities from LFW with a seed;
  2. materialise their photos as PNGs under the work dir;
  3. cloak the *train* photos of the *protected* identities with the chosen cloaker;
  4. for each evaluator (the adversary's own detect + align + embed pipeline) embed everything,
     train a logistic-regression probe on the train embeddings (cloaked for protected identities)
     and measure how often it still recognises the clean test photos of the protected identities.

Run:  uv run python eval/harness.py --cloaker legacy --mode mid [--jpeg 75] [--smoke]
"""
import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import sys
import time
import warnings
from dataclasses import dataclass, field, asdict

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_WORKDIR = os.path.join(ROOT, "eval", "work")
RESULTS_DIR = os.path.join(ROOT, "eval", "results")

# 1:1 verification threshold on cosine similarity. insightface's ArcFace-family packs (buffalo_l
# w600k_r50, antelopev2 glintr100) are usually deployed with a threshold in the 0.25-0.35 range;
# 0.3 is the conventional middle value and is used for every evaluator here.
VERIFICATION_THRESHOLD = 0.3

BUILTIN_EVALUATORS = {"buffalo_l": "w600k_r50", "antelopev2": "glintr100"}

CLOAKED_SUFFIX = "_cloaked"

# insightface 2.0 still calls the deprecated skimage SimilarityTransform.estimate on every alignment
warnings.filterwarnings("ignore", category=FutureWarning, module="insightface")


# --------------------------------------------------------------------------- data selection

@dataclass
class Identity:
    name: str            # LFW folder name, e.g. "George_W_Bush"
    train: list = field(default_factory=list)   # source photo file names
    test: list = field(default_factory=list)


@dataclass
class Split:
    protected: list      # list[Identity]
    clean: list          # list[Identity]
    target: Identity     # extra identity for the v2 cloaker's target


def select_identities(counts, seed, n_protected, n_clean, train_per_id, test_per_id):
    """Deterministically choose protected / clean / target identities and split their photos.

    `counts` maps identity name -> sorted list of its photo file names. Only identities with
    enough photos are candidates. Returns a Split; all three groups are disjoint.
    """
    rng = np.random.default_rng(seed)
    need = train_per_id + test_per_id
    candidates = sorted(name for name, files in counts.items() if len(files) >= need)
    n_total = n_protected + n_clean + 1
    if len(candidates) < n_total:
        raise SystemExit(f"need {n_total} identities with >= {need} photos, only {len(candidates)} available")
    order = rng.permutation(len(candidates))
    chosen = [candidates[i] for i in order[:n_total]]

    def make(name):
        files = list(counts[name])
        perm = rng.permutation(len(files))
        train = sorted(files[i] for i in perm[:train_per_id])
        test = sorted(files[i] for i in perm[train_per_id:need])
        return Identity(name=name, train=train, test=test)

    identities = [make(name) for name in chosen]
    return Split(protected=identities[:n_protected],
                 clean=identities[n_protected:n_protected + n_clean],
                 target=identities[-1])


def load_lfw():
    """Download LFW (funneled) via scikit-learn and return {identity: [jpeg paths]} for identities
    with >= 20 photos. The pixels are read from the extracted 250x250 JPEGs rather than from the
    125x94 benchmark crops sklearn returns, so the cloakers see whole photos."""
    from sklearn.datasets import fetch_lfw_people
    from sklearn.datasets import get_data_home

    lfw = fetch_lfw_people(min_faces_per_person=20, color=True, resize=1.0, funneled=True)
    folder = os.path.join(get_data_home(), "lfw_home", "lfw_funneled")
    counts = {}
    for display_name in lfw.target_names:
        name = display_name.replace(" ", "_")
        person_dir = os.path.join(folder, name)
        files = sorted(f for f in os.listdir(person_dir) if f.lower().endswith(".jpg"))
        counts[name] = [os.path.join(person_dir, f) for f in files]
    return counts


def write_png(src, dst):
    """Convert `src` to a PNG at `dst` unless a non-empty one exists; writes atomically so an
    interrupted run cannot leave a truncated file behind."""
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return
    tmp = dst + ".tmp"
    Image.open(src).convert("RGB").save(tmp, "PNG")
    os.replace(tmp, dst)


def materialise(split, workdir):
    """Write <workdir>/<identity>/<train|test>/<n>.png for every identity, plus <workdir>/_target/."""
    for ident in split.protected + split.clean:
        for part in ("train", "test"):
            out_dir = os.path.join(workdir, ident.name, part)
            os.makedirs(out_dir, exist_ok=True)
            for n, src in enumerate(getattr(ident, part)):
                write_png(src, os.path.join(out_dir, f"{n}.png"))
    target_dir = os.path.join(workdir, "_target")
    os.makedirs(target_dir, exist_ok=True)
    for n, src in enumerate(split.target.train):
        write_png(src, os.path.join(target_dir, f"{n}.png"))
    return target_dir


def photo_path(workdir, ident, part, n):
    return os.path.join(workdir, ident.name, part, f"{n}.png")


def cloaked_path(clean_png):
    base, _ = os.path.splitext(clean_png)
    return base + CLOAKED_SUFFIX + ".png"


# --------------------------------------------------------------------------- cloakers

def remove_stale_cloaks(paths):
    for p in paths:
        d = os.path.dirname(p)
        for f in os.listdir(d):
            if CLOAKED_SUFFIX in f:
                os.remove(os.path.join(d, f))


def cloak_none(paths, args, target_dir):
    for p in paths:
        shutil.copyfile(p, cloaked_path(p))


def cloak_legacy(paths, args, target_dir):
    from fawkes.protection import Fawkes
    rc = Fawkes(gpu=None, mode=args.mode).run_protection(paths, batch_size=args.batch_size)
    if rc == 2:
        print("legacy cloaker: no face detected in any photo")
    elif rc == 3:
        raise SystemExit("legacy cloaker: no images found")


def cloak_v2(paths, args, target_dir):
    try:
        from fawkes.protection import Fawkes
        protector = Fawkes(mode=args.mode, target_dir=target_dir)
    except (ImportError, TypeError) as e:
        raise SystemExit(f"v2 cloaker API (Fawkes(mode=..., target_dir=...)) is not available yet: {e!r}")
    rc = protector.run_protection(paths, batch_size=args.batch_size)
    if rc not in (None, 1):
        print(f"v2 cloaker returned {rc}")


CLOAKERS = {"none": cloak_none, "legacy": cloak_legacy, "v2": cloak_v2}


def run_cloaker(args, paths, target_dir):
    """Cloak `paths` in place (writes <n>_cloaked.png next to each). Returns seconds per photo and the
    list of photos for which the cloaker produced no output (they are evaluated uncloaked)."""
    remove_stale_cloaks(paths)
    t0 = time.perf_counter()
    CLOAKERS[args.cloaker](paths, args, target_dir)
    seconds = time.perf_counter() - t0
    missing = [p for p in paths if not os.path.exists(cloaked_path(p))]
    for p in missing:
        shutil.copyfile(p, cloaked_path(p))
    return seconds / max(len(paths), 1), missing


def jpeg_reencode(png_path, quality):
    """Re-encode a PNG as JPEG at `quality`; returns the new path."""
    out = os.path.splitext(png_path)[0] + f"_q{quality}.jpg"
    Image.open(png_path).convert("RGB").save(out, "JPEG", quality=quality)
    return out


# --------------------------------------------------------------------------- evaluators

def largest_face(faces):
    if not faces:
        return None
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


class Evaluator:
    """An adversary's recogniser: insightface detection + 5-point alignment + the pack's ArcFace model."""

    def __init__(self, pack):
        from insightface.app import FaceAnalysis
        self.name = pack
        self.app = FaceAnalysis(name=pack, allowed_modules=["detection", "recognition"],
                                providers=["CPUExecutionProvider"])
        self.app.prepare(ctx_id=-1, det_size=(640, 640))

    def embed_one(self, rgb):
        """Returns a (512,) L2-normalised embedding or None if no face is detected."""
        face = largest_face(self.app.get(rgb[:, :, ::-1]))  # insightface expects BGR
        if face is None:
            return None
        e = face.normed_embedding.astype(np.float32)
        return e / max(np.linalg.norm(e), 1e-12)

    def embed(self, images):
        out = [self.embed_one(img) for img in images]
        return np.stack([e for e in out if e is not None]) if any(e is not None for e in out) else np.zeros((0, 512))


class ExternalEvaluator(Evaluator):
    """Detection and alignment from insightface (buffalo_l's det_10g), embedding from
    fawkes.models.load_evaluator(key). The model receives aligned 112x112 RGB uint8 crops, either
    through `.embed(list_of_crops)` or by being called with the list."""

    def __init__(self, key):
        from insightface.app import FaceAnalysis
        from fawkes.models import load_evaluator
        self.name = key
        self.model = load_evaluator(key)
        self.app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection"], providers=["CPUExecutionProvider"])
        self.app.prepare(ctx_id=-1, det_size=(640, 640))

    def embed_one(self, rgb):
        from insightface.utils.face_align import norm_crop
        face = largest_face(self.app.get(rgb[:, :, ::-1]))
        if face is None:
            return None
        crop = norm_crop(rgb[:, :, ::-1], landmark=face.kps, image_size=112)[:, :, ::-1]  # back to RGB
        fn = getattr(self.model, "embed", self.model)
        e = np.asarray(fn([np.ascontiguousarray(crop)]), dtype=np.float32).reshape(-1)
        return e / max(np.linalg.norm(e), 1e-12)


def make_evaluator(key):
    if key in BUILTIN_EVALUATORS:
        return Evaluator(key)
    try:
        import fawkes.models  # noqa: F401
    except ImportError as e:
        raise SystemExit(f"evaluator {key!r} is not built in and fawkes.models is not importable: {e}")
    return ExternalEvaluator(key)


class EmbeddingCache:
    """sqlite cache: (evaluator, sha256 of file bytes) -> embedding blob, or NULL when no face was found."""

    def __init__(self, path):
        self.db = sqlite3.connect(path)
        self.db.execute("CREATE TABLE IF NOT EXISTS emb (evaluator TEXT, sha TEXT, vec BLOB, PRIMARY KEY (evaluator, sha))")

    def embed_files(self, evaluator, paths):
        """Returns {path: embedding or None}. Only files missing from the cache are embedded."""
        result, todo = {}, []
        for p in paths:
            with open(p, "rb") as f:
                sha = hashlib.sha256(f.read()).hexdigest()
            row = self.db.execute("SELECT vec FROM emb WHERE evaluator=? AND sha=?", (evaluator.name, sha)).fetchone()
            if row is None:
                todo.append((p, sha))
            else:
                result[p] = None if row[0] is None else np.frombuffer(row[0], dtype=np.float32)
        for i, (p, sha) in enumerate(todo):
            e = evaluator.embed_one(np.asarray(Image.open(p).convert("RGB")))
            self.db.execute("INSERT OR REPLACE INTO emb VALUES (?,?,?)",
                            (evaluator.name, sha, None if e is None else e.astype(np.float32).tobytes()))
            result[p] = e
            if (i + 1) % 50 == 0:
                self.db.commit()
                print(f"  [{evaluator.name}] embedded {i + 1}/{len(todo)}")
        self.db.commit()
        return result


# --------------------------------------------------------------------------- metrics

def probe_metrics(train_x, train_y, test_x, test_y, protected):
    """Linear-probe protection rate.

    Trains logistic regression on (train_x, train_y) and returns
      protection_rate = 1 - top-1 accuracy on the test rows whose label is in `protected`,
      clean_accuracy  = top-1 accuracy on the remaining (clean-identity) test rows.
    """
    from sklearn.linear_model import LogisticRegression
    probe = LogisticRegression(C=1.0, max_iter=2000)
    probe.fit(train_x, train_y)
    pred = probe.predict(test_x)
    test_y = np.asarray(test_y)
    prot = np.isin(test_y, list(protected))
    acc_prot = float((pred[prot] == test_y[prot]).mean()) if prot.any() else float("nan")
    acc_clean = float((pred[~prot] == test_y[~prot]).mean()) if (~prot).any() else float("nan")
    return {"protection_rate": 1.0 - acc_prot, "protected_test_accuracy": acc_prot, "clean_test_accuracy": acc_clean}


def verification_metrics(cloaked, clean_train, test_by_id, threshold=VERIFICATION_THRESHOLD):
    """cloaked / clean_train: list of (identity, embedding); test_by_id: {identity: [embeddings]}.
    Cosine of each train embedding to the clean test centroid of the same identity."""
    centroids = {}
    for ident, embs in test_by_id.items():
        if embs:
            c = np.mean(embs, axis=0)
            centroids[ident] = c / max(np.linalg.norm(c), 1e-12)

    def cos_list(pairs):
        return np.array([float(e @ centroids[i]) for i, e in pairs if i in centroids])

    cc, cl = cos_list(cloaked), cos_list(clean_train)
    return {"cos_cloaked_to_clean_centroid": float(cc.mean()) if len(cc) else float("nan"),
            "cos_clean_to_clean_centroid": float(cl.mean()) if len(cl) else float("nan"),
            "frac_cloaked_below_threshold": float((cc < threshold).mean()) if len(cc) else float("nan"),
            "frac_clean_below_threshold": float((cl < threshold).mean()) if len(cl) else float("nan"),
            "threshold": threshold}


def image_quality(pairs):
    """pairs: list of (clean_png, cloaked_png). Mean PSNR and DSSIM over the whole photo, plus DSSIM
    over the bounding box of the changed pixels (the face crop, for cloakers that only touch the face)."""
    from skimage.metrics import structural_similarity, peak_signal_noise_ratio
    psnr, dssim, dssim_face = [], [], []
    for clean_p, cloaked_p in pairs:
        a = np.asarray(Image.open(clean_p).convert("RGB"))
        b = np.asarray(Image.open(cloaked_p).convert("RGB"))
        if a.shape != b.shape:
            continue
        if np.array_equal(a, b):
            dssim.append(0.0), dssim_face.append(0.0)
            continue
        psnr.append(peak_signal_noise_ratio(a, b, data_range=255))
        dssim.append((1 - structural_similarity(a, b, channel_axis=-1, data_range=255)) / 2)
        ys, xs = np.where((a != b).any(axis=-1))
        fa, fb = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1], b[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        if min(fa.shape[:2]) >= 7:
            dssim_face.append((1 - structural_similarity(fa, fb, channel_axis=-1, data_range=255)) / 2)
    mean = lambda v: float(np.mean(v)) if len(v) else None
    return {"psnr": mean(psnr), "dssim": mean(dssim), "dssim_face": mean(dssim_face)}


# --------------------------------------------------------------------------- evaluation

def evaluate(evaluator, cache, split, workdir, cloaked_files):
    """Embed everything with one evaluator and compute the probe + verification metrics."""
    prot_names = {i.name for i in split.protected}
    train_rows, test_rows = [], []          # (identity, path)
    for ident in split.protected + split.clean:
        for n in range(len(ident.train)):
            p = photo_path(workdir, ident, "train", n)
            train_rows.append((ident.name, cloaked_files.get(p, p)))
        for n in range(len(ident.test)):
            test_rows.append((ident.name, photo_path(workdir, ident, "test", n)))
    clean_train_rows = [(i.name, photo_path(workdir, i, "train", n)) for i in split.protected for n in range(len(i.train))]

    paths = sorted({p for _, p in train_rows + test_rows + clean_train_rows})
    emb = cache.embed_files(evaluator, paths)
    undetected = sorted(p for p in paths if emb[p] is None)
    keep = lambda rows: [(i, emb[p]) for i, p in rows if emb[p] is not None]
    train, test, clean_train = keep(train_rows), keep(test_rows), keep(clean_train_rows)

    metrics = probe_metrics(np.stack([e for _, e in train]), [i for i, _ in train],
                            np.stack([e for _, e in test]), [i for i, _ in test], prot_names)
    test_by_id = {}
    for i, e in test:
        if i in prot_names:
            test_by_id.setdefault(i, []).append(e)
    metrics.update(verification_metrics([(i, e) for i, e in train if i in prot_names], clean_train, test_by_id))
    metrics["n_undetected"] = len(undetected)
    metrics["undetected"] = [os.path.relpath(p, workdir) for p in undetected]
    return metrics


def fmt(v, nd=3):
    return "-" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{nd}f}"


def markdown_report(results):
    lines = ["| evaluator | protection rate | clean-id acc | cos cloaked->centroid | cos clean->centroid "
             f"| frac cloaked < {VERIFICATION_THRESHOLD} | undetected |",
             "|---|---|---|---|---|---|---|"]
    for name, m in results["evaluators"].items():
        lines.append(f"| {name} | {fmt(m['protection_rate'])} | {fmt(m['clean_test_accuracy'])} | "
                     f"{fmt(m['cos_cloaked_to_clean_centroid'])} | {fmt(m['cos_clean_to_clean_centroid'])} | "
                     f"{fmt(m['frac_cloaked_below_threshold'])} | {m['n_undetected']} |")
    q = results["quality"]
    lines += ["", "| cloaked photos | uncloaked (cloaker gave no output) | PSNR dB | DSSIM photo | DSSIM face box | s/photo |",
              "|---|---|---|---|---|---|",
              f"| {q['n_cloaked']} | {q['n_uncloaked']} | {fmt(q['psnr'], 1)} | {fmt(q['dssim'], 4)} | "
              f"{fmt(q['dssim_face'], 4)} | {fmt(q['seconds_per_photo'], 1)} |"]
    return "\n".join(lines)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cloaker", choices=sorted(CLOAKERS), default="none")
    ap.add_argument("--mode", default="mid", help="cloaker mode (low/mid/high)")
    ap.add_argument("--batch-size", type=int, default=1, help="cloaker optimisation batch size")
    ap.add_argument("--jpeg", type=int, default=None, metavar="Q", help="re-encode cloaked photos as JPEG quality Q")
    ap.add_argument("--evaluator", action="append", default=None,
                    help="evaluator key; repeatable. Built in: " + ", ".join(BUILTIN_EVALUATORS) +
                         "; anything else is resolved with fawkes.models.load_evaluator")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-protected", type=int, default=10)
    ap.add_argument("--n-clean", type=int, default=10)
    ap.add_argument("--train-per-id", type=int, default=10)
    ap.add_argument("--test-per-id", type=int, default=5)
    ap.add_argument("--smoke", action="store_true", help="2+2 identities, 4 train + 2 test photos each")
    ap.add_argument("--workdir", default=DEFAULT_WORKDIR)
    ap.add_argument("--skip-cloak", action="store_true", help="reuse the *_cloaked.png files already in the workdir")
    ap.add_argument("--seconds-per-photo", type=float, default=None,
                    help="with --skip-cloak: record this previously measured cloaking time in the results")
    args = ap.parse_args(argv)
    if args.smoke:
        args.n_protected, args.n_clean, args.train_per_id, args.test_per_id = 2, 2, 4, 2
    if args.evaluator is None:
        args.evaluator = list(BUILTIN_EVALUATORS)
    return args


def main(argv=None):
    args = parse_args(argv)
    sys.stdout.reconfigure(line_buffering=True)  # progress shows up promptly when redirected to a file
    workdir = os.path.join(args.workdir, "smoke" if args.smoke else "full")
    os.makedirs(workdir, exist_ok=True)

    print("loading LFW ...")
    counts = load_lfw()
    split = select_identities(counts, args.seed, args.n_protected, args.n_clean, args.train_per_id, args.test_per_id)
    target_dir = materialise(split, workdir)
    print(f"protected: {[i.name for i in split.protected]}\nclean: {[i.name for i in split.clean]}\ntarget: {split.target.name}")

    to_cloak = [photo_path(workdir, i, "train", n) for i in split.protected for n in range(len(i.train))]
    if args.skip_cloak:
        seconds_per_photo = args.seconds_per_photo
        uncloaked = [p for p in to_cloak if not os.path.exists(cloaked_path(p))]
    else:
        print(f"cloaking {len(to_cloak)} photos with {args.cloaker} (mode {args.mode}) ...")
        seconds_per_photo, uncloaked = run_cloaker(args, to_cloak, target_dir)
        print(f"cloaked in {fmt(seconds_per_photo, 1)} s/photo; {len(uncloaked)} photos got no cloak")

    quality = image_quality([(p, cloaked_path(p)) for p in to_cloak])
    quality.update(n_cloaked=len(to_cloak) - len(uncloaked), n_uncloaked=len(uncloaked), seconds_per_photo=seconds_per_photo)

    cloaked_files = {p: cloaked_path(p) for p in to_cloak}
    if args.jpeg is not None:
        cloaked_files = {p: jpeg_reencode(c, args.jpeg) for p, c in cloaked_files.items()}

    cache = EmbeddingCache(os.path.join(workdir, "embeddings.sqlite"))
    results = {"args": vars(args), "split": asdict(split), "quality": quality, "evaluators": {}}
    for key in args.evaluator:
        print(f"evaluating with {key} ...")
        results["evaluators"][key] = evaluate(make_evaluator(key), cache, split, workdir, cloaked_files)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    tag = f"{time.strftime('%Y%m%d-%H%M%S')}_{args.cloaker}_{args.mode}" + (f"_jpeg{args.jpeg}" if args.jpeg else "") + ("_smoke" if args.smoke else "")
    out = os.path.join(RESULTS_DIR, tag + ".json")
    with open(out, "w") as f:
        json.dump(results, f, indent=1, default=str)
    report = markdown_report(results)
    print(f"\n### {args.cloaker} / {args.mode}" + (f" / jpeg {args.jpeg}" if args.jpeg else "") + "\n\n" + report)
    print(f"\nwritten {os.path.relpath(out, ROOT)}")
    return results


if __name__ == "__main__":
    main()
