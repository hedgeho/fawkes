#!/usr/bin/env python
"""LFW evaluation harness: how well do Fawkes cloaks transfer to held-out face recognisers?

Protocol (see eval/README.md):
  1. pick protected, clean and (for v2) target identities from LFW with a seed;
  2. materialise their photos as PNGs under the work dir;
  3. cloak the *train* photos of the *protected* identities with the chosen cloaker;
  4. for each evaluator (the adversary's own detect + align + embed pipeline) embed everything,
     train a logistic-regression probe on the train embeddings (cloaked for protected identities)
     and measure how often it still recognises the clean test photos of the protected identities.
     With --clean-gallery the roles are swapped: the probe is trained on the clean test photos and
     asked to recognise the cloaked train photos (an adversary that already holds clean photos).

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
    targets: list        # list[Identity]: one shared target, or one per protected identity

    def target_for(self, k):
        """Target identity of protected identity number k."""
        return self.targets[k % len(self.targets)]


def select_identities(counts, seed, n_protected, n_clean, train_per_id, test_per_id, n_targets=1):
    """Deterministically choose protected / clean / target identities and split their photos.

    `counts` maps identity name -> sorted list of its photo file names. Only identities with
    enough photos are candidates. Returns a Split; all three groups are disjoint. The protected
    and clean sets do not depend on `n_targets`.
    """
    rng = np.random.default_rng(seed)
    need = train_per_id + test_per_id
    candidates = sorted(name for name, files in counts.items() if len(files) >= need)
    n_total = n_protected + n_clean + n_targets
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
                 targets=identities[n_protected + n_clean:])


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
    """Write <workdir>/<identity>/<train|test>/<n>.png for every identity and
    <workdir>/_targets/<target>/<n>.png for every target. Returns {protected name: target dir}."""
    for ident in split.protected + split.clean:
        for part in ("train", "test"):
            out_dir = os.path.join(workdir, ident.name, part)
            os.makedirs(out_dir, exist_ok=True)
            for n, src in enumerate(getattr(ident, part)):
                write_png(src, os.path.join(out_dir, f"{n}.png"))
    target_dirs = {}
    for k, ident in enumerate(split.protected):
        target = split.target_for(k)
        target_dir = os.path.join(workdir, "_targets", target.name)
        os.makedirs(target_dir, exist_ok=True)
        for n, src in enumerate(target.train):
            write_png(src, os.path.join(target_dir, f"{n}.png"))
        target_dirs[ident.name] = target_dir
    return target_dirs


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


def cloak_none(paths, args, target_dirs):
    for p in paths:
        shutil.copyfile(p, cloaked_path(p))


def cloak_legacy(paths, args, target_dirs):
    from fawkes.protection import Fawkes
    rc = Fawkes(gpu=None, mode=args.mode).run_protection(paths, batch_size=args.batch_size)
    if rc == 2:
        print("legacy cloaker: no face detected in any photo")
    elif rc == 3:
        raise SystemExit("legacy cloaker: no images found")


def parse_cloak_args(items):
    """['steps=120', 'self_weight=1.0', 'models=a,b'] -> {'steps': 120, 'self_weight': 1.0, 'models': ['a', 'b']}"""
    out = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"--cloak-arg expects name=value, got {item!r}")
        name, value = item.split("=", 1)
        if name == "models":
            out[name] = value.split(",")
            continue
        for cast in (int, float):
            try:
                out[name] = cast(value)
                break
            except ValueError:
                continue
        else:
            out[name] = value
    return out


def cloak_v2(paths, args, target_dirs):
    """One Fawkes instance; the photos are grouped by target so the models load once. With a
    --target-strategy other than random, each protected identity's photos are cloaked together and
    Fawkes picks one target for them from the pool (fawkes.target_pool), excluding every harness identity;
    one_person stops Fawkes from splitting an identity into several "people" by embedding."""
    if args.target_strategy != "random":
        from fawkes.protection import Fawkes
        by_identity = {}
        for p in paths:
            by_identity.setdefault(os.path.basename(os.path.dirname(os.path.dirname(p))), []).append(p)
        protector = Fawkes(mode=args.mode, target_dir=None, batch_size=args.batch_size,
                           target_strategy=args.target_strategy, target_exclude=args.harness_identities,
                           remember_targets=False, one_person=True,
                           **parse_cloak_args(args.cloak_arg))
        for ident, group in by_identity.items():
            n_before = len(protector.chosen_targets)
            rc = protector.run_protection(group)
            args.chosen_targets[ident] = [r for _, r in protector.chosen_targets[n_before:]]
            if rc not in (None, 1):
                print(f"v2 cloaker returned {rc}")
        return
    groups = {}
    for p in paths:
        groups.setdefault(target_dirs[p], []).append(p)
    protector = None
    for target_dir, group in groups.items():
        if protector is None:
            from fawkes.protection import Fawkes
            protector = Fawkes(mode=args.mode, target_dir=target_dir, batch_size=args.batch_size,
                               **parse_cloak_args(args.cloak_arg))
        else:
            protector.set_target_dir(target_dir)
        rc = protector.run_protection(group)
        if rc not in (None, 1):
            print(f"v2 cloaker returned {rc}")


CLOAKERS = {"none": cloak_none, "legacy": cloak_legacy, "v2": cloak_v2}


def run_cloaker(args, paths, target_dirs):
    """Cloak `paths` in place (writes <n>_cloaked.png next to each); `target_dirs` maps each path to
    its target directory. Returns seconds per photo and the list of photos for which the cloaker
    produced no output (they are evaluated uncloaked)."""
    remove_stale_cloaks(paths)
    t0 = time.perf_counter()
    CLOAKERS[args.cloaker](paths, args, target_dirs)
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
        from fawkes.models import onnx_session_options
        self.name = pack
        self.app = FaceAnalysis(name=pack, allowed_modules=["detection", "recognition"],
                                providers=["CPUExecutionProvider"], sess_options=onnx_session_options())
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
        from fawkes.models import onnx_session_options
        self.app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection"], providers=["CPUExecutionProvider"],
                                sess_options=onnx_session_options())
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
        self.db = sqlite3.connect(path, timeout=600)  # several evaluation jobs may share one cache
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


def chroma_stats(a, b, lf_fraction=0.02):
    """Visible-tint statistics of the difference b - a (HxWx3 uint8 arrays, the face box).

    The difference is split into luma and Cb/Cr chroma (BT.601). Returns the rms chroma magnitude,
    and the rms of the chroma and luma after a Gaussian low-pass with sigma = `lf_fraction` of the
    longer side (2 percent of the face: the smooth colour patches a viewer sees as a tint, as
    opposed to the fine texture SSIM already measures)."""
    from scipy.ndimage import gaussian_filter
    d = b.astype(np.float32) - a.astype(np.float32)
    r, g, bl = d[..., 0], d[..., 1], d[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * bl
    cb = -0.168736 * r - 0.331264 * g + 0.5 * bl
    cr = 0.5 * r - 0.418688 * g - 0.081312 * bl
    sigma = max(0.5, lf_fraction * max(d.shape[:2]))
    lp = lambda c: gaussian_filter(c, sigma)
    rms = lambda c: float(np.sqrt(np.mean(c ** 2)))
    return {"chroma_rms": rms(np.sqrt(cb ** 2 + cr ** 2)),
            "chroma_lf_rms": rms(np.sqrt(lp(cb) ** 2 + lp(cr) ** 2)),
            "luma_lf_rms": rms(lp(y))}


def image_quality(pairs):
    """pairs: list of (clean_png, cloaked_png). Mean PSNR and DSSIM over the whole photo, plus DSSIM
    and the luma / chroma statistics of `chroma_stats` over the bounding box of the changed pixels
    (the face crop, for cloakers that only touch the face)."""
    from skimage.metrics import structural_similarity, peak_signal_noise_ratio
    psnr, dssim, dssim_face = [], [], []
    tint = {"chroma_rms": [], "chroma_lf_rms": [], "luma_lf_rms": []}
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
            for k, v in chroma_stats(fa, fb).items():
                tint[k].append(v)
    mean = lambda v: float(np.mean(v)) if len(v) else None
    out = {"psnr": mean(psnr), "dssim": mean(dssim), "dssim_face": mean(dssim_face)}
    out.update({k + "_face": mean(v) for k, v in tint.items()})
    return out


# Held-out age judge: MiVOLO v2 (VOLO transformer, Kuprashevich et al. 2024; weights Apache-2.0),
# age output only, as an ONNX export of iitolstykh/mivolo_v2. tests/test_harness.py checks it against
# the official PyTorch model (eval/README.md has the recipe). Neither its architecture nor its data
# is shared with insightface genderage, which the cloak's age penalty optimises against.
MIVOLO_REPO = "Sam-Apostel/mivolo-v2-age-onnx"
MIVOLO_FILE = "mivolo_v2_age.onnx"
MIVOLO_REVISION = "8eb4cd8f5dd4bd28df2c9a43b24bc54d6d139d36"
MIVOLO_SHA256 = "2db5e05be33b3f120518a86f29a4a65eb0a01c2967219b8787a7dead9b796951"


class MiVoloAge:
    """Apparent age of a face crop with MiVOLO v2, face input only (MiVOLO's own 'no body' input)."""

    SIZE = 384
    MEAN = np.array([0.485, 0.456, 0.406], np.float32)
    STD = np.array([0.229, 0.224, 0.225], np.float32)

    def __init__(self):
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from fawkes.models import model_dir, onnx_session_options, sha256sum
        path = model_dir() / "mivolo_v2_age" / MIVOLO_FILE
        if not path.exists():
            path = hf_hub_download(MIVOLO_REPO, MIVOLO_FILE, revision=MIVOLO_REVISION,
                                   local_dir=str(model_dir() / "mivolo_v2_age"))
            if sha256sum(path) != MIVOLO_SHA256:
                os.remove(path)
                raise RuntimeError(f"sha256 mismatch for {MIVOLO_FILE}; file removed")
        self.sess = ort.InferenceSession(str(path), onnx_session_options(), providers=["CPUExecutionProvider"])
        self.input = self.sess.get_inputs()[0].name
        self.no_body = np.broadcast_to((-self.MEAN / self.STD)[:, None, None], (3, self.SIZE, self.SIZE))

    def letterbox(self, rgb):
        """MiVOLO's class_letterbox: scale the longer side to SIZE, pad with black, centred."""
        import cv2
        h, w = rgb.shape[:2]
        r = min(self.SIZE / h, self.SIZE / w)
        nw, nh = int(round(w * r)), int(round(h * r))
        if (nw, nh) != (w, h):
            rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
        dw, dh = (self.SIZE - nw) / 2, (self.SIZE - nh) / 2
        top, bottom, left, right = (int(round(dh - 0.1)), int(round(dh + 0.1)),
                                    int(round(dw - 0.1)), int(round(dw + 0.1)))
        return cv2.copyMakeBorder(rgb, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    def predict_crop(self, rgb_crop):
        x = (self.letterbox(np.ascontiguousarray(rgb_crop)).astype(np.float32) / 255.0 - self.MEAN) / self.STD
        blob = np.concatenate([x.transpose(2, 0, 1), self.no_body])[None].astype(np.float32)
        return float(self.sess.run(None, {self.input: blob})[0].reshape(-1)[0])

    def predict(self, rgb, bbox):
        """Age in years of the face in `bbox` (x0, y0, x1, y1) of an RGB photo."""
        h, w = rgb.shape[:2]
        x0, y0, x1, y1 = [int(round(v)) for v in bbox]
        return self.predict_crop(rgb[max(y0, 0):min(y1, h), max(x0, 0):min(x1, w)])


def perceptual_quality(pairs):
    """LPIPS (AlexNet) over the face box of the change and the apparent-age shift of the largest face
    (on the clean face's box, in years, cloaked minus clean) by insightface genderage and by the
    held-out MiVOLO v2. Genderage is not a held-out measure once a cloak penalises age; LPIPS is not
    once a cloak penalises LPIPS."""
    import torch
    from fawkes.detect import Detector
    from fawkes.target_pool import AgeGender
    try:
        from fawkes.cloak import lpips_distance, lpips_model
        net = lpips_model(torch.device("cpu"))
    except ImportError:
        net = None
    detector, ages, mivolo = Detector(), AgeGender(), MiVoloAge()
    lp, d_age, d_mivolo = [], [], []
    for clean_p, cloaked_p in pairs:
        a = np.asarray(Image.open(clean_p).convert("RGB"))
        b = np.asarray(Image.open(cloaked_p).convert("RGB"))
        if a.shape != b.shape or np.array_equal(a, b):
            continue
        faces = detector.detect(a)
        if faces:
            d_age.append(ages.predict(b, faces[0].bbox)[0] - ages.predict(a, faces[0].bbox)[0])
            d_mivolo.append(mivolo.predict(b, faces[0].bbox) - mivolo.predict(a, faces[0].bbox))
        if net is not None:
            ys, xs = np.where((a != b).any(axis=-1))
            box = np.s_[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
            ta = torch.from_numpy(a[box].astype(np.float32)).permute(2, 0, 1)[None]
            tb = torch.from_numpy(b[box].astype(np.float32)).permute(2, 0, 1)[None]
            with torch.no_grad():
                lp.append(float(lpips_distance(net, tb, ta)[0]))
    mean = lambda v: float(np.mean(v)) if len(v) else None
    pct = lambda v, q: float(np.percentile(v, q)) if len(v) else None
    return {"lpips_face": mean(lp), "age_shift": mean(d_age), "age_shift_p90": pct(d_age, 90),
            "age_shift_mivolo": mean(d_mivolo), "age_shift_mivolo_p10": pct(d_mivolo, 10),
            "age_shift_mivolo_p90": pct(d_mivolo, 90)}


# --------------------------------------------------------------------------- evaluation

def probe_rows(split, workdir, cloaked_files, clean_gallery=False):
    """(identity, path) rows for the linear probe: the gallery it is trained on and the probes it
    must recognise. Default: gallery = train photos (cloaked for protected identities), probes =
    clean test photos. --clean-gallery: gallery = clean test photos, probes = train photos (cloaked
    for protected identities), i.e. the adversary enrolled the person before the cloaks existed."""
    train_rows, test_rows = [], []
    for ident in split.protected + split.clean:
        for n in range(len(ident.train)):
            p = photo_path(workdir, ident, "train", n)
            train_rows.append((ident.name, cloaked_files.get(p, p)))
        for n in range(len(ident.test)):
            test_rows.append((ident.name, photo_path(workdir, ident, "test", n)))
    return (test_rows, train_rows) if clean_gallery else (train_rows, test_rows)


def evaluate(evaluator, cache, split, workdir, cloaked_files, clean_gallery=False):
    """Embed everything with one evaluator and compute the probe + verification metrics."""
    prot_names = {i.name for i in split.protected}
    gallery_rows, probe_list = probe_rows(split, workdir, cloaked_files, clean_gallery)
    clean_train_rows = [(i.name, photo_path(workdir, i, "train", n)) for i in split.protected for n in range(len(i.train))]
    cloaked_rows = [(i, cloaked_files.get(p, p)) for i, p in clean_train_rows]
    clean_test_rows = [(i.name, photo_path(workdir, i, "test", n)) for i in split.protected for n in range(len(i.test))]

    paths = sorted({p for _, p in gallery_rows + probe_list + clean_train_rows + cloaked_rows + clean_test_rows})
    emb = cache.embed_files(evaluator, paths)
    undetected = sorted(p for p in paths if emb[p] is None)
    keep = lambda rows: [(i, emb[p]) for i, p in rows if emb[p] is not None]
    gallery, probes = keep(gallery_rows), keep(probe_list)

    metrics = probe_metrics(np.stack([e for _, e in gallery]), [i for i, _ in gallery],
                            np.stack([e for _, e in probes]), [i for i, _ in probes], prot_names)
    metrics["gallery"] = "clean test photos" if clean_gallery else "cloaked train photos"
    # 1:1 verification is always cloaked train photo against the clean test centroid of the same identity
    test_by_id = {}
    for i, e in keep(clean_test_rows):
        test_by_id.setdefault(i, []).append(e)
    metrics.update(verification_metrics(keep(cloaked_rows), keep(clean_train_rows), test_by_id))
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
    lines += ["", "| cloaked photos | uncloaked (cloaker gave no output) | PSNR dB | DSSIM photo | DSSIM face box "
              "| chroma rms face | chroma LF rms face | luma LF rms face | LPIPS face | age shift (p90) "
              "| MiVOLO age shift (p10, p90) | s/photo |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|",
              f"| {q['n_cloaked']} | {q['n_uncloaked']} | {fmt(q['psnr'], 1)} | {fmt(q['dssim'], 4)} | "
              f"{fmt(q['dssim_face'], 4)} | {fmt(q.get('chroma_rms_face'), 2)} | {fmt(q.get('chroma_lf_rms_face'), 2)} | "
              f"{fmt(q.get('luma_lf_rms_face'), 2)} | {fmt(q.get('lpips_face'), 3)} | "
              f"{fmt(q.get('age_shift'), 1)} ({fmt(q.get('age_shift_p90'), 1)}) | "
              f"{fmt(q.get('age_shift_mivolo'), 1)} ({fmt(q.get('age_shift_mivolo_p10'), 1)}, "
              f"{fmt(q.get('age_shift_mivolo_p90'), 1)}) | {fmt(q['seconds_per_photo'], 1)} |"]
    if results.get("chosen_targets"):
        lines += ["", "Targets: " + "; ".join(f"{k}: {', '.join(v)}" for k, v in results["chosen_targets"].items())]
    return "\n".join(lines)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cloaker", choices=sorted(CLOAKERS), default="none")
    ap.add_argument("--mode", default="mid", help="cloaker mode (low/mid/high)")
    ap.add_argument("--batch-size", type=int, default=1, help="cloaker optimisation batch size")
    ap.add_argument("--cloak-arg", action="append", default=None, metavar="NAME=VALUE",
                    help="v2 cloaker parameter override, repeatable (e.g. steps=120, self_weight=1.0)")
    ap.add_argument("--target-strategy", choices=["random", "near", "far"], default="random",
                    help="random: a random LFW identity per protected identity (the default protocol); near / far: "
                         "Fawkes picks a matched target (sex, age, skin tone) from its pool, see fawkes.target_pool")
    ap.add_argument("--shared-target", action="store_true",
                    help="all protected identities mimic one target (default: one target per identity)")
    ap.add_argument("--tag", default=None, help="extra label for the results file and report header")
    ap.add_argument("--jpeg", type=int, default=None, metavar="Q", help="re-encode cloaked photos as JPEG quality Q")
    ap.add_argument("--clean-gallery", action="store_true",
                    help="train the probe on the clean test photos and recognise the cloaked train photos "
                         "(the adversary enrolled the person from clean photos before the cloaks existed)")
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
    ap.add_argument("--cloak-only", action="store_true",
                    help="cloak and stop (GPU job); evaluate later with --skip-cloak on a CPU node")
    ap.add_argument("--cache", default=None, help="embedding cache file (default: <workdir>/embeddings.sqlite); "
                    "entries are keyed by file hash, so workdirs may share one")
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
    split = select_identities(counts, args.seed, args.n_protected, args.n_clean, args.train_per_id, args.test_per_id,
                              n_targets=1 if args.shared_target else args.n_protected)
    target_dirs = materialise(split, workdir)
    print(f"protected: {[i.name for i in split.protected]}\nclean: {[i.name for i in split.clean]}\n"
          f"targets: {[i.name for i in split.targets]}")

    args.harness_identities = [i.name for i in split.protected + split.clean + split.targets]
    args.chosen_targets = {}
    to_cloak = [photo_path(workdir, i, "train", n) for i in split.protected for n in range(len(i.train))]
    path_targets = {photo_path(workdir, i, "train", n): target_dirs[i.name]
                    for i in split.protected for n in range(len(i.train))}
    info_path = os.path.join(workdir, "cloak_info.json")
    if args.skip_cloak:
        seconds_per_photo = args.seconds_per_photo
        uncloaked = [p for p in to_cloak if not os.path.exists(cloaked_path(p))]
        if os.path.exists(info_path):  # written by an earlier --cloak-only run
            info = json.load(open(info_path))
            seconds_per_photo = seconds_per_photo or info["seconds_per_photo"]
            args.chosen_targets = info["chosen_targets"]
    else:
        print(f"cloaking {len(to_cloak)} photos with {args.cloaker} (mode {args.mode}) ...")
        seconds_per_photo, uncloaked = run_cloaker(args, to_cloak, path_targets)
        print(f"cloaked in {fmt(seconds_per_photo, 1)} s/photo; {len(uncloaked)} photos got no cloak")
        with open(info_path, "w") as f:
            json.dump({"seconds_per_photo": seconds_per_photo, "chosen_targets": args.chosen_targets}, f)
        if args.cloak_only:
            print(f"cloak only: evaluate later with --skip-cloak --workdir {args.workdir}")
            return None

    quality = image_quality([(p, cloaked_path(p)) for p in to_cloak])
    quality.update(perceptual_quality([(p, cloaked_path(p)) for p in to_cloak]))
    quality.update(n_cloaked=len(to_cloak) - len(uncloaked), n_uncloaked=len(uncloaked), seconds_per_photo=seconds_per_photo)

    cloaked_files = {p: cloaked_path(p) for p in to_cloak}
    if args.jpeg is not None:
        cloaked_files = {p: jpeg_reencode(c, args.jpeg) for p, c in cloaked_files.items()}

    cache = EmbeddingCache(args.cache or os.path.join(workdir, "embeddings.sqlite"))
    results = {"args": vars(args), "split": asdict(split), "quality": quality, "evaluators": {},
               "chosen_targets": args.chosen_targets}
    for key in args.evaluator:
        print(f"evaluating with {key} ...")
        results["evaluators"][key] = evaluate(make_evaluator(key), cache, split, workdir, cloaked_files,
                                              clean_gallery=args.clean_gallery)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    tag = (f"{time.strftime('%Y%m%d-%H%M%S')}_{args.cloaker}_{args.mode}" + (f"_{args.tag}" if args.tag else "")
           + (f"_jpeg{args.jpeg}" if args.jpeg else "") + ("_cleangallery" if args.clean_gallery else "")
           + ("_smoke" if args.smoke else ""))
    out = os.path.join(RESULTS_DIR, tag + ".json")
    with open(out, "w") as f:
        json.dump(results, f, indent=1, default=str)
    report = markdown_report(results)
    print(f"\n### {args.cloaker} / {args.mode}" + (f" / {args.tag}" if args.tag else "")
          + (f" / jpeg {args.jpeg}" if args.jpeg else "") + (" / clean gallery" if args.clean_gallery else "")
          + "\n\n" + report)
    print(f"\nwritten {os.path.relpath(out, ROOT)}")
    return results


if __name__ == "__main__":
    main()
