"""Automatic target choice: a pool of public identities (LFW) described by their surrogate
embeddings, apparent age, sex and skin tone, and a rule that picks, for the faces to protect, a
different person who looks broadly alike.

Why: the cloak moves a face toward its target in every surrogate's feature space, and the cheapest
directions are the attributes that differ. A target decades older with another skin tone makes the
optimiser add shading that reads as wrinkles and a tint (decision 13). A target of the same sex,
similar age and skin tone, but a different identity, leaves only identity to change.

    python -m fawkes.target_pool build [--min-photos 5]   # once; needs the `eval` extra (LFW via sklearn)
    python -m fawkes.target_pool show                     # list the pool
"""
import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fawkes.align import make_crop, norm_crop

POOL_FILE = "target_pool/lfw_pool.npz"
GENDERAGE_REPO = "public-data/insightface"
GENDERAGE_FILE = "models/buffalo_l/genderage.onnx"
GENDERAGE_SHA256 = "4fde69b1c810857b88c64a335084f1c3fe8f01246c9a191b48c7bb756d6652fb"
# same-person threshold for grouping the faces to protect (cosine of the first surrogate)
SAME_PERSON = 0.4


def pool_path():
    from fawkes.models import model_dir
    return model_dir() / POOL_FILE


# ----------------------------------------------------------------------------- face attributes

def genderage_path():
    """Local path of insightface's genderage.onnx (1.3 MB), downloaded and checked on first use."""
    from huggingface_hub import hf_hub_download
    from fawkes.models import model_dir, sha256sum
    path = model_dir() / GENDERAGE_FILE
    if not path.exists():
        path = Path(hf_hub_download(GENDERAGE_REPO, GENDERAGE_FILE, local_dir=str(model_dir())))
        if sha256sum(path) != GENDERAGE_SHA256:
            path.unlink()
            raise RuntimeError("sha256 mismatch for genderage.onnx; file removed")
    return path


def genderage_matrix(bbox, size=96):
    """2x3 map from image pixels to genderage's input: the box centre to the middle, its longer side
    to size / 1.5, no rotation (insightface's Attribute.get)."""
    x0, y0, x1, y1 = [float(v) for v in bbox]
    s = size / (max(x1 - x0, y1 - y0) * 1.5)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return np.array([[s, 0, size / 2 - s * cx], [0, s, size / 2 - s * cy]], np.float32)


class TorchAge:
    """genderage as a differentiable torch function: apparent age in years of (N,3,H,W) [0,255] RGB
    crops, each with its genderage matrix (N,2,3). Used by the cloak's age penalty."""

    def __init__(self, device):
        from fawkes.onnx_torch import OnnxModule
        self.net = OnnxModule(genderage_path()).to(device).eval()

    def __call__(self, x, matrices):
        from fawkes.align import warp_to_template
        aligned = warp_to_template(x / 255.0, matrices, image_size=96) * 255.0
        return self.net(aligned)[:, 2] * 100.0


class AgeGender:
    """insightface's genderage (buffalo_l, 96x96 MobileNet) on CPU: apparent age and sex of a face."""

    def __init__(self):
        from insightface.model_zoo import get_model
        from fawkes.models import onnx_session_options
        self.model = get_model(str(genderage_path()), providers=["CPUExecutionProvider"],
                               sess_options=onnx_session_options())
        self.model.prepare(ctx_id=-1)

    def predict(self, rgb, bbox):
        """(age in years, is_male) of the face in `bbox` of an RGB photo."""
        from insightface.app.common import Face
        face = Face(bbox=np.asarray(bbox, np.float32))
        bgr = np.ascontiguousarray(np.asarray(rgb)[..., ::-1]).astype(np.uint8)
        gender, age = self.model.get(bgr, face)
        return float(age), bool(gender == 1)


# cheek patches of the 112x112 ArcFace template (x0, y0, x1, y1): skin away from eyes, brows, beard line
_CHEEKS = ((28, 60, 42, 76), (70, 60, 84, 76))


def skin_lab(rgb, kps):
    """Median CIELab (L 0-100, a, b) of the cheeks of the face with landmarks `kps`."""
    import cv2
    aligned = norm_crop(np.clip(np.asarray(rgb), 0, 255).astype(np.uint8), np.asarray(kps, np.float32))
    lab = cv2.cvtColor(aligned.astype(np.float32) / 255.0, cv2.COLOR_RGB2Lab)
    pixels = np.concatenate([lab[y0:y1, x0:x1].reshape(-1, 3) for x0, y0, x1, y1 in _CHEEKS])
    return np.median(pixels, axis=0).astype(np.float32)


@dataclass
class Person:
    """What the target rule needs about one person: unit centroid per surrogate and attributes."""
    embedding: dict  # {key: (d,) unit vector}
    age: float
    male: bool
    lab: np.ndarray  # (3,)


def describe(faces, images, crops, cloaker, age_model, embeddings=None):
    """A Person from several photos of one person. `faces`: DetectedFace per crop, `images`: the photo
    each face comes from, `crops`: FaceCrops; `embeddings` ({key: (n, d)}) if already computed."""
    embs = embeddings if embeddings is not None else cloaker.embed(crops)
    centroid = {}
    for k, e in embs.items():
        c = e.mean(axis=0)
        centroid[k] = (c / np.linalg.norm(c)).astype(np.float32)
    ages, males, labs = [], [], []
    for face, img in zip(faces, images):
        age, male = age_model.predict(img, face.bbox)
        ages.append(age)
        males.append(male)
        labs.append(skin_lab(img, face.kps))
    return Person(embedding=centroid, age=float(np.median(ages)), male=bool(np.mean(males) >= 0.5),
                  lab=np.median(np.stack(labs), axis=0))


def group_faces(embeddings, key, threshold=SAME_PERSON):
    """Greedy grouping of faces into people by cosine under surrogate `key`: list of index lists."""
    e = embeddings[key]
    groups, centroids = [], []
    for i in range(len(e)):
        sims = [float(e[i] @ c) for c in centroids]
        if sims and max(sims) >= threshold:
            j = int(np.argmax(sims))
            groups[j].append(i)
            c = e[groups[j]].mean(axis=0)
            centroids[j] = c / np.linalg.norm(c)
        else:
            groups.append([i])
            centroids.append(e[i] / np.linalg.norm(e[i]))
    return groups


# ----------------------------------------------------------------------------- the pool

@dataclass
class Pool:
    names: list
    age: np.ndarray  # (n,)
    male: np.ndarray  # (n,) bool
    lab: np.ndarray  # (n, 3)
    embedding: dict  # {key: (n, d)}

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, names=np.array(self.names), age=self.age, male=self.male, lab=self.lab,
                 **{"emb_" + k: v for k, v in self.embedding.items()})

    @classmethod
    def load(cls, path=None):
        path = Path(path or pool_path())
        if not path.exists():
            raise FileNotFoundError(f"no target pool at {path}; build it with `python -m fawkes.target_pool build` "
                                    "or pass --target-dir")
        d = np.load(path)
        emb = {k[4:]: d[k] for k in d.files if k.startswith("emb_")}
        return cls(names=d["names"].tolist(), age=d["age"], male=d["male"].astype(bool), lab=d["lab"], embedding=emb)

    def target(self, i, keys):
        missing = [k for k in keys if k not in self.embedding]
        if missing:
            raise KeyError(f"the target pool has no embeddings for {missing}; rebuild it")
        return {k: self.embedding[k][i].astype(np.float32) for k in keys}


def select(pool, person, keys, strategy="near", exclude=(), max_cos=0.2, age_tol=(8, 12, 20), n_skin=5):
    """Index of the pool identity to use as target for `person`, and a one-line reason.

    Candidates must be of the same apparent sex and a different person (cosine below `max_cos` under
    every surrogate in `keys`); the age window widens through `age_tol` until `n_skin` remain; of those
    the `n_skin` nearest in skin colour (CIE76 delta E) are kept, and `strategy` picks the one most
    ("near") or least ("far") similar in feature space. Near targets need the smallest change to
    reach; far targets were the original Fawkes choice. Raises ValueError if no identity qualifies.
    """
    if strategy not in ("near", "far"):
        raise ValueError(f"strategy must be 'near' or 'far', got {strategy!r}")
    cos = np.stack([pool.embedding[k] @ person.embedding[k] for k in keys], axis=1)  # (n, n_keys)
    ok = (cos.max(axis=1) < max_cos) & (pool.male == person.male)
    ok &= ~np.isin(np.array(pool.names), list(exclude))
    age_gap = np.abs(pool.age - person.age)
    for tol in tuple(age_tol) + (np.inf,):
        cand = np.flatnonzero(ok & (age_gap <= tol))
        if len(cand) >= n_skin:
            break
    if len(cand) == 0:
        raise ValueError("no identity in the target pool qualifies; pass --target-dir")
    delta_e = np.linalg.norm(pool.lab[cand] - person.lab[None], axis=1)
    cand = cand[np.argsort(delta_e, kind="stable")[:n_skin]]
    mean_cos = cos[cand].mean(axis=1)
    i = int(cand[np.argmax(mean_cos) if strategy == "near" else np.argmin(mean_cos)])
    reason = "{} (age {:.0f} vs {:.0f}, skin dE {:.1f}, mean cosine {:.2f})".format(
        pool.names[i], pool.age[i], person.age, float(np.linalg.norm(pool.lab[i] - person.lab)),
        float(cos[i].mean()))
    return i, reason


ASSIGNMENTS_FILE = "target_pool/assignments.json"


class Assignments:
    """Remembers which pool target each person got, so later batches of photos of the same person are
    pushed toward the same target (a recogniser must see one consistent false identity). A person is
    recognised by the cosine of their centroid under one surrogate (SAME_PERSON)."""

    def __init__(self, path=None):
        from fawkes.models import model_dir
        import json
        self.path = Path(path or model_dir() / ASSIGNMENTS_FILE)
        self.entries = json.loads(self.path.read_text()) if self.path.exists() else []

    def lookup(self, key, centroid, threshold=SAME_PERSON):
        best, name = threshold, None
        for e in self.entries:
            if e["key"] == key:
                c = float(np.dot(e["centroid"], centroid))
                if c >= best:
                    best, name = c, e["target"]
        return name

    def add(self, key, centroid, target):
        import json
        self.entries.append({"key": key, "centroid": [float(v) for v in centroid], "target": target})
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.entries))


def build(keys, device=None, min_photos=5, max_photos=8, lfw_dir=None, verbose=True):
    """Describe every LFW identity with at least `min_photos` single-face photos (using at most
    `max_photos` of them) under the surrogates `keys`."""
    from fawkes.cloak import Cloaker
    from fawkes.detect import Detector
    from fawkes.models import load_surrogate, default_device
    from fawkes.utils import load_image
    if lfw_dir is None:
        from sklearn.datasets import fetch_lfw_people, get_data_home
        fetch_lfw_people(min_faces_per_person=20, color=True, resize=1.0, funneled=True)  # downloads all of LFW
        lfw_dir = os.path.join(get_data_home(), "lfw_home", "lfw_funneled")
    device = device or default_device()
    cloaker = Cloaker({k: load_surrogate(k, device) for k in keys})
    detector, age_model = Detector(), AgeGender()
    names = sorted(n for n in os.listdir(lfw_dir)
                   if os.path.isdir(os.path.join(lfw_dir, n)) and len(os.listdir(os.path.join(lfw_dir, n))) >= min_photos)
    people, kept = [], []
    for j, name in enumerate(names):
        files = sorted(f for f in os.listdir(os.path.join(lfw_dir, name)) if f.lower().endswith(".jpg"))
        faces, images, crops = [], [], []
        for f in files:
            img = load_image(os.path.join(lfw_dir, name, f))
            found = detector.detect(img) if img is not None else []
            # LFW photos are centred on the person; the largest face is them, a second one is background
            if not found:
                continue
            faces.append(found[0])
            images.append(img)
            crops.append(make_crop(img, found[0].bbox, found[0].kps))
            if len(crops) >= max_photos:
                break
        if len(crops) < min(min_photos, max_photos):
            continue
        people.append(describe(faces, images, crops, cloaker, age_model))
        kept.append(name)
        if verbose and (j + 1) % 50 == 0:
            print(f"  {j + 1}/{len(names)} identities")
    return Pool(names=kept, age=np.array([p.age for p in people], np.float32),
                male=np.array([p.male for p in people]), lab=np.stack([p.lab for p in people]).astype(np.float32),
                embedding={k: np.stack([p.embedding[k] for p in people]) for k in keys})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="describe the LFW identities and save the pool")
    b.add_argument("--min-photos", type=int, default=5)
    b.add_argument("--max-photos", type=int, default=8)
    b.add_argument("--lfw-dir", default=None, help="an extracted lfw_funneled directory (default: download via sklearn)")
    b.add_argument("--device", default=None)
    sub.add_parser("show", help="print the pool")
    args = ap.parse_args(argv)
    if args.cmd == "build":
        from fawkes.models import SURROGATES
        pool = build(list(SURROGATES), device=args.device, min_photos=args.min_photos, max_photos=args.max_photos,
                     lfw_dir=args.lfw_dir)
        pool.save(pool_path())
        print(f"{len(pool.names)} identities ({int(pool.male.sum())} male), saved to {pool_path()}")
    else:
        pool = Pool.load()
        for i, n in enumerate(pool.names):
            print(f"{n:40s} age {pool.age[i]:4.0f} {'M' if pool.male[i] else 'F'} Lab {np.round(pool.lab[i], 1)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
