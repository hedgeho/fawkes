"""Target identity: embed a few photos of the person to mimic, and persist the result so that
later runs push new photos toward the same point in feature space."""
import hashlib
import os
from pathlib import Path

import numpy as np

from fawkes.align import make_crop

TARGET_FILE = "fawkes_target.npz"
# cosine above which the user's own faces are already close to the target under a surrogate
TOO_SIMILAR = 0.35


def _weights_tag(keys):
    """Short identifier of the surrogate set and their weight files, so a stale cache is not reused."""
    from fawkes.models import SURROGATES, download
    h = hashlib.sha256()
    for k in sorted(keys):
        h.update(k.encode())
        spec = SURROGATES[k]
        h.update((spec.sha256 or Path(download(k)).stat().st_size.__str__()).encode())
    return h.hexdigest()[:16]


def target_crops(image_paths, images, detector, verbose=True):
    """One aligned crop per target photo. Photos with no face or with several faces are skipped."""
    crops = []
    for path, img in zip(image_paths, images):
        faces = detector.detect(img)
        if len(faces) != 1:
            if verbose:
                print("Target photo {}: {} faces found, skipped (need exactly one)".format(
                    os.path.basename(path), len(faces)))
            continue
        crops.append(make_crop(img, faces[0].bbox, faces[0].kps))
    return crops


def build_target(crops, cloaker):
    """Mean unit embedding per surrogate: {key: (d,) float32}."""
    if not crops:
        raise ValueError("no usable target photo (each must contain exactly one face)")
    embs = cloaker.embed(crops)
    out = {}
    for k, e in embs.items():
        mean = e.mean(axis=0)
        out[k] = (mean / np.linalg.norm(mean)).astype(np.float32)
    return out


def save_target(path, target, keys, tag=None):
    np.savez(path, __tag__=np.array(tag or _weights_tag(keys)), __keys__=np.array(sorted(keys)), **target)


def load_target(path, keys, tag=None):
    """The persisted target for these surrogate keys, or None if absent or built for other weights."""
    if not os.path.exists(path):
        return None
    data = np.load(path)
    if str(data["__tag__"]) != (tag or _weights_tag(keys)):
        return None
    if set(data["__keys__"].tolist()) != set(keys):
        return None
    return {k: data[k] for k in keys}


def get_or_build_target(target_dir, keys, detector, cloaker, verbose=True):
    """Load `<target_dir>/fawkes_target.npz` if it matches the surrogates, else build and save it."""
    from fawkes.utils import filter_image_paths
    path = os.path.join(target_dir, TARGET_FILE)
    tag = _weights_tag(keys)
    target = load_target(path, keys, tag)
    if target is not None:
        if verbose:
            print("Using saved target embeddings from {}".format(path))
        return target
    paths = sorted(p for p in Path(target_dir).iterdir() if p.is_file() and p.name != TARGET_FILE)
    paths, images = filter_image_paths([str(p) for p in paths])
    crops = target_crops(paths, images, detector, verbose)
    target = build_target(crops, cloaker)
    save_target(path, target, keys, tag)
    if verbose:
        print("Built target from {} photo(s), saved to {}".format(len(crops), path))
    return target


def similarity_warning(user_embeddings, target):
    """Message if the user's faces already resemble the target under any surrogate, else None."""
    lines = []
    for k, e in user_embeddings.items():
        if len(e) == 0:
            continue
        centroid = e.mean(axis=0)
        centroid /= np.linalg.norm(centroid)
        c = float(centroid @ target[k])
        if c > TOO_SIMILAR:
            lines.append("{}: cosine {:.2f}".format(k, c))
    if not lines:
        return None
    return ("Warning: the faces to protect already look like the target to some models ({}). "
            "A target that looks less like you gives stronger protection.".format(", ".join(lines)))
