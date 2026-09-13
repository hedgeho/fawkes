"""Fast tests for eval/harness.py; none of them download data or load a model."""
import importlib.util
import os

import numpy as np
import pytest
from PIL import Image

HARNESS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval", "harness.py")
spec = importlib.util.spec_from_file_location("harness", HARNESS)
harness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(harness)


def fake_counts(n_ids=30, n_photos=20):
    return {f"Person_{i:02d}": [f"Person_{i:02d}_{j:04d}.jpg" for j in range(1, n_photos + 1)] for i in range(n_ids)}


def test_split_is_deterministic_and_disjoint():
    counts = fake_counts()
    a = harness.select_identities(counts, seed=0, n_protected=10, n_clean=10, train_per_id=10, test_per_id=5)
    b = harness.select_identities(counts, seed=0, n_protected=10, n_clean=10, train_per_id=10, test_per_id=5)
    c = harness.select_identities(counts, seed=1, n_protected=10, n_clean=10, train_per_id=10, test_per_id=5)
    assert a == b
    assert a != c
    names = [i.name for i in a.protected] + [i.name for i in a.clean] + [t.name for t in a.targets]
    assert len(names) == len(set(names)) == 21
    for ident in a.protected + a.clean + a.targets:
        assert len(ident.train) == 10 and len(ident.test) == 5
        assert not set(ident.train) & set(ident.test)
        assert set(ident.train + ident.test) <= set(counts[ident.name])


def test_split_skips_identities_with_too_few_photos():
    counts = fake_counts(n_ids=5, n_photos=20)
    counts["Small_Person"] = ["Small_Person_0001.jpg"] * 3
    s = harness.select_identities(counts, seed=0, n_protected=2, n_clean=2, train_per_id=4, test_per_id=2)
    assert "Small_Person" not in [i.name for i in s.protected + s.clean + s.targets]
    with pytest.raises(SystemExit):
        harness.select_identities(counts, seed=0, n_protected=5, n_clean=5, train_per_id=4, test_per_id=2)


def synthetic_embeddings(n_ids=10, per_id=8, dim=64, seed=0):
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(n_ids, dim))
    x = np.concatenate([c + 0.05 * rng.normal(size=(per_id, dim)) for c in centres])
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    y = np.repeat([f"id{i}" for i in range(n_ids)], per_id)
    return x, y


def test_probe_protection_rate_separable_is_zero():
    x, y = synthetic_embeddings()
    m = harness.probe_metrics(x, y, x, y, protected={"id0", "id1", "id2"})
    assert m["protection_rate"] == pytest.approx(0.0)
    assert m["clean_test_accuracy"] == pytest.approx(1.0)


def test_probe_protection_rate_shuffled_labels_is_near_one():
    x, y = synthetic_embeddings()
    rng = np.random.default_rng(1)
    protected = {"id0", "id1", "id2"}
    # protected identities' train labels are shuffled among themselves: the probe learns nothing useful
    train_y = y.copy()
    idx = np.where(np.isin(y, list(protected)))[0]
    train_y[idx] = train_y[rng.permutation(idx)]
    m = harness.probe_metrics(x, train_y, x, y, protected)
    assert m["protection_rate"] >= 0.5
    assert m["clean_test_accuracy"] == pytest.approx(1.0)


def test_verification_metrics():
    e = np.eye(4, dtype=np.float32)
    test_by_id = {"a": [e[0], e[0]], "b": [e[1]]}
    clean = [("a", e[0]), ("b", e[1])]
    cloaked = [("a", e[2]), ("b", (e[1] + e[3]) / np.sqrt(2))]
    m = harness.verification_metrics(cloaked, clean, test_by_id)
    assert m["cos_clean_to_clean_centroid"] == pytest.approx(1.0)
    assert m["cos_cloaked_to_clean_centroid"] == pytest.approx((0 + 1 / np.sqrt(2)) / 2)
    assert m["frac_cloaked_below_threshold"] == pytest.approx(0.5)
    assert m["frac_clean_below_threshold"] == pytest.approx(0.0)


def test_probe_rows_swap_gallery_and_probes():
    split = harness.select_identities(fake_counts(), seed=0, n_protected=2, n_clean=1, train_per_id=3, test_per_id=2)
    work = "/w"
    to_cloak = [harness.photo_path(work, i, "train", n) for i in split.protected for n in range(3)]
    cloaked = {p: harness.cloaked_path(p) for p in to_cloak}
    gallery, probes = harness.probe_rows(split, work, cloaked)
    gallery_c, probes_c = harness.probe_rows(split, work, cloaked, clean_gallery=True)
    assert (gallery_c, probes_c) == (probes, gallery)
    # default: the gallery holds the cloaks, the probes are clean; clean identities are clean on both sides
    assert sorted(p for _, p in gallery if harness.CLOAKED_SUFFIX in p) == sorted(cloaked.values())
    assert not any(harness.CLOAKED_SUFFIX in p for _, p in probes)
    assert len(gallery) == 3 * 3 and len(probes) == 3 * 2
    clean_name = split.clean[0].name
    assert all(harness.CLOAKED_SUFFIX not in p for i, p in gallery if i == clean_name)


def test_jpeg_reencode_changes_bytes(tmp_path):
    rng = np.random.default_rng(0)
    png = tmp_path / "3.png"
    Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)).save(png)
    out = harness.jpeg_reencode(str(png), 75)
    assert out.endswith("3_q75.jpg")
    assert png.read_bytes() != open(out, "rb").read()
    assert not np.array_equal(np.asarray(Image.open(png)), np.asarray(Image.open(out)))


def test_image_quality_on_identical_and_perturbed(tmp_path):
    rng = np.random.default_rng(0)
    a = rng.integers(0, 255, (80, 80, 3), dtype=np.uint8)
    b = a.copy()
    b[20:50, 20:50] = np.clip(b[20:50, 20:50].astype(int) + 20, 0, 255)
    Image.fromarray(a).save(tmp_path / "a.png")
    Image.fromarray(b).save(tmp_path / "b.png")
    same = harness.image_quality([(str(tmp_path / "a.png"), str(tmp_path / "a.png"))])
    assert same["dssim"] == 0.0 and same["psnr"] is None
    diff = harness.image_quality([(str(tmp_path / "a.png"), str(tmp_path / "b.png"))])
    assert 0 < diff["dssim"] < diff["dssim_face"] and diff["psnr"] > 0


def test_per_identity_targets_keep_the_protected_and_clean_sets():
    counts = fake_counts()
    shared = harness.select_identities(counts, seed=0, n_protected=4, n_clean=3, train_per_id=10, test_per_id=5)
    per_id = harness.select_identities(counts, seed=0, n_protected=4, n_clean=3, train_per_id=10, test_per_id=5,
                                       n_targets=4)
    assert [i.name for i in shared.protected] == [i.name for i in per_id.protected]
    assert [i.name for i in shared.clean] == [i.name for i in per_id.clean]
    assert len(shared.targets) == 1 and len(per_id.targets) == 4
    assert per_id.targets[0].name == shared.targets[0].name
    names = [i.name for i in per_id.protected + per_id.clean + per_id.targets]
    assert len(set(names)) == len(names)
    assert per_id.target_for(3) is per_id.targets[3]
    assert shared.target_for(3) is shared.targets[0]


def test_parse_cloak_args():
    assert harness.parse_cloak_args(["steps=120", "self_weight=1.0", "models=a,b", "name=x"]) == {
        "steps": 120, "self_weight": 1.0, "models": ["a", "b"], "name": "x"}
    assert harness.parse_cloak_args(None) == {}
