import numpy as np
import pytest

from fawkes import align, cloak, target
from test_cloak import BBOX, KPS, DummySurrogate, _face_photo


class FakeDetector:
    def __init__(self, counts):
        self.counts = list(counts)

    def detect(self, img):
        n = self.counts.pop(0)
        face = type("F", (), {"bbox": BBOX, "kps": KPS})()
        return [face] * n


def _cloaker():
    return cloak.Cloaker({"a": DummySurrogate(1), "b": DummySurrogate(2)})


def test_target_crops_skip_photos_without_exactly_one_face(capsys):
    imgs = [_face_photo(seed=i) for i in range(3)]
    crops = target.target_crops(["a.png", "b.png", "c.png"], imgs, FakeDetector([1, 0, 2]))
    assert len(crops) == 1
    out = capsys.readouterr().out
    assert "b.png: 0 faces" in out and "c.png: 2 faces" in out


def test_build_target_is_unit_mean_and_roundtrips(tmp_path):
    cl = _cloaker()
    crops = [align.make_crop(_face_photo(seed=i), BBOX, KPS) for i in range(3)]
    t = target.build_target(crops, cl)
    assert set(t) == {"a", "b"}
    for v in t.values():
        assert abs(np.linalg.norm(v) - 1) < 1e-5
    path = tmp_path / target.TARGET_FILE
    target.save_target(path, t, ["a", "b"], tag="x")
    back = target.load_target(path, ["a", "b"], tag="x")
    np.testing.assert_array_equal(back["a"], t["a"])
    assert target.load_target(path, ["a", "b"], tag="other-weights") is None
    assert target.load_target(path, ["a"], tag="x") is None


def test_build_target_without_crops_raises():
    with pytest.raises(ValueError):
        target.build_target([], _cloaker())


def test_similarity_warning():
    t = {"a": np.array([1.0, 0, 0], np.float32)}
    far = {"a": np.array([[0, 1.0, 0], [0, 0.9, 0.1]], np.float32)}
    near = {"a": np.array([[0.9, 0.1, 0]], np.float32)}
    assert target.similarity_warning(far, t) is None
    assert "cosine" in target.similarity_warning(near, t)
