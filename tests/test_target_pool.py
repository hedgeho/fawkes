import numpy as np
import pytest

from fawkes import target_pool as tp


def _unit(v):
    v = np.asarray(v, np.float32)
    return v / np.linalg.norm(v)


def _pool():
    n, d = 8, 16
    eye = np.eye(d, dtype=np.float32)
    # identity i > 0 has cosine about 0.02 * i to identity 0: all distinct, but at graded distances
    emb = {k: np.stack([eye[0]] + [_unit(eye[i] + 0.02 * i * eye[0]) for i in range(1, n)]) for k in ("a", "b")}
    return tp.Pool(names=[f"p{i}" for i in range(n)],
                   age=np.array([30, 32, 60, 29, 31, 33, 28, 30], np.float32),
                   male=np.array([1, 1, 1, 0, 1, 1, 1, 1], bool),
                   lab=np.array([[60, 10, 15], [61, 11, 15], [60, 10, 15], [60, 10, 15],
                                 [40, 20, 25], [60, 10, 16], [59, 10, 15], [60, 10, 15]], np.float32),
                   embedding=emb)


def _person(pool, like, male=True, age=30.0):
    return tp.Person(embedding={k: pool.embedding[k][like] for k in ("a", "b")}, age=age, male=male,
                     lab=np.array([60, 10, 15], np.float32))


def test_select_excludes_same_person_other_sex_and_age():
    pool = _pool()
    person = _person(pool, like=0)  # the person is identity p0 itself
    i, reason = tp.select(pool, person, ["a", "b"], "near", n_skin=3)
    assert pool.names[i] not in ("p0", "p2", "p3", "p4"), reason  # self, age 60, female, other skin
    assert pool.names[i] in reason


def test_select_near_and_far_differ_and_respect_exclude():
    pool = _pool()
    person = _person(pool, like=0)
    near, _ = tp.select(pool, person, ["a", "b"], "near", n_skin=4)
    far, _ = tp.select(pool, person, ["a", "b"], "far", n_skin=4)
    cos = lambda i: np.mean([pool.embedding[k][i] @ person.embedding[k] for k in ("a", "b")])
    assert cos(near) >= cos(far)
    other, _ = tp.select(pool, person, ["a", "b"], "near", exclude=[pool.names[near]], n_skin=4)
    assert other != near


def test_select_widens_age_window_and_raises_when_empty():
    pool = _pool()
    person = _person(pool, like=0, age=45.0)  # nobody within 8 or 12 years
    i, _ = tp.select(pool, person, ["a", "b"], "near", n_skin=2)
    assert pool.male[i]
    with pytest.raises(ValueError):
        tp.select(pool, person, ["a", "b"], "near", exclude=pool.names)


def test_pool_roundtrip(tmp_path):
    pool = _pool()
    pool.save(tmp_path / "p.npz")
    back = tp.Pool.load(tmp_path / "p.npz")
    assert back.names == pool.names and back.male.dtype == bool
    np.testing.assert_array_equal(back.target(3, ["a"])["a"], pool.embedding["a"][3])
    with pytest.raises(KeyError):
        back.target(0, ["c"])


def test_group_faces():
    a, b = _unit([1, 0, 0, 0]), _unit([0, 1, 0, 0])
    e = {"k": np.stack([a, b, _unit(a + 0.1 * b), _unit(b + 0.05)])}
    assert tp.group_faces(e, "k") == [[0, 2], [1, 3]]


def test_skin_lab_on_uniform_face():
    img = np.full((200, 200, 3), (200, 150, 120), np.float32)
    kps = np.array([[70, 90], [130, 90], [100, 120], [75, 150], [125, 150]], np.float32)
    lab = tp.skin_lab(img, kps)
    assert 60 < lab[0] < 75 and lab[1] > 5 and lab[2] > 10


def test_assignments_remember_a_person(tmp_path):
    a = tp.Assignments(tmp_path / "a.json")
    me = _unit([1, 0.1, 0, 0])
    assert a.lookup("k", me) is None
    a.add("k", me, "p3")
    again = tp.Assignments(tmp_path / "a.json")
    assert again.lookup("k", _unit([1, 0.2, 0, 0])) == "p3"
    assert again.lookup("k", _unit([0, 1, 0, 0])) is None
    assert again.lookup("other", me) is None
