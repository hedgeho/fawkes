import numpy as np
from fawkes import utils


def test_resize_preserves_intensity_range():
    img = np.full((50, 50, 3), 100.0, np.float32)
    img[10:20, 10:20] = 140
    out = utils.resize(img, (50, 50))
    assert 99 <= out.min() and out.max() <= 141, (out.min(), out.max())
