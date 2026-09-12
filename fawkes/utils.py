#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-05-17
# @Author  : Shawn Shan (shansixiong@cs.uchicago.edu)
# @Link    : https://www.shawnshan.com/
"""Image loading and saving."""
import os

import numpy as np
import PIL
from PIL import Image, ImageOps


def load_image(path):
    """Load `path` as an RGB float32 array with EXIF orientation applied, or None if it is not an image."""
    try:
        img = Image.open(path)
    except (PIL.UnidentifiedImageError, IsADirectoryError, PermissionError, FileNotFoundError):
        return None

    try:
        img = ImageOps.exif_transpose(img)
    except OSError:
        return None

    return np.asarray(img.convert('RGB'), dtype=np.float32)


def filter_image_paths(image_paths):
    """Load every readable image; returns (paths, images) for the ones that are images."""
    print("Identify {} files in the directory".format(len(image_paths)))
    new_image_paths = []
    new_images = []
    for p in image_paths:
        img = load_image(p)
        if img is None:
            print("{} is not an image file, skipped".format(os.path.basename(p)))
            continue
        new_image_paths.append(p)
        new_images.append(img)
    print("Identify {} images in the directory".format(len(new_image_paths)))
    return new_image_paths, new_images


def dump_image(x, filename, format="png"):
    """Save an RGB array in [0, 255] to `filename`."""
    arr = np.clip(np.asarray(x, dtype=np.float32), 0, 255).round().astype(np.uint8)
    Image.fromarray(arr).save(filename, format)
