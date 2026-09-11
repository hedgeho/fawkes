#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-05-17
# @Author  : Shawn Shan (shansixiong@cs.uchicago.edu)
# @Link    : https://www.shawnshan.com/


import hashlib
import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import urlretrieve

import PIL
import keras
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
from keras.models import Model
from keras.utils import Progbar, img_to_array, array_to_img

from fawkes.align_face import align

MODEL_URL_BASE = "https://mirror.cs.uchicago.edu/fawkes/files"
IMG_SIZE = 112
PREPROCESS = 'raw'
IMAGENET_MEAN_BGR = np.array([103.939, 116.779, 123.68], dtype=np.float32)


def load_image(path):
    """Load `path` as an RGB float array with EXIF orientation applied, or None if it is not an image."""
    try:
        img = Image.open(path)
    except (PIL.UnidentifiedImageError, IsADirectoryError, PermissionError):
        return None

    try:
        img = ImageOps.exif_transpose(img)
    except OSError:
        return None

    return img_to_array(img.convert('RGB'))


def filter_image_paths(image_paths):
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


class Faces(object):
    """Crops faces out of images, pads them to IMG_SIZE squares, and merges cloaks back into the originals."""

    def __init__(self, image_paths, loaded_images, aligner, verbose=1, preprocessing=True, no_align=False):
        self.image_paths = image_paths
        self.verbose = verbose
        self.no_align = no_align
        self.aligner = aligner
        self.margin = 30
        self.org_faces = []
        self.cropped_faces = []
        self.cropped_faces_shape = []
        self.cropped_index = []
        self.start_end_ls = []
        self.callback_idx = []
        self.images_without_face = []
        self.cloaked_cropped_faces = None
        self.cloaked_faces = []
        for i, (cur_img, p) in enumerate(zip(loaded_images, image_paths)):
            self.org_faces.append(cur_img)

            if not no_align:
                align_img = align(cur_img, self.aligner)
                if align_img is None:
                    cur_faces, cur_index = [], []
                else:
                    cur_faces, cur_index = align_img
            else:
                # treat the whole image as one face so it is merged back at full size
                cur_faces = [cur_img]
                cur_index = [[0, 0, cur_img.shape[0], cur_img.shape[1]]]

            cur_faces = [face for face in cur_faces if face.shape[0] != 0 and face.shape[1] != 0]
            if verbose and not no_align:
                print("Find {} face(s) in {}".format(len(cur_faces), os.path.basename(p)))
            if not cur_faces:
                self.images_without_face.append(i)
                continue

            cur_faces_square = []
            for img in cur_faces:
                long_size = max([img.shape[1], img.shape[0]]) + self.margin
                base = np.ones((long_size, long_size, 3)) * np.mean(img, axis=(0, 1))

                start1, end1 = get_ends(long_size, img.shape[0])
                start2, end2 = get_ends(long_size, img.shape[1])
                base[start1:end1, start2:end2, :] = img
                self.start_end_ls.append((start1, end1, start2, end2))

                cur_faces_square.append(resize(base, (IMG_SIZE, IMG_SIZE)))

            self.cropped_faces.extend(cur_faces_square)
            self.cropped_faces_shape.extend(f.shape[:-1] for f in cur_faces)
            self.cropped_index.extend(cur_index[:len(cur_faces_square)])
            self.callback_idx.extend([i] * len(cur_faces_square))

        if len(self.cropped_faces) == 0:
            return

        self.cropped_faces = np.array(self.cropped_faces)

        if preprocessing:
            self.cropped_faces = preprocess(self.cropped_faces, PREPROCESS)

        self.cloaked_faces = [np.copy(f) for f in self.org_faces]

    def merge_faces(self, protected_images, original_images):
        self.cloaked_faces = [np.copy(f) for f in self.org_faces]

        for i in range(len(self.cropped_faces)):
            cur_protected = protected_images[i]
            cur_original = original_images[i]

            org_shape = self.cropped_faces_shape[i]

            old_square_shape = max([org_shape[0], org_shape[1]]) + self.margin

            cur_protected = resize(cur_protected, (old_square_shape, old_square_shape))
            cur_original = resize(cur_original, (old_square_shape, old_square_shape))

            start1, end1, start2, end2 = self.start_end_ls[i]

            reshape_cloak = cur_protected - cur_original
            reshape_cloak = reshape_cloak[start1:end1, start2:end2, :]

            callback_id = self.callback_idx[i]
            bb = self.cropped_index[i]
            self.cloaked_faces[callback_id][bb[0]:bb[2], bb[1]:bb[3], :] += reshape_cloak

        for i in range(0, len(self.cloaked_faces)):
            self.cloaked_faces[i] = np.clip(self.cloaked_faces[i], 0.0, 255.0)
        return self.cloaked_faces, self.images_without_face


def get_ends(longsize, window):
    start = (longsize - window) // 2
    end = start + window
    return start, end


def resize(img, sz):
    assert np.min(img) >= 0 and np.max(img) <= 255.0
    im_data = array_to_img(img).resize((sz[1], sz[0]))
    im_data = img_to_array(im_data)
    return im_data


def init_gpu(gpu):
    ''' code to initialize gpu in tf2'''
    if isinstance(gpu, list):
        gpu_num = ','.join([str(i) for i in gpu])
    else:
        gpu_num = str(gpu)
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        print('GPU already initiated')
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_num
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[0], True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            print(e)


def preprocess(X, method):
    assert method in {'raw', 'imagenet'}
    if method == 'imagenet':
        return imagenet_preprocessing(X)
    return X


def reverse_preprocess(X, method):
    assert method in {'raw', 'imagenet'}
    if method == 'imagenet':
        return imagenet_reverse_preprocessing(X)
    return X


def imagenet_preprocessing(x):
    """RGB in [0, 255] -> BGR zero-centred by the ImageNet mean (channels last)."""
    return np.array(x, dtype=np.float32)[..., ::-1] - IMAGENET_MEAN_BGR


def imagenet_reverse_preprocessing(x):
    return (np.array(x, dtype=np.float32) + IMAGENET_MEAN_BGR)[..., ::-1]


def load_extractor(name):
    hash_map = {"extractor_2": "ce703d481db2b83513bbdafa27434703",
                "extractor_0": "94854151fd9077997d69ceda107f9c6b"}
    assert name in ["extractor_2", 'extractor_0']
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
    os.makedirs(model_dir, exist_ok=True)
    model_file = get_file("{}.h5".format(name), "{}/{}.h5".format(MODEL_URL_BASE, name),
                          cache_dir=model_dir, cache_subdir='', md5_hash=hash_map[name])

    model = load_legacy_model(model_file)
    model = Extractor(model)
    return model


def _sanitize_legacy_config(config):
    """Strip Keras 2 layer arguments that Keras 3 no longer accepts (in place)."""
    for layer_cfg in config.get("layers", []):
        if layer_cfg["class_name"] in ("Functional", "Model"):
            _sanitize_legacy_config(layer_cfg["config"])
        elif layer_cfg["class_name"] == "DepthwiseConv2D":
            layer_cfg["config"].pop("groups", None)


def load_legacy_model(model_file):
    """Load a Keras 2 HDF5 model whose top-level graph nests Functional sub-models.

    Keras 3's legacy HDF5 loader assumes nested Functional models start with a
    built-in inbound node (the Keras 2 convention), so the saved node index of 1
    is out of range and loading fails. The top-level graph of the extractors is
    a simple chain, so it is rebuilt here from the saved config and the weights
    are then loaded topologically from the same file.
    """
    import h5py
    with h5py.File(model_file, "r") as f:
        config = json.loads(f.attrs["model_config"])["config"]

    _sanitize_legacy_config(config)
    tensors = {}
    for layer_cfg in config["layers"]:
        name = layer_cfg["name"]
        if layer_cfg["class_name"] == "InputLayer":
            shape = layer_cfg["config"]["batch_input_shape"][1:]
            tensors[name] = keras.Input(shape=shape, name=name)
            continue
        if layer_cfg["class_name"] in ("Functional", "Model"):
            layer = Model.from_config(layer_cfg["config"])
        else:
            layer = keras.layers.deserialize(layer_cfg)
        inbound = [tensors[node[0]] for node in layer_cfg["inbound_nodes"][0]]
        tensors[name] = layer(inbound[0] if len(inbound) == 1 else inbound)

    inputs = [tensors[node[0]] for node in config["input_layers"]]
    outputs = [tensors[node[0]] for node in config["output_layers"]]
    model = Model(inputs[0] if len(inputs) == 1 else inputs,
                  outputs[0] if len(outputs) == 1 else outputs, name=config["name"])
    model.load_weights(model_file)
    _refresh_normalization_layers(model)
    return model


def _refresh_normalization_layers(model):
    """Re-read Normalization statistics from the loaded weights.

    Keras 3's Normalization layer snapshots its mean and variance when it is
    built, which happens before load_weights() runs, so the snapshot has to be
    refreshed or the layer keeps normalizing with the initial zeros and ones.
    """
    for layer in model.layers:
        if isinstance(layer, Model):
            _refresh_normalization_layers(layer)
        elif isinstance(layer, keras.layers.Normalization):
            layer.finalize_state()


class Extractor(object):
    def __init__(self, model):
        self.model = model

    def predict(self, imgs):
        imgs = imgs / 255.0
        embeds = l2_norm(self.model(imgs))
        return embeds

    def __call__(self, x):
        return self.predict(x)


def dump_image(x, filename, format="png", scale=False):
    img = array_to_img(x, scale=scale)
    img.save(filename, format)
    return


def l2_norm(x, axis=1):
    """l2 norm"""
    norm = tf.norm(x, axis=axis, keepdims=True)
    output = x / norm
    return output


def get_file(fname, origin, md5_hash=None, cache_subdir='datasets', cache_dir=None):
    """Download `origin` into cache_dir/cache_subdir/fname unless a copy with a matching MD5 is already there.

    Trimmed from tf.keras.utils.get_file (TensorFlow 2.3). Returns the local path.
    """
    if cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser('~'), '.keras')
    datadir_base = os.path.expanduser(cache_dir)
    if not os.access(datadir_base, os.W_OK):
        datadir_base = os.path.join('/tmp', '.keras')
    datadir = os.path.join(datadir_base, cache_subdir)
    os.makedirs(datadir, exist_ok=True)
    fpath = os.path.join(datadir, fname)

    if os.path.exists(fpath):
        if md5_hash is None or validate_file(fpath, md5_hash, algorithm='md5'):
            return fpath
        print('A local file was found, but it seems to be incomplete or outdated because the md5 file hash '
              'does not match the original value of ' + md5_hash + ' so we will re-download the data.')

    print('Downloading data from', origin)
    progbar = []

    def dl_progress(count, block_size, total_size):
        if not progbar:
            progbar.append(Progbar(None if total_size == -1 else total_size))
        else:
            progbar[0].update(count * block_size)

    error_msg = 'URL fetch failure on {}: {} -- {}'
    try:
        try:
            urlretrieve(origin, fpath, dl_progress)
        except HTTPError as e:
            raise RuntimeError(error_msg.format(origin, e.code, e.msg)) from e
        except URLError as e:
            raise RuntimeError(error_msg.format(origin, e.errno, e.reason)) from e
    except (Exception, KeyboardInterrupt):
        if os.path.exists(fpath):
            os.remove(fpath)
        raise
    return fpath


def validate_file(fpath, file_hash, algorithm='auto', chunk_size=65535):
    """Validate a file against an md5 or sha256 hash; 'auto' picks the algorithm from the hash length."""
    if algorithm == 'auto':
        algorithm = 'sha256' if len(file_hash) == 64 else 'md5'
    return _hash_file(fpath, algorithm, chunk_size) == str(file_hash)


def _hash_file(fpath, algorithm='sha256', chunk_size=65535):
    hasher = hashlib.new(algorithm)
    with open(fpath, 'rb') as fpath_file:
        for chunk in iter(lambda: fpath_file.read(chunk_size), b''):
            hasher.update(chunk)
    return hasher.hexdigest()
