#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Fawkes v2 driver: detect, align, cloak toward a target identity, paste back, save.
import argparse
import dataclasses
import glob
import os
import sys

import numpy as np

from fawkes.align import make_crop, paste_back, template_crop
from fawkes.cloak import CloakParams, Cloaker
from fawkes.utils import dump_image, filter_image_paths

# steps / eps (L-inf, 0-255) / dssim_budget: perturbation size, eot_samples: robustness copies per step,
# stop_cos: a face is done once every surrogate sees the target at this cosine
MODES = {
    'low': dict(models=["adaface_ir101"], steps=30, eps=10.0, dssim_budget=0.008, eot_samples=0,
                stop_cos=0.8),
    'mid': dict(models=["arcface_r100", "adaface_ir101", "lvface_b"], steps=60, eps=12.0,
                dssim_budget=0.012, eot_samples=2, stop_cos=0.9),
    'high': dict(models=["arcface_r100", "adaface_ir101", "lvface_b"], steps=120, eps=16.0,
                 dssim_budget=0.017, eot_samples=2, stop_cos=0.95),
}


class Fawkes(object):
    def __init__(self, mode="mid", target_dir=None, models=None, steps=None, eps=None, dssim_budget=None,
                 eot_samples=None, stop_cos=None, threads=None, seed=0, batch_size=8, verbose=False,
                 device=None, **cloak_params):
        """`cloak_params`: any further CloakParams field (self_weight, laggard, lr, patience, ...)."""
        if mode not in MODES:
            raise ValueError("mode must be one of {}, got {!r}".format(", ".join(repr(m) for m in MODES), mode))
        if target_dir is None:
            raise ValueError("target_dir is required: a directory with a few photos of the person to mimic")
        if not os.path.isdir(target_dir):
            raise ValueError("target_dir {!r} is not a directory".format(target_dir))
        params = dict(MODES[mode])
        for name, value in (("steps", steps), ("eps", eps), ("dssim_budget", dssim_budget),
                            ("eot_samples", eot_samples), ("stop_cos", stop_cos)):
            if value is not None:
                params[name] = value
        if models:
            params["models"] = list(models)
        fields = {f.name for f in dataclasses.fields(CloakParams)}
        unknown = [k for k in cloak_params if k not in fields]
        if unknown:
            raise TypeError("unknown cloak parameter(s) {}; known: {}".format(unknown, ", ".join(sorted(fields))))
        params.update({k: v for k, v in cloak_params.items() if v is not None})

        from fawkes.models import SURROGATES
        unknown = [m for m in params["models"] if m not in SURROGATES]
        if unknown:
            raise ValueError("unknown surrogate model(s) {}; known: {}".format(unknown, ", ".join(SURROGATES)))

        import torch
        if threads:
            torch.set_num_threads(int(threads))

        from fawkes.models import default_device
        self.device = device or default_device()
        self.mode = mode
        self.target_dir = target_dir
        self.model_keys = params.pop("models")
        self.params = CloakParams(seed=seed, batch_size=batch_size, **params)
        self.verbose = verbose

        self._detector = None
        self._cloaker = None
        self._target = None

    # models are loaded lazily so that argument errors surface before any download starts
    @property
    def detector(self):
        if self._detector is None:
            from fawkes.detect import Detector
            self._detector = Detector()
        return self._detector

    @property
    def cloaker(self):
        if self._cloaker is None:
            from fawkes.models import load_surrogate
            surrogates = {k: load_surrogate(k, self.device) for k in self.model_keys}
            self._cloaker = Cloaker(surrogates, self.params, verbose=self.verbose)
        return self._cloaker

    @property
    def target(self):
        if self._target is None:
            from fawkes.target import get_or_build_target
            self._target = get_or_build_target(self.target_dir, self.model_keys, self.detector, self.cloaker)
        return self._target

    def set_target_dir(self, target_dir):
        """Switch to another target identity, keeping the loaded models."""
        if not os.path.isdir(target_dir):
            raise ValueError("target_dir {!r} is not a directory".format(target_dir))
        self.target_dir = target_dir
        self._target = None

    def run_protection(self, image_paths, format='png', no_align=False, debug=False):
        """Cloak every face in `image_paths`, writing <name>_cloaked.<format> next to each input.

        Returns 1 on success, 2 if no face was found, 3 if no image was found.
        """
        from fawkes.target import similarity_warning

        image_paths, images = filter_image_paths(image_paths)
        if not image_paths:
            print("No images in the directory")
            return 3

        crops, owner = [], []
        for i, (path, img) in enumerate(zip(image_paths, images)):
            if no_align:
                faces = [None]
                crops.append(template_crop(img))
                owner.append(i)
                continue
            faces = self.detector.detect(img)
            print("Find {} face(s) in {}".format(len(faces), os.path.basename(path)))
            for face in faces:
                crops.append(make_crop(img, face.bbox, face.kps))
                owner.append(i)
        if not crops:
            print("No face detected. ")
            return 2

        target = self.target
        warning = similarity_warning(self.cloaker.embed(crops), target)
        if warning:
            print(warning)

        result = self.cloaker.cloak(crops, target)
        if debug:
            for i, crop in enumerate(crops):
                print("face {} ({}): steps {} dssim {:.4f} cos {}".format(
                    i, os.path.basename(image_paths[owner[i]]), result.steps[i], result.dssim[i],
                    " ".join("{}={:.2f}".format(k, c) for k, c in zip(result.models, result.cos[i]))))
        print("protection cost {:.1f} s ({:.1f} s/face)".format(result.seconds, result.seconds / len(crops)))

        outputs = {}
        for crop, cloaked, i in zip(crops, result.images, owner):
            base = outputs.get(i, images[i])
            outputs[i] = paste_back(base, crop, cloaked)
        for i, out in outputs.items():
            file_name = "{}_cloaked.{}".format(os.path.splitext(image_paths[i])[0], format)
            dump_image(out, file_name, format=format)

        print("Done!")
        return 1


def main(*argv):
    if not argv:
        argv = list(sys.argv)

    try:
        import signal
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    except (AttributeError, ValueError):  # no SIGPIPE on Windows, or not on the main thread
        pass

    parser = argparse.ArgumentParser(description="Cloak the faces in a directory of images so that facial "
                                                 "recognition models see a chosen target person instead.")
    parser.add_argument('--directory', '-d', type=str, default='imgs/',
                        help='the directory that contains images to run protection')
    parser.add_argument('--target-dir', '-t', type=str, required=True,
                        help='directory with a few photos (one face each) of the person to mimic')
    parser.add_argument('--mode', '-m', choices=list(MODES), default='mid',
                        help='tradeoff between perturbation size, run time and protection strength')
    parser.add_argument('--models', type=str, default=None,
                        help='comma-separated surrogate keys overriding the mode (see `python -m fawkes.models`)')
    parser.add_argument('--steps', type=int, default=None, help='maximum optimisation steps per face')
    parser.add_argument('--eps', type=float, default=None, help='maximum per-pixel change (0-255)')
    parser.add_argument('--th', type=float, default=None, help='DSSIM budget for the perturbation')
    parser.add_argument('--no-eot', action='store_true',
                        help='skip the blur/resize/JPEG robustness copies (faster, less robust)')
    parser.add_argument('--self-weight', type=float, default=None,
                        help="weight of the penalty on the face's remaining similarity to itself")
    parser.add_argument('--batch-size', type=int, default=8, help="number of faces optimised together")
    parser.add_argument('--threads', type=int, default=None, help='CPU threads for torch')
    parser.add_argument('--device', type=str, default=None,
                        help="torch device for the surrogates, e.g. cuda or cpu (default: cuda if available)")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--no-align', action='store_true',
                        help="inputs are already 112x112 aligned faces: skip detection and cloak each whole image")
    parser.add_argument('--debug', action='store_true',
                        help="print per-face statistics and copy/paste the stdout when reporting an issue")
    parser.add_argument('--format', choices=['png', 'jpg', 'jpeg'], default="png",
                        help="format of the output image")

    args = parser.parse_args(argv[1:])
    if args.format == 'jpg':
        args.format = 'jpeg'

    image_paths = [path for path in glob.glob(os.path.join(args.directory, "*"))
                   if "_cloaked" not in os.path.basename(path)]

    protector = Fawkes(mode=args.mode, target_dir=args.target_dir,
                       models=args.models.split(",") if args.models else None,
                       steps=args.steps, eps=args.eps, dssim_budget=args.th,
                       eot_samples=0 if args.no_eot else None, threads=args.threads, seed=args.seed,
                       batch_size=args.batch_size, verbose=args.debug, device=args.device,
                       self_weight=args.self_weight)
    return protector.run_protection(image_paths, format=args.format, no_align=args.no_align, debug=args.debug)


if __name__ == '__main__':
    main(*sys.argv)
