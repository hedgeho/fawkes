#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-05-17
# @Author  : Shawn Shan (shansixiong@cs.uchicago.edu)
# @Link    : https://www.shawnshan.com/

import argparse
import glob
import logging
import os
import sys

logging.getLogger('tensorflow').setLevel(logging.ERROR)
os.environ["KMP_AFFINITY"] = "noverbose"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(3)

import numpy as np
from fawkes.differentiator import FawkesMaskGeneration
from fawkes.utils import init_gpu, dump_image, reverse_preprocess, Faces, filter_image_paths, load_extractor, \
    IMG_SIZE, PREPROCESS

from fawkes.align_face import aligner

# th: DSSIM budget for the perturbation, max_step: optimisation steps, lr: learning rate,
# sd: penalty constant applied when the perturbation exceeds the budget.
MODES = {
    'low': dict(th=0.004, max_step=40, lr=25, sd=1e7, extractors=["extractor_2"]),
    'mid': dict(th=0.012, max_step=75, lr=20, sd=1e7, extractors=["extractor_0", "extractor_2"]),
    'high': dict(th=0.017, max_step=150, lr=15, sd=1e7, extractors=["extractor_0", "extractor_2"]),
}
# 'custom' takes th, max_step, lr and sd from the caller and optimises against both extractors.
CUSTOM_EXTRACTORS = ["extractor_0", "extractor_2"]


class Fawkes(object):
    def __init__(self, gpu=None, mode="low", th=None, max_step=None, lr=None, sd=None):
        if mode in MODES:
            params = MODES[mode]
        elif mode == 'custom':
            missing = [name for name, value in (("th", th), ("max_step", max_step), ("lr", lr), ("sd", sd))
                       if value is None]
            if missing:
                raise ValueError("mode 'custom' needs th, max_step, lr and sd (missing: {})".format(", ".join(missing)))
            params = dict(th=th, max_step=max_step, lr=lr, sd=sd, extractors=CUSTOM_EXTRACTORS)
        else:
            raise ValueError("mode must be one of {} or 'custom', got {!r}".format(
                ", ".join(repr(m) for m in MODES), mode))

        self.mode = mode
        self.th = params['th']
        self.max_step = params['max_step']
        self.lr = params['lr']
        self.sd = params['sd']
        self.gpu = gpu
        if gpu is not None:
            init_gpu(gpu)

        self.aligner = aligner()

        self.protector = None
        self.protector_param = None
        self.feature_extractors_ls = [load_extractor(name) for name in params['extractors']]

    def run_protection(self, image_paths, batch_size=1, format='png', debug=False, no_align=False,
                       maximize=True, save_last_on_failed=True):
        """Cloak every face in `image_paths`, writing <name>_cloaked.<format> next to each input.

        Returns 1 on success, 2 if no face was found, 3 if no image was found.
        """
        current_param = "-".join(str(x) for x in [self.th, self.sd, self.lr, self.max_step, batch_size,
                                                  debug, maximize, save_last_on_failed])

        image_paths, loaded_images = filter_image_paths(image_paths)

        if not image_paths:
            print("No images in the directory")
            return 3

        faces = Faces(image_paths, loaded_images, self.aligner, verbose=1, no_align=no_align)
        original_images = faces.cropped_faces

        if len(original_images) == 0:
            print("No face detected. ")
            return 2
        original_images = np.array(original_images)

        if current_param != self.protector_param:
            self.protector_param = current_param
            if batch_size == -1:
                batch_size = len(original_images)
            self.protector = FawkesMaskGeneration(self.feature_extractors_ls,
                                                  batch_size=batch_size,
                                                  intensity_range=PREPROCESS,
                                                  initial_const=self.sd,
                                                  learning_rate=self.lr,
                                                  max_iterations=self.max_step,
                                                  l_threshold=self.th,
                                                  verbose=debug,
                                                  maximize=maximize,
                                                  image_shape=(IMG_SIZE, IMG_SIZE, 3),
                                                  save_last_on_failed=save_last_on_failed,
                                                  )
        protected_images = self.protector.compute(original_images)
        faces.cloaked_cropped_faces = protected_images

        final_images, images_without_face = faces.merge_faces(
            reverse_preprocess(protected_images, PREPROCESS),
            reverse_preprocess(original_images, PREPROCESS))

        for i, (p_img, path) in enumerate(zip(final_images, image_paths)):
            if i in images_without_face:
                continue
            file_name = "{}_cloaked.{}".format(os.path.splitext(path)[0], format)
            dump_image(p_img, file_name, format=format)

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

    parser = argparse.ArgumentParser(description="Cloak the faces in a directory of images against facial "
                                                 "recognition models.")
    parser.add_argument('--directory', '-d', type=str,
                        help='the directory that contains images to run protection', default='imgs/')
    parser.add_argument('--gpu', '-g', type=str,
                        help='the GPU id when using GPU for optimization', default='0')
    parser.add_argument('--mode', '-m', choices=['low', 'mid', 'high', 'custom'], default='low',
                        help='cloak generation mode. The higher the mode is, the more perturbation added and '
                             'stronger protection. custom uses --th, --max-step, --lr and --sd')
    parser.add_argument('--th', help='only relevant with mode=custom, DSSIM threshold for perturbation', type=float,
                        default=0.01)
    parser.add_argument('--max-step', help='only relevant with mode=custom, number of steps for optimization', type=int,
                        default=1000)
    parser.add_argument('--sd', type=float, help='only relevant with mode=custom, penalty number, read more in the paper',
                        default=1e6)
    parser.add_argument('--lr', type=float, help='only relevant with mode=custom, learning rate', default=2)
    parser.add_argument('--batch-size', help="number of images to run optimization together", type=int, default=1)
    parser.add_argument('--no-align', help="skip face detection and cloak each whole image",
                        action='store_true')
    parser.add_argument('--debug', help="turn on debug and copy/paste the stdout when reporting an issue on github",
                        action='store_true')
    parser.add_argument('--format', choices=['png', 'jpg', 'jpeg'], default="png",
                        help="format of the output image")

    args = parser.parse_args(argv[1:])

    if args.format == 'jpg':
        args.format = 'jpeg'

    image_paths = [path for path in glob.glob(os.path.join(args.directory, "*"))
                   if "_cloaked" not in os.path.basename(path)]

    protector = Fawkes(gpu=args.gpu, mode=args.mode, th=args.th, max_step=args.max_step, lr=args.lr, sd=args.sd)

    protector.run_protection(image_paths, batch_size=args.batch_size, format=args.format,
                             debug=args.debug, no_align=args.no_align)


if __name__ == '__main__':
    main(*sys.argv)
