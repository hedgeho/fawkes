#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-10-21
# @Author  : Emily Wenger (ewenger@uchicago.edu)

import datetime
import time

import numpy as np
import tensorflow as tf
from fawkes.utils import preprocess, reverse_preprocess
from keras.utils import Progbar


class FawkesMaskGeneration:
    # number of iterations to perform gradient descent
    MAX_ITERATIONS = 10000
    # larger values converge faster to less accurate results
    LEARNING_RATE = 1e-2
    # the initial constant c to pick as a first guess
    INITIAL_CONST = 1
    # pixel intensity range
    INTENSITY_RANGE = 'imagenet'
    # threshold for distance
    L_THRESHOLD = 0.03
    # max_val of image
    MAX_VAL = 255
    MAXIMIZE = False
    IMAGE_SHAPE = (112, 112, 3)

    def __init__(self, bottleneck_model_ls, batch_size=1, learning_rate=LEARNING_RATE,
                 max_iterations=MAX_ITERATIONS, initial_const=INITIAL_CONST,
                 intensity_range=INTENSITY_RANGE, l_threshold=L_THRESHOLD,
                 max_val=MAX_VAL, maximize=MAXIMIZE, image_shape=IMAGE_SHAPE, verbose=1,
                 save_last_on_failed=True):

        assert intensity_range in {'raw', 'imagenet'}

        # constant used for tanh transformation to avoid corner cases

        self.it = 0
        self.tanh_constant = 2 - 1e-6
        self.save_last_on_failed = save_last_on_failed
        self.max_iterations = max_iterations
        self.initial_const = initial_const
        self.batch_size = batch_size
        self.intensity_range = intensity_range
        self.l_threshold = l_threshold
        self.max_val = max_val
        self.verbose = verbose
        self.maximize = maximize
        self.learning_rate = learning_rate
        self.single_shape = list(image_shape)
        self.bottleneck_models = bottleneck_model_ls

        # compiled step, its variables and optimizer; built lazily and reused across batches of the same shape
        self._step = None
        self._step_shape = None
        self._optimizer = None
        self._optimizer_initial_state = None

    @staticmethod
    def resize_tensor(input_tensor, model_input_shape):
        if input_tensor.shape[1:] == model_input_shape or model_input_shape[1] is None:
            return input_tensor
        resized_tensor = tf.image.resize(input_tensor, model_input_shape[:2])
        return resized_tensor

    def preprocess_arctanh(self, imgs):
        """ Do tan preprocess """
        imgs = reverse_preprocess(imgs, self.intensity_range)
        imgs = imgs / 255.0
        imgs = imgs - 0.5
        imgs = imgs * self.tanh_constant
        tanh_imgs = np.arctanh(imgs)
        return tanh_imgs

    def reverse_arctanh(self, imgs):
        raw_img = (tf.tanh(imgs) / self.tanh_constant + 0.5) * 255
        return raw_img

    def input_space_process(self, img):
        if self.intensity_range == 'imagenet':
            mean = tf.constant([103.939, 116.779, 123.68], dtype=tf.float32)
            raw_img = img[..., ::-1] - mean
        else:
            raw_img = img
        return raw_img

    def clipping(self, imgs):
        imgs = reverse_preprocess(imgs, self.intensity_range)
        imgs = np.clip(imgs, 0, self.max_val)
        imgs = preprocess(imgs, self.intensity_range)
        return imgs

    def calc_dissim(self, source_raw, source_mod_raw):
        msssim_split = tf.image.ssim(source_raw, source_mod_raw, max_val=255.0)
        dist_raw = (1.0 - tf.stack(msssim_split)) / 2.0
        dist = tf.maximum(dist_raw - self.l_threshold, 0.0)
        dist_raw_avg = tf.reduce_mean(dist_raw)
        dist_sum = tf.reduce_sum(dist)

        return dist, dist_raw, dist_sum, dist_raw_avg

    def extract_features(self, imgs):
        """ Feature-space representation of `imgs` from every bottleneck model. """
        imgs = self.resize_tensor(imgs, self.single_shape)
        return [bottleneck_model(imgs) for bottleneck_model in self.bottleneck_models]

    def calc_bottlesim(self, source_input, reference_features):
        """ original Fawkes loss function: normalised feature distance to fixed reference features. """
        bottlesim = 0.0
        for feature_a, feature_ref in zip(self.extract_features(source_input), reference_features):
            scale_factor = tf.sqrt(tf.reduce_sum(tf.square(feature_ref), axis=1))
            cur_bottlesim = tf.reduce_sum(tf.square(feature_a - feature_ref), axis=1)
            bottlesim += cur_bottlesim / scale_factor
        return bottlesim

    def compute_feature_loss(self, aimg_raw, simg_raw, aimg_input, reference_features):
        """ Compute input space + feature space loss.
        """
        input_space_loss, dist_raw, input_space_loss_sum, input_space_loss_raw_avg = self.calc_dissim(aimg_raw,
                                                                                                      simg_raw)
        feature_space_loss = self.calc_bottlesim(aimg_input, reference_features)

        if self.maximize:
            loss = self.const * tf.square(input_space_loss) - feature_space_loss * self.const_diff
        else:
            loss = self.const * tf.square(input_space_loss) + 1000 * feature_space_loss

        loss_sum = tf.reduce_sum(loss)
        return loss_sum, feature_space_loss, input_space_loss_raw_avg, dist_raw

    def _build_step(self, optimizer):
        """ One compiled optimisation step: forward, loss, gradient and optimizer update. """

        @tf.function
        def step(modifier, simg_tanh, simg_raw, reference_features):
            with tf.GradientTape() as tape:
                # Convert from tanh for DISSIM
                aimg_raw = self.reverse_arctanh(simg_tanh + modifier)
                actual_modifier = tf.clip_by_value(aimg_raw - simg_raw, -15.0, 15.0)
                aimg_raw = simg_raw + actual_modifier

                # Convert further preprocess for bottleneck
                aimg_input = self.input_space_process(aimg_raw)

                loss, internal_dist, input_dist_avg, dist_raw = self.compute_feature_loss(
                    aimg_raw, simg_raw, aimg_input, reference_features)

            grad = tape.gradient(loss, modifier)
            optimizer.apply_gradients([(grad, modifier)])
            return loss, internal_dist, input_dist_avg, dist_raw, aimg_input, grad

        return step

    def _prepare_step(self, batch_shape):
        """ (Re)build the compiled step for `batch_shape` or reset its state for a fresh batch.

        Tracing the step costs seconds per extractor, so the same trace is reused for every batch of the
        same shape; only the variable contents and the optimizer state are reset between batches.
        """
        nb_imgs = batch_shape[0]
        if self._step is None or self._step_shape != batch_shape:
            self.modifier = tf.Variable(tf.zeros(batch_shape, dtype=tf.float32))
            self.const = tf.Variable(tf.zeros(nb_imgs, dtype=tf.float32))
            self.const_diff = tf.Variable(tf.zeros(nb_imgs, dtype=tf.float32))
            self._optimizer = tf.keras.optimizers.Adadelta(float(self.learning_rate))
            self._optimizer.build([self.modifier])
            self._optimizer_initial_state = [v.numpy() for v in self._optimizer.variables]
            self._step = self._build_step(self._optimizer)
            self._step_shape = batch_shape
        else:
            for var, value in zip(self._optimizer.variables, self._optimizer_initial_state):
                var.assign(value)

        self.modifier.assign(np.random.uniform(-1, 1, batch_shape) * 1e-4)
        self.const.assign(np.ones(nb_imgs) * self.initial_const)
        self.const_diff.assign(np.ones(nb_imgs))
        return self._step

    def compute(self, source_imgs, target_imgs=None):
        """ Main function that runs cloak generation. """
        start_time = time.time()
        adv_imgs = []
        for idx in range(0, len(source_imgs), self.batch_size):
            print('processing image %d at %s' % (idx + 1, datetime.datetime.now()))
            adv_img = self.compute_batch(source_imgs[idx:idx + self.batch_size],
                                         target_imgs[idx:idx + self.batch_size] if target_imgs is not None else None)
            adv_imgs.extend(adv_img)
        elapsed_time = time.time() - start_time
        print('protection cost %f s' % elapsed_time)
        return np.array(adv_imgs)

    def compute_batch(self, source_imgs, target_imgs=None):
        """ TF2 method to generate the cloak. """
        nb_imgs = source_imgs.shape[0]

        # make sure source/target images are an array
        source_imgs = np.array(source_imgs, dtype=np.float32)
        if target_imgs is not None:
            target_imgs = np.array(target_imgs, dtype=np.float32)

        # metrics to test
        best_bottlesim = np.zeros(nb_imgs) if self.maximize else np.full(nb_imgs, np.inf)
        # fall back to the unmodified image if no iteration lands inside the threshold
        best_adv = np.copy(source_imgs)

        # convert to tanh-space; the source in raw space is constant across iterations
        simg_tanh = tf.constant(self.preprocess_arctanh(source_imgs), dtype=tf.float32)
        simg_raw = self.reverse_arctanh(simg_tanh)

        step = self._prepare_step(tuple([nb_imgs] + self.single_shape))
        const_diff_numpy = np.ones(nb_imgs)

        # the features being moved away from (maximize) or towards (mimic) never change,
        # so extract them once instead of on every iteration
        reference_raw = simg_raw if self.maximize else tf.constant(target_imgs, dtype=tf.float32)
        reference_features = self.extract_features(self.input_space_process(reference_raw))

        progressbar = None
        if self.verbose == 0:
            progressbar = Progbar(self.max_iterations, width=30, verbose=1)

        # run the attack
        outside_list = np.ones(nb_imgs)
        dist_raw_np = internal_dist_np = aimg_np = None
        self.it = 0

        while self.it < self.max_iterations:

            self.it += 1
            loss, internal_dist, input_dist_avg, dist_raw, aimg_input, grad = step(
                self.modifier, simg_tanh, simg_raw, reference_features)

            if self.it == 1:
                self.modifier.assign(self.modifier - tf.sign(grad) * 0.01)

            # pull the per-image metrics to the host once per iteration
            dist_raw_np = dist_raw.numpy()
            internal_dist_np = internal_dist.numpy()
            aimg_np = None

            for e in range(nb_imgs):
                input_dist = dist_raw_np[e]
                feature_d = internal_dist_np[e]

                if input_dist <= self.l_threshold * 0.9 and const_diff_numpy[e] <= 129:
                    const_diff_numpy[e] *= 2
                    if outside_list[e] == -1:
                        const_diff_numpy[e] = 1
                    outside_list[e] = 1
                elif input_dist >= self.l_threshold * 1.1 and const_diff_numpy[e] >= 1 / 129:
                    const_diff_numpy[e] /= 2

                    if outside_list[e] == 1:
                        const_diff_numpy[e] = 1
                    outside_list[e] = -1
                else:
                    const_diff_numpy[e] = 1.0
                    outside_list[e] = 0

                if input_dist <= self.l_threshold * 1.1 and (
                        (feature_d < best_bottlesim[e] and (not self.maximize)) or (
                        feature_d > best_bottlesim[e] and self.maximize)):
                    best_bottlesim[e] = feature_d
                    if aimg_np is None:
                        aimg_np = aimg_input.numpy()
                    best_adv[e] = aimg_np[e]

            self.const_diff.assign(const_diff_numpy)

            if self.verbose == 1:
                print("ITER {:0.2f}  Total Loss: {:.2f} {:0.4f} raw; diff: {:.4f}".format(
                    self.it, float(loss), float(input_dist_avg), np.mean(internal_dist_np)))

            if progressbar is not None:
                progressbar.update(self.it)
        if self.verbose == 1:
            print("Final diff: {:.4f}".format(np.mean(best_bottlesim)))
        print("\n")

        if self.save_last_on_failed and dist_raw_np is not None:
            if aimg_np is None:
                aimg_np = aimg_input.numpy()
            for e, diff in enumerate(best_bottlesim):
                if diff < 0.3 and dist_raw_np[e] < 0.015 and internal_dist_np[e] > diff:
                    best_adv[e] = aimg_np[e]

        best_adv = self.clipping(best_adv[:nb_imgs])
        return best_adv
