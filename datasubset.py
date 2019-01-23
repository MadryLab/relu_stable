
"""
Utilities for importing the MNIST dataset.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import sys
import random
import tensorflow as tf
version = sys.version_info

import numpy as np

class DataSubset(object):
    def __init__(self, xs, ys, size):
        if size < 55000:
            xs, ys = self._per_class_subsample(xs, ys, size)
        self.xs = xs
        self.n = xs.shape[0]
        self.ys = ys
        self.batch_start = 0
        self.cur_order = np.random.permutation(self.n)

    def get_next_batch(self, batch_size, multiple_passes=False, reshuffle_after_pass=True):
        if self.n < batch_size:
            raise ValueError('Batch size can be at most the dataset size')
        if not multiple_passes:
            actual_batch_size = min(batch_size, self.n - self.batch_start)
            if actual_batch_size <= 0:
                raise ValueError('Pass through the dataset is complete.')
            batch_end = self.batch_start + actual_batch_size
            batch_xs = self.xs[self.cur_order[self.batch_start : batch_end], ...]
            batch_ys = self.ys[self.cur_order[self.batch_start : batch_end], ...]
            self.batch_start += actual_batch_size
            return batch_xs, batch_ys
        actual_batch_size = min(batch_size, self.n - self.batch_start)
        if actual_batch_size < batch_size:
            if reshuffle_after_pass:
                self.cur_order = np.random.permutation(self.n)
            self.batch_start = 0
        batch_end = self.batch_start + batch_size
        batch_xs = self.xs[self.cur_order[self.batch_start : batch_end], ...]
        batch_ys = self.ys[self.cur_order[self.batch_start : batch_end], ...]
        self.batch_start += actual_batch_size
        return batch_xs, batch_ys

    @staticmethod
    def _per_class_subsample(xs, ys, size):
        per_label = {i: [] for i in range(10)}
        for x, y in zip(xs, ys):
            per_label[y].append(x)
        xs, ys = [], []
        for i in range(10):
            k = size // 10
            if i < (size % 10):
                k += 1
            xs.extend(random.sample(per_label[i], k))
            ys.extend(i for _ in range(k))
        return np.array(xs), np.array(ys)
