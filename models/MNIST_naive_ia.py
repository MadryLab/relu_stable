"""
The model is adapted from the tensorflow tutorial:
https://www.tensorflow.org/get_started/mnist/pros
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

class Model(object):
    def __init__(self, config):
        filters = config["filters"]
        filter_size = config["filter_size"]

        self.x_input = tf.placeholder(tf.float32, shape = [None, 784])
        self.y_input = tf.placeholder(tf.int64, shape = [None])
        self.x_input_natural = tf.placeholder(tf.float32, shape = [None, 784])
        self.x_input_natural_reshaped = tf.reshape(self.x_input_natural, [-1, 28, 28, 1])
        self.x_image = tf.reshape(self.x_input, [-1, 28, 28, 1])

        # first convolutional layer
        self.W_conv1 = self._weight_variable([filter_size, filter_size, 1, filters[0]],
                          sparsity = config["sparse_init"])
        b_conv1 = self._bias_variable([filters[0]])
        self.h_1 = self._conv2d_2x2_strided(self.x_image, self.W_conv1) + b_conv1
        self.h_conv1 = tf.nn.relu(self.h_1)

        # second convolutional layer
        self.W_conv2 = self._weight_variable([filter_size, filter_size, filters[0], filters[1]],
                          sparsity = config["sparse_init"])
        b_conv2 = self._bias_variable([filters[1]])
        self.h_2 = self._conv2d_2x2_strided(self.h_conv1, self.W_conv2) + b_conv2
        self.h_conv2 = tf.nn.relu(self.h_2)

        # first fc layer
        self.W_fc1 = self._weight_variable([7 * 7 * filters[1], filters[2]])
        b_fc1 = self._bias_variable([filters[2]])
        h_conv2_flat = tf.reshape(self.h_conv2, [-1, 7 * 7 * filters[1]])
        self.h_fc_pre_relu = tf.matmul(h_conv2_flat, self.W_fc1) + b_fc1
        self.h_fc1 = tf.nn.relu(self.h_fc_pre_relu)

        # l1 loss, with weights scaled based on the fully-connected matrix each conv layer represents
        self.l1_loss = 14 * 14 * self._l1(self.W_conv1) + 7 * 7 * self._l1(self.W_conv2) + self._l1(self.W_fc1)

        # relu lb/ub estimation for layer 0
        self.lb_0 = tf.maximum(self.x_input_natural_reshaped - config["eval_epsilon"], 0)
        self.ub_0 = tf.minimum(self.x_input_natural_reshaped + config["eval_epsilon"], 1)

        # relu lb/ub estimation for layer 1
        self.lb_1, self.ub_1 = self._interval_arithmetic_conv_2x2_strided(self.lb_0, self.ub_0, self.W_conv1, b_conv1)
        self.lbh_1, self.ubh_1 = tf.nn.relu(self.lb_1), tf.nn.relu(self.ub_1)

        # relu lb/ub estimation for layer 2
        self.lb_2, self.ub_2 = self._interval_arithmetic_conv_2x2_strided(self.lbh_1, self.ubh_1, self.W_conv2, b_conv2)
        self.lbh_2, self.ubh_2 = tf.nn.relu(self.lb_2), tf.nn.relu(self.ub_2)
        self.lbh_2_flat = tf.reshape(self.lbh_2, [-1, 7 * 7 * filters[1]])
        self.ubh_2_flat = tf.reshape(self.ubh_2, [-1, 7 * 7 * filters[1]])

        # relu lb/ub estimation for layer 3
        self.lb_3, self.ub_3 = self._interval_arithmetic(self.lbh_2_flat, self.ubh_2_flat, self.W_fc1, b_fc1)

        # unstable relus estimation
        self.unstable1 = self._num_unstable(self.lb_1, self.ub_1)
        self.unstable2 = self._num_unstable(self.lb_2, self.ub_2)
        self.unstable3 = self._num_unstable(self.lb_3, self.ub_3)

        # unstable relus loss
        self.un1loss = self._l_relu_stable(self.lb_1, self.ub_1)
        self.un2loss = self._l_relu_stable(self.lb_2, self.ub_2)
        self.un3loss = self._l_relu_stable(self.lb_3, self.ub_3)
        self.rsloss = self.un1loss + self.un2loss + self.un3loss

        # output layer
        self.W_fc_out = self._weight_variable([filters[2],10])
        b_fc_out = self._bias_variable([10])
        self.pre_softmax = tf.matmul(self.h_fc1, self.W_fc_out) + b_fc_out
        y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.y_input, logits=self.pre_softmax)

        # xent loss
        self.xent = tf.reduce_mean(y_xent)

        # Final prediction
        self.y_pred = tf.argmax(self.pre_softmax, 1)
        correct_prediction = tf.equal(self.y_pred, self.y_input)
        self.num_correct = tf.reduce_sum(tf.cast(correct_prediction, tf.int64))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Assumes shapes of Bxm, Bxm, mxn, n
    def _interval_arithmetic(self, lb, ub, W, b):
        W_max = tf.maximum(W, 0.0)
        W_min = tf.minimum(W, 0.0)
        new_lb = tf.matmul(lb, W_max) + tf.matmul(ub, W_min) + b
        new_ub = tf.matmul(ub, W_max) + tf.matmul(lb, W_min) + b
        return new_lb, new_ub

    def _interval_arithmetic_conv_2x2_strided(self, lb, ub, W, b):
        W_max = tf.maximum(W, 0.0)
        W_min = tf.minimum(W, 0.0)
        new_lb = self._conv2d_2x2_strided(lb, W_max) + self._conv2d_2x2_strided(ub, W_min) + b
        new_ub = self._conv2d_2x2_strided(ub, W_max) + self._conv2d_2x2_strided(lb, W_min) + b
        return new_lb, new_ub

    @staticmethod
    def _weight_variable(shape, sparsity=-1.0):
        initial = tf.truncated_normal(shape, stddev=0.1)
        if sparsity > 0:
            mask = tf.cast(tf.random_uniform(shape) < sparsity, tf.float32)
            initial *= mask
        return tf.Variable(initial)

    @staticmethod
    def _bias_variable(shape):
        initial = tf.constant(0.1, shape = shape)
        return tf.Variable(initial)

    @staticmethod
    def _conv2d_2x2_strided(x, W):
        return tf.nn.conv2d(x, W, strides=[1,2,2,1], padding='SAME')

    """L1 weight decay loss."""
    @staticmethod 
    def _l1(var):
        return  tf.reduce_sum(tf.abs(var))

    """RS Loss"""
    @staticmethod
    def _l_relu_stable(lb, ub, norm_constant=1.0):
        loss = -tf.reduce_mean(tf.reduce_sum(tf.tanh(1.0+ norm_constant * lb * ub), axis=-1))
        return loss

    """Count number of unstable ReLUs"""
    @staticmethod
    def _num_unstable(lb, ub):
        is_unstable = tf.cast(lb * ub < 0.0, tf.int32)
        all_but_first_dim = np.arange(len(is_unstable.shape))[1:]
        result = tf.reduce_sum(is_unstable, all_but_first_dim)
        return result

