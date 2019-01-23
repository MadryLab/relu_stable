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
    def __init__(self, config, c1_ops, c2_ops, fc_ops):
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
        self.h_conv1 = self.get_anti_relu_layer(self.h_1, c1_ops)

        # second convolutional layer
        self.W_conv2 = self._weight_variable([filter_size, filter_size, filters[0], filters[1]],
                          sparsity = config["sparse_init"])
        b_conv2 = self._bias_variable([filters[1]])
        self.h_2 = self._conv2d_2x2_strided(self.h_conv1, self.W_conv2) + b_conv2
        self.h_conv2 = self.get_anti_relu_layer(self.h_2, c2_ops)

        # first fc layer
        self.W_fc1 = self._weight_variable([7 * 7 * filters[1], filters[2]])
        b_fc1 = self._bias_variable([filters[2]])
        h_conv2_flat = tf.reshape(self.h_conv2, [-1, 7 * 7 * filters[1]])
        self.h_fc_pre_relu = tf.matmul(h_conv2_flat, self.W_fc1) + b_fc1
        self.h_fc1 = self.get_anti_relu_layer(self.h_fc_pre_relu, fc_ops)

        # relu lb/ub estimation for layer 0
        self.lb_0 = tf.maximum(self.x_input_natural_reshaped - config["eval_epsilon"], 0)
        self.ub_0 = tf.minimum(self.x_input_natural_reshaped + config["eval_epsilon"], 1)

        # relu lb/ub estimation for layer 1
        self.lb_1, self.ub_1 = self._interval_arithmetic_conv_2x2_strided(self.lb_0, self.ub_0, self.W_conv1, b_conv1)
        self.lbh_1, self.ubh_1 = self.get_anti_relu_layer(self.lb_1, c1_ops), self.get_anti_relu_layer(self.ub_1, c1_ops)

        # relu lb/ub estimation for layer 2
        self.lb_2, self.ub_2 = self._interval_arithmetic_conv_2x2_strided(self.lbh_1, self.ubh_1, self.W_conv2, b_conv2)
        self.lbh_2, self.ubh_2 = self.get_anti_relu_layer(self.lb_2, c2_ops), self.get_anti_relu_layer(self.ub_2, c2_ops)
        self.lbh_2_flat = tf.reshape(self.lbh_2, [-1, 7 * 7 * filters[1]])
        self.ubh_2_flat = tf.reshape(self.ubh_2, [-1, 7 * 7 * filters[1]])

        # relu lb/ub estimation for layer 3
        self.lb_3, self.ub_3 = self._interval_arithmetic(self.lbh_2_flat, self.ubh_2_flat, self.W_fc1, b_fc1)

        # unstable relus estimation
        self.unstable1 = self._num_unstable(self.lb_1, self.ub_1, c1_ops)
        self.unstable2 = self._num_unstable(self.lb_2, self.ub_2, c2_ops)
        self.unstable3 = self._num_unstable(self.lb_3, self.ub_3, fc_ops)

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

    """Count number of unstable ReLUs"""
    @staticmethod
    def _num_unstable(lb, ub, ops=None):
        is_unstable = tf.cast(lb * ub < 0.0, tf.int32)
        if ops is not None:
            is_relu = (ops == 0)
            is_unstable_relu = is_relu * is_unstable
        else:
            is_unstable_relu = is_unstable
        all_but_first_dim = np.arange(len(is_unstable_relu.shape))[1:]
        result = tf.reduce_sum(is_unstable_relu, all_but_first_dim)
        return result

    def get_anti_relu_layer(self, activations, ops):
        assert (activations.shape[1:] == ops.shape)
    
        # Hacky solution because tensorflow is hard
        shape = activations.shape.as_list()
        shape[0] = -1
        num_elements = np.prod(shape[1:])
        flat_shape = [-1, num_elements]
        flat_activations = tf.reshape(activations, flat_shape)
        flat_ops = ops.flatten()
        flat_output = [0 for i in range(num_elements)]

        for i in range(num_elements):
            op = flat_ops[i]
            if op == -1:
                flat_output[i] = 0 * flat_activations[:,i]
            elif op == 1:
                flat_output[i] = flat_activations[:,i]
            elif op == 0:
                flat_output[i] = tf.nn.relu(flat_activations[:,i])
            else:
                raise ValueError("Ops should be -1, 1, or 0, but it is not")

        # Transpose is necessary for batch size > 1
        output = tf.reshape(tf.transpose(flat_output), shape)
        return output
