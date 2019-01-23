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

    # l1 loss
    self.l1_loss = 14 * 14 * self._l1(self.W_conv1) + 7 * 7 * self._l1(self.W_conv2) + self._l1(self.W_fc1)

    # relu lb/ub estimation for layer 0
    self.lb_0 = tf.maximum(self.x_input_natural - config["eval_epsilon"], 0)
    self.ub_0 = tf.minimum(self.x_input_natural + config["eval_epsilon"], 1)
    lb_0_e = tf.expand_dims(self.lb_0, 1)
    ub_0_e = tf.expand_dims(self.ub_0, 1)

    # Convert conv1 to fc
    id_reshaped = tf.reshape(tf.eye(28 * 28), [-1, 28, 28, 1])
    W_conv1_fc = self._conv2d_2x2_strided(id_reshaped, self.W_conv1)
    self.W_conv1_fc = tf.reshape(W_conv1_fc, [-1, 14 * 14 * filters[0]])
    b_conv1_expanded = tf.reshape(tf.expand_dims(tf.ones([14, 14]), 2) * b_conv1, [-1, 14 * 14 * filters[0]])[0]

    # relu lb/ub estimation for layer 1
    self.lb_1, self.ub_1 = self._interval_arithmetic(self.lb_0, self.ub_0, self.W_conv1_fc, b_conv1_expanded)
    lb_1_e = tf.expand_dims(self.get_anti_relu_layer(self.lb_1, c1_ops.flatten()), 1)
    ub_1_e = tf.expand_dims(self.get_anti_relu_layer(self.ub_1, c1_ops.flatten()), 1)

    # Convert conv2 to fc
    id_reshaped = tf.reshape(tf.eye(14 * 14 * filters[0]), [-1, 14, 14, filters[0]])
    W_conv2_fc = self._conv2d_2x2_strided(id_reshaped, self.W_conv2)
    self.W_conv2_fc = tf.reshape(W_conv2_fc, [-1, 7 * 7 * filters[1]])
    b_conv2_expanded = tf.reshape(tf.expand_dims(tf.ones([7, 7]), 2) * b_conv2, [-1, 7 * 7 * filters[1]])[0]

    # relu lb/ub estimation for layer 2
    self.lb_2, self.ub_2 = self._compute_bounds_2_layers(lb_1_e, ub_1_e, self.W_conv2_fc, b_conv2_expanded, lb_0_e, ub_0_e, self.W_conv1_fc, b_conv1_expanded)
    lb_2_temp = self.lb_2[:, 0, :]
    ub_2_temp = self.ub_2[:, 0, :]
    lb_2_e = tf.expand_dims(self.get_anti_relu_layer(lb_2_temp, c2_ops.flatten()), 1)
    ub_2_e = tf.expand_dims(self.get_anti_relu_layer(ub_2_temp, c2_ops.flatten()), 1)

    # relu lb/ub estimation for layer 3
    self.lb_3, self.ub_3 = self._compute_bounds_3_layers(lb_2_e, ub_2_e, self.W_fc1, b_fc1, lb_1_e, ub_1_e, self.W_conv2_fc, b_conv2_expanded, lb_0_e, ub_0_e, self.W_conv1_fc, b_conv1_expanded)

    # unstable relus estimation
    self.unstable1 = self._num_unstable(self.lb_1, self.ub_1, c1_ops.flatten())
    self.unstable2 = self._num_unstable(self.lb_2, self.ub_2, c2_ops.flatten())
    self.unstable3 = self._num_unstable(self.lb_3, self.ub_3, fc_ops.flatten())
    
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

    self.xent = tf.reduce_mean(y_xent)

    self.y_pred = tf.argmax(self.pre_softmax, 1)

    correct_prediction = tf.equal(self.y_pred, self.y_input)

    self.num_correct = tf.reduce_sum(tf.cast(correct_prediction, tf.int64))
    self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Naive IA
    self.lb_1_naive, self.ub_1_naive = self.lb_1, self.ub_1
    lb_1_naive_h = self.get_anti_relu_layer(self.lb_1_naive, c1_ops.flatten())
    ub_1_naive_h = self.get_anti_relu_layer(self.ub_1_naive, c1_ops.flatten())

    self.lb_2_naive, self.ub_2_naive = self._interval_arithmetic(lb_1_naive_h, ub_1_naive_h, self.W_conv2_fc, b_conv2_expanded)

    lb_2_naive_h = self.get_anti_relu_layer(self.lb_2_naive, c2_ops.flatten())
    ub_2_naive_h = self.get_anti_relu_layer(self.ub_2_naive, c2_ops.flatten())

    lb_2_naive_h_flat = tf.reshape(lb_2_naive_h, [-1, 7 * 7 * filters[1]])
    ub_2_naive_h_flat = tf.reshape(ub_2_naive_h, [-1, 7 * 7 * filters[1]])

    # naive lb/ub estimation for layer 3
    self.lb_3_naive, self.ub_3_naive = self._interval_arithmetic(lb_2_naive_h_flat, ub_2_naive_h_flat, self.W_fc1, b_fc1)

    # unstable relus estimation
    self.unstable1_naive = self._num_unstable(self.lb_1_naive, self.ub_1_naive, c1_ops.flatten())
    self.unstable2_naive = self._num_unstable(self.lb_2_naive, self.ub_2_naive, c2_ops.flatten())
    self.unstable3_naive = self._num_unstable(self.lb_3_naive, self.ub_3_naive, fc_ops.flatten())


  def _compute_bounds_2_layers(self, lb, ub, W, bias, lb_m1, ub_m1, prev_W, prev_bias):
      # Assumes lb_m1/ub_m1 have shape B x 1 x n, lb/ub have shape B x 1 x m, 
      # prev_W is n x m, W is m x p -> convert to B x n x m

      out_dim = W.shape[1].value
      lb_reshaped = tf.transpose(lb, [1, 0, 2])[0] # This should be B x m now
      NL_mask_unexpanded = tf.cast(tf.less_equal(lb_reshaped, 0), dtype=tf.float32)
      NL_mask = tf.tile(tf.expand_dims(NL_mask_unexpanded, 2), [1, 1, out_dim]) # This should be B x m x p
      L_mask_unexpanded = 1.0 - NL_mask_unexpanded
      L_mask = 1.0 - NL_mask
      
      W_fc_NL = tf.multiply(W, NL_mask) # B x m x p
      W_fc_L = tf.multiply(W, L_mask) # Should be B x m x p
      W_fc_L_prod = tf.einsum('nm,bmp->bnp', prev_W, W_fc_L)

      extra_bias = tf.expand_dims(tf.einsum('m,bmp->bp', prev_bias, W_fc_L), 1)

      lb_NL, ub_NL = self._interval_arithmetic(lb, ub, W_fc_NL, bias)
      lb_L, ub_L = self._interval_arithmetic(lb_m1, ub_m1, W_fc_L_prod, extra_bias)

      new_lb, new_ub = lb_NL + lb_L, ub_NL + ub_L
      return new_lb, new_ub

  def _compute_bounds_3_layers(self, lb, ub, W, bias, lb_m1, ub_m1, W_m1, bias_m1, lb_m2, ub_m2, W_m2, bias_m2):
      # Assumes lb_m1/ub_m1 have shape B x 1 x n, lb/ub have shape B x 1 x m, 
      # prev_W is n x m, W is m x p -> convert to B x n x m

      out_dim = W.shape[1].value
      out_dim_m1 = W_m1.shape[1].value

      lb_reshaped = tf.transpose(lb, [1, 0, 2])[0] # This should be B x m now
      NL_mask_unexpanded = tf.cast(tf.less_equal(lb_reshaped, 0), dtype=tf.float32)
      NL_mask = tf.tile(tf.expand_dims(NL_mask_unexpanded, 2), [1, 1, out_dim]) # This should be B x m x p
      L_mask_unexpanded = 1.0 - NL_mask_unexpanded
      L_mask = 1.0 - NL_mask
      
      W_fc_NL = tf.multiply(W, NL_mask) # B x m x p
      W_fc_L = tf.multiply(W, L_mask) # Should be B x m x p

      # Same thing for m1
      lb_m1_reshaped = tf.transpose(lb_m1, [1, 0, 2])[0] # This should be B x n now
      NL_m1_mask_unexpanded = tf.cast(tf.less_equal(lb_m1_reshaped, 0), dtype=tf.float32)
      NL_m1_mask = tf.tile(tf.expand_dims(NL_m1_mask_unexpanded, 2), [1, 1, out_dim_m1]) # This should be B x m x p
      L_m1_mask_unexpanded = 1.0 - NL_m1_mask_unexpanded
      L_m1_mask = 1.0 - NL_m1_mask
      
      W_m1_fc_NL = tf.multiply(W_m1, NL_m1_mask) # B x n x m
      W_m1_fc_L = tf.multiply(W_m1, L_m1_mask) # Should be B x n x m

      W_fc_L_prod = tf.einsum('bnm,bmp->bnp', W_m1_fc_NL, W_fc_L)
      W_fc_L_prod_L = tf.einsum('bnm,bmp->bnp', W_m1_fc_L, W_fc_L)
      W_fc_L_mega_prod = tf.einsum('kn,bnp->bkp', W_m2, W_fc_L_prod_L)

      bias_prop_m1 = tf.expand_dims(tf.einsum('m,bmp->bp', bias_m1, W_fc_L), 1)
      bias_prop_m2 = tf.expand_dims(tf.einsum('n,bnp->bp', bias_m2, W_fc_L_prod_L), 1)

      lb_m2, ub_m2 = self._interval_arithmetic(lb_m2, ub_m2, W_fc_L_mega_prod, bias_prop_m2)
      lb_m1, ub_m1 = self._interval_arithmetic(lb_m1, ub_m1, W_fc_L_prod, bias_prop_m1)
      lb, ub = self._interval_arithmetic(lb, ub, W_fc_NL, bias)

      new_lb, new_ub = lb + lb_m1 + lb_m2, ub + ub_m1 + ub_m2
      return new_lb, new_ub

  @staticmethod
  def _interval_arithmetic(lb, ub, W, b):
      W_max = tf.maximum(W, 0.0)
      W_min = tf.minimum(W, 0.0)
      new_lb = tf.matmul(lb, W_max) + tf.matmul(ub, W_min) + b
      new_ub = tf.matmul(ub, W_max) + tf.matmul(lb, W_min) + b
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
  def _l1(var):
    """L1 weight decay loss."""
    return  tf.reduce_sum(tf.abs(var))

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

  @staticmethod
  def _l_relu_stable(lb, ub, norm_constant=1.0):
    loss = -tf.reduce_mean(tf.reduce_sum(tf.tanh(1.0+ norm_constant * lb * ub), axis=-1))
    return loss

  @staticmethod
  def _conv2d_2x2_strided(x, W):
      return tf.nn.conv2d(x, W, strides=[1,2,2,1], padding='SAME')

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
