"""
Implementation of attack methods. Running this file as a program will
apply the attack to the model specified by the config file and store
the examples in an .npy file.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np


class LinfPGDAttack:
  def __init__(self, model, epsilon, k, a, random_start, loss_func, incremental = False, starting_epsilon = 0.01):
    """Attack parameter initialization. The attack performs k steps of
       size a, while always staying within epsilon from the initial
       point."""
    self.model = model
    self.epsilon = epsilon
    self.k = k
    self.a = a
    self.rand = random_start
    self.incremental = incremental
    self.starting_epsilon = starting_epsilon

    if loss_func == 'xent':
      loss = model.xent
    elif loss_func == 'cw':
      label_mask = tf.one_hot(model.y_input,
                              10,
                              on_value=1.0,
                              off_value=0.0,
                              dtype=tf.float32)
      correct_logit = tf.reduce_sum(label_mask * model.pre_softmax, axis=1)
      wrong_logit = tf.reduce_max((1-label_mask) * model.pre_softmax, axis=1)
      loss = -tf.nn.relu(correct_logit - wrong_logit + 50)
    else:
      print('Unknown loss function. Defaulting to cross-entropy')
      loss = model.xent

    self.grad = tf.gradients(loss, model.x_input)[0]

  def perturb(self, x_nat, y, sess, train_frac = 1):
    """Given a set of examples (x_nat, y), returns a set of adversarial
       examples within epsilon of x_nat in l_infinity norm."""
    """In the case of incremental PGD
       If train_frac < 0.5, interpolate between self.starting_epsilon and self.epsilon
       If train_frac > 0.5, use self.epsilon"""
    if self.incremental and train_frac < 0.5:
      epsilon = self.epsilon * 2 * train_frac + self.starting_epsilon * (1 - 2 * train_frac)
    else:
      epsilon = self.epsilon

    if self.rand:
      x = x_nat + np.random.uniform(-epsilon, epsilon, x_nat.shape)
    else:
      x = np.copy(x_nat)

    for i in range(self.k):
      grad = sess.run(self.grad, feed_dict={self.model.x_input: x,
                                            self.model.y_input: y})

      x += self.a * np.sign(grad)

      np.clip(x, x_nat - epsilon, x_nat + epsilon, out=x) 
      np.clip(x, 0, 1, out=x) # ensure valid pixel range

    return x
