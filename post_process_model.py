from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
from datetime import datetime
import json
import math
import os
import sys
import time
import operator
from tensorflow.python import pywrap_tensorflow

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import scipy.io as sio

from pgd_attack import LinfPGDAttack
import models.MNIST_naive_ia
import models.MNIST_naive_ia_masked

'''
NOTE: This file assumes an architecture involving a 3 layer DNN with
two 2x2-strided convolutions and a fully connected layers. It also
assumes the filter size is 5. Check #TODO for where to edit to change
the filter size.
'''

import argparse
parser = argparse.ArgumentParser(description='Pass in post-processing options. Type -h for details')
parser.add_argument('--model_dir', dest='model_dir', help='specify which saved model to load')
parser.add_argument('--no_weight_prune', dest='weight_prune', action='store_false', help='use this flag to turn off weight pruning')
parser.set_defaults(weight_prune=True)
parser.add_argument('--weight_thresh', dest='weight_thresh', default=1e-3, help='set pruning threshold for small weights (default 1e-3)')
parser.add_argument('--no_relu_prune', dest='relu_prune', action='store_false', help='use this flag to turn off relu pruning')
parser.set_defaults(relu_prune=True)
parser.add_argument('--relu_prune_frac', dest='relu_prune_frac', default=0.1, help='set pruning threshold for relus (default 0.1)')
parser.add_argument('--do_eval', dest='do_eval', action='store_true', help='use this flag to evaluate test accuracy, PGD adversarial accuracy, and ReLU stability after each post-processing step')
parser.set_defaults(do_eval=False)
parser.add_argument('--output', dest='output', help='set the name of the output .mat file')
  
args = parser.parse_args()
if args.output is None:
  raise ValueError('Need to specify output .mat filename')

model_dir = args.model_dir
weight_prune = args.weight_prune
relu_prune = args.relu_prune
do_eval = args.do_eval
relu_prune_frac = float(args.relu_prune_frac)
weight_thresh = float(args.weight_thresh)

if not os.path.isdir(model_dir):
  raise ValueError('The model directory was not found')

# Set up the data, hyperparameters, and the model
mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

with open('config.json') as config_file:
    config = json.load(config_file)

num_training_examples = config['num_training_examples']
num_eval_examples = config['num_eval_examples']
eval_batch_size = config['eval_batch_size']

model = models.MNIST_naive_ia.Model(config)
attack = LinfPGDAttack(model, 
                       config['epsilon'],
                       config['k'],
                       config['a'],
                       config['random_start'],
                       config['loss_func'])

global_step = tf.contrib.framework.get_or_create_global_step()
saver = tf.train.Saver()

def convert_conv_2x2_to_fc(input_dims, conv_filter, conv_bias, conv_mask=None):
    # Example:
    # input_dims = [28, 28]
    # conv_filter = 4D numpy array
    # conv_bias = vector
    # conv_mask = numpy array, dimensions should be [14, 14, conv_filter[-1]]
    output_dims = [(x+1)//2 for x in input_dims]
    flat_input_dim = input_dims[0] * input_dims[1]
    flat_output_dim = output_dims[0] * output_dims[1]

    num_filters = conv_filter.shape[3]
    num_in_channels = conv_filter.shape[2]
    output_dim = flat_output_dim * num_filters
    input_dim = flat_input_dim * num_in_channels
    if conv_mask is not None:
      assert(conv_mask.shape == (output_dims[0], output_dims[1], num_filters))

    fc_weights = np.zeros([input_dim, output_dim])
    fc_biases = np.zeros([output_dim])
    fc_masks = np.zeros([output_dim])

    # Look through every output pixel
    for i in range(num_filters):
        for row in range(output_dims[0]):
            for col in range(output_dims[1]):

                output_ind = to_fc_index(row, output_dims[0], col, output_dims[1], i, num_filters)
                fc_biases[output_ind] = conv_bias[i]
                if conv_mask is not None:
                  fc_masks[output_ind] = conv_mask[row, col, i]

                # TODO: May have to change this to support filter sizes other than 5x5
                # This part encodes the 2x2 strided conv
                corr_input_center = (2*row+1, 2*col+1)

                # Shift by up to 5, based on 5x5 filters
                for x_shift in range(-2, 3):
                    for y_shift in range(-2, 3):
                        new_x = corr_input_center[0] + x_shift
                        new_y = corr_input_center[1] + y_shift
                        # Skip if out of bounds
                        if new_x < 0 or new_x >= input_dims[0] or new_y < 0 or new_y >= input_dims[1]:
                            continue
                        for input_channel in range(num_in_channels):
                            conv_coeff = conv_filter[x_shift+2,y_shift+2, input_channel, i]
                            input_ind = to_conv_index(new_x, input_dims[0],
                                                        new_y, input_dims[1],
                                                        input_channel, num_in_channels)
                            fc_weights[input_ind][output_ind] = conv_coeff
    return fc_weights, fc_biases, fc_masks

def to_fc_index(x, size_x, y, size_y, filter_num, num_filters):
    return num_filters*(size_y * x + y) + filter_num

def to_conv_index(x, size_x, y, size_y, input_channel, num_in_channels):
    return num_in_channels*(size_y * x + y) + input_channel

def prune_small_weights(tf_vars, sess, tolerance):
  for tf_var in tf_vars:
    weights = sess.run(tf_var)
    weights[np.where(abs(weights) < tolerance)] = 0
    print("remaining nonzero weights: {}".format(len(np.where(abs(weights) != 0)[0])))
    print("remaining weights proportion: {}".format(len(np.where(abs(weights) != 0)[0])/len(weights.flatten())))
    tf_var.assign(weights).eval()

# A function for evaluating a single checkpoint
def evaluate_checkpoint(filename, weight_prune, tolerance, relu_prune, relu_prune_frac):
  with tf.Session() as sess:
    # Restore the checkpoint
    saver.restore(sess, filename)
    print('restored checkpoint for {}'.format(filename))
    print('First eval - no changes')

    x_single_train = mnist.train.images[0:1, :]
    y_single_train = mnist.train.labels[0:1]
    dict_nat_single = { model.x_input: x_single_train,
                        model.x_input_natural: x_single_train,
                        model.y_input: y_single_train}

    # Get the variables
    c1_v = [x for x in tf.global_variables() if x.op.name=='Variable'][0]
    c1_b = [x for x in tf.global_variables() if x.op.name=='Variable_1'][0]
    c2_v = [x for x in tf.global_variables() if x.op.name=='Variable_2'][0]
    c2_b = [x for x in tf.global_variables() if x.op.name=='Variable_3'][0]
    fc_v = [x for x in tf.global_variables() if x.op.name=='Variable_4'][0]
    fc_b = [x for x in tf.global_variables() if x.op.name=='Variable_5'][0]
    sm_v = [x for x in tf.global_variables() if x.op.name=='Variable_6'][0]
    sm_b = [x for x in tf.global_variables() if x.op.name=='Variable_7'][0]

    # Save values in the final variables
    c1, c1b, c2, c2b, fc, fcb, sm, smb = sess.run([c1_v, c1_b,
      c2_v, c2_b, fc_v, fc_b, sm_v, sm_b], feed_dict = dict_nat_single)

    if do_eval:
      # Iterate over the eval samples batch-by-batch
      num_batches = int(math.ceil(num_eval_examples / eval_batch_size))
      total_corr_nat = 0
      total_corr_adv = 0
      tot_unstable1n = 0
      tot_unstable2n = 0
      tot_unstable3n = 0

      for ibatch in range(num_batches):
        bstart = ibatch * eval_batch_size
        bend = min(bstart + eval_batch_size, num_eval_examples)

        x_batch = mnist.test.images[bstart:bend, :]
        y_batch = mnist.test.labels[bstart:bend]

        dict_nat = {model.x_input: x_batch,
                    model.x_input_natural: x_batch,
                    model.y_input: y_batch}

        x_batch_adv = attack.perturb(x_batch, y_batch, sess)

        dict_adv = {model.x_input: x_batch_adv,
                    model.x_input_natural: x_batch,
                    model.y_input: y_batch}

        cur_corr_nat = sess.run(model.num_correct, feed_dict = dict_nat)
        cur_corr_adv = sess.run(model.num_correct, feed_dict = dict_adv)      
        
        total_corr_nat += cur_corr_nat
        total_corr_adv += cur_corr_adv

        un1n, un2n, un3n = \
          sess.run([model.unstable1, model.unstable2, \
                    model.unstable3],
                    feed_dict = dict_nat)

        tot_unstable1n += np.sum(un1n)
        tot_unstable2n += np.sum(un2n)
        tot_unstable3n += np.sum(un3n)

      avg_un1n = tot_unstable1n / num_eval_examples
      avg_un2n = tot_unstable2n / num_eval_examples
      avg_un3n = tot_unstable3n / num_eval_examples
      acc_nat = total_corr_nat / num_eval_examples
      acc_adv = total_corr_adv / num_eval_examples

      print('natural: {:.2f}%'.format(100 * acc_nat))
      print('adversarial: {:.2f}%'.format(100 * acc_adv))
      print('  un1n, un2n, un3n: {}, {}, {}'.format(avg_un1n,
              avg_un2n, avg_un3n))

    if weight_prune:
      print('Second eval - prune small weights')
      
      # Hardcoded variables
      prune_small_weights([c1_v, c2_v, fc_v], sess, tolerance)

      # These are the correct values (no need to refix-nonzeros) for the masked models
      c1, c1b, c2, c2b, fc, fcb, sm, smb = sess.run([c1_v, c1_b,
        c2_v, c2_b, fc_v, fc_b, sm_v, sm_b], feed_dict = dict_nat_single)
      
      if do_eval:
        # Iterate over the eval samples batch-by-batch
        num_batches = int(math.ceil(num_eval_examples / eval_batch_size))
        total_corr_nat = 0
        total_corr_adv = 0
        tot_unstable1n = 0
        tot_unstable2n = 0
        tot_unstable3n = 0

        for ibatch in range(num_batches):
          bstart = ibatch * eval_batch_size
          bend = min(bstart + eval_batch_size, num_eval_examples)

          x_batch = mnist.test.images[bstart:bend, :]
          y_batch = mnist.test.labels[bstart:bend]

          dict_nat = {model.x_input: x_batch,
                      model.x_input_natural: x_batch,
                      model.y_input: y_batch}

          x_batch_adv = attack.perturb(x_batch, y_batch, sess)

          dict_adv = {model.x_input: x_batch_adv,
                      model.x_input_natural: x_batch,
                      model.y_input: y_batch}

          cur_corr_nat = sess.run(model.num_correct, feed_dict = dict_nat)
          cur_corr_adv = sess.run(model.num_correct, feed_dict = dict_adv)      
          
          total_corr_nat += cur_corr_nat
          total_corr_adv += cur_corr_adv

          un1n, un2n, un3n = \
            sess.run([model.unstable1, model.unstable2, \
                      model.unstable3],
                      feed_dict = dict_nat)

          tot_unstable1n += np.sum(un1n)
          tot_unstable2n += np.sum(un2n)
          tot_unstable3n += np.sum(un3n)

        avg_un1n = tot_unstable1n / num_eval_examples
        avg_un2n = tot_unstable2n / num_eval_examples
        avg_un3n = tot_unstable3n / num_eval_examples
        acc_nat = total_corr_nat / num_eval_examples
        acc_adv = total_corr_adv / num_eval_examples

        print('natural: {:.2f}%'.format(100 * acc_nat))
        print('adversarial: {:.2f}%'.format(100 * acc_adv))
        print('  un1n, un2n, un3n: {}, {}, {}'.format(avg_un1n,
                avg_un2n, avg_un3n))

    if relu_prune:
      print('Third eval - prune relus')

      # Get locations of where relus are equal (or close) to 0 or 55000
      h1_rc = tf.reduce_sum(tf.cast(model.h_1>0, tf.int32), axis = 0)
      h2_rc = tf.reduce_sum(tf.cast(model.h_2>0, tf.int32), axis = 0)
      hfc_rc = tf.reduce_sum(tf.cast(model.h_fc_pre_relu>0, tf.int32), axis = 0)

      # Iterate over the training samples batch-by-batch to do relu count
      num_training_batches = int(math.ceil(num_training_examples / eval_batch_size))

      # Only do relu count for adv training examples only, since DNN is trained on adv
      tot_rc1 = 0
      tot_rc2 = 0
      tot_rfc = 0

      for ibatch in range(num_training_batches):
        bstart = ibatch * eval_batch_size
        bend = min(bstart + eval_batch_size, num_training_examples)

        x_batch = mnist.train.images[bstart:bend, :]
        y_batch = mnist.train.labels[bstart:bend]
        x_batch_adv = attack.perturb(x_batch, y_batch, sess)

        dict_adv = {model.x_input: x_batch_adv,
                    model.x_input_natural: x_batch,
                    model.y_input: y_batch}

        rc1_adv = sess.run(h1_rc,feed_dict = dict_adv)
        rc2_adv = sess.run(h2_rc,feed_dict = dict_adv)
        rfc_adv = sess.run(hfc_rc,feed_dict = dict_adv)
        tot_rc1 += rc1_adv
        tot_rc2 += rc2_adv
        tot_rfc += rfc_adv
      
      def get_ops(adv, relu_prune_frac):
          num_to_remove = int(num_training_examples * relu_prune_frac)
          assert(num_to_remove <= num_training_examples/2 + 1)
          linear_relus = adv >= (num_training_examples - num_to_remove)
          zero_relus = adv <= num_to_remove
          ops = np.zeros(adv.shape)
          ops[linear_relus] = 1
          ops[zero_relus] = -1
          print("number of relus left: ", len(ops[ops==0]))
          return ops

      c1_ops = get_ops(tot_rc1, relu_prune_frac)
      c2_ops = get_ops(tot_rc2, relu_prune_frac)
      fc_ops = get_ops(tot_rfc, relu_prune_frac)

      if do_eval:
        mask_model = models.MNIST_naive_ia_masked.Model(config, c1_ops, c2_ops, fc_ops)
        mask_model_attack = LinfPGDAttack(mask_model, 
                               config['epsilon'],
                               config['k'],
                               config['a'],
                               config['random_start'],
                               config['loss_func'])

        print("Created masked model")

        # Copy variables over from main model
        new_c1_v = [x for x in tf.global_variables() if x.op.name=='Variable_8'][0]
        new_c1_b = [x for x in tf.global_variables() if x.op.name=='Variable_9'][0]
        new_c2_v = [x for x in tf.global_variables() if x.op.name=='Variable_10'][0]
        new_c2_b = [x for x in tf.global_variables() if x.op.name=='Variable_11'][0]
        new_fc_v = [x for x in tf.global_variables() if x.op.name=='Variable_12'][0]
        new_fc_b = [x for x in tf.global_variables() if x.op.name=='Variable_13'][0]
        new_sm_v = [x for x in tf.global_variables() if x.op.name=='Variable_14'][0]
        new_sm_b = [x for x in tf.global_variables() if x.op.name=='Variable_15'][0]

        new_c1_v.assign(c1).eval()
        new_c1_b.assign(c1b).eval()
        new_c2_v.assign(c2).eval()
        new_c2_b.assign(c2b).eval()
        new_fc_v.assign(fc).eval()
        new_fc_b.assign(fcb).eval()
        new_sm_v.assign(sm).eval()
        new_sm_b.assign(smb).eval()

        # Iterate over the eval samples batch-by-batch
        num_batches = int(math.ceil(num_eval_examples / eval_batch_size))
        total_corr_nat = 0
        total_corr_adv = 0
        tot_unstable1n = 0
        tot_unstable2n = 0
        tot_unstable3n = 0

        for ibatch in range(num_batches):
          bstart = ibatch * eval_batch_size
          bend = min(bstart + eval_batch_size, num_eval_examples)

          x_batch = mnist.test.images[bstart:bend, :]
          y_batch = mnist.test.labels[bstart:bend]

          dict_nat = {mask_model.x_input: x_batch,
                      mask_model.x_input_natural: x_batch,
                      mask_model.y_input: y_batch}

          x_batch_adv = mask_model_attack.perturb(x_batch, y_batch, sess)

          dict_adv = {mask_model.x_input: x_batch_adv,
                      mask_model.x_input_natural: x_batch,
                      mask_model.y_input: y_batch}

          cur_corr_nat = sess.run(mask_model.num_correct, feed_dict = dict_nat)
          cur_corr_adv = sess.run(mask_model.num_correct, feed_dict = dict_adv)      
          
          total_corr_nat += cur_corr_nat
          total_corr_adv += cur_corr_adv

          un1n, un2n, un3n = \
            sess.run([mask_model.unstable1, mask_model.unstable2, \
                      mask_model.unstable3],
                      feed_dict = dict_nat)

          tot_unstable1n += np.sum(un1n)
          tot_unstable2n += np.sum(un2n)
          tot_unstable3n += np.sum(un3n)

        avg_un1n = tot_unstable1n / num_eval_examples
        avg_un2n = tot_unstable2n / num_eval_examples
        avg_un3n = tot_unstable3n / num_eval_examples
        acc_nat = total_corr_nat / num_eval_examples
        acc_adv = total_corr_adv / num_eval_examples

        print('natural: {:.2f}%'.format(100 * acc_nat))
        print('adversarial: {:.2f}%'.format(100 * acc_adv))
        print('  un1n, un2n, un3n: {}, {}, {}'.format(avg_un1n,
                avg_un2n, avg_un3n))

    new_model_weights = { 'c1_w': c1,
                          'c1_b': c1b,
                          'c2_w': c2,
                          'c2_b': c2b,
                          'fc_w': fc,
                          'fc_b': fcb,
                          'sm_w': sm,
                          'sm_b': smb,
                          }
    if relu_prune:
      new_model_weights['c1_m'] = c1_ops
      new_model_weights['c2_m'] = c2_ops
      new_model_weights['fc_m'] = fc_ops
  return new_model_weights


print("Processing model from {}".format(model_dir))
cur_checkpoint = tf.train.latest_checkpoint(model_dir)
new_model = evaluate_checkpoint(cur_checkpoint, weight_prune, weight_thresh,
                                                relu_prune, relu_prune_frac)

if relu_prune:
  fc1_weight, fc1_bias, fc1_mask = convert_conv_2x2_to_fc([28, 28], 
            new_model['c1_w'], new_model['c1_b'], new_model['c1_m'])
  fc2_weight, fc2_bias, fc2_mask = convert_conv_2x2_to_fc([14, 14], 
            new_model['c2_w'], new_model['c2_b'], new_model['c2_m'])
else:
  fc1_weight, fc1_bias, fc1_mask = convert_conv_2x2_to_fc([28, 28], 
            new_model['c1_w'], new_model['c1_b'])
  fc2_weight, fc2_bias, fc2_mask = convert_conv_2x2_to_fc([14, 14], 
            new_model['c2_w'], new_model['c2_b'])

print("Saving model now")

all_weights = {     'fc1/weight':       fc1_weight,
                    'fc1/bias':         fc1_bias,
                    'fc2/weight':       fc2_weight,
                    'fc2/bias':         fc2_bias,
                    'fc3/weight':       new_model['fc_w'],
                    'fc3/bias':         new_model['fc_b'],
                    'softmax/weight':   new_model['sm_w'],
                    'softmax/bias':     new_model['sm_b']     }

if relu_prune:
  all_weights['fc1/mask'] = fc1_mask
  all_weights['fc2/mask'] = fc2_mask
  all_weights['fc3/mask'] = new_model['fc_m']

if not os.path.exists('model_mats'):
    os.makedirs('model_mats')

sio.savemat('model_mats/{}'.format(args.output), all_weights)

