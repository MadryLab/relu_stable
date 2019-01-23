"""Trains a model, saving checkpoints and tensorboard summaries along
   the way."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import json
import math
import os
import shutil
from timeit import default_timer as timer

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tqdm import trange

from datasubset import DataSubset
import models.MNIST_improved_ia
import models.MNIST_naive_ia
from pgd_attack import LinfPGDAttack

with open('config.json') as config_file:
    config = json.load(config_file)
if os.path.exists('job_parameters.json'):
    with open('job_parameters.json') as config_file:
        param_config = json.load(config_file)
    for k in param_config.keys():
        assert k in config.keys()
    config.update(param_config)

# Setting up training parameters
tf.set_random_seed(config['random_seed'])

# Training parameters
max_num_training_steps = config['max_num_training_steps']
num_output_steps = config['num_output_steps']
num_summary_steps = config['num_summary_steps']
num_checkpoint_steps = config['num_checkpoint_steps']
num_eval_steps = config['num_eval_steps']
dataset_size = config['num_training_examples']
batch_size = config['training_batch_size']
eval_during_training = config['eval_during_training']
adv_training = config['adversarial_training']
w_l1 = config["w_l1"]
w_rsloss = config["w_rsloss"]

# Eval parameters
num_eval_examples = config['num_eval_examples']
eval_batch_size = config['eval_batch_size']

# Output directory
model_dir = config['model_dir']

# Setting up the training data
mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
mnist_train = DataSubset(mnist.train.images,
                         mnist.train.labels,
                         dataset_size)
global_step = tf.contrib.framework.get_or_create_global_step()

# Setting up the model
if config["estimation_method"] == 'improved_ia':
    model = models.MNIST_improved_ia.Model(config)
elif config["estimation_method"] == 'naive_ia':
    model = models.MNIST_naive_ia.Model(config)
else:
    print("Defaulting to Naive IA for ReLU bound estimation")
    model = models.MNIST_naive_ia.Model(config)

# Setting up the optimizer
train_step = tf.train.AdamOptimizer(1e-4).minimize(model.xent + \
                                                    w_l1 * model.l1_loss + \
                                                    w_rsloss * model.rsloss,
                                                    global_step=global_step)

# Set up adversary
attack = LinfPGDAttack(model, 
                       config['epsilon'],
                       config['k'],
                       config['a'],
                       config['random_start'],
                       config['loss_func'],
                       config['incremental'])

# Set up eval adversary in the case of an incremental training schedule
eval_attack = LinfPGDAttack(model,
                       config['eval_epsilon'],
                       40,
                       config['eval_epsilon']/10.0,
                       config['random_start'],
                       config['loss_func'])

# Setting up the Tensorboard and checkpoint outputs
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
eval_dir = os.path.join(model_dir, 'eval')
if eval_during_training and not os.path.exists(eval_dir):
    os.makedirs(eval_dir)

# Keep track of accuracies in Tensorboard
saver = tf.train.Saver(max_to_keep=3)
tf.summary.scalar('accuracy_adv_train', model.accuracy, collections = ['adv'])
tf.summary.scalar('accuracy_adv', model.accuracy, collections = ['adv'])
tf.summary.scalar('xent_adv_train', model.xent, collections = ['adv'])
tf.summary.scalar('xent_adv', model.xent, collections = ['adv'])
adv_summaries = tf.summary.merge_all('adv')

tf.summary.scalar('accuracy_nat_train', model.accuracy, collections = ['nat'])
tf.summary.scalar('accuracy_nat', model.accuracy, collections = ['nat'])
tf.summary.scalar('xent_nat_train', model.xent, collections = ['nat'])
tf.summary.scalar('xent_nat', model.xent, collections = ['nat'])
nat_summaries = tf.summary.merge_all('nat')

# Keep track of number of unstable relus and the RS Loss
tf.summary.scalar('avg_un1', model.unstable1, collections = ['unstable'])
tf.summary.scalar('avg_un2', model.unstable2, collections = ['unstable'])
tf.summary.scalar('avg_un3', model.unstable3, collections = ['unstable'])
tf.summary.scalar('avg_un1l', model.un1loss, collections = ['unstable'])
tf.summary.scalar('avg_un2l', model.un2loss, collections = ['unstable'])
tf.summary.scalar('avg_un3l', model.un3loss, collections = ['unstable'])
unstable_summaries = tf.summary.merge_all('unstable')

shutil.copy('config.json', model_dir)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    # Initialize the summary writer, global variables, and our time counter.
    summary_writer = tf.summary.FileWriter(model_dir, sess.graph)
    if eval_during_training:
        summary_writer_eval = tf.summary.FileWriter(eval_dir)
    sess.run(tf.global_variables_initializer())
    training_time = 0.0

    # Main training loop
    for ii in range(max_num_training_steps + 1):
        x_batch, y_batch = mnist_train.get_next_batch(batch_size,
                                                      multiple_passes=True)

        # Compute Adversarial Perturbations
        start = timer()
        if adv_training:
            x_batch_adv = attack.perturb(x_batch, y_batch, sess, ii/max_num_training_steps)
        else:
            x_batch_adv = x_batch
        end = timer()
        training_time += end - start

        nat_dict = {model.x_input: x_batch,
                    model.x_input_natural: x_batch,
                    model.y_input: y_batch}

        adv_dict = {model.x_input: x_batch_adv,
                    model.x_input_natural: x_batch,
                    model.y_input: y_batch}

        # Output to stdout
        if ii % num_output_steps == 0:
            nat_acc = sess.run(model.accuracy, feed_dict=nat_dict)
            adv_acc = sess.run(model.accuracy, feed_dict=adv_dict)
            print('    Step {}:    ({})'.format(ii, datetime.now()))
            print('    training nat accuracy {:.4}%'.format(nat_acc * 100))
            print('    training adv accuracy {:.4}%'.format(adv_acc * 100))
            if ii != 0:
                print('    {} examples per second'.format(
                    num_output_steps * batch_size / training_time))
                training_time = 0.0
       
        # Tensorboard summaries
        if ii % num_summary_steps == 0:
            summary = sess.run(adv_summaries, feed_dict=adv_dict)
            summary_writer.add_summary(summary, global_step.eval(sess))
            summary = sess.run(nat_summaries, feed_dict=nat_dict)
            summary_writer.add_summary(summary, global_step.eval(sess))

        # Write a checkpoint
        if ii % num_checkpoint_steps == 0:
            saver.save( sess,
                        os.path.join(model_dir, 'checkpoint'),
                        global_step=global_step)

        # Evaluate
        if eval_during_training and ii % num_eval_steps == 0 and ii > 0:
            num_batches = int(math.ceil(num_eval_examples / eval_batch_size))
            total_xent_nat = 0.
            total_xent_adv = 0.
            total_corr_nat = 0
            total_corr_adv = 0
            tot_unstable1 = 0
            tot_unstable2 = 0
            tot_unstable3 = 0
            tot_unstable1l = 0
            tot_unstable2l = 0
            tot_unstable3l = 0

            for ibatch in trange(num_batches):
                bstart = ibatch * eval_batch_size
                bend = min(bstart + eval_batch_size, num_eval_examples)

                x_batch_eval = mnist.test.images[bstart:bend, :]
                y_batch_eval = mnist.test.labels[bstart:bend]

                dict_nat_eval = {model.x_input: x_batch_eval,
                               model.x_input_natural: x_batch_eval,
                               model.y_input: y_batch_eval}

                x_batch_eval_adv = eval_attack.perturb(x_batch_eval, y_batch_eval, sess)

                dict_adv_eval = {model.x_input: x_batch_eval_adv,
                               model.x_input_natural: x_batch_eval,
                               model.y_input: y_batch_eval}

                cur_corr_nat, cur_xent_nat = sess.run(
                                              [model.num_correct,model.xent],
                                              feed_dict = dict_nat_eval)
                cur_corr_adv, cur_xent_adv = sess.run(
                                              [model.num_correct,model.xent],
                                              feed_dict = dict_adv_eval)
                un1, un2, un3 = \
                sess.run([model.unstable1, model.unstable2, \
                          model.unstable3],
                          feed_dict = dict_nat_eval)
                un1l, un2l, un3l = \
                sess.run([model.un1loss, model.un2loss, \
                          model.un3loss],
                          feed_dict = dict_nat_eval)
                tot_unstable1 += np.sum(un1)
                tot_unstable2 += np.sum(un2)
                tot_unstable3 += np.sum(un3)
                tot_unstable1l += w_rsloss * un1l
                tot_unstable2l += w_rsloss * un2l
                tot_unstable3l += w_rsloss * un3l

                total_xent_nat += cur_xent_nat
                total_xent_adv += cur_xent_adv
                total_corr_nat += cur_corr_nat
                total_corr_adv += cur_corr_adv

            avg_un1 = tot_unstable1 / num_eval_examples
            avg_un2 = tot_unstable2 / num_eval_examples
            avg_un3 = tot_unstable3 / num_eval_examples
            avg_un1l = tot_unstable1l / num_eval_examples
            avg_un2l = tot_unstable2l / num_eval_examples
            avg_un3l = tot_unstable3l / num_eval_examples

            avg_xent_nat = total_xent_nat / num_eval_examples
            avg_xent_adv = total_xent_adv / num_eval_examples
            acc_nat = total_corr_nat / num_eval_examples
            acc_adv = total_corr_adv / num_eval_examples

            summary = tf.Summary(value=[
                  tf.Summary.Value(tag='xent_adv_eval', simple_value= avg_xent_adv),
                  tf.Summary.Value(tag='xent_adv', simple_value= avg_xent_adv),
                  tf.Summary.Value(tag='xent_nat_eval', simple_value= avg_xent_nat),
                  tf.Summary.Value(tag='xent_nat', simple_value= avg_xent_nat),
                  tf.Summary.Value(tag='accuracy_adv_eval', simple_value= acc_adv),
                  tf.Summary.Value(tag='accuracy_adv', simple_value= acc_adv),
                  tf.Summary.Value(tag='accuracy_nat_eval', simple_value= acc_nat),
                  tf.Summary.Value(tag='accuracy_nat', simple_value= acc_nat),
                  tf.Summary.Value(tag='avg_un1l', simple_value= avg_un1l),
                  tf.Summary.Value(tag='avg_un2l', simple_value= avg_un2l),
                  tf.Summary.Value(tag='avg_un3l', simple_value= avg_un3l),
                  tf.Summary.Value(tag='avg_un1', simple_value= avg_un1),
                  tf.Summary.Value(tag='avg_un2', simple_value= avg_un2),
                  tf.Summary.Value(tag='avg_un3', simple_value= avg_un3)])
            summary_writer_eval.add_summary(summary, global_step.eval(sess))

            print('Eval at {}:'.format(ii))
            print('  natural: {:.2f}%'.format(100 * acc_nat))
            print('  adversarial: {:.2f}%'.format(100 * acc_adv))
            print('  avg nat loss: {:.4f}'.format(avg_xent_nat))
            print('  avg adv loss: {:.4f}'.format(avg_xent_adv))
            print('  unstablerelus1, unstablerelus2, unstablerelus3: {}, {}, {}'.format(avg_un1,
                avg_un2, avg_un3))
            print('  ur1loss, ur2loss, ur3loss: {}, {}, {}'.format(avg_un1l,
                avg_un2l, avg_un3l))
            results = {'natural': 100 * acc_nat,
                       'adversarial': 100 * acc_adv,
                       'unstablerelus1': avg_un1,
                       'unstablerelus2': avg_un2,
                       'unstablerelus3': avg_un3,
                       }
            with open('job_result.json', 'w') as result_file:
                json.dump(results, result_file, sort_keys=True, indent=4)

        # Actual training step
        start = timer()
        sess.run(train_step, feed_dict=adv_dict)
        end = timer()
        training_time += end - start
