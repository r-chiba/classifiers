from __future__ import print_function
from __future__ import division

import os
import sys
import time
import math
import random
import glob
import numpy as np
import tensorflow as tf
import cv2

sys.path.append('/home/chiba/research/tensorflow/dl_utils/')
from utils import *
from data import *

def residual_block(input_, output_dim, khs=[5, 5], kws=[5, 5], sths=[1, 1], stws=[1, 1],
                    sd=0.02, padding='SAME', bias=True, bn=False, projection=False, is_training=True,
                    with_w=False, name='resblock'):

    input_dim = input_.get_shape().as_list()[-1]
    with tf.variable_scope(name):
        if bn:
            input_ = tf.contrib.layers.batch_norm(input_, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True, is_training=is_training)
        #input_ = lrelu(input_)
        conv1, w1, b1 = conv2d(input_, output_dim, khs[0], kws[0], sths[0], stws[0],
                                sd, padding, bias, with_w=True, name='conv2d_1')

        if bn:
            conv1 = tf.contrib.layers.batch_norm(conv1, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True, is_training=is_training)
        conv1 = tf.nn.relu(conv1)
        conv2, w2, b2 = conv2d(conv1, output_dim, khs[1], kws[1], sths[1], stws[1],
                                sd, padding, bias, with_w=True, name='conv2d_2')

        if input_dim < output_dim:
            if projection:
                base = conv2d(input_, output_dim, kh=1, kw=1, sth=2, stw=2, name='projection')
            else:
                base = tf.pad(input_, [[0, 0], [0, 0], [0, 0], [0, output_dim - input_dim]])
        else:
            base = input_

        if with_w:
            return conv2 + base, [w1, w2], [b1, b2]
        else:
            return conv2 + base

class Classifier:
    def __init__(self, sess, batch_size, input_shape, cnn_dim, save_dir, augment=False):
        self.sess = sess
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.cnn_dim = cnn_dim
        self.learning_rate = 0.01
        self.save_dir = save_dir
        self.checkpoint_dir = os.path.join(self.save_dir, 'checkpoints')
        self.image_dir = os.path.join(self.save_dir, 'images')
        self.augment = augment

    def model(self, x, training=True, reuse=False):
        n_stack = 10

        with tf.variable_scope('classifier') as scope:
            if reuse == True:
                scope.reuse_variables()

            self.weight_decays = []

            self.h0, h0_w, _ = conv2d(x, self.cnn_dim, kh=3, kw=3, sth=1, stw=1, name='h0', with_w=True)
            #self.h0 = tf.nn.relu(self.h0)
            self.weight_decays += [h0_w]

            self.h1 = self.h0
            for i in xrange(n_stack):
                self.h1, h1_ws, _ = residual_block(self.h1, self.cnn_dim, khs=[3, 3], kws=[3, 3], sths=[1, 1], stws=[1, 1],
                                            bn=True, projection=False, is_training=training, with_w=True, name='h1_%d'%i)
                self.weight_decays += h1_ws
            self.h1 = tf.nn.avg_pool(self.h1, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

            self.h2 = self.h1
            for i in xrange(n_stack):
                self.h2, h2_ws, _ = residual_block(self.h2, self.cnn_dim*2, khs=[3, 3], kws=[3, 3], sths=[1, 1], stws=[1, 1],
                                            bn=True, projection=False, is_training=training, with_w=True, name='h2_%d'%i)
                self.weight_decays += h2_ws
            self.h2 = tf.nn.avg_pool(self.h2, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

            self.h3 = self.h2
            for i in xrange(n_stack):
                self.h3, h3_ws, _ = residual_block(self.h3, self.cnn_dim*4, khs=[3, 3], kws=[3, 3], sths=[1, 1], stws=[1, 1],
                                            bn=True, projection=False, is_training=training, with_w=True, name='h3_%d'%i)
                self.weight_decays += h3_ws
            self.h3 = tf.nn.avg_pool(self.h3, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

            # global average pooling
            self.h3 = tf.reduce_mean(self.h3, reduction_indices=[1, 2])

            self.h4 = linear(self.h3, 10, 'h4')

            self.output_shapes = []
            self.output_shapes.append(self.h4.get_shape().as_list()[1:])
            self.output_shapes.append(self.h3.get_shape().as_list()[1:])
            self.output_shapes.append(self.h2.get_shape().as_list()[1:])
            self.output_shapes.append(self.h1.get_shape().as_list()[1:])
            self.output_shapes.append(self.h0.get_shape().as_list()[1:])
            self.n_layers = len(self.output_shapes)

            return self.h4, self.h3, self.h2, self.h1, self.h0

    def build_model(self):
        self.x = tf.placeholder(tf.float32, 
            [None, self.input_shape[0], self.input_shape[1], self.input_shape[2]], name='x')
        self.label = tf.placeholder(tf.int32, [None, 10], name='label')
        self.codes = self.model(self.x)
        self.logits = self.codes[0]
        self.codes_test = self.model(self.x, training=False, reuse=True)
        self.logits_test = self.codes_test[0]

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.logits, labels=self.label))

        for w in self.weight_decays:
            self.loss += 0.0001 * tf.nn.l2_loss(w)

        #self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)
        #self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        self.optimizer = tf.train.MomentumOptimizer(self.learning_rate, 0.9)
        self.update_op = self.optimizer.minimize(self.loss)

        self.prediction = tf.argmax(self.logits_test, 1)
        self.answer = tf.argmax(self.label, 1)

        self.correct_prediction = tf.equal(self.prediction, self.answer)
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        var_save = tf.get_collection(tf.GraphKeys.VARIABLES, scope='classifier')
        self.saver = tf.train.Saver(var_save)

    #def train(self, dataset, n_steps):
    def train(self, dataset, n_epochs):
        self.sess.run(tf.initialize_all_variables())

        step = 1
        epoch = 1
        n_steps_per_epoch = len(dataset.train_images) // self.batch_size
        start = time.time()
        #while step <= n_steps:
        while epoch <= n_epochs:
            batch_images, batch_labels, _ = dataset.train_batch(self.batch_size)
            if self.augment:
                batch_images = random_crop(batch_images, self.input_shape)
                batch_images = random_horizontal_flip(batch_images)
            _, loss = self.sess.run(
                [self.update_op, self.loss],
                feed_dict={self.x: batch_images, self.label: batch_labels})

            if step % 100 == 0:
                elapsed = (time.time() - start) / 100
                print("epoch %3d(%6d): loss=%.4e; time/step=%.2f sec" %
                        (epoch, step, loss, elapsed))
                start = time.time()

            #if step % 1000 == 0:
            if step % n_steps_per_epoch == 0:
                test_images, test_labels = dataset.test_images, dataset.test_labels
                if self.augment:
                    test_images = crop_images(test_images, self.input_shape)
                acc = self.sess.run(self.accuracy,
                    feed_dict={self.x: test_images, self.label: test_labels})
                print("epoch %3d: accuracy=%.4e" % (epoch, acc))

                if epoch == 150 or epoch == 250:
                    self.optimizer._learning_rate *= 0.1
                epoch += 1

            if epoch % 50 == 0 and step % n_steps_per_epoch == 0:
                ckpt_name = 'classifier_epoch_%d.ckpt'%epoch
                self.saver.save(self.sess, os.path.join(self.checkpoint_dir, ckpt_name))
                print('save trained model to ' + ckpt_name)

            step += 1
            sys.stdout.flush()
            sys.stderr.flush()

