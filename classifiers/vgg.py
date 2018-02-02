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

    def model(self, x, keep_prob, training=True, reuse=False):
        ht0, wh0 = pad_out_size_same(self.input_shape[0], 2), pad_out_size_same(self.input_shape[0], 2)
        ht1, wh1 = pad_out_size_same(ht0, 2), pad_out_size_same(wh0, 2)
        ht2, wh2 = pad_out_size_same(ht1, 2), pad_out_size_same(wh1, 2)
        ht3, wh3 = pad_out_size_same(ht2, 2), pad_out_size_same(wh2, 2)
        ht4, wh4 = pad_out_size_same(ht3, 2), pad_out_size_same(wh3, 2)

        with tf.variable_scope('classifier') as scope:
            if reuse == True:
                scope.reuse_variables()

            self.weight_decays = []

            self.h0 = x
            for i in xrange(2):
                self.h0, h0_w, _ = conv2d(self.h0, self.cnn_dim, kh=3, kw=3, with_w=True, name='h0_%d'%i)
                self.h0 = tf.contrib.layers.batch_norm(self.h0, decay=0.9,
                    updates_collections=None, epsilon=1e-5, scale=True, is_training=training)
                self.h0 = tf.nn.relu(self.h0)
                self.weight_decays.append(h0_w)
            self.h0 = tf.nn.max_pool(self.h0, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

            self.h1 = self.h0
            for i in xrange(2):
                self.h1, h1_w, _ = conv2d(self.h1, self.cnn_dim*8, kh=3, kw=3, with_w=True, name='h1_%d'%i)
                self.h1 = tf.contrib.layers.batch_norm(self.h1, decay=0.9,
                    updates_collections=None, epsilon=1e-5, scale=True, is_training=training)
                self.h1 = tf.nn.relu(self.h1)
                self.weight_decays.append(h1_w)
            self.h1 = tf.nn.max_pool(self.h1, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

            self.h2 = self.h1
            for i in xrange(3):
                self.h2, h2_w, _ = conv2d(self.h2, self.cnn_dim*16, kh=3, kw=3, with_w=True, name='h2_%d'%i)
                self.h2 = tf.contrib.layers.batch_norm(self.h2, decay=0.9,
                    updates_collections=None, epsilon=1e-5, scale=True, is_training=training)
                self.h2 = tf.nn.relu(self.h2)
                self.weight_decays.append(h2_w)
            self.h2 = tf.nn.max_pool(self.h2, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

            self.h3 = self.h2
            for i in xrange(3):
                self.h3, h3_w, _ = conv2d(self.h3, self.cnn_dim*32, kh=3, kw=3, with_w=True, name='h3_%d'%i)
                self.h3 = tf.contrib.layers.batch_norm(self.h3, decay=0.9,
                    updates_collections=None, epsilon=1e-5, scale=True, is_training=training)
                self.h3 = tf.nn.relu(self.h3)
                self.weight_decays.append(h3_w)
            self.h3 = tf.nn.max_pool(self.h3, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

            self.h4 = self.h3
            for i in xrange(3):
                self.h4, h4_w, _ = conv2d(self.h4, self.cnn_dim*32, kh=3, kw=3, with_w=True, name='h4_%d'%i)
                self.h4 = tf.contrib.layers.batch_norm(self.h4, decay=0.9,
                    updates_collections=None, epsilon=1e-5, scale=True, is_training=training)
                self.h4 = tf.nn.relu(self.h4)
                self.weight_decays.append(h4_w)
            self.h4 = tf.nn.max_pool(self.h4, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

            self.h4_flatten = tf.reshape(self.h4, [-1, ht4*wh4*self.cnn_dim*32])

            self.h5 = linear(self.h4_flatten, self.cnn_dim*32, 'h5')
            self.h5 = tf.nn.dropout(self.h5, keep_prob)
            self.h5 = tf.nn.relu(self.h5)

            self.h6 = linear(self.h5, self.cnn_dim*32, 'h6')
            self.h6 = tf.nn.dropout(self.h6, keep_prob)
            self.h6 = tf.nn.relu(self.h6)

            self.h7 = linear(self.h6, 10, 'h7')

            self.output_shapes = []
            self.output_shapes.append(self.h7.get_shape().as_list()[1:])
            self.output_shapes.append(self.h6.get_shape().as_list()[1:])
            self.output_shapes.append(self.h5.get_shape().as_list()[1:])
            self.output_shapes.append(self.h4.get_shape().as_list()[1:])
            self.output_shapes.append(self.h3.get_shape().as_list()[1:])
            self.output_shapes.append(self.h2.get_shape().as_list()[1:])
            self.output_shapes.append(self.h1.get_shape().as_list()[1:])
            self.output_shapes.append(self.h0.get_shape().as_list()[1:])
            self.n_layers = len(self.output_shapes)

            return self.h7, self.h6, self.h5, self.h4, self.h3, self.h2, self.h1, self.h0

    def build_model(self):
        self.x = tf.placeholder(tf.float32, 
            [None, self.input_shape[0], self.input_shape[1], self.input_shape[2]], name='x')
        self.label = tf.placeholder(tf.int32, [None, 10], name='label')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.codes = self.model(self.x, self.keep_prob)
        self.logits = self.codes[0]
        self.codes_test = self.model(self.x, self.keep_prob, training=False, reuse=True)
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
                feed_dict={self.x: batch_images, self.label: batch_labels, self.keep_prob: 0.5})

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
                acc = self.sess.run(
                        self.accuracy,
                        feed_dict={self.x: test_images, self.label: test_labels, self.keep_prob: 1.0})
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

