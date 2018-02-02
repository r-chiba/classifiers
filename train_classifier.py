from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os
import pprint
import numpy as np
import tensorflow as tf

sys.path.append('/home/chiba/research/tensorflow/dl_utils/')
from data import *

flags = tf.app.flags
flags.DEFINE_integer('batch_size', 100, 'batch size')
#flags.DEFINE_integer('n_steps', 100000, 'train steps')
flags.DEFINE_integer('n_epochs', 100, 'epochs to train')
flags.DEFINE_string('dataset', 'fmnist', 'dataset name [fmnist, cifar10]')
flags.DEFINE_string('nn_type', 'resnet', 'neural networks type [resnet, vgg]')
flags.DEFINE_string('save_dir', 'save', 'save directory')
flags.DEFINE_string('gpu_list', '0', 'gpu numbers to use')
FLAGS = flags.FLAGS

def main(argv):
    if FLAGS.dataset == 'fmnist':
        FLAGS.input_shape = (28, 28, 1)
        FLAGS.cnn_dim = 4
        FLAGS.augment = False
        dataset = FashionMnistDataset(code_dim=0, code_init=None)
    elif FLAGS.dataset == 'cifar10':
        #FLAGS.input_shape = (32, 32, 3)
        FLAGS.input_shape = (24, 24, 3)
        FLAGS.cnn_dim = 16
        FLAGS.augment = True
        dataset = Cifar10Dataset('/home/chiba/data/cifar10/cifar-10-batches-py', code_dim=0, code_init=None)
    else:
        raise ValueError('Dataset %s is unsupported.'%FLAGS.dataset)

    if FLAGS.nn_type == 'resnet':
        from classifiers.resnet import Classifier
    elif FLAGS.nn_type == 'vgg':
        from classifiers.vgg import Classifier
    else:
        raise ValueError('Neural Network %s is unsupported.'%FLAGS.nn_type)

    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(FLAGS.__flags)

    if not os.path.exists(FLAGS.save_dir):
        os.makedirs(FLAGS.save_dir)
    FLAGS.image_dir = os.path.join(FLAGS.save_dir, 'images')
    if not os.path.exists(FLAGS.image_dir):
        os.makedirs(FLAGS.image_dir)
    FLAGS.checkpoint_dir = os.path.join(FLAGS.save_dir, 'checkpoints')
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = FLAGS.gpu_list

    with tf.Session(config=config) as sess:
        clf = Classifier(sess, FLAGS.batch_size, FLAGS.input_shape, FLAGS.cnn_dim, FLAGS.save_dir, FLAGS.augment)
        clf.build_model()
        #clf.train(dataset=dataset, n_steps=FLAGS.n_steps)
        clf.train(dataset=dataset, n_epochs=FLAGS.n_epochs)

if __name__ == '__main__':
    tf.app.run()

