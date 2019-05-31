

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: cifar10-resnet.py
# Author: Yuxin Wu

import argparse
import os
import tensorflow as tf
from getcifa100 import getData

from tensorpack import *
from tensorpack.dataflow import dataset
from tensorpack.tfutils.summary import add_moving_summary, add_param_summary
from tensorpack.utils.gpu import get_num_gpu

"""
CIFAR10 ResNet example. See:
Deep Residual Learning for Image Recognition, arxiv:1512.03385
This implementation uses the variants proposed in:
Identity Mappings in Deep Residual Networks, arxiv:1603.05027

I can reproduce the results on 2 TitanX for
n=5, about 7.1% val error after 67k steps (20.4 step/s)
n=18, about 5.95% val error after 80k steps (5.6 step/s, not converged)
n=30: a 182-layer network, about 5.6% val error after 51k steps (3.4 step/s)
This model uses the whole training set instead of a train-val split.

To train:
    ./cifar10-resnet.py --gpu 0,1
"""

BATCH_SIZE = 128
NUM_UNITS = None


def convnormrelu(x, name, chan):
    x = Conv2D(name, x, chan, 3)
    x = BatchNorm(name + '_bn', x)
    x = tf.nn.relu(x, name=name + '_relu')
    return x

class VGG16(ModelDesc):

    def __init__(self, n):
        super(VGG16, self).__init__()

    def inputs(self):
        return [tf.TensorSpec([None, 32, 32, 3], tf.float32, 'input'),
                tf.TensorSpec([None], tf.int32, 'label')]

    def build_graph(self, image, label):
        image = image / 128.0
        assert tf.test.is_gpu_available()
        image = tf.transpose(image, [0, 3, 1, 2])
        logits = VGG16.get_logits(image)
        tf.nn.softmax(logits, name='output')
        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        cost = tf.reduce_mean(cost, name='cross_entropy_loss')

        wrong = tf.cast(tf.logical_not(tf.nn.in_top_k(logits, label, 1)), tf.float32, name='wrong_vector')
        # monitor training error
        add_moving_summary(tf.reduce_mean(wrong, name='train_error'))
        # weight decay on all W of fc layers
        wd_w = tf.train.exponential_decay(0.0002, get_global_step_var(),
                                          480000, 0.2, True)
        wd_cost = tf.multiply(wd_w, regularize_cost('.*/W', tf.nn.l2_loss), name='wd_cost')
        add_moving_summary(cost, wd_cost)
        add_param_summary(('.*/W', ['histogram']))   # monitor W
        return tf.add_n([cost, wd_cost], name='cost')

    def get_logits(self, image):
        """
        Args:
            image: 4D tensor of ``self.input_shape`` in ``self.data_format``
        Returns:
            Nx#class logits
        """
        with argscope(Conv2D, kernel_initializer=tf.variance_scaling_initializer(scale=2.)), \
                argscope([Conv2D, MaxPooling, BatchNorm], data_format='channels_last'), \
             argscope(MaxPooling, padding='SAME'):
            logits = (LinearWrap(image)
                      .apply(convnormrelu, 'conv1_1', 64)
                      .apply(convnormrelu, 'conv1_2', 64)
                      .MaxPooling('pool1', 2)
                      # 16
                      .apply(convnormrelu, 'conv2_1', 128)
                      .apply(convnormrelu, 'conv2_2', 128)
                      .MaxPooling('pool2', 2)
                      # 8
                      .apply(convnormrelu, 'conv3_1', 256)
                      .apply(convnormrelu, 'conv3_2', 256)
                      .apply(convnormrelu, 'conv3_3', 256)
                      .MaxPooling('pool3', 2)
                      # 4
                      .apply(convnormrelu, 'conv4_1', 512)
                      .apply(convnormrelu, 'conv4_2', 512)
                      .apply(convnormrelu, 'conv4_3', 512)
                      .MaxPooling('pool4', 2)
                      #2
                      .FullyConnected('fc0',2048,activation=tf.nn.relu)
                      .BatchNorm('fc0b0')
                      .FullyConnected('fc1', 2048, activation=tf.nn.relu)
                      .BatchNorm('fc1b1')
                      .FullyConnected('fc2', 100, activation=tf.identity)())

        add_param_summary(('.*', ['histogram', 'rms']))
        return logits

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=0.01, trainable=False)
        opt = tf.train.MomentumOptimizer(lr, 0.9)
        return opt

def get_data(train_or_test):
    isTrain = train_or_test == 'train'
    ds = dataset.Cifar100(train_or_test)
    pp_mean = ds.get_per_pixel_mean(('train',))
    if isTrain:
        augmentors = [
            imgaug.CenterPaste((40, 40)),
            imgaug.RandomCrop((32, 32)),
            imgaug.Flip(horiz=True),
            imgaug.MapImage(lambda x: x - pp_mean),
        ]
    else:
        augmentors = [
            imgaug.MapImage(lambda x: x - pp_mean)
        ]
    ds = AugmentImageComponent(ds, augmentors)
    ds = BatchData(ds, BATCH_SIZE, remainder=not isTrain)
    if isTrain:
        ds = PrefetchData(ds, 3, 2)
    return ds

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('-n', '--num_units',
                        help='number of units in each stage',
                        type=int, default=18)
    parser.add_argument('--load', help='load model for training')
    args = parser.parse_args()
    NUM_UNITS = args.num_units

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    logger.auto_set_dir()

    dataset_train = get_data('train')
    dataset_test = get_data('test')

    config = TrainConfig(
        model=Model(n=NUM_UNITS),
        dataflow=dataset_train,
        callbacks=[
            ModelSaver(),
            InferenceRunner(dataset_test,
                            [ScalarStats('cost'), ClassificationError('wrong_vector')]),
            ScheduledHyperParamSetter('learning_rate',
                                      [(1, 0.1), (82, 0.01), (123, 0.001), (300, 0.0002)])
        ],
        max_epoch=400,
        session_init=SaverRestore(args.load) if args.load else None
    )
    num_gpu = max(get_num_gpu(), 1)
    launch_train_with_config(config, SyncMultiGPUTrainerParameterServer(num_gpu))











