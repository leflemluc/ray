from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

from ray.rllib.models.model import Model
from ray.rllib.models.misc import normc_initializer

def flatten(x):
    return tf.reshape(x, [-1, np.prod(x.get_shape().as_list()[1:])])

"""
class VisionNetwork(Model):

    def _init(self, inputs, num_outputs, options):
        filters = options.get("conv_filters", [
            [16, [8, 8], 4],
            [32, [4, 4], 2],
            [512, [10, 10], 1],
        ])
        with tf.name_scope("vision_net"):
            for i, (out_size, kernel, stride) in enumerate(filters[:-1], 1):
                inputs = slim.conv2d(
                    inputs, out_size, kernel, stride,
                    scope="conv{}".format(i))
            out_size, kernel, stride = filters[-1]
            fc1 = slim.conv2d(
                inputs, out_size, kernel, stride, padding="VALID", scope="fc1")
            fc2 = slim.conv2d(fc1, num_outputs, [1, 1], activation_fn=None,
                              normalizer_fn=None, scope="fc2")
            return tf.squeeze(fc2, [1, 2]), tf.squeeze(fc1, [1, 2])
"""

class VisionNetwork(Model):
    """Generic vision network, followed by a projection"""

    def _init(self, inputs, num_outputs, options):
        filters = options.get("conv_filters", [
            [16, [8, 8], 4],
            [32, [4, 4], 2],
            [512, [10, 10], 1],
        ])

        hidden_z = options.get("fcnet_hiddens", [256])
        fcnet_activation = options.get("fcnet_activation", "tanh")
        if fcnet_activation == "tanh":
            activation = tf.nn.tanh
        elif fcnet_activation == "relu":
            activation = tf.nn.relu

        with tf.name_scope("vision_net"):
            for i, (out_size, kernel, stride) in enumerate(filters[:-1], 1):
                inputs = slim.conv2d(
                    inputs, out_size, kernel, stride,
                    scope="conv{}".format(i), activation_fn=activation)
            out_size, kernel, stride = filters[-1]
            fc1 = slim.conv2d(
                inputs, out_size, kernel, stride, scope="fc1", activation_fn=activation)

            flattened_filters = flatten(fc1)

        with tf.name_scope("z_layer"):
            i = 1
            z_layer = flattened_filters
            for size in hidden_z:
                label = "z_layer{}".format(i)
                z_layer = slim.fully_connected(
                    z_layer, size,
                    weights_initializer=normc_initializer(1.0),
                    activation_fn=activation,
                    scope=label)

        with tf.name_scope("s_layer"):
            s_layer = slim.fully_connected(
                z_layer, num_outputs,
                    weights_initializer=normc_initializer(1.0),
                    activation_fn=tf.nn.relu,
                    scope="s")


        return s_layer, z_layer