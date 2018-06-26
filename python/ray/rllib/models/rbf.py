from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from ray.rllib.models.model import Model


class RBF(Model):
    """Generic fully connected network."""

    def _init(self, inputs, num_outputs, options):
        hiddens = options.get("fcnet_hiddens", 50)

        last_layer = tf.get_variable(name="weights", shape=hiddens, dtype=tf.float32,
                                     initializer=tf.initializers.truncated_normal, trainable=True)

        feat_mat = tf.contrib.layers.fully_connected(
            inputs,
            hiddens,
            activation_fn=tf.sin,
            weights_initializer=tf.initializers.random_normal,
            biases_initializer=tf.initializers.random_uniform(minval=-np.pi, maxval=np.pi),
            trainable=False,
        )

        output = tf.matmul(feat_mat, last_layer)

        return output, last_layer
