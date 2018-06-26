# Code in this file is copied and adapted from
# https://github.com/openai/evolution-strategies-starter.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
import numpy as np
import tensorflow as tf

import ray
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.filter import get_filter
import tensorflow.contrib.slim as slim
from ray.rllib.models.misc import normc_initializer


def dcos(a,b):
    norme_a = np.multiply(a,a).sum(axis=1)
    norme_b  = np.multiply(b,b).sum(axis=1)
    return np.multiply(a,b).sum(axis=1) / (1e-11 + np.sqrt(norme_a * norme_b) )

def compute_internal_rewards(c, liste_s, liste_g):
    shifted_s = []
    shifted_g = []
    liste_s = np.squeeze(liste_s)
    liste_g= np.squeeze(liste_g)
    for i in range(1, c+1):
        padding_s = np.zeros(shape=(i, liste_s.shape[1]))
        padding_g = np.zeros(shape=(i, liste_g.shape[1]))

        shifted_s.append(np.append(padding_s, liste_s[:-i], axis=0))
        shifted_g.append(np.append(padding_g, liste_g[:-i], axis=0))

    internal_rewards = 0
    for i in range(c):
        internal_rewards += dcos(liste_s- shifted_s[i], shifted_g[i])

    return internal_rewards / c


def rollout(config, policy, env, timestep_limit=None, add_noise=False):
    """Do a rollout.

    If add_noise is True, the rollout will take noisy actions with
    noise drawn from that stream. Otherwise, no action noise will be added.
    """
    env_timestep_limit = env.spec.max_episode_steps
    timestep_limit = (env_timestep_limit if timestep_limit is None
                      else min(timestep_limit, env_timestep_limit))
    rews = []
    t = 0
    c = config["c"]
    observation = env.reset()
    for step in range(timestep_limit or 999999):
        g, s, z_carried = policy.compute_manager(observation)
        if step == 0:
            s_s = np.array([s])
            g_s = np.array([g])
            g_sum = g
        elif step < c:
            s_s = np.append(s_s, [s], axis=0)
            g_s = np.append(g_s, [g], axis=0)
            g_sum = g_s.sum(axis=0)
        else:
            s_s = np.append(s_s, [s], axis=0)
            g_s = np.append(g_s, [g], axis=0)
            g_sum = g_s[-(c + 1):].sum(axis=0)
        ac = policy.compute(g_sum, z_carried, add_noise)
        observation, rew, done, _ = env.step(ac)
        rews.append(rew)
        t += 1
        if done:
            break
    rews = np.array(rews, dtype=np.float32)
    internal_rews = compute_internal_rewards(c, s_s, g_s)
    return rews, internal_rews, t




class HRLPolicy(object):
    def __init__(self, config, registry, sess, action_space, preprocessor, action_noise_std):
        self.sess = sess
        self.action_space = action_space
        self.action_noise_std = action_noise_std
        self.preprocessor = preprocessor
        self.observation_filter = get_filter(
            config["observation_filter"], self.preprocessor.shape)
        self.inputs = tf.placeholder(
            tf.float32, [None] + list(self.preprocessor.shape))
        self.g_sum = tf.placeholder(tf.float32, (None, config["g_dim"]))
        self.carried_z = tf.placeholder(tf.float32, (None, config["z_dimension"]))


        # Policy network.
        dist_class, logit_dim = ModelCatalog.get_action_dist(
            self.action_space, dist_type="deterministic")

        model_manager_filter = config["model_manager_filter"].copy()
        model_manager = ModelCatalog.get_model(
            registry, self.inputs, config["g_dim"], model_manager_filter)
        self.z = model_manager.last_layer
        self.s = model_manager.outputs

        with tf.variable_scope("goal_designer"):
            model_manager_goal_config = config["model_manager_goal"].copy()
            g_hat = ModelCatalog.get_model(
                registry, self.s, config["g_dim"], model_manager_goal_config).outputs

            self.g = tf.nn.l2_normalize(g_hat, dim=1)

        with tf.variable_scope("goal_embedding"):
            phi = tf.get_variable("phi", (config["g_dim"], config["k_dim"]))
            w = tf.matmul(self.g_sum, phi)
            w = tf.expand_dims(w, [2])

            model_worker_goal_config = config["model_worker_goal"].copy()
            flat_logits = ModelCatalog.get_model(
                registry, self.carried_z, logit_dim * config["k_dim"], model_worker_goal_config).outputs

            U = tf.reshape(flat_logits, [-1, logit_dim, config["k_dim"]])


        with tf.variable_scope("policy_worker"):

            self.curr_logits = tf.reshape(tf.matmul(U,w),[-1,logit_dim])
            self.curr_dist = dist_class(self.curr_logits)
            self.sampler = self.curr_dist.sample()


        self.variables_manager = ray.experimental.TensorFlowVariables(
            self.g, self.sess)

        self.variables_worker = ray.experimental.TensorFlowVariables(
            self.curr_logits, self.sess)

        self.num_params_manager = sum([np.prod(variable.shape.as_list())
                               for _, variable
                               in self.variables_manager.variables.items()])

        self.num_params_worker = sum([np.prod(variable.shape.as_list())
                                       for _, variable
                                       in self.variables_worker.variables.items()])
        self.sess.run(tf.global_variables_initializer())

    def compute_manager(self, observation, update=True):
        observation = self.preprocessor.transform(observation)
        observation = self.observation_filter(observation[None], update=update)
        g, s, z = self.sess.run([self.g, self.s, self.z],
                               feed_dict={self.inputs: observation})
        return g[0], s[0], z[0]

    def compute(self, g_sum, z_carried, add_noise=False):
        action = self.sess.run(self.sampler,
                               feed_dict={self.g_sum: [g_sum], self.carried_z: [z_carried]})
        if add_noise and isinstance(self.action_space, gym.spaces.Box):
            action += np.random.randn(*action.shape) * self.action_noise_std
        return action


    def set_weights_manager(self, x):
        self.variables_manager.set_flat(x)

    def get_weights_manager(self):
        return self.variables_manager.get_flat()


    def set_weights_worker(self, x):
        self.variables_worker.set_flat(x)

    def get_weights_worker(self):
        return self.variables_worker.get_flat()