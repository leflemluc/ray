from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from ray.rllib.models import ModelCatalog


class ProximalPolicyGraph(object):


    is_recurrent = False

    def __init__(
            self, observation_space, action_space, global_step,
            observations, value_targets_manager, value_targets_worker,
            advantages_manager, advantages_worker, actions,
            prev_logits, prev_manager_vf_preds, prev_worker_vf_preds,
            g_sum, carried_z, s_diff, logit_dim,
            kl_coeff, distribution_class, config, sess, registry):

        self.prev_dist = distribution_class(prev_logits)

        self.g_sum = g_sum
        self.carried_z = carried_z
        self.s_diff = s_diff

        # Saved so that we can compute actions given different observations
        self.observations = observations

        model_manager_filter = config["model_manager_filter"].copy()
        model_manager = ModelCatalog.get_model(
            registry, observations, config["g_dim"], model_manager_filter)
        self.z = model_manager.last_layer
        self.s = model_manager.outputs

        with tf.variable_scope("value_function_manager"):
            model_manager_VF = config["model_manager_VF"].copy()
            self.value_function_manager = ModelCatalog.get_model(
                registry, self.s, 1, model_manager_VF).outputs
        self.value_function_manager = tf.reshape(self.value_function_manager, [-1])

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

        with tf.variable_scope("value_function_worker"):
            model_worker_VF = config["model_worker_VF"].copy()
            self.value_function_worker = ModelCatalog.get_model(
                registry, flat_logits, 1, model_worker_VF).outputs
        self.value_function_worker = tf.reshape(self.value_function_worker, [-1])


        with tf.variable_scope("policy_worker"):

            self.curr_logits = tf.reshape(tf.matmul(U,w),[-1,logit_dim])
            self.curr_dist = distribution_class(self.curr_logits)
            self.sampler = self.curr_dist.sample()



        # Make loss functions.

        # First the Manager Loss
        # (i) The policy loss
        dot = tf.reduce_sum(tf.multiply(self.s_diff, self.g), axis=1)
        mag = tf.norm(self.s_diff, axis=1) + .0001
        self.manager_loss = -tf.reduce_mean(advantages_manager * dot / mag)

        # (ii) The VF loss
        vf_loss1_manager = tf.square(self.value_function_manager - value_targets_manager)
        vf_clipped_manager = prev_manager_vf_preds + tf.clip_by_value(
            self.value_function_manager - prev_manager_vf_preds,
            -config["clip_param"], config["clip_param"])
        vf_loss2_manager = tf.square(vf_clipped_manager - value_targets_manager)
        self.manager_vf_loss = tf.reduce_mean(tf.minimum(vf_loss1_manager, vf_loss2_manager))

        # Second the Worker Loss

        # (i) KL and Entropy
        kl = self.prev_dist.kl(self.curr_dist)
        self.mean_kl = tf.reduce_mean(kl)
        entropy = self.curr_dist.entropy()
        beta = tf.train.polynomial_decay(config["entropy_coeff"]+0.00001, global_step,
                                         end_learning_rate=0,
                                         decay_steps=1000,
                                         power=1)
        self.mean_entropy = tf.reduce_mean(entropy)

        # (ii) The policy loss
        self.ratio = tf.exp(self.curr_dist.logp(actions) -
                            self.prev_dist.logp(actions))
        surr1 = self.ratio * advantages_worker
        surr2 = tf.clip_by_value(self.ratio, 1 - config["clip_param"],
                                      1 + config["clip_param"]) * advantages_worker
        self.worker_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))

        # (iii) The VF loss
        vf_loss1_worker = tf.square(self.value_function_worker - value_targets_worker)
        vf_clipped_worker = prev_worker_vf_preds + tf.clip_by_value(
            self.value_function_worker - prev_worker_vf_preds,
            -config["clip_param"], config["clip_param"])
        vf_loss2_worker = tf.square(vf_clipped_worker - value_targets_worker)
        self.worker_vf_loss = tf.reduce_mean(tf.minimum(vf_loss1_worker, vf_loss2_worker))


        ### THE BIG FINAL LOSS ###

        self.loss = self.manager_loss + config["manager_vf_loss_coeff"] * self.manager_vf_loss + \
                    self.worker_loss + config["worker_vf_loss_coeff"] * self.worker_vf_loss - \
                    beta * self.mean_entropy + kl_coeff * self.mean_kl


        self.sess = sess

        self.policy_results_manager = [
                self.z, self.s, self.g, self.value_function_manager]

        self.policy_results_worker = [
            self.sampler, self.curr_logits, self.value_function_worker]

    def compute_warmup_manager(self, observation):
        z, s, g, vfm = self.sess.run(
            self.policy_results_manager,
            feed_dict={self.observations: [observation]})
        return z[0], s[0], g[0], vfm[0]

    def compute_single_action(self, g_sum, carried_z):
        action, logprobs, vfw = self.sess.run(
            self.policy_results_worker,
            feed_dict={self.g_sum: [g_sum], self.carried_z: [carried_z]})
        return action[0], logprobs[0], vfw[0]

    def get_initial_state(self):
        return []

    def loss(self):
        return self.loss
