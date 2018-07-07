from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
import numpy as np
import six.moves.queue as queue
import threading

from ray.rllib.optimizers.sample_batch import SampleBatchBuilder


CompletedRollout = namedtuple("CompletedRollout",
                              ["episode_length", "episode_reward"])


class SyncSampler(object):
    """This class interacts with the environment and tells it what to do.

    Note that batch_size is only a unit of measure here. Batches can
    accumulate and the gradient can be calculated on up to 5 batches.

    This class provides data on invocation, rather than on a separate
    thread."""
    _async = False

    def __init__(
            self, env, policy, obs_filter, num_local_steps, horizon=None,
            pack=False, ADB=False):
        self.num_local_steps = num_local_steps
        self.horizon = horizon
        self.env = env
        self.policy = policy
        self._obs_filter = obs_filter
        self.rollout_provider = _env_runner(self.env, self.policy,
                                                    self.num_local_steps, self.horizon,
                                                    self._obs_filter, pack, ADB)

        self.metrics_queue = queue.Queue()

    def get_data(self):
        while True:
            item = next(self.rollout_provider)
            if isinstance(item, CompletedRollout):
                self.metrics_queue.put(item)
            else:
                return item

    def get_metrics(self):
        completed = []
        while True:
            try:
                completed.append(self.metrics_queue.get_nowait())
            except queue.Empty:
                break
        return completed


class AsyncSampler(threading.Thread):
    """This class interacts with the environment and tells it what to do.

    Note that batch_size is only a unit of measure here. Batches can
    accumulate and the gradient can be calculated on up to 5 batches."""
    _async = True

    def __init__(
            self, env, policy, obs_filter, num_local_steps, horizon=None,
            pack=False):
        assert getattr(
            obs_filter, "is_concurrent",
            False), ("Observation Filter must support concurrent updates.")
        threading.Thread.__init__(self)
        self.queue = queue.Queue(5)
        self.metrics_queue = queue.Queue()
        self.num_local_steps = num_local_steps
        self.horizon = horizon
        self.env = env
        self.policy = policy
        self._obs_filter = obs_filter
        self.started = False
        self.daemon = True
        self.pack = pack

    def run(self):
        self.started = True
        try:
            self._run()
        except BaseException as e:
            self.queue.put(e)
            raise e

    def _run(self):
        rollout_provider = _env_runner(self.env, self.policy,
                                       self.num_local_steps, self.horizon,
                                       self._obs_filter, self.pack)
        while True:
            # The timeout variable exists because apparently, if one worker
            # dies, the other workers won't die with it, unless the timeout is
            # set to some large number. This is an empirical observation.
            item = next(rollout_provider)
            if isinstance(item, CompletedRollout):
                self.metrics_queue.put(item)
            else:
                self.queue.put(item, timeout=600.0)

    def get_data(self):
        """Gets currently accumulated data.

        Returns:
            rollout (SampleBatch): trajectory data (unprocessed)
        """
        assert self.started, "Sampler never started running!"
        rollout = self.queue.get(timeout=600.0)
        if isinstance(rollout, BaseException):
            raise rollout
        while not rollout["dones"][-1]:
            try:
                part = self.queue.get_nowait()
                if isinstance(part, BaseException):
                    raise rollout
                rollout = rollout.concat(part)
            except queue.Empty:
                break
        return rollout

    def get_metrics(self):
        completed = []
        while True:
            try:
                completed.append(self.metrics_queue.get_nowait())
            except queue.Empty:
                break
        return completed


def _env_runner(env, policy, num_local_steps, horizon, obs_filter, pack, ADB):
    """This implements the logic of the thread runner.

    It continually runs the policy, and as long as the rollout exceeds a
    certain length, the thread runner appends the policy to the queue. Yields
    when `timestep_limit` is surpassed, environment terminates, or
    `num_local_steps` is reached.

    Args:
        env: Environment generated by env_creator
        policy: Policy used to interact with environment. Also sets fields
            to be included in `SampleBatch`
        num_local_steps: Number of steps before `SampleBatch` is yielded. Set
            to infinity to yield complete episodes.
        horizon: Horizon of the episode.
        obs_filter: Filter used to process observations.
        pack: Whether to pack multiple episodes into each batch. This
            guarantees batches will be exactly `num_local_steps` in size.

    Yields:
        rollout (SampleBatch): Object containing state, action, reward,
            terminal condition, and other fields as dictated by `policy`.
    """
    last_observation = obs_filter(env.reset())
    try:
        horizon = horizon if horizon else env.spec.max_episode_steps
    except Exception:
        print("Warning, no horizon specified, assuming infinite")
    if not horizon:
        horizon = 999999
    last_features = policy.get_initial_state()
    features = last_features
    length = 0
    rewards = 0
    rollout_number = 0

    while True:
        batch_builder = SampleBatchBuilder()
        observations_ = []
        actions_ = []
        for _ in range(num_local_steps):
            # Assume batch size one for now
            action, features, pi_info = policy.compute_single_action(
                last_observation, last_features, is_training=True)
            for i, state_value in enumerate(last_features):
                pi_info["state_in_{}".format(i)] = state_value
            for i, state_value in enumerate(features):
                pi_info["state_out_{}".format(i)] = state_value
            observation, reward, terminal, info = env.step(action)
            observation = obs_filter(observation)

            length += 1
            rewards += reward
            if length >= horizon:
                terminal = True

            # Concatenate multiagent actions
            if isinstance(action, list):
                action = np.concatenate(action, axis=0).flatten()

            # Collect the experience.
            batch_builder.add_values(
                obs=last_observation,
                actions=action,
                rewards=reward,
                dones=terminal,
                new_obs=observation,
                **pi_info)
            observations_.append(last_observation)
            actions_.append(action)

            last_observation = observation
            last_features = features

            if terminal:
                Q_functions = np.transpose(np.array(policy.compute_Q_fuctions(observations_, actions_)))
                batch_builder.set_Q(Q_functions)
                yield CompletedRollout(length, rewards)

                if (length >= horizon or
                        not env.metadata.get("semantics.autoreset")):
                    last_observation = obs_filter(env.reset())
                    last_features = policy.get_initial_state()
                    rollout_number += 1
                    length = 0
                    rewards = 0
                    if not pack:
                        break

        # Once we have enough experience, yield it, and have the ThreadRunner
        # place it on a queue.
        yield batch_builder.build()
