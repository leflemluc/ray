from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.signal
from ray.rllib.optimizers import SampleBatch


def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


def dcos(a,b):
    norme_a = np.multiply(a,a).sum(axis=1)
    norme_b  = np.multiply(b,b).sum(axis=1)
    return np.multiply(a,b).sum(axis=1) / (1e-11 + np.sqrt(norme_a * norme_b) )

def compute_internal_rewards(c, liste_s, liste_g):
    shifted_s = []
    shifted_g = []
    for i in range(1, c+1):
        padding_s = np.zeros(shape=(i, liste_s.shape[1]))
        padding_g = np.zeros(shape=(i, liste_g.shape[1]))
        shifted_s.append(np.append(padding_s, liste_s[:-i], axis=0))
        shifted_g.append(np.append(padding_g, liste_g[:-i], axis=0))

    internal_rewards = 0
    for i in range(c):
        internal_rewards += dcos(liste_s- shifted_s[i], shifted_g[i])

    return internal_rewards / c


def compute_advantages(rollout, last_r, gamma, lambda_=1.0, use_gae=True):
    """Given a rollout, compute its value targets and the advantage.

    Args:
        rollout (PartialRollout): Partial Rollout Object
        last_r (float): Value estimation for last observation
        gamma (float): Parameter for GAE
        lambda_ (float): Parameter for GAE
        use_gae (bool): Using Generalized Advantage Estamation

    Returns:
        SampleBatch (SampleBatch): Object with experience from rollout and
            processed rewards."""

    traj = {}
    trajsize = len(rollout["actions"])
    for key in rollout:
        traj[key] = np.stack(rollout[key])

    if use_gae:
        assert "vf_preds" in rollout, "Values not found!"
        vpred_t = np.concatenate([rollout["vf_preds"], np.array([last_r])])
        delta_t = traj["rewards"] + gamma * vpred_t[1:] - vpred_t[:-1]
        # This formula for the advantage comes
        # "Generalized Advantage Estimation": https://arxiv.org/abs/1506.02438
        traj["advantages"] = discount(delta_t, gamma * lambda_)
        traj["value_targets"] = traj["advantages"] + traj["vf_preds"]
    else:
        rewards_plus_v = np.concatenate(
            [rollout["rewards"], np.array([last_r])])
        traj["advantages"] = discount(rewards_plus_v, gamma)[:-1]

    traj["advantages"] = traj["advantages"].copy()

    assert all(val.shape[0] == trajsize for val in traj.values()), \
        "Rollout stacked incorrectly!"
    return SampleBatch(traj)



def compute_advantages_feudal(rollout, last_r, gamma, gamma_internal, lambda_, tradeoff_coeff=0.5, c=10):
    """Given a rollout, compute its value targets and the advantage.

    Args:
        rollout (PartialRollout): Partial Rollout Object
        last_r (float): Value estimation for last observation
        gamma (float): Parameter for GAE
        lambda_ (float): Parameter for GAE
        use_gae (bool): Using Generalized Advantage Estamation

    Returns:
        SampleBatch (SampleBatch): Object with experience from rollout and
            processed rewards."""

    traj = {}
    trajsize = len(rollout["actions"])
    for key in rollout:
        traj[key] = np.stack(rollout[key])

    assert "manager_vf_preds" in rollout, "Values not found for MANAGER!"
    manager_vf_preds = np.concatenate([rollout["manager_vf_preds"], np.array([last_r])])
    manager_delta_t = traj["rewards"] + gamma * manager_vf_preds[1:] - manager_vf_preds[:-1]
    # This formula for the advantage comes
    # "Generalized Advantage Estimation": https://arxiv.org/abs/1506.02438
    traj["advantages_manager"] = discount(manager_delta_t, gamma * lambda_)
    traj["manager_value_targets"] = traj["advantages_manager"] + traj["manager_vf_preds"]

    traj["advantages_manager"] = traj["advantages_manager"].copy()

    assert "worker_vf_preds" in rollout, "Values not found for WORKER!"
    internal_rewards = compute_internal_rewards(c, traj["s"], traj["g"])
    worker_vf_preds = np.concatenate([rollout["worker_vf_preds"], np.array([last_r])])
    worker_delta_t = internal_rewards + gamma_internal * worker_vf_preds[1:] - worker_vf_preds[:-1]
    # This formula for the advantage comes
    # "Generalized Advantage Estimation": https://arxiv.org/abs/1506.02438
    advantage_internal = discount(worker_delta_t, gamma_internal * lambda_)
    traj["advantages_worker"] = traj["advantages_manager"] + tradeoff_coeff * advantage_internal
    traj["worker_value_targets"] = advantage_internal + traj["worker_vf_preds"]

    traj["advantages_worker"] = traj["advantages_worker"].copy()



    diff_1 = np.append(traj["s"][c:], np.array([traj["s"][-1] for _ in range(c)]), axis=0)
    diff = diff_1 - traj["s"]
    traj["s_diff"] = diff.copy()

    del traj["s"]

    gsum = []
    g_dim = traj["g"].shape[1]
    for i in range(c + 1):
        constant = np.array([traj["g"][i] for _ in range(c - i)])
        zeros = np.zeros((i, g_dim))
        if i == 0:
            tensor = np.append(constant, traj["g"][i:i - c], axis=0)
        elif i == c:
            tensor = np.append(zeros, traj["g"][i:], axis=0)
        else:
            padding = np.append(zeros, constant, axis=0)
            tensor = np.append(padding, traj["g"][i:i - c], axis=0)

        gsum.append(tensor)

    traj["g_sum"] = np.array(gsum).sum(axis=0)



    assert all(val.shape[0] == trajsize for val in traj.values()), \
        "Rollout stacked incorrectly!"
    return SampleBatch(traj)

