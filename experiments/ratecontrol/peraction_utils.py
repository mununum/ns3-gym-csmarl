import numpy as np
import scipy.signal
from functools import partial

from ray.rllib.agents.ppo.ppo import UpdateKL, warn_about_bad_reward_scales
from ray.rllib.execution.metric_ops import StandardMetricsReporting
from ray.rllib.execution.rollout_ops import ParallelRollouts, ConcatBatches, StandardizeFields, SelectExperiences
from ray.rllib.execution.train_ops import TrainOneStep, TrainTFMultiGPU
from ray.rllib.models.tf.tf_action_dist import MultiActionDistribution, Categorical
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils import try_import_tf, try_import_tree

from ns3_import import import_graph_module

tf = try_import_tf()
tree = try_import_tree()
graph = import_graph_module()


def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


def peraction_compute_advantages(rollout,
                                 last_r,
                                 gamma=0.9,
                                 lambda_=1.0,
                                 use_gae=True,
                                 use_critic=True):
    """
    Given a rollout, compute its value targets and the advantage.

    Args:
        rollout (SampleBatch): SampleBatch of a single trajectory
        last_r (float): Value estimation for last observation
        gamma (float): Discount factor.
        lambda_ (float): Parameter for GAE
        use_gae (bool): Using Generalized Advantage Estimation
        use_critic (bool): Whether to use critic (value estimates). Setting
                           this to False will use 0 as baseline.
    
    Returns:
        SampleBatch (SampleBatch): Object with experience from rollout and
            processed rewards.
    """

    traj = {}
    trajsize = len(rollout[SampleBatch.ACTIONS])
    for key in rollout:
        traj[key] = np.stack(rollout[key])
    
    assert "values_0" in rollout and use_critic
    assert use_gae

    vpred_0_t = np.concatenate(
        [rollout["values_0"],
         np.array([last_r])]
    )
    delta_0_t = (
        traj[SampleBatch.REWARDS] + gamma * vpred_0_t[1:] - vpred_0_t[:-1]
    )
    traj["advantages_0"] = discount(delta_0_t, gamma * lambda_)
    traj["value_targets_0"] = (
        traj["advantages_0"] +
        traj["values_0"]).copy().astype(np.float32)

    vpred_1_t = np.concatenate(
        [rollout["values_1"],
         np.array([last_r])]
    )
    delta_1_t = (
        traj[SampleBatch.REWARDS] + gamma * vpred_1_t[1:] - vpred_1_t[:-1]
    )
    traj["advantages_1"] = discount(delta_1_t, gamma * lambda_)
    traj["value_targets_1"] = (
        traj["advantages_1"] + traj["values_1"]
    ).copy().astype(np.float32)

    traj["advantages_0"] = traj["advantages_0"].copy().astype(np.float32)
    traj["advantages_1"] = traj["advantages_1"].copy().astype(np.float32)

    assert all(val.shape[0] == trajsize for val in traj.values()), \
        "Rollout stacked incorrectly!"
    return SampleBatch(traj)


class PPOPerActionLoss:
    def __init__(self,
                 dist_class,
                 model,
                 value_targets_0,
                 value_targets_1,
                 advantages_0,
                 advantages_1,
                 actions,
                 prev_logits,
                 vf_preds_0,
                 vf_preds_1,
                 curr_action_dist,
                 value_fn_0,
                 value_fn_1,
                 cur_kl_coeff,
                 valid_mask,
                 entropy_coeff=0,
                 clip_param=0.1,
                 vf_clip_param=0.1,
                 vf_loss_coeff=1.0,
                 use_gae=True):
        """Constructs the loss for Proximal Policy Objective.

        """
        if valid_mask is not None:

            def reduce_mean_valid(t):
                return tf.reduce_mean(tf.boolean_mask(t, valid_mask))

        else:

            def reduce_mean_valid(t):
                return tf.reduce_mean(t)

        prev_dist = dist_class(prev_logits, model)
        # Make loss functions.

        assert isinstance(curr_action_dist, MultiActionDistribution)
        def peraction_logp(multi_dist, x):
            # Single tensor input (all merged)
            if isinstance(x, (tf.Tensor, np.ndarray)):
                split_indices = []
                for dist in multi_dist.flat_child_distributions:
                    if isinstance(dist, Categorical):
                        split_indices.append(1)
                    else:
                        split_indices.append(tf.shape(dist.sample())[1])
                split_x = tf.split(x, split_indices, axis=1)
            # Structured or flattened (by single action component) input
            else:
                split_x = tree.flatten(x)
            
            def map_(val, dist):
                # Remove extra categorical dimension.
                if isinstance(dist, Categorical):
                    val = tf.cast(tf.squeeze(val, axis=-1), tf.int32)
                return dist.logp(val)
            
            # Remove extra categorical dimension and take the logp of each
            # component.
            flat_logps = tree.map_structure(map_, split_x,
                                            multi_dist.flat_child_distributions)

            return flat_logps

        flat_logps = peraction_logp(curr_action_dist, actions)
        assert len(flat_logps) == 2

        prev_flat_logps = peraction_logp(prev_dist, actions)
        assert len(prev_flat_logps) == 2

        logp_ratio_0 = tf.exp(flat_logps[0] - prev_flat_logps[0])
        logp_ratio_1 = tf.exp(flat_logps[1] - prev_flat_logps[1])

        def peraction_kl(dist_0, dist_1):
            kl_list = [
                d.kl(o) for d, o in zip(dist_0.flat_child_distributions,
                                        dist_1.flat_child_distributions)
            ]
            return kl_list

        action_kl_0, action_kl_1 = peraction_kl(prev_dist, curr_action_dist)
        self.mean_kl_0 = reduce_mean_valid(action_kl_0)
        self.mean_kl_1 = reduce_mean_valid(action_kl_1)
        self.mean_kl = (self.mean_kl_0 + self.mean_kl_1) / 2

        def peraction_entropy(dist):
            entropy_list = [d.entropy() for d in dist.flat_child_distributions]
            return entropy_list

        curr_entropy_0, curr_entropy_1 = peraction_entropy(curr_action_dist)
        self.mean_entropy_0 = reduce_mean_valid(curr_entropy_0)
        self.mean_entropy_1 = reduce_mean_valid(curr_entropy_1)

        surrogate_loss_0 = tf.minimum(
            advantages_0 * logp_ratio_0,
            advantages_0 * tf.clip_by_value(logp_ratio_0, 1 - clip_param,
                                            1 + clip_param))
        surrogate_loss_1 = tf.minimum(
            advantages_1 * logp_ratio_1,
            advantages_1 * tf.clip_by_value(logp_ratio_1, 1 - clip_param,
                                            1 + clip_param))
        self.mean_policy_loss_0 = reduce_mean_valid(-surrogate_loss_0)
        self.mean_policy_loss_1 = reduce_mean_valid(-surrogate_loss_1)
        self.mean_policy_loss = (self.mean_policy_loss_0 + self.mean_policy_loss_1) / 2

        assert use_gae
        vf_loss1_0 = tf.square(value_fn_0 - value_targets_0)
        vf_clipped_0 = vf_preds_0 + tf.clip_by_value(
            value_fn_0 - vf_preds_0, -vf_clip_param, vf_clip_param)
        vf_loss2_0 = tf.square(vf_clipped_0 - value_targets_0)
        vf_loss_0 = tf.maximum(vf_loss1_0, vf_loss2_0)

        vf_loss1_1 = tf.square(value_fn_1 - value_targets_1)
        vf_clipped_1 = vf_preds_1 + tf.clip_by_value(
            value_fn_1 - vf_preds_1, -vf_clip_param, vf_clip_param)
        vf_loss2_1 = tf.square(vf_clipped_1 - value_targets_1)
        vf_loss_1 = tf.maximum(vf_loss1_1, vf_loss2_1)

        self.mean_vf_loss_0 = reduce_mean_valid(vf_loss_0)
        self.mean_vf_loss_1 = reduce_mean_valid(vf_loss_1)
        self.mean_vf_loss = (self.mean_vf_loss_0 + self.mean_vf_loss_1) / 2

        loss_0 = reduce_mean_valid(
            -surrogate_loss_0 + cur_kl_coeff * action_kl_0 +
            vf_loss_coeff * vf_loss_0 - entropy_coeff * curr_entropy_0)
        loss_1 = reduce_mean_valid(
            -surrogate_loss_1 + cur_kl_coeff * action_kl_1 +
            vf_loss_coeff * vf_loss_1 - entropy_coeff * curr_entropy_1)

        loss = loss_0 + loss_1

        self.loss_0 = loss_0
        self.loss_1 = loss_1

        self.loss = loss


def renew_graph(config, item):
    # change the topology in a master thread
    if config["env_config"]["topology"] == "random":
        exp_name = config["env_config"].get("exp_name", "default")
        topology_file = config["env_config"]["topology"] + "-" + exp_name
        graph.gen_link_graph(topology_file, threshold=config["env_config"].get("threshold", 0.3),
                                            sigma=config["env_config"].get("sigma", 0))
    
    return item


def ns3_peraction_execution_plan(workers, config):
    rollouts = ParallelRollouts(workers, mode="bulk_sync")

    # Collect large batches of relevant experiences & standardize.
    rollouts = rollouts.for_each(
        SelectExperiences(workers.trainable_policies()))
    rollouts = rollouts.combine(
        ConcatBatches(min_batch_size=config["train_batch_size"]))
    rollouts = rollouts.for_each(StandardizeFields(["advantages_0", "advantages_1"]))

    if config["simple_optimizer"]:
        train_op = rollouts.for_each(
            TrainOneStep(
                workers,
                num_sgd_iter=config["num_sgd_iter"],
                sgd_minibatch_size=config["sgd_minibatch_size"]))
    else:
        train_op = rollouts.for_each(
            TrainTFMultiGPU(
                workers,
                sgd_minibatch_size=config["sgd_minibatch_size"],
                num_sgd_iter=config["num_sgd_iter"],
                num_gpus=config["num_gpus"],
                rollout_fragment_length=config["rollout_fragment_length"],
                num_envs_per_worker=config["num_envs_per_worker"],
                train_batch_size=config["train_batch_size"],
                shuffle_sequences=config["shuffle_sequences"],
                _fake_gpus=config["_fake_gpus"]))

    # change the graph topology
    train_op = train_op.for_each(partial(renew_graph, config))

    # Update KL after each round of training.
    train_op = train_op.for_each(lambda t: t[1]).for_each(UpdateKL(workers))

    return StandardMetricsReporting(train_op, workers, config) \
        .for_each(lambda result: warn_about_bad_reward_scales(config, result))