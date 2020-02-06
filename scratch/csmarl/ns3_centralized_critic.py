import os
import gym
import numpy as np
import ray
import argparse
from ray import tune
from ray.tune.registry import register_env

from ray.rllib.agents.ppo.ppo import PPOTrainer
from ray.rllib.agents.ppo.ppo_policy import PPOTFPolicy, KLCoeffMixin, PPOLoss, BEHAVIOUR_LOGITS
from ray.rllib.evaluation.postprocessing import compute_advantages, Postprocessing
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.tf_policy import LearningRateSchedule, EntropyCoeffSchedule, ACTION_LOGP
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet_v2 import FullyConnectedNetwork
from ray.rllib.utils.explained_variance import explained_variance
from ray.rllib.utils.tf_ops import make_tf_callable
from ray.rllib.utils import try_import_tf

from ns3multiagentenv import Ns3MultiAgentEnv, on_episode_start, on_episode_step, on_episode_end

tf = try_import_tf()

OTHER_OBS = "other_obs"
OTHER_ACTION = "other_action"


class CentralizedCriticModel(TFModelV2):
    """Multi-agent model that implements a centralized VF."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(CentralizedCriticModel, self).__init__(
            obs_space, action_space, num_outputs, model_config, name)

        # Policy network
        # Base of the model
        self.model = FullyConnectedNetwork(
            obs_space, action_space, num_outputs, model_config, name)
        self.register_variables(self.model.variables())

        self.n_agents_in_critic = model_config["custom_options"]["n_agents_in_critic"]

        # Value network
        # Central VF maps (obs, other_obs, other_act, agent_id) -> vf_pred
        # Referenced from COMA critic function

        # MYTODO change static dimenstion assignments
        self.obs_dim = 4
        self.act_dim = 10

        obs = tf.keras.layers.Input(shape=(4,), name="obs")  # Box(4)
        other_obs = tf.keras.layers.Input(
            shape=(self.obs_dim * (self.n_agents_in_critic - 1),), name="other_obs")  # Box(4) * 2
        other_act = tf.keras.layers.Input(
            shape=(self.act_dim * (self.n_agents_in_critic - 1),), name="other_act")  # Discrete(10) * 2
        agent_id = tf.keras.layers.Input(
            shape=(self.n_agents_in_critic,), name="agent_id")  # 3 agents

        # NN definition
        # MYTODO configure
        concat_input = tf.keras.layers.Concatenate(
            axis=1)([obs, other_obs, other_act, agent_id])
        central_vf_dense = tf.keras.layers.Dense(
            32, activation=tf.nn.tanh, name="c_vf_dense")(concat_input)
        central_vf_out = tf.keras.layers.Dense(
            1, activation=None, name="c_vf_out")(central_vf_dense)
        self.central_vf = tf.keras.Model(
            inputs=[obs, other_obs, other_act, agent_id], outputs=central_vf_out)
        self.register_variables(self.central_vf.variables)

    def forward(self, input_dict, state, seq_lens):
        return self.model.forward(input_dict, state, seq_lens)

    def central_value_function(self, obs, other_obs, other_actions, agent_id):
        # XXX Currently concatenated onehot for joint other-actions
        other_actions_onehot = tf.reshape(
            tf.one_hot(other_actions, self.act_dim), [-1, self.act_dim * (self.n_agents_in_critic - 1)])
        agent_id_onehot = tf.one_hot(agent_id, self.n_agents_in_critic)
        return tf.reshape(
            self.central_vf([obs, other_obs, other_actions_onehot, agent_id_onehot]), [-1])

    def value_function(self):
        return self.model.value_function()  # not used


class CentralizedValueMixin:
    """Add method to evaluate the central value function from the model."""

    def __init__(self):
        self.compute_central_vf = make_tf_callable(self.get_session())(
            self.model.central_value_function)


# Grabs the opponent obs/act and includes it in the experience train_batch,
# and computes GAE using the central vf predictions.
def centralized_critic_postprocessing(policy,
                                      sample_batch,
                                      other_agent_batches=None,  # batches from other agents
                                      episode=None):

    n_agents_in_critic = policy.model.model_config["custom_options"]["n_agents_in_critic"]

    if policy.loss_initialized():
        assert sample_batch["dones"][-1], "Not implemented for train_batch_mode=truncate_episodes"
        assert other_agent_batches is not None
        other_batches = [other_batch for _,
                         other_batch in other_agent_batches.values()]
        # [(_, other_batch)] = list(other_agent_batches.values())

        # also record the opponent obs and actions in the trajectory
        # sample_batch[OTHER_OBS] = other_batch[SampleBatch.CUR_OBS]
        # sample_batch[OTHER_ACTION] = other_batch[SampleBatch.ACTIONS]

        # NOTE shape of batches
        # b[SampleBatch.CUR_OBS] = (T, 4)
        # b[SampleBatch.ACTIONS] = (T,)
        sample_batch[OTHER_OBS] = np.concatenate(
            [b[SampleBatch.CUR_OBS] for b in other_batches], axis=-1)
        sample_batch[OTHER_ACTION] = np.stack(
            [b[SampleBatch.ACTIONS] for b in other_batches], axis=-1)

        # print([b[SampleBatch.CUR_OBS].shape for b in other_batches])
        # print([b[SampleBatch.ACTIONS].shape for b in other_batches])
        # print(sample_batch[OTHER_OBS].shape)
        # print(sample_batch[OTHER_ACTION].shape)

        # overwrite default VF prediction with the central VF
        sample_batch[SampleBatch.VF_PREDS] = policy.compute_central_vf(
            sample_batch[SampleBatch.CUR_OBS], sample_batch[OTHER_OBS],
            sample_batch[OTHER_ACTION], sample_batch[SampleBatch.AGENT_INDEX])
    else:
        # policy hasn't initialized yet, use zeros
        # This initialization will be placeholders in model definition
        # sample_batch[OTHER_OBS] = np.zeros_like(
        #     sample_batch[SampleBatch.CUR_OBS])
        # sample_batch[OTHER_ACTION] = np.zeros_like(
        #     sample_batch[SampleBatch.ACTIONS])

        sample_batch[OTHER_OBS] = np.zeros_like(
            np.tile(sample_batch[SampleBatch.CUR_OBS], n_agents_in_critic - 1)
        )  # (1, 8)
        # NOTE This works for action space with shape ()
        sample_batch[OTHER_ACTION] = np.zeros_like(
            np.tile(sample_batch[SampleBatch.ACTIONS],
                    (1, n_agents_in_critic - 1))
        )  # (1, 2)

        sample_batch[SampleBatch.VF_PREDS] = np.zeros_like(
            sample_batch[SampleBatch.ACTIONS], dtype=np.float32)

        # NOTE It is okay to override SampleBatch.AGENT_INDEX
        sample_batch[SampleBatch.AGENT_INDEX] = np.zeros((1,), dtype=int)

        # print(sample_batch[SampleBatch.CUR_OBS].shape)    # (1, 4)
        # print(sample_batch[SampleBatch.ACTIONS].shape)    # (1,)
        # print(np.tile(sample_batch[SampleBatch.CUR_OBS],
        #               n_agents_in_critic - 1).shape)      # (1, 8)
        # print(np.tile(sample_batch[SampleBatch.ACTIONS],  # (1, 2)
        #               (1, n_agents_in_critic - 1)).shape)

    # Run GAE to compute advantages
    # VF_PREDS are used in here
    train_batch = compute_advantages(
        sample_batch,
        0.0,
        policy.config["gamma"],
        policy.config["lambda"],
        use_gae=policy.config["use_gae"])
    return train_batch


# Copied from PPO but opimizing the central value function
def loss_with_central_critic(policy, model, dist_class, train_batch):
    CentralizedValueMixin.__init__(policy)

    # GRAPH INITIALIZATION: This function is called only once
    # central_value_function: graph definition
    # compute_central_vf: called on execution

    # # central_value_function requires tensor input ...

    logits, state = model.from_batch(train_batch)
    action_dist = dist_class(logits, model)
    policy.central_value_out = policy.model.central_value_function(
        train_batch[SampleBatch.CUR_OBS], train_batch[OTHER_OBS],
        train_batch[OTHER_ACTION], train_batch[SampleBatch.AGENT_INDEX])

    # VALUE_TARGETS = VF_PREDS + ADVANTAGES
    policy.loss_obj = PPOLoss(
        policy.action_space,
        dist_class,
        model,
        train_batch[Postprocessing.VALUE_TARGETS],
        train_batch[Postprocessing.ADVANTAGES],
        train_batch[SampleBatch.ACTIONS],
        train_batch[BEHAVIOUR_LOGITS],
        train_batch[ACTION_LOGP],
        train_batch[SampleBatch.VF_PREDS],
        action_dist,
        policy.central_value_out,
        policy.kl_coeff,
        tf.ones_like(train_batch[Postprocessing.ADVANTAGES], dtype=tf.bool),
        entropy_coeff=policy.entropy_coeff,
        clip_param=policy.config["clip_param"],
        vf_clip_param=policy.config["vf_clip_param"],
        vf_loss_coeff=policy.config["vf_loss_coeff"],
        use_gae=policy.config["use_gae"],
        model_config=policy.config["model"])

    return policy.loss_obj.loss


def setup_mixins(policy, obs_space, action_space, config):
    # Copied from PPO
    KLCoeffMixin.__init__(policy, config)
    EntropyCoeffSchedule.__init__(policy, config["entropy_coeff"],
                                  config["entropy_coeff_schedule"])
    LearningRateSchedule.__init__(policy, config["lr"], config["lr_schedule"])


def central_vf_stats(policy, train_batch, grads):
    # Report the explained variance of the central value function
    return {
        "vf_explained_var": explained_variance(
            train_batch[Postprocessing.VALUE_TARGETS],
            policy.central_value_out),
    }


CCPPO = PPOTFPolicy.with_updates(
    name="CCPPO",
    postprocess_fn=centralized_critic_postprocessing,
    loss_fn=loss_with_central_critic,
    before_loss_init=setup_mixins,
    grad_stats_fn=central_vf_stats,
    mixins=[
        LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin,
        CentralizedValueMixin
    ]
)

CCTrainer = PPOTrainer.with_updates(name="CCPPOTrainer", default_policy=CCPPO)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stop", help="number of timesteps, default 1e8", type=int, default=1e8)
    parser.add_argument(
        "--debug", help="debug indicator, default false", type=bool, default=False)

    args = parser.parse_args()

    if args.debug:
        ray.init()
    else:
        ray.init(log_to_driver=False)

    cwd = os.path.dirname(os.path.abspath(__file__))

    ModelCatalog.register_custom_model("cc_model", CentralizedCriticModel)
    tune.run(
        CCTrainer,
        stop={
            "timesteps_total": args.stop,
        },
        config={
            "env": Ns3MultiAgentEnv,
            "batch_mode": "complete_episodes",
            "log_level": "DEBUG",
            "env_config": {
                "n_agents": 3,  # environment configuration
                "cwd": cwd,
                "debug": not args.debug
            },
            "num_workers": 0 if args.debug else 16,
            "model": {
                "custom_model": "cc_model",
                "custom_options": {
                    "n_agents_in_critic": 3,  # n_agents in critic
                }
            },
            "callbacks": {
                "on_episode_start": on_episode_start,
                "on_episode_step": on_episode_step,
                "on_episode_end": on_episode_end
            }
        }
    )
