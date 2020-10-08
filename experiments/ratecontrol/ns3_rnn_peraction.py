import argparse
import numpy as np

import ray
from ray import tune

from ray.rllib.agents.ppo.ppo import PPOTrainer
from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy, KLCoeffMixin
from ray.rllib.models import ModelCatalog
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.recurrent_net import RecurrentNetwork
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.tf_policy import LearningRateSchedule, EntropyCoeffSchedule
from ray.rllib.utils import merge_dicts
from ray.rllib.utils.annotations import override
from ray.rllib.utils.explained_variance import explained_variance
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.tf_ops import make_tf_callable

from ns3_import import import_graph_module
from ns3_multiagent_env import MyCallbacks
from peraction_utils import peraction_compute_advantages, PPOPerActionLoss, ns3_peraction_execution_plan

graph = import_graph_module()
tf = try_import_tf()


class PerActionModel(RecurrentNetwork):

    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 hiddens_size=256,
                 cell_size=64,
                 n_agents=3):
        super(PerActionModel, self).__init__(obs_space, action_space, num_outputs,
                                                   model_config, name)
        self.cell_size = cell_size

        self.n_agents = n_agents
        self.obs_dim = obs_space.shape[0]

        # Define input layers
        inputs = tf.keras.layers.Input(
            shape=(None, self.obs_dim), name="inputs")
        state_in_h = tf.keras.layers.Input(shape=(cell_size,), name="h")
        state_in_c = tf.keras.layers.Input(shape=(cell_size,), name="c")
        seq_in = tf.keras.layers.Input(shape=(), name="seq_in", dtype=tf.int32)

        # Preprocess observation with a hidden layer and send to LSTM cell
        dense = tf.keras.layers.Dense(
            hiddens_size, activation=tf.nn.relu, name="dense")(inputs)
        lstm_out, state_h, state_c = tf.keras.layers.LSTM(
            cell_size, return_sequences=True, return_state=True, name="lstm")(
                inputs=dense,
                mask=tf.sequence_mask(seq_in),
                initial_state=[state_in_h, state_in_c])

        # Postprocess LSTM output with another hidden layer and compute values
        # logits: output of policy network
        logits = tf.keras.layers.Dense(
            self.num_outputs,
            activation=None,
            name="logits")(lstm_out)
        # values: output of value network
        values = tf.keras.layers.Dense(
            1, activation=None, name="values")(lstm_out)

        # Create the RNN model
        self.rnn_model = tf.keras.Model(
            inputs=[inputs, seq_in, state_in_h, state_in_c],
            outputs=[logits, values, state_h, state_c])
        self.register_variables(self.rnn_model.variables)

        self.act_0_dim = action_space.spaces[0].n
        self.act_1_dim = action_space.spaces[1].n

        action_0 = tf.keras.layers.Input(
            shape=(None, self.act_0_dim), name="action_0")
        action_1 = tf.keras.layers.Input(
            shape=(None, self.act_1_dim), name="action_1")

        act_0_dense = tf.keras.layers.Dense(
            hiddens_size, activation=tf.nn.relu, name="act_0_dense")(action_0)
        act_1_dense = tf.keras.layers.Dense(
            hiddens_size, activation=tf.nn.relu, name="act_1_dense")(action_1)

        obs_and_act_0 = tf.concat([lstm_out, act_0_dense], axis=-1)  # obs_seq + act_0
        obs_and_act_1 = tf.concat([lstm_out, act_1_dense], axis=-1)  # obs_seq + act_1

        values_0 = tf.keras.layers.Dense(
            1, activation=None, name="values_0")(obs_and_act_1)  # counterfactuals
        values_1 = tf.keras.layers.Dense(
            1, activation=None, name="values_1")(obs_and_act_0)
        
        self.rnn_peraction_vf = tf.keras.Model(
            inputs=[inputs, seq_in, state_in_h, state_in_c, action_0, action_1],
            outputs=[values_0, values_1])
        self.register_variables(self.rnn_peraction_vf.variables)


    @override(RecurrentNetwork)
    def forward_rnn(self, inputs, state, seq_lens):
        # state: [h, c]
        model_out, self._dummy_value_out, h, c = self.rnn_model(
            [inputs, seq_lens] + state)
        return model_out, [h, c]

    @override(ModelV2)
    def get_initial_state(self):
        # initial [h, c]
        return [
            np.zeros(self.cell_size, np.float32),
            np.zeros(self.cell_size, np.float32)
        ]

    def peraction_value_function(self, inputs, state_h, state_c, seq_lens, action_0, action_1):
        act_0_onehot = tf.one_hot(action_0, self.act_0_dim)
        act_1_onehot = tf.one_hot(action_1, self.act_1_dim)

        values_0, values_1 = self.rnn_peraction_vf(
            [inputs, seq_lens, state_h, state_c, act_0_onehot, act_1_onehot]
        )
        return tf.reshape(values_0, [-1]), tf.reshape(values_1, [-1])

    @override(ModelV2)
    def value_function(self):
        return tf.reshape(self._dummy_value_out, [-1])


class PerActionValueMixin:

    def __init__(self):
        self.compute_peraction_vf = make_tf_callable(self.get_session())(
            self.model.peraction_value_function
        )


def peraction_critic_postprocessing(policy, sample_batch, other_agent_batches=None, episode=None):

    if not policy.loss_initialized():

        sample_batch["action_0"] = np.zeros((1,), dtype=np.int32)
        sample_batch["action_1"] = np.zeros((1,), dtype=np.int32)

        sample_batch["values_0"] = np.zeros_like(
            sample_batch[SampleBatch.REWARDS], dtype=np.float32)
        sample_batch["values_1"] = np.zeros_like(
            sample_batch[SampleBatch.REWARDS], dtype=np.float32)

    else:
        assert sample_batch["dones"][-1], "Not implemented for train_batch_mode=truncated_episodes"

        sample_batch["action_0"] = sample_batch[SampleBatch.ACTIONS][:, 0]
        sample_batch["action_1"] = sample_batch[SampleBatch.ACTIONS][:, 1]

        seq_lens = np.array([sample_batch[SampleBatch.CUR_OBS].shape[0],])

        # construct a single batch with time indicies
        obs_with_time = np.expand_dims(sample_batch[SampleBatch.CUR_OBS], axis=0)
        act_0_with_time = np.expand_dims(sample_batch["action_0"], axis=0)
        act_1_with_time = np.expand_dims(sample_batch["action_1"], axis=0)
        h, c = policy.model.get_initial_state()
        h = np.expand_dims(h, axis=0)
        c = np.expand_dims(c, axis=0)
        
        sample_batch["values_0"], sample_batch["values_1"] = \
            policy.compute_peraction_vf(
                obs_with_time, h, c, seq_lens, act_0_with_time, act_1_with_time)

    train_batch = peraction_compute_advantages(
        sample_batch,
        0.0,
        policy.config["gamma"],
        policy.config["lambda"],
        use_gae=policy.config["use_gae"]
    )
    return train_batch


def loss_with_peraction_critic(policy, model, dist_class, train_batch):
    PerActionValueMixin.__init__(policy)

    logits, _ = model.from_batch(train_batch)
    action_dist = dist_class(logits, model)

    states = [train_batch["state_in_0"], train_batch["state_in_1"]]

    seq_lens = train_batch["seq_lens"]

    obs_with_time = add_time_dimension(train_batch[SampleBatch.CUR_OBS], seq_lens)
    act_0_with_time = add_time_dimension(train_batch["action_0"], seq_lens)
    act_1_with_time = add_time_dimension(train_batch["action_1"], seq_lens)

    policy.peraction_value_out_0, policy.peraction_value_out_1 = \
        policy.model.peraction_value_function(obs_with_time, states[0], states[1], 
                                              seq_lens, act_0_with_time, act_1_with_time)

    mask = tf.ones_like(train_batch["advantages_0"], dtype=tf.bool)

    # VALUE_TARGETS = VF_PREDS + ADVANTAGES
    policy.loss_obj = PPOPerActionLoss(
        dist_class,
        model,
        train_batch["value_targets_0"],
        train_batch["value_targets_1"],
        train_batch["advantages_0"],
        train_batch["advantages_1"],
        train_batch[SampleBatch.ACTIONS],
        train_batch[SampleBatch.ACTION_DIST_INPUTS],
        train_batch["values_0"],
        train_batch["values_1"],
        action_dist,
        policy.peraction_value_out_0,
        policy.peraction_value_out_1,
        policy.kl_coeff,
        mask,
        entropy_coeff=policy.entropy_coeff,
        clip_param=policy.config["clip_param"],
        vf_clip_param=policy.config["vf_clip_param"],
        vf_loss_coeff=policy.config["vf_loss_coeff"],
        use_gae=policy.config["use_gae"]
    )

    return policy.loss_obj.loss


def setup_mixins(policy, obs_space, action_space, config):
    # Copied from PPO
    KLCoeffMixin.__init__(policy, config)
    EntropyCoeffSchedule.__init__(policy, config["entropy_coeff"],
                                  config["entropy_coeff_schedule"])
    LearningRateSchedule.__init__(policy, config["lr"], config["lr_schedule"])


def peraction_stats(policy, train_batch):
    return {
        "cur_kl_coeff": tf.cast(policy.kl_coeff, tf.float64),
        "cur_lr": tf.cast(policy.cur_lr, tf.float64),
        "total_loss": policy.loss_obj.loss,
        "policy_loss_0": policy.loss_obj.mean_policy_loss_0,
        "policy_loss_1": policy.loss_obj.mean_policy_loss_1,
        "policy_loss": policy.loss_obj.mean_policy_loss,
        "vf_loss_0": policy.loss_obj.mean_vf_loss_0,
        "vf_loss_1": policy.loss_obj.mean_vf_loss_1,
        "vf_loss": policy.loss_obj.mean_vf_loss,
        "vf_explained_var_0": explained_variance(
            train_batch["value_targets_0"],
            policy.peraction_value_out_0,
        ),
        "vf_explained_var_1": explained_variance(
            train_batch["value_targets_1"],
            policy.peraction_value_out_1,
        ),
        "kl_0": policy.loss_obj.mean_kl_0,
        "kl_1": policy.loss_obj.mean_kl_1,
        "kl": policy.loss_obj.mean_kl,
        "entropy_0": policy.loss_obj.mean_entropy_0,
        "entropy_1": policy.loss_obj.mean_entropy_1,
        "entropy_coeff": tf.cast(policy.entropy_coeff, tf.float64),
    }


PerActionPPO = PPOTFPolicy.with_updates(
    name="PerActionPPO",
    postprocess_fn=peraction_critic_postprocessing,
    loss_fn=loss_with_peraction_critic,
    before_loss_init=setup_mixins,
    stats_fn=peraction_stats,
    mixins=[
        LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin,
        PerActionValueMixin
    ]
)

def get_policy_class(config):
    return PerActionPPO

PerActionPPOTrainer = PPOTrainer.with_updates(
    name="PerActionPPOTrainer",
    default_policy=PerActionPPO,
    get_policy_class=get_policy_class,
    execution_plan=ns3_peraction_execution_plan
)

ModelCatalog.register_custom_model("peraction_model", PerActionModel)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--topology", type=str, default="complex")
    args = parser.parse_args()

    ray.init(log_to_driver=args.debug, local_mode=args.debug)

    env_config ={
        "topology": args.topology,
        "exp_name": __file__.split(".")[0],
    }

    _, n_agents = graph.read_graph(env_config["topology"])

    NUM_GPUS = 4
    num_workers = 8
    num_gpus_per_worker = NUM_GPUS / num_workers
    timesteps_total = 3e8

    config = {
        "env": "ns3_multiagent_env",
        "batch_mode": "complete_episodes",
        "log_level": "WARN",
        "num_workers": num_workers,
        "num_gpus_per_worker": num_gpus_per_worker,
        "sgd_minibatch_size": 2000,
        "lr_schedule": [[0, 5e-5], [timesteps_total, 0.0]],
        "env_config": env_config,
        "callbacks": MyCallbacks,
        "model": {
            "custom_model": "peraction_model",
            "custom_model_config": {
                "n_agents": n_agents,
            }
        }
    }

    if env_config["topology"] == "random":
        # add evaluation config
        num_eval_workers = 4
        num_gpus_per_worker = NUM_GPUS / (num_workers + num_eval_workers)
        eval_config = {
            "evaluation_interval": 1,
            "evaluation_num_workers": num_eval_workers,
            "evaluation_num_episodes": num_eval_workers,
            "evaluation_config": {
                "env_config": {"topology": "complex"},
                "explore": False,
            },
            "num_gpus_per_worker": num_gpus_per_worker,
        }
        config = merge_dicts(config, eval_config)

    if args.debug:
        env_config["simTime"] = 1
        config["log_level"] = "DEBUG"
        config["num_workers"] = 0

    stop = {
        "timesteps_total": timesteps_total,
    }

    tune.run(PerActionPPOTrainer, config=config, stop=stop, checkpoint_freq=10, keep_checkpoints_num=1)