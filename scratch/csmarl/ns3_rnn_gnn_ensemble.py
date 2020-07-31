import os
import random
import gym
import numpy as np

import ray
from ray import tune

from ray.rllib.models import ModelCatalog
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.recurrent_net import RecurrentNetwork
from ray.rllib.utils.annotations import override
from ray.rllib.utils import try_import_tf

from ns3gym import ns3env
import ns3_multiagent_env
from ns3_rnn_gnn import GNNPPOTrainer, MyCallbacks
from link_graph import read_graph

tf = try_import_tf()
import tf_slim as slim


class RecurrentEnsemblePolicyModel(RecurrentNetwork):

    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 topology="fim",
                 hiddens_size=256,
                 cell_size=64):
        super(RecurrentEnsemblePolicyModel, self).__init__(obs_space, action_space, num_outputs,
                                                           model_config, name)
        self.cell_size = cell_size

        _, self.n_agents = read_graph(topology)
        self.obs_dim = obs_space.shape[0]
        self.act_dim = self.num_outputs

        ### RECURRENT POLICY NETWORK ###

        input_layer = tf.keras.layers.Input(
            shape=(None, self.obs_dim), name="inputs")

        state_in_h = tf.keras.layers.Input(shape=(cell_size,), name="h")
        state_in_c = tf.keras.layers.Input(shape=(cell_size,), name="c")
        seq_in = tf.keras.layers.Input(shape=(), name="seq_in", dtype=tf.int32)

        # Preprocess observation with a hidden layer and send to LSTM cell
        dense1 = tf.keras.layers.Dense(
            hiddens_size, activation=tf.nn.relu, name="dense1")(input_layer)
        lstm_out, state_h, state_c = tf.keras.layers.LSTM(
            cell_size, return_sequences=True, return_state=True, name="lstm")(
                inputs=dense1,
                mask=tf.sequence_mask(seq_in),
                initial_state=[state_in_h, state_in_c])

        # Postprocess LSTM output with another hidden layer and compute values
        # logits: output of policy network
        logits = tf.keras.layers.Dense(
            self.num_outputs,
            activation=tf.keras.activations.linear,
            name="logits")(lstm_out)
        # values: output of value network
        # dummy value; replaced with gnn critic
        values = tf.keras.layers.Dense(
            1, activation=None, name="values")(lstm_out)

        # Create RNN model
        self.rnn_model = tf.keras.Model(
            inputs=[input_layer, seq_in, state_in_h, state_in_c],
            outputs=[logits, values, state_h, state_c])

        self.register_variables(self.rnn_model.variables)

        ### GRAPH-CONVOLUTIONAL VALUE NETWORK ###

        obs = tf.keras.layers.Input(
            shape=(self.obs_dim,), name="obs")

        # used for graph convolution
        neighbor_num = tf.keras.layers.Input(
            shape=(1,), name="neighbor_num")
        neighbor_obs = tf.keras.layers.Input(
            shape=(self.n_agents, self.obs_dim,), name="neighbor_obs")
        neighbor_act = tf.keras.layers.Input(
            shape=(self.n_agents, self.act_dim,), name="neighbor_act")

        def value_fn(args):

            obs, neighbor_num, neighbor_obs, neighbor_act = args

            with tf.variable_scope(
                    tf.VariableScope(tf.AUTO_REUSE, "shared"),
                    reuse=tf.AUTO_REUSE,
                    auxiliary_name_scope=False):

                obs_dense = slim.fully_connected(obs, hiddens_size, activation_fn=tf.nn.tanh, scope="obs_dense")

                neighbor_feat = tf.concat([neighbor_obs, neighbor_act], axis=-1, name="neighbor_feat")
                neighbor_dense = slim.fully_connected(neighbor_feat, hiddens_size, activation_fn=tf.nn.tanh, biases_initializer=None, scope="neighbor_dense")

                neighbor_sum = tf.math.reduce_sum(neighbor_dense, axis=-2, name="neighor_sum")

                gnn_vf_in = tf.cond(neighbor_num[0][0] > 0, lambda: obs_dense + neighbor_sum / neighbor_num, lambda: obs_dense, name="gnn_vf_in")
                gnn_vf_dense = slim.fully_connected(gnn_vf_in, hiddens_size, activation_fn=tf.nn.tanh, scope="gnn_vf_dense")
                gnn_vf_out = slim.fully_connected(gnn_vf_dense, 1, activation_fn=None, scope="gnn_vf_out")

            return gnn_vf_out

        self.gnn_vf = value_fn

        # forward pass once to get the variables
        _ = self.gnn_vf([obs, neighbor_num, neighbor_obs, neighbor_act])

        var_set = slim.get_model_variables()
        self.register_variables(var_set)

    @override(RecurrentNetwork)
    def forward_rnn(self, inputs, rnn_state, seq_lens):
        # rnn_state: [h, c]
        model_out, self._dummy_value_out, h, c = self.rnn_model(
            [inputs, seq_lens] + rnn_state)
        return model_out, [h, c]

    @override(ModelV2)
    def get_initial_state(self):
        # initial [h, c]
        return [
            np.zeros(self.cell_size, np.float32),
            np.zeros(self.cell_size, np.float32),
        ]

    # directly calling this function establishes tf_op connection between tensors
    def gnn_value_function(self, obs, neighbor_num, neighbor_obs, neighbor_act):
        neighbor_act_onehot = tf.one_hot(neighbor_act, self.act_dim)
        return tf.reshape(
            self.gnn_vf([obs, neighbor_num, neighbor_obs, neighbor_act_onehot]), [-1])

    @override(ModelV2)
    def value_function(self):
        # unused
        return tf.reshape(self._dummy_value_out, [-1])


ModelCatalog.register_custom_model(
    "recurrent_ensemble_model", RecurrentEnsemblePolicyModel)


if __name__ == "__main__":

    debug = False

    ray.init(log_to_driver=debug)
    cwd = os.path.dirname(os.path.abspath(__file__))
    topology = "complex"

    NUM_GPUS = 4
    num_workers = 16

    # params_list = [0]
    params_list = [5e-3, 5e-4, 5e-5, 5e-6]

    num_samples = 1

    num_workers_total = num_workers * \
        len(params_list) * num_samples  # <= 32 is recommended
    num_gpus_per_worker = NUM_GPUS / num_workers_total

    # temporary single environment for extracting single obs/act dim
    with ns3env.Ns3Env(port=0, 
                       startSim=True, 
                       simArgs={
                           "--algorithm": "rl", 
                           "--simTime": 1, 
                           "--topology": "single", 
                           "--debug": False}, 
                       cwd=cwd) as single_env:
        multi_obs_space = single_env.observation_space
        multi_act_space = single_env.action_space
        obs_space = gym.spaces.Box(
            low=multi_obs_space.low[0], high=multi_obs_space.high[0], dtype=multi_obs_space.dtype)
        act_space = multi_act_space.spaces[0]

    tune.run(
        GNNPPOTrainer,
        stop={
            "timesteps_total": 3e8,
        },
        config={
            "env": "ns3_multiagent_env",
            "batch_mode": "complete_episodes",
            "log_level": "DEBUG" if debug else "WARN",
            "env_config": {
                "cwd": cwd,
                "debug": False,
                "reward": "indiv",
                "topology": topology,
            },
            "num_workers": num_workers,
            "num_gpus_per_worker": num_gpus_per_worker,
            "gamma": 0.5,
            "lr": 5e-4,
            # "lr": tune.grid_search(params_list),
            "use_gae": True,
            "sgd_minibatch_size": 2000,
            "model": {
                "custom_model": "recurrent_ensemble_model",
                "max_seq_len": 20,
                "custom_model_config": {
                    "topology": topology,
                }
            },
            "multiagent": {
                "policies": {
                    "ensemble_0": (None, obs_space, act_space, {}),
                    "ensemble_1": (None, obs_space, act_space, {}),
                    "ensemble_2": (None, obs_space, act_space, {}),
                },
                "policy_mapping_fn": lambda agent_id: random.choice(["ensemble_0", "ensemble_1", "ensemble_2"]),  # random picking ensemble
            },
            "callbacks": MyCallbacks,
        },
        num_samples=num_samples,
        checkpoint_freq=10,
        keep_checkpoints_num=1,
        checkpoint_score_attr="episode_reward_mean",
    )
