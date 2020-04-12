import os
import gym
from gym.spaces import Discrete
import numpy as np
import argparse

import ray
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.recurrent_tf_modelv2 import RecurrentTFModelV2
from ray.rllib.utils.tf_ops import make_tf_callable
from ray.rllib.utils.annotations import override
from ray.rllib.utils import try_import_tf

from ns3_multiagent_env import Ns3MultiAgentEnv, on_episode_start, on_episode_step, on_episode_end
from ns3_centralized_critic import CCTrainer
from graph import read_graph

tf = try_import_tf()


class CentralizedCriticRNNModel(RecurrentTFModelV2):

    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 hiddens_size=256,
                 cell_size=64):
        super(CentralizedCriticRNNModel, self).__init__(obs_space, action_space, num_outputs,
                                                        model_config, name)
        self.cell_size = cell_size

        self.n_agents_in_critic = read_graph(model_config["custom_options"]["topology"])
        self.obs_dim = obs_space.shape[0]
        self.act_dim = self.num_outputs
        # MYTODO make proper mapping on agent_id
        agent_id = tf.keras.layers.Input(
            shape=(self.n_agents_in_critic,), name="agent_id")

        ### RECURRENT POLICY NETWORK ###

        # Define input layers
        # option 1: without agent_id
        input_layer = tf.keras.layers.Input(
            shape=(None, obs_space.shape[0]), name="inputs")

        # option 2: Input layers with agent_id
        # input_layer_raw = tf.keras.layers.Input(
        #     shape=(None, obs_space.shape[0]), name="inputs")
        # agent_id_actor = tf.cast(input_layer_raw[:,:,-1], tf.int32)  # (None,)
        # agent_id_actor_onehot = tf.one_hot(agent_id_actor, self.n_agents_in_critic, axis=-1)  # (None, n_ag)
        # input_layer = tf.keras.layers.Concatenate(
        #     axis=-1)([input_layer_raw[:,:,:-1], agent_id_actor_onehot])
        #

        state_in_h = tf.keras.layers.Input(shape=(cell_size, ), name="h")
        state_in_c = tf.keras.layers.Input(shape=(cell_size, ), name="c")
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
        # dummy value; replaced with centralized critic
        values = tf.keras.layers.Dense(
            1, activation=None, name="values")(lstm_out)

        # Create the RNN model

        # option 1: without agent_id
        self.rnn_model = tf.keras.Model(
            inputs=[input_layer, seq_in, state_in_h, state_in_c],
            outputs=[logits, values, state_h, state_c])

        # option 2: include agent_id
        # self.rnn_model = tf.keras.Model(
        #     inputs=[input_layer_raw, seq_in, state_in_h, state_in_c],
        #     outputs=[logits, values, state_h, state_c])

        self.register_variables(self.rnn_model.variables)
        self.rnn_model.summary()  # dump the model info

        ### VALUE NETWORK ###


        obs = tf.keras.layers.Input(
            shape=(self.obs_dim,), name="obs")
        state = tf.keras.layers.Input(
            shape=(self.obs_dim * self.n_agents_in_critic, ), name="state")
        other_act = tf.keras.layers.Input(
            shape=(self.act_dim * (self.n_agents_in_critic - 1), ), name="other_act")

        # NN definition
        # concat_input = tf.keras.layers.Concatenate(
        #     axis=1)([obs, state, other_act, agent_id])  # with agent_id
        concat_input = tf.keras.layers.Concatenate(
            axis=1)([obs, state, other_act])  # without agent_id
        central_vf_dense = tf.keras.layers.Dense(
            hiddens_size, activation=tf.nn.tanh, name="c_vf_dense")(concat_input)
        central_vf_out = tf.keras.layers.Dense(
            1, activation=None, name="c_vf_out")(central_vf_dense)
        self.central_vf = tf.keras.Model(
            inputs=[obs, state, other_act, agent_id], outputs=central_vf_out)
        self.register_variables(self.central_vf.variables)

    @override(RecurrentTFModelV2)
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

    def central_value_function(self, obs, state, other_act, agent_id):
        other_act_onehot = tf.reshape(
            tf.one_hot(other_act, self.act_dim), [-1, self.act_dim * (self.n_agents_in_critic - 1)])
        agent_id_onehot = tf.one_hot(agent_id, self.n_agents_in_critic)
        return tf.reshape(
            self.central_vf([obs, state, other_act_onehot, agent_id_onehot]), [-1])

    @override(ModelV2)
    def value_function(self):
        return tf.reshape(self._dummy_value_out, [-1])


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stop", help="number of timesteps, default 3e8", type=int, default=3e8)
    parser.add_argument(
        "--debug", help="debug indicator, default false", type=bool, default=False)

    args = parser.parse_args()

    ray.init(log_to_driver=args.debug)

    # MYTODO: make it configurable
    cwd = os.path.dirname(os.path.abspath(__file__))
    topology = "complex"

    ModelCatalog.register_custom_model(
        "cc_rnn_model", CentralizedCriticRNNModel)

    config_params = [0]
    env_config = {  # environment configuration
        "cwd": cwd,
        "debug": args.debug,
        "reward": "shared",
        "topology": topology,
        "traffic": "cbr",
    }
    # config_params = [FILL]  # for env config testing
    # env_config = tune.grid_search(config_params)


    NUM_GPUS = 4
    num_workers = 16

    if args.debug:
        params_list = [0]
    else:
        params_list = [0]
        # params_list = [5e-4, 5e-5, 5e-6, 5e-7]  # for parameter testing
        # params_list = [1, 0.8, 0.5, 0]

    num_samples = 1
    
    num_workers_total = num_workers * \
        len(params_list) * len(config_params) * num_samples  # <= 32 is recommended
    num_gpus_per_worker = NUM_GPUS / num_workers_total

    # MYTODO combine simulation codes
    tune.run(
        CCTrainer,
        stop={
            "timesteps_total": args.stop,
        },
        config={
            "env": Ns3MultiAgentEnv,
            "batch_mode": "complete_episodes",
            "log_level": "DEBUG" if args.debug else "WARN",
            "env_config": env_config,
            "num_workers": 0 if args.debug else num_workers,
            "num_gpus_per_worker": num_gpus_per_worker,
            "lr": 5e-4,
            # "lr": 5e-4 if args.debug else tune.grid_search(params_list),
            # "lambda": 1 if args.debug else tune.grid_search(params_list),
            "use_gae": False,
            "sgd_minibatch_size": 2000,
            "model": {
                "custom_model": "cc_rnn_model",
                "max_seq_len": 20,
                "custom_options": {
                    "topology": topology,
                }
            },
            "callbacks": {
                "on_episode_start": on_episode_start,
                "on_episode_step": on_episode_step,
                "on_episode_end": on_episode_end
            }
        },
        num_samples=1 if args.debug else num_samples,
    )
