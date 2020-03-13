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

        ### RECURRENT POLICY NETWORK ###

        # Define input layers
        input_layer = tf.keras.layers.Input(
            shape=(None, obs_space.shape[0]), name="inputs")
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
        self.rnn_model = tf.keras.Model(
            inputs=[input_layer, seq_in, state_in_h, state_in_c],
            outputs=[logits, values, state_h, state_c])
        self.register_variables(self.rnn_model.variables)
        self.rnn_model.summary()  # dump the model info

        ### VALUE NETWORK ###

        self.n_agents_in_critic = model_config["custom_options"]["n_agents_in_critic"]
        self.obs_dim = obs_space.shape[0]
        self.act_dim = self.num_outputs

        obs = tf.keras.layers.Input(
            shape=(self.obs_dim,), name="obs")
        other_obs = tf.keras.layers.Input(
            shape=(self.obs_dim * (self.n_agents_in_critic - 1), ), name="other_obs")
        other_act = tf.keras.layers.Input(
            shape=(self.act_dim * (self.n_agents_in_critic - 1), ), name="other_act")
        # MYTODO make proper mapping on agent_id
        agent_id = tf.keras.layers.Input(
            shape=(self.n_agents_in_critic,), name="agent_id")

        # NN definition
        # XXX no agent_id here
        concat_input = tf.keras.layers.Concatenate(
            axis=1)([obs, other_obs, other_act])
        central_vf_dense = tf.keras.layers.Dense(
            hiddens_size, activation=tf.nn.tanh, name="c_vf_dense")(concat_input)
        central_vf_out = tf.keras.layers.Dense(
            1, activation=None, name="c_vf_out")(central_vf_dense)
        self.central_vf = tf.keras.Model(
            inputs=[obs, other_obs, other_act, agent_id], outputs=central_vf_out)
        self.register_variables(self.central_vf.variables)

    @override(RecurrentTFModelV2)
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
            np.zeros(self.cell_size, np.float32),
        ]

    def central_value_function(self, obs, other_obs, other_actions, agent_id):
        other_actions_onehot = tf.reshape(
            tf.one_hot(other_actions, self.act_dim), [-1, self.act_dim * (self.n_agents_in_critic - 1)])
        agent_id_onehot = tf.one_hot(agent_id, self.n_agents_in_critic)
        return tf.reshape(
            self.central_vf([obs, other_obs, other_actions_onehot, agent_id_onehot]), [-1])

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

    ModelCatalog.register_custom_model(
        "cc_rnn_model", CentralizedCriticRNNModel)

    config_params = [0]
    env_config = {  # environment configuration
        "n_agents": 3,
        "cwd": cwd,
        "debug": args.debug,
        "reward": "shared",
        "topology": "fim",
    }
    # config_params = [FILL]  # for env config testing
    # env_config = tune.grid_search(config_params)


    NUM_GPUS = 4
    num_workers = 8
    params_list = [0]
    # params_list = [FILL]  # for parameter testing
    num_samples = 2
    num_workers_total = num_workers * \
        len(params_list) * len(config_params) * num_samples  # <= 32 is recommended
    num_gpus_per_worker = NUM_GPUS / num_workers_total

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
            "sgd_minibatch_size": 2000,
            "model": {
                "custom_model": "cc_rnn_model",
                "custom_options": {
                    "n_agents_in_critic": 3,  # n_agents in critic
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
