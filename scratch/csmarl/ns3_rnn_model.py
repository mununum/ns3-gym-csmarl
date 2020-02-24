import os
import gym
from gym.spaces import Discrete
import numpy as np
import random
import argparse

import ray
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.recurrent_tf_modelv2 import RecurrentTFModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils import try_import_tf

from ns3multiagentenv import Ns3MultiAgentEnv, on_episode_start, on_episode_step, on_episode_end

tf = try_import_tf()


class MyKerasRNN(RecurrentTFModelV2):

    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 hiddens_size=256,
                 cell_size=64):
        super(MyKerasRNN, self).__init__(obs_space, action_space, num_outputs,
                                         model_config, name)
        self.cell_size = cell_size

        # print("CUDA_VISIBLE_DEVICES: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
        # print("ray.get_gpu_ids(): {}".format(ray.get_gpu_ids()))

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
        values = tf.keras.layers.Dense(
            1, activation=None, name="values")(lstm_out)

        # Create the RNN model
        self.rnn_model = tf.keras.Model(
            inputs=[input_layer, seq_in, state_in_h, state_in_c],
            outputs=[logits, values, state_h, state_c])
        self.register_variables(self.rnn_model.variables)
        self.rnn_model.summary()  # dump the model info

    @override(RecurrentTFModelV2)
    def forward_rnn(self, inputs, state, seq_lens):
        # state: [h, c]
        model_out, self._value_out, h, c = self.rnn_model([inputs, seq_lens] +
                                                          state)
        return model_out, [h, c]

    @override(ModelV2)
    def get_initial_state(self):
        # initial [h, c]
        return [
            np.zeros(self.cell_size, np.float32),
            np.zeros(self.cell_size, np.float32),
        ]

    @override(ModelV2)
    def value_function(self):
        return tf.reshape(self._value_out, [-1])


if __name__ == "__main__":
    ray.init(log_to_driver=False)

    ModelCatalog.register_custom_model("rnn", MyKerasRNN)

    cwd = os.path.dirname(os.path.abspath(__file__))

    tune.run(
        "PPO",
        # stop={"training_iteration": 1000},
        stop={"timesteps_total": 3e8},
        config={
            "env": Ns3MultiAgentEnv,
            "batch_mode": "complete_episodes",
            "log_level": "DEBUG",
            "env_config": {
                "n_agents": 12,
                "cwd": cwd,
                "debug": False,
                "reward": "shared",
                "topology": "fc",
            },
            "num_workers": 16,
            # "num_gpus": 4,
            "num_gpus_per_worker": 0.25,
            "sgd_minibatch_size": 4000,
            "model": {
                "custom_model": "rnn",
                "max_seq_len": 20,
            },
            "callbacks": {
                "on_episode_start": on_episode_start,
                "on_episode_step": on_episode_step,
                "on_episode_end": on_episode_end,
            }
        }
    )