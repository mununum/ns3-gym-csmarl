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
from ray.rllib.utils.annotations import override
from ray.rllib.utils import try_import_tf

from ns3_multiagent_env import on_episode_start, on_episode_step, on_episode_end

tf = try_import_tf()


class Ns3RNNModel(RecurrentTFModelV2):

    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 hiddens_size=256,
                 cell_size=64):
        super(Ns3RNNModel, self).__init__(obs_space, action_space, num_outputs,
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


ModelCatalog.register_custom_model("ns3_rnn_model", Ns3RNNModel)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stop", help="number of timesteps, default 3e8", type=int, default=3e8)
    parser.add_argument(
        "--debug", help="debug indicator, default false", type=bool, default=False)

    args = parser.parse_args()

    ray.init(log_to_driver=args.debug)

    cwd = os.path.dirname(os.path.abspath(__file__))
    topology = "complex"

    NUM_GPUS = 4
    num_workers = 16

    if args.debug:
        params_list = [0]
    else:
        params_list = [0]
        # params_list = [5e-4, 5e-5, 5e-6, 5e-7]  # for parameter testing

    num_samples = 1
    
    num_workers_total = num_workers * len(params_list) * num_samples # <= 32 is recommended
    num_gpus_per_worker = NUM_GPUS / num_workers_total

    tune.run(
        "PPO",
        stop={
            "timesteps_total": args.stop
        },
        config={
            "env": "ns3_multiagent_env",
            "batch_mode": "complete_episodes",
            "log_level": "DEBUG" if args.debug else "WARN",
            "env_config": {
                "cwd": cwd,
                "debug": args.debug,
                "reward": "shared",
                "topology": topology,
                "traffic": "cbr",
                "fixedFlow": False,
            },
            "num_workers": 0 if args.debug else num_workers,
            "num_gpus_per_worker": num_gpus_per_worker,
            "lr": 5e-4,
            # "lr": 5e-4 if args.debug else tune.grid_search(params_list),
            "use_gae": True,
            "sgd_minibatch_size": 2000,  # For maximum parallelism, MYTODO check whether suboptimality happens because of this
            "model": {
                "custom_model": "ns3_rnn_model",
                "max_seq_len": 20,
            },
            "callbacks": {
                "on_episode_start": on_episode_start,
                "on_episode_step": on_episode_step,
                "on_episode_end": on_episode_end,
            }
        },
        num_samples=1 if args.debug else num_samples,
        checkpoint_freq=10,
        keep_checkpoints_num=1,
        checkpoint_score_attr="episode_reward_mean",
    )