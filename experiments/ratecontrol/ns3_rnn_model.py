import argparse
import numpy as np

import ray
from ray import tune

from ray.rllib.agents.ppo.ppo import PPOTrainer
from ray.rllib.models import ModelCatalog
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.recurrent_net import RecurrentNetwork
from ray.rllib.utils import merge_dicts
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf

from ns3_import import import_graph_module
from ns3_multiagent_env import MyCallbacks, ns3_execution_plan

graph = import_graph_module()

tf = try_import_tf()

class Ns3RNNModel(RecurrentNetwork):

    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 hiddens_size=256,
                 cell_size=64,
                 n_agents=3):
        super(Ns3RNNModel, self).__init__(obs_space, action_space, num_outputs,
                                          model_config, name)
        self.cell_size = cell_size

        self.n_agents = n_agents
        self.obs_dim = obs_space.shape[0]
        self.act_dim = self.num_outputs

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

    @override(RecurrentNetwork)
    def forward_rnn(self, inputs, state, seq_lens):
        # state: [h, c]
        model_out, self._value_out, h, c = self.rnn_model(
            [inputs, seq_lens] + state)
        return model_out, [h, c]

    @override(ModelV2)
    def get_initial_state(self):
        # initial [h, c]
        return [
            np.zeros(self.cell_size, np.float32),
            np.zeros(self.cell_size, np.float32)
        ]

    @override(ModelV2)
    def value_function(self):
        return tf.reshape(self._value_out, [-1])

ModelCatalog.register_custom_model(
    "rnn_model", Ns3RNNModel
)

Ns3PPOTrainer = PPOTrainer.with_updates(
    execution_plan=ns3_execution_plan,
)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--topology", type=str, default="complex")
    args = parser.parse_args()

    ray.init(log_to_driver=args.debug, local_mode=args.debug)

    env_config = {
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
            "custom_model": "rnn_model",
            "custom_model_config": {
                "n_agents": n_agents,
            }
        },
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

    tune.run(Ns3PPOTrainer, config=config, stop=stop, checkpoint_freq=10, keep_checkpoints_num=1)