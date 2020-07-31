import os
import gym
from gym.spaces import Discrete
import numpy as np
import networkx as nx
import argparse
from collections import defaultdict
from typing import Dict

import ray
from ray import tune

from ray.rllib.agents.ppo.ppo import PPOTrainer
from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy, KLCoeffMixin, PPOLoss
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.evaluation.postprocessing import compute_advantages, Postprocessing
from ray.rllib.models import ModelCatalog
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.tf_policy import LearningRateSchedule, EntropyCoeffSchedule
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.recurrent_net import RecurrentNetwork
from ray.rllib.utils.explained_variance import explained_variance
from ray.rllib.utils.tf_ops import make_tf_callable
from ray.rllib.utils.annotations import override
from ray.rllib.utils import try_import_tf

from ns3_multiagent_env import Ns3MultiAgentEnv
from link_graph import read_graph

tf = try_import_tf()

NEIGHBOR_NUM = "neighbor_num"
NEIGHBOR_OBS = "neighbor_obs"
NEIGHBOR_ACT = "neighbor_act"
SEGMENT_IDS = "segment_ids"


class GraphConvolutionalCriticRNNModel(RecurrentNetwork):

    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 topology="fim",
                 hiddens_size=256,
                 cell_size=64):
        super(GraphConvolutionalCriticRNNModel, self).__init__(obs_space, action_space, num_outputs,
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

        obs_dense = tf.keras.layers.Dense(
            hiddens_size, activation=tf.nn.tanh, name="obs_dense")(obs)

        neighbor_feat = tf.keras.layers.Concatenate(
            axis=-1, name="neighbor_feat")([neighbor_obs, neighbor_act])
        neighbor_dense = tf.keras.layers.Dense(
            hiddens_size, activation=tf.nn.tanh, use_bias=False, name="neighbor_dense")(neighbor_feat)

        neighbor_sum = tf.math.reduce_sum(neighbor_dense, axis=-2, name="neighbor_sum")

        # conditional expression considering island nodes
        gnn_vf_in = tf.cond(neighbor_num[0][0] > 0, lambda: obs_dense + neighbor_sum / neighbor_num, lambda: obs_dense, name="gnn_vf_in")
        gnn_vf_dense = tf.keras.layers.Dense(
            hiddens_size, activation=tf.nn.tanh, name="gnn_vf_dense")(gnn_vf_in)
        gnn_vf_out = tf.keras.layers.Dense(
            1, activation=None, name="gnn_vf_out")(gnn_vf_dense)

        self.gnn_vf = tf.keras.Model(
            inputs=[obs, neighbor_num, neighbor_obs, neighbor_act], outputs=gnn_vf_out)
        self.register_variables(self.gnn_vf.variables)

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
    "rnn_gnn_model", GraphConvolutionalCriticRNNModel)


class GraphConvolutionalValueMixin:
    """Add method to evaluate the gnn value function from the model."""

    def __init__(self):
        # this function call runs within tf session
        self.compute_gnn_vf = make_tf_callable(self.get_session(), dynamic_shape=True)(
            self.model.gnn_value_function)


def graph_convolutional_critic_postprocessing(policy, sample_batch, other_agent_batches=None, episode=None):

    n_agents = policy.model.n_agents

    if not policy.loss_initialized():
        # policy hasn't initialized yet, use zeros

        sample_batch[NEIGHBOR_NUM] = np.zeros((1, 1), dtype=np.float32)

        obs_dim = sample_batch[SampleBatch.CUR_OBS].shape[-1]

        sample_batch[NEIGHBOR_OBS] = np.zeros((1, n_agents, obs_dim), dtype=np.float32)
        sample_batch[NEIGHBOR_ACT] = np.zeros((1, n_agents), dtype=np.int32)

        sample_batch[SampleBatch.VF_PREDS] = np.zeros_like(sample_batch[SampleBatch.ACTIONS], dtype=np.float32)

    else:
        assert sample_batch["dones"][-1], "Not implemented for train_batch_mode=truncated_episodes"
        assert other_agent_batches is not None

        # find the id that is not in other_agent_batches
        agent_id = [i for i in range(n_agents) if i not in other_agent_batches.keys()]
        assert len(agent_id) == 1
        agent_id = agent_id[0]

        # all_batches: {agent_index: sample_batch} dict
        # discard policy object
        all_batches = {k: v for k, (_, v) in other_agent_batches.items()}
        all_batches[agent_id] = sample_batch
        assert len(all_batches.keys()) == n_agents

        # identify neighbors and its features from the graph
        G = episode.user_data["graph"]  # networkx graph

        # MYNOTE reward separation can also happen here ...

        batch_size = sample_batch[SampleBatch.CUR_OBS].shape[0]
        obs_dim = sample_batch[SampleBatch.CUR_OBS].shape[-1]

        sample_batch[NEIGHBOR_NUM] = np.ones((batch_size, 1), dtype=np.float32) * len(G[agent_id])

        sample_batch[NEIGHBOR_OBS] = np.zeros(((batch_size, n_agents, obs_dim)), dtype=np.float32)
        sample_batch[NEIGHBOR_ACT] = np.zeros(((batch_size, n_agents)), dtype=np.int32)

        for i, n in enumerate(G[agent_id]):
            sample_batch[NEIGHBOR_OBS][:,i,:] = all_batches[n][SampleBatch.CUR_OBS]
            sample_batch[NEIGHBOR_ACT][:,i] = all_batches[n][SampleBatch.ACTIONS]

        # forward-propagate the value network for prediction
        sample_batch[SampleBatch.VF_PREDS] = policy.compute_gnn_vf(
            sample_batch[SampleBatch.CUR_OBS], sample_batch[NEIGHBOR_NUM],
            sample_batch[NEIGHBOR_OBS], sample_batch[NEIGHBOR_ACT])

    # Run GAE to compute advantages
    # VF_PREDS are used in here
    train_batch = compute_advantages(
        sample_batch,
        0.0,
        policy.config["gamma"],
        policy.config["lambda"],
        use_gae=policy.config["use_gae"])
    return train_batch


# Graph initialization: this function is called only once
def loss_with_graph_convolutional_critic(policy, model, dist_class, train_batch):
    GraphConvolutionalValueMixin.__init__(policy)

    # gnn_value_function: computation graph connection
    # compute_gnn_vf: invokes a session for actual execution

    logits, state = model.from_batch(train_batch)
    action_dist = dist_class(logits, model)

    # computation graph for value calculation
    policy.gnn_value_out = policy.model.gnn_value_function(
        train_batch[SampleBatch.CUR_OBS], train_batch[NEIGHBOR_NUM],
        train_batch[NEIGHBOR_OBS], train_batch[NEIGHBOR_ACT])

    if state:
        max_seq_len = tf.reduce_max(train_batch["seq_lens"])
        mask = tf.sequence_mask(train_batch["seq_lens"], max_seq_len)
        mask = tf.reshape(mask, [-1])
    else:
        mask = tf.ones_like(
            train_batch[Postprocessing.ADVANTAGES], dtype=tf.bool)

    # VALUE_TARGETS = VF_PREDS + ADVANTAGES
    policy.loss_obj = PPOLoss(
        dist_class,
        model,
        train_batch[Postprocessing.VALUE_TARGETS],
        train_batch[Postprocessing.ADVANTAGES],
        train_batch[SampleBatch.ACTIONS],
        train_batch[SampleBatch.ACTION_DIST_INPUTS],
        train_batch[SampleBatch.ACTION_LOGP],
        train_batch[SampleBatch.VF_PREDS],
        action_dist,
        policy.gnn_value_out,
        policy.kl_coeff,
        mask,
        entropy_coeff=policy.entropy_coeff,
        clip_param=policy.config["clip_param"],
        vf_clip_param=policy.config["vf_clip_param"],
        vf_loss_coeff=policy.config["vf_loss_coeff"],
        use_gae=policy.config["use_gae"])

    return policy.loss_obj.loss


def setup_mixins(policy, obs_space, action_space, config):
    # Copied from PPO
    KLCoeffMixin.__init__(policy, config)
    EntropyCoeffSchedule.__init__(policy, config["entropy_coeff"],
                                  config["entropy_coeff_schedule"])
    LearningRateSchedule.__init__(policy, config["lr"], config["lr_schedule"])


def gnn_vf_stats(policy, train_batch, grads):
    # Report the explained variance of the gnn value function
    return {
        "vf_explained_var": explained_variance(
            train_batch[Postprocessing.VALUE_TARGETS],
            policy.gnn_value_out),
    }

def step_callback(trainer, fetches):
    # things to do in a master thread
    print("step callback")


class MyCallbacks(DefaultCallbacks):

    def on_episode_start(self, worker: RolloutWorker, base_env: BaseEnv,
                         policies: Dict[str, Policy],
                         episode: MultiAgentEpisode, **kwargs):
        env = base_env.envs[0]
        episode.user_data["graph"], _ = read_graph(env.topology)
        episode.user_data["stat"] = defaultdict(list)

    def on_episode_step(self, worker: RolloutWorker, base_env: BaseEnv,
                        episode: MultiAgentEpisode, **kwargs):
        n_agents = base_env.envs[0].n_agents

        # process per_agent info
        for i in range(n_agents):
            for k, v in episode.last_info_for(i).items():
                if k != "common":
                    episode.user_data["stat"][k+"_"+str(i)].append(v)

        # process common info
        for k, v in episode.last_info_for(0).get("common", {}).items():
            if k != "state":
                episode.user_data["stat"][k].append(v)

    def on_episode_end(self, worker: RolloutWorker, base_env: BaseEnv,
                       policies: Dict[str, Policy], episode: MultiAgentEpisode,
                       **kwargs):
        for k, v in episode.user_data["stat"].items():
            if "loss_rate" in k:
                # MYTODO np.mean is not technically correct at this point
                agg = np.mean(v)
            else:
                agg = np.sum(v)
            episode.custom_metrics[k] = agg


GNNPPO = PPOTFPolicy.with_updates(
    name="GNNPPO",
    postprocess_fn=graph_convolutional_critic_postprocessing,
    loss_fn=loss_with_graph_convolutional_critic,
    before_loss_init=setup_mixins,
    grad_stats_fn=gnn_vf_stats,
    mixins=[
        LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin,
        GraphConvolutionalValueMixin
    ]
)

def get_policy_class(config):
    return GNNPPO

GNNPPOTrainer = PPOTrainer.with_updates(
    name="GNNPPOTrainer",
    default_policy=GNNPPO,
    get_policy_class=get_policy_class,
    after_optimizer_step=step_callback,
)

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

    num_workers_total = num_workers * len(params_list) * num_samples  # <= 32 is recommended
    num_gpus_per_worker = NUM_GPUS / num_workers_total

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
                "custom_model": "rnn_gnn_model",
                "max_seq_len": 20,
                "custom_model_config": {
                    "topology": topology,
                }
            },
            "callbacks": MyCallbacks,
        },
        num_samples=num_samples,
        checkpoint_freq=10,
        keep_checkpoints_num=1,
        checkpoint_score_attr="episode_reward_mean",
    )