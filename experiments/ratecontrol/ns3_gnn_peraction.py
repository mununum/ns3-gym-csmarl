import argparse
import networkx as nx
import numpy as np

import ray
from ray import tune

from ray.rllib.agents.ppo.ppo import PPOTrainer
from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy, KLCoeffMixin
from ray.rllib.models import ModelCatalog
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.recurrent_net import RecurrentNetwork
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.tf_policy import LearningRateSchedule, EntropyCoeffSchedule
from ray.rllib.utils import merge_dicts
from ray.rllib.utils.annotations import override
from ray.rllib.utils.explained_variance import explained_variance
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.tf_ops import make_tf_callable

import common
from ns3_import import import_graph_module
from ns3_multiagent_env import MyCallbacks, ns3_execution_plan
from peraction_utils import peraction_compute_advantages, PPOPerActionLoss, ns3_peraction_execution_plan

graph = import_graph_module()

tf = try_import_tf()

ALL_OBS = "all_obs"
ALL_ACT = "all_act"
ADJ = "adj"


class GraphConvLayer(tf.keras.layers.Layer):
    def __init__(
            self,
            input_dim,
            output_dim,
            activation=None,
            use_bias=False,
            name="graph_conv"):
        super(GraphConvLayer, self).__init__(name=name)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        self.use_bias = use_bias

        with tf.variable_scope(self.name):
            self.w = tf.get_variable(
                name="w",
                shape=(self.input_dim, self.output_dim),
                initializer=tf.initializers.glorot_uniform())

            if self.use_bias:
                self.b = tf.get_variable(
                    name="b",
                    initializer=tf.constant(0.1, shape=(self.output_dim,)))

    def call(self, adj_norm, x):
        x = tf.matmul(x, self.w)    # XW
        x = tf.matmul(adj_norm, x)  # AXW

        if self.use_bias:
            x = tf.add(x, self.b)   # AXW + B
        
        if self.activation is not None:
            x = self.activation(x)  # activation(AXW + B)
        
        return x


class Ns3GNNPerActionModel(RecurrentNetwork):

    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 hiddens_size=256,
                 cell_size=64,
                 n_agents=3):
        super(Ns3GNNPerActionModel, self).__init__(
            obs_space, action_space, num_outputs, model_config, name)
        self.cell_size = cell_size

        self.n_agents = n_agents
        self.obs_dim = obs_space.shape[0]
        self.act_0_dim = action_space.spaces[0].n
        self.act_1_dim = action_space.spaces[1].n

        # policy network
        self.rnn_model = self._policy_network(hiddens_size, cell_size)
        self.register_variables(self.rnn_model.variables)

        # value network
        self.gnn_vf = self._value_network(32)
        self.register_variables(self.gnn_vf.variables)

    def _policy_network(self, hiddens_size, cell_size):

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

        # Create RNN model
        return tf.keras.Model(
            inputs=[inputs, seq_in, state_in_h, state_in_c],
            outputs=[logits, values, state_h, state_c])

    def _value_network(self, hiddens_size):

        all_obs = tf.keras.layers.Input(
            shape=(self.n_agents, self.obs_dim), name="all_obs")  # (B, N, O)
        all_act = tf.keras.layers.Input(
            shape=(self.n_agents, self.num_outputs), name="all_act")  # (B, N, A)
        adj = tf.keras.layers.Input(
            shape=(self.n_agents, self.n_agents), name="adj")  # (B, N, N)

        other_feat = tf.concat(
            [all_obs[:, 1:, :], all_act[:, 1:, :]], axis=-1)  # (B, N-1, O+A)
        other_dense = tf.keras.layers.Dense(
            hiddens_size, activation=tf.nn.tanh, name="other_dense")(other_feat)  # (B, N-1, H)
        
        # exclude my actions for counterfactual calculation
        my_act_0 = all_act[:, :1, :self.act_0_dim]  # (B, 1, A0)
        my_act_1 = all_act[:, :1, self.act_0_dim:]  # (B, 1, A1)
        my_obs_and_act_0 = tf.concat([all_obs[:, :1, :], my_act_0], axis=-1)  # (B, 1, O+A0)
        my_obs_and_act_1 = tf.concat([all_obs[:, :1, :], my_act_1], axis=-1)  # (B, 1, O+A1)

        # per-action counterfactual
        obs_dense_0 = tf.keras.layers.Dense(
            hiddens_size, activation=tf.nn.tanh, name="obs_dense_0")(my_obs_and_act_1)  # (B, 1, H)
        obs_dense_1 = tf.keras.layers.Dense(
            hiddens_size, activation=tf.nn.tanh, name="obs_dense_1")(my_obs_and_act_0)  # (B, 1, H)

        # gather features for graph operation
        gnn_vf_in_0 = tf.concat([obs_dense_0, other_dense], axis=1)  # (B, N, H)
        gnn_vf_in_1 = tf.concat([obs_dense_1, other_dense], axis=1)  # (B, N, H)

        # GCN operation
        gnn_vf_dense1_0 = GraphConvLayer(
                            input_dim=hiddens_size,
                            output_dim=hiddens_size,
                            name="gcn1_0",
                            activation=tf.nn.tanh)(adj, gnn_vf_in_0)  # (B, N, H)
        
        gnn_vf_dense2_0 = GraphConvLayer(
                            input_dim=hiddens_size,
                            output_dim=hiddens_size,
                            name="gcn2_0",
                            activation=tf.nn.tanh)(adj, gnn_vf_dense1_0)  # (B, N, H)

        gnn_vf_dense1_1 = GraphConvLayer(
                            input_dim=hiddens_size,
                            output_dim=hiddens_size,
                            name="gcn1_1",
                            activation=tf.nn.tanh)(adj, gnn_vf_in_1)  # (B, N, H)
        
        gnn_vf_dense2_1 = GraphConvLayer(
                            input_dim=hiddens_size,
                            output_dim=hiddens_size,
                            name="gcn2_1",
                            activation=tf.nn.tanh)(adj, gnn_vf_dense1_1)  # (B, N, H)

        gnn_vf_concat_0 = tf.concat([gnn_vf_in_0, gnn_vf_dense1_0, gnn_vf_dense2_0], axis=-1)  # (B, N, H*3)
        gnn_vf_concat_1 = tf.concat([gnn_vf_in_1, gnn_vf_dense1_1, gnn_vf_dense2_1], axis=-1)  # (B, N, H*3)

        gnn_vf_agg_0 = tf.keras.layers.Dense(
            hiddens_size, activation=tf.nn.tanh, name="gnn_vf_agg_0")(gnn_vf_concat_0)  # (B, N, H)
        gnn_vf_agg_0 = tf.reduce_sum(gnn_vf_agg_0, axis=1)  # (B, H)

        gnn_vf_agg_1 = tf.keras.layers.Dense(
            hiddens_size, activation=tf.nn.tanh, name="gnn_vf_agg_1")(gnn_vf_concat_1)  # (B, N, H)
        gnn_vf_agg_1 = tf.reduce_sum(gnn_vf_agg_1, axis=1)  # (B, H)

        gnn_vf_out_0 = tf.keras.layers.Dense(
            1, activation=None, name="gnn_vf_out_0")(gnn_vf_agg_0)
        gnn_vf_out_1 = tf.keras.layers.Dense(
            1, activation=None, name="gnn_vf_out_1")(gnn_vf_agg_1)
        
        return tf.keras.Model(
            inputs=[all_obs, all_act, adj], outputs=[gnn_vf_out_0, gnn_vf_out_1])

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

    def gnn_peraction_value_function(self, all_obs, all_act, adj):
        all_act_onehot_0 = tf.one_hot(all_act[:, :, 0], self.act_0_dim)
        all_act_onehot_1 = tf.one_hot(all_act[:, :, 1], self.act_1_dim)
        all_act_onehot = tf.concat([all_act_onehot_0, all_act_onehot_1], axis=-1)

        values_0, values_1 = self.gnn_vf([all_obs, all_act_onehot, adj])

        return tf.reshape(values_0, [-1]), tf.reshape(values_1, [-1])

    @override(ModelV2)
    def value_function(self):
        return tf.reshape(self._dummy_value_out, [-1])


class GNNPerActionValueMixin:

    def __init__(self):
        self.compute_gnn_peraction_vf = make_tf_callable(self.get_session())(
            self.model.gnn_peraction_value_function)


def gnn_peraction_critic_postprocessing(policy, sample_batch, other_agent_batches=None, episode=None):

    n_agents = policy.model.n_agents
    obs_dim = sample_batch[SampleBatch.CUR_OBS].shape[-1]
    act_dim = sample_batch[SampleBatch.ACTIONS].shape[-1]

    if not policy.loss_initialized():

        sample_batch[ALL_OBS] = np.zeros(
            (1, n_agents, obs_dim), dtype=np.float32)
        sample_batch[ALL_ACT] = np.zeros((1, n_agents, act_dim), dtype=np.int32)
        sample_batch[ADJ] = np.zeros((1, n_agents, n_agents), dtype=np.float32)

        sample_batch["values_0"] = np.zeros_like(
            sample_batch[SampleBatch.REWARDS], dtype=np.float32)
        sample_batch["values_1"] = np.zeros_like(
            sample_batch[SampleBatch.REWARDS], dtype=np.float32)
    
    else:
        assert sample_batch["dones"][-1], "Not implemented for train_batch_mode=truncated_episodes"
        assert other_agent_batches is not None

        # find the id that is not in other_agent_batches
        agent_id = [i for i in range(n_agents)
            if i not in other_agent_batches.keys()]
        assert len(agent_id) == 1
        agent_id = agent_id[0]

        # other_batches: {agent_idx: sample_batch} dict
        # discard policy object
        other_batches = {k: v for k, (_, v) in other_agent_batches.items()}
        assert len(other_batches) == n_agents - 1

        G = episode.user_data["graph"]

        batch_size = sample_batch[SampleBatch.CUR_OBS].shape[0]

        sample_batch[ALL_OBS] = np.zeros(
            (batch_size, n_agents, obs_dim), dtype=np.float32)
        sample_batch[ALL_ACT] = np.zeros(
            (batch_size, n_agents, act_dim), dtype=np.int32)

        sample_batch[ALL_OBS][:, 0, :] = sample_batch[SampleBatch.CUR_OBS]
        for i, other_id in enumerate(other_batches):
            sample_batch[ALL_OBS][:, i+1, :] = other_batches[other_id][SampleBatch.CUR_OBS]
            sample_batch[ALL_ACT][:, i+1, :] = other_batches[other_id][SampleBatch.ACTIONS]

        # adjacency matrix
        adj = nx.adjacency_matrix(G, nodelist=[agent_id]+list(other_batches)).todense()
        adj = np.array(adj)

        # preprocess adjacency matrix
        adj_tilde = adj + np.identity(n_agents)
        d_tilde_diag = np.sum(adj_tilde, axis=1)
        d_tilde_inv_sqrt_diag = np.power(d_tilde_diag, -1/2)
        d_tilde_inv_sqrt = np.diag(d_tilde_inv_sqrt_diag)
        adj_norm = np.dot(np.dot(d_tilde_inv_sqrt, adj_tilde), d_tilde_inv_sqrt)

        sample_batch[ADJ] = np.array([adj_norm] * batch_size)

        sample_batch["values_0"], sample_batch["values_1"] = policy.compute_gnn_peraction_vf(
            sample_batch[ALL_OBS], sample_batch[ALL_ACT], sample_batch[ADJ])

    train_batch = peraction_compute_advantages(
        sample_batch,
        0.0,
        policy.config["gamma"],
        policy.config["lambda"],
        use_gae=policy.config["use_gae"])
    return train_batch


def loss_with_gnn_peraction_critic(policy, model, dist_class, train_batch):
    GNNPerActionValueMixin.__init__(policy)

    logits, state = model.from_batch(train_batch)
    action_dist = dist_class(logits, model)

    policy.gnn_peraction_value_out_0, policy.gnn_peraction_value_out_1 = \
        policy.model.gnn_peraction_value_function(train_batch[ALL_OBS], train_batch[ALL_ACT], train_batch[ADJ])
    
    mask = tf.ones_like(train_batch["advantages_0"], dtype=tf.bool)

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
        policy.gnn_peraction_value_out_0,
        policy.gnn_peraction_value_out_1,
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
            policy.gnn_peraction_value_out_0,
        ),
        "vf_explained_var_1": explained_variance(
            train_batch["value_targets_1"],
            policy.gnn_peraction_value_out_1,
        ),
        "kl_0": policy.loss_obj.mean_kl_0,
        "kl_1": policy.loss_obj.mean_kl_1,
        "kl": policy.loss_obj.mean_kl,
        "entropy_0": policy.loss_obj.mean_entropy_0,
        "entropy_1": policy.loss_obj.mean_entropy_1,
        "entropy_coeff": tf.cast(policy.entropy_coeff, tf.float64),
    }


GNNPerActionPPO = PPOTFPolicy.with_updates(
    name="GNNPerActionPPO",
    postprocess_fn=gnn_peraction_critic_postprocessing,
    loss_fn=loss_with_gnn_peraction_critic,
    before_loss_init=setup_mixins,
    stats_fn=peraction_stats,
    mixins=[
        LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin,
        GNNPerActionValueMixin
    ]
)

def get_policy_class(config):
    return GNNPerActionPPO

GNNPerActionPPOTrainer = PPOTrainer.with_updates(
    name="GNNPerActionPPOTrainer",
    default_policy=GNNPerActionPPO,
    get_policy_class=get_policy_class,
    execution_plan=ns3_peraction_execution_plan,
)

ModelCatalog.register_custom_model("gnn_peraction_model", Ns3GNNPerActionModel)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--topology", type=str, default="complex")
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--sigma", type=float, default=0)
    parser.add_argument("--id", type=int, default=0)
    args = parser.parse_args()

    ray.init(log_to_driver=args.debug, local_mode=args.debug, include_webui=False)

    config, stop = common.config(args)
    _, n_agents = graph.read_link_graph(args.topology)

    config["env_config"]["exp_name"] = __file__.split(".")[0]+"-"+str(args.id)
    config["model"] = {
        "custom_model": "gnn_peraction_model",
        "custom_model_config": {
            "n_agents": n_agents,
        }
    }

    tune.run(GNNPerActionPPOTrainer, config=config, stop=stop, checkpoint_freq=10, keep_checkpoints_num=1)