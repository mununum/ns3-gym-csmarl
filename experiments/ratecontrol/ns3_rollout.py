import argparse
import collections
import copy
import glob
import json
import os
import pickle

import gym
import ray

from ray.rllib.env import MultiAgentEnv
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.utils.spaces.space_utils import flatten_to_single_ndarray
from ray.tune.registry import get_trainable_cls
from ray.tune.utils import merge_dicts

import ns3_rnn_model
import ns3_gnn_model
import ns3_gnn_peraction

def prepare(args, parser):
    config = {}
    # Load configuration from checkpoint file.
    config_dir = os.path.dirname(args.checkpoint)
    config_path = os.path.join(config_dir, "params.pkl")
    # Try parent directory.
    if not os.path.exists(config_path):
        config_path = os.path.join(config_dir, "../params.pkl")

    # If no pkl file found, require command line `--config`.
    if not os.path.exists(config_path):
        if not args.config:
            raise ValueError(
                "Could not find params.pkl in either the checkpoint dir or "
                "its parent directory AND no config given on command line!")
    
    # Load the config from pickled.
    else:
        with open(config_path, "rb") as f:
            config = pickle.load(f)

    # Set num_workers to be at least 2.

    # Merge with `evaluation_config`.
    evaluation_config = copy.deepcopy(config.get("evaluation_config", {}))
    config = merge_dicts(config, evaluation_config)
    # Merge with command line `--config` settings.
    config = merge_dicts(config, args.config)
    if not args.env:
        if not config.get("env"):
            parser.error("the following arguments are required: --env")
        args.env = config.get("env")

    ray.init()

    if args.run == "GNNPPOTrainer":
        cls = ns3_gnn_model.GNNPPOTrainer
    elif args.run == "GNNPerActionPPOTrainer":
        cls = ns3_gnn_peraction.GNNPerActionPPOTrainer
    else:
        cls = get_trainable_cls(args.run)
    
    agent = cls(env=args.env, config=config)
    # Load state from checkpoint.
    agent.restore(args.checkpoint)

    return agent


class DefaultMapping(collections.defaultdict):
    """default_factory now takes as an argument the missing key."""

    def __missing__(self, key):
        self[key] = value = self.default_factory(key)
        return value


def default_policy_agent_mapping(unused_agent_id):
    return DEFAULT_POLICY_ID

def keep_going(steps, num_steps, episodes, num_episodes):
    """Determine whether we've collected enough data"""
    # if num_episodes is set, this overrides num_steps
    if num_episodes:
        return episodes < num_episodes
    # if num_steps is set, continue until we reach the limit
    if num_steps:
        return steps < num_steps
    # oterhwise keep going forever
    return True


def rollout(agent,
            env_name,
            num_episodes=1,
            newargs={}):
    policy_agent_mapping = default_policy_agent_mapping

    env = agent.workers.local_worker().env
    multiagent = isinstance(env, MultiAgentEnv)

    policy_map = agent.workers.local_worker().policy_map
    state_init = {p: m.get_initial_state() for p, m in policy_map.items()}
    use_lstm = {p: len(s) > 0 for p, s in state_init.items()}
    action_init = {
        p: flatten_to_single_ndarray(m.action_space.sample())
        for p, m in policy_map.items()
    }

    steps = 0
    episodes = 0
    while episodes < num_episodes:
        mapping_cache = {}  # in case policy_agent_agent_mapping is stochastic
        obs = env.reset(newargs)

        agent_states = DefaultMapping(
            lambda agent_id: state_init[mapping_cache[agent_id]])
        prev_actions = DefaultMapping(
            lambda agent_id: action_init[mapping_cache[agent_id]])
        prev_rewards = collections.defaultdict(lambda: 0.)

        done = False
        reward_total = 0.0
        while not done:
            multi_obs = obs
            action_dict = {}
            for agent_id, a_obs in multi_obs.items():
                if a_obs is not None:
                    policy_id = mapping_cache.setdefault(
                        agent_id, policy_agent_mapping(agent_id))
                    p_use_lstm = use_lstm[policy_id]
                    if p_use_lstm:
                        a_action, p_state, _ = agent.compute_action(
                            a_obs,
                            state=agent_states[agent_id],
                            prev_action=prev_actions[agent_id],
                            prev_reward=prev_rewards[agent_id],
                            policy_id=policy_id)
                        agent_states[agent_id] = p_state
                    else:
                        a_action = agent.compute_action(
                            a_obs,
                            prev_action=prev_actions[agent_id],
                            prev_reward=prev_rewards[agent_id],
                            policy_id=policy_id)
                    a_action = flatten_to_single_ndarray(a_action)
                    action_dict[agent_id] = a_action
                    prev_actions[agent_id] = a_action
            action = action_dict

            next_obs, reward, done, info = env.step(action)
            if multiagent:
                for agent_id, r in reward.items():
                    prev_rewards[agent_id] = r

            if multiagent:
                done = done["__all__"]
                reward_total += sum(reward.values())
            steps += 1
            obs = next_obs
        # print("Episode #{}: reward: {}".format(episodes, reward_total))
        print("Episode #{} done".format(episodes))
        if done:
            episodes += 1
    
    env.close()


# def cleanup(agent):
#     env = agent.workers.local_worker().env


def load_checkpoint(exp_str):

    alg = exp_str.split("_")[0]

    exp_str_pattern = "~/ray_results/" + alg + "/" + exp_str + "*"
    path = glob.glob(os.path.expanduser(exp_str_pattern))
    if len(path) != 1:
        raise ValueError("ambiguous path")
    path = path[0]
    chp_pattern = path + "/checkpoint*"
    checkpoint_files = glob.glob(chp_pattern)

    r = [int(f.split('/')[-1].split('_')[-1]) for f in checkpoint_files]
    chpno = max(r)

    checkpoint = path+"/checkpoint_"+str(chpno)+"/checkpoint-"+str(chpno)

    return alg, checkpoint

if __name__ == "__main__":
    
    exp_str = "FILL_HERE"

    parser = argparse.ArgumentParser()

    parser.add_argument("--run", type=str)
    parser.add_argument("--env", type=str)
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--config", default="{}", type=json.loads)
    parser.add_argument("--topology", type=str, default=None)
    # parser.add_argument("--topology2", type=str, default=None)

    args = parser.parse_args()

    args.run, args.checkpoint = load_checkpoint(exp_str)

    args.config = {
        "num_workers": 0,
        "env_config": {
            "debug": True,
            # "simName": "csmarl_dynamic",
            # "toplogy2": None,
        },
        "multiagent": {
            "policies_to_train": ["nothing"]
        },
        "explore": True,
        "evaluation_interval": None,
    }

    if args.topology is not None:
        print("using custom topology", args.topology)
        args.config["env_config"]["topology"] = args.topology

    # load checkpoint
    agent = prepare(args, parser)

    # Do the actual rollout.
    rollout(agent, args.env, num_episodes=args.episodes)

    # Intensity experiment.
    # newargs = {"--intensity": 0.1}
    # for i in range(10):
    #     rollout(agent, args.env, newargs=newargs)
    #     newargs["--intensity"] += 0.1