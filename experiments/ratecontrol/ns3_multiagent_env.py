import os
import sys
import gym
import numpy as np
from collections import defaultdict
from functools import partial

from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.agents.ppo.ppo import UpdateKL, warn_about_bad_reward_scales
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.execution.metric_ops import StandardMetricsReporting
from ray.rllib.execution.rollout_ops import ParallelRollouts, ConcatBatches, StandardizeFields, SelectExperiences
from ray.rllib.execution.train_ops import TrainOneStep, TrainTFMultiGPU
from ray.tune.registry import register_env
from ray.tune.utils import merge_dicts
from ns3gym import ns3env

from ns3_import import import_graph_module

class Ns3MultiAgentEnv(MultiAgentEnv):

    def __init__(self, env_config):

        graph = import_graph_module()

        simName = env_config.get("simName", "csmarl3")

        self.topology = env_config.get("topology", None)
        assert self.topology, "topology is needed to be specified"

        if simName == "csmarl3":
            # use link graph topology for training
            _, self.n_agents = graph.read_link_graph(self.topology)
        elif simName == "csmarl_test":
            if env_config["testArgs"]["--loss"] == "graph":
                if env_config["testArgs"]["--layout"] == "node":
                    _, self.n_agents = graph.read_node_graph(self.topology)
                elif env_config["testArgs"]["--layout"] == "link":
                    _, self.n_agents = graph.read_link_graph(self.topology)
            elif env_config["testArgs"]["--loss"] == "geometric":
                # "topology" argument must be the form of N,d
                if len(env_config["topology"].split(",")) != 2:
                    print("invalid configuration, \"topology\" argument must be the form of N,d")
                    sys.exit(1)
                self.n_agents = int(env_config["topology"].split(",")[0])
        port = 0

        d = os.path.dirname(os.path.abspath(__file__))
        cwd = os.path.join(d, "../../scratch/" + simName)

        simTime = env_config.get("simTime", None)
        stepTime = env_config.get("stepTime", None)
        intensity = env_config.get("intensity", None)
        self.debug = env_config.get("debug", False)

        # random topology generation
        self.exp_name = env_config.get("exp_name", "default")
        if simName == "csmarl3" and self.topology == "random":
            # random topology
            self.topology_file = self.topology + "-" + self.exp_name
            if env_config.worker_index == 0:
                graph.gen_link_graph(self.topology_file, sigma=env_config.get("sigma", 0))
        else:
            self.topology_file = self.topology

        simArgs = {
            "--topology": self.topology_file,
            "--simTime": simTime,
            "--stepTime": stepTime,
            "--debug": self.debug,
            "--algorithm": "rl",
        }

        # arguments for csmarl_test
        if simName == "csmarl_test":
            testArgs = env_config.get("testArgs", {})
            simArgs = merge_dicts(simArgs, testArgs)

        # remove the argument with None value
        for k in list(simArgs):
            if simArgs[k] is None:
                del simArgs[k]

        seed = env_config.get("seed", 0)

        self._env = ns3env.Ns3Env(port=port, startSim=True, simSeed=seed, simArgs=simArgs, debug=self.debug, cwd=cwd)

        self.multi_obs_space = self._env.observation_space
        self.multi_act_space = self._env.action_space

        self.observation_space = self.multi_obs_space.spaces[0]
        self.action_space = self.multi_act_space.spaces[0]

    def obs_to_dict(self, multi_obs):
        return {i: np.array(multi_obs[i]) for  i in range(self.n_agents)}

    def info_to_rew(self, info):
        pass

    def parse_info(self, info, obs):
        common_info = {}
        for kv in info.split():
            k, v = kv.split("=")
            common_info[k] = float(v)

        ret = {}
        for i in range(self.n_agents):
            o = obs[i]
            ret[i] = {
                "rate": o[0],
                "common": common_info
            }
        return ret

    def reset(self, newArgs={}):
        # overwrite simArgs if necessary
        oldArgs = {}
        for key in newArgs:
            if key in self._env.simArgs:
                oldArgs[key] = self._env.simArgs[key]
            self._env.simArgs[key] = newArgs[key]
        if newArgs is not {}:
            # Force restart the environment when simArgs is changed
            self._env.envDirty = True

        obs = self._env.reset()

        # restore old args
        for key in newArgs:
            if key in oldArgs:
                self._env.simArgs[key] = oldArgs[key]
            else:
                del self._env.simArgs[key]
        return self.obs_to_dict(obs)

    def step(self, action_dict):
        action = [action_dict[i] for i in range(self.n_agents)]
        o, r, d, i = self._env.step(action)
        obs = self.obs_to_dict(o)
        rew = {i: r for i in range(self.n_agents)}
        done = {"__all__": d}
        info = self.parse_info(i, obs)
        return obs, rew, done, info
    
    def close(self):
        self._env.close()


graph = import_graph_module()

class MyCallbacks(DefaultCallbacks):

    def on_episode_start(self, worker, base_env, policies, episode, **kwargs):
        env = base_env.envs[0]
        episode.user_data["graph"], _ = graph.read_link_graph(env.topology_file)
        episode.user_data["stat"] = defaultdict(list)

    def on_episode_step(self, worker, base_env, episode, **kwargs):
        env = base_env.envs[0]

        # process per_agent info
        for i in range(env.n_agents):
            for k, v in episode.last_info_for(i).items():
                if k != "common":
                    episode.user_data["stat"][k+"_"+str(i)].append(v)
        
        # process common info
        for k, v in episode.last_info_for(0).get("common", {}).items():
            episode.user_data["stat"][k].append(v)

    def on_episode_end(self, worker, base_env, policies, episode, **kwargs):
        for k, v in episode.user_data["stat"].items():
            episode.custom_metrics[k] = np.sum(v)

register_env("ns3_multiagent_env", lambda env_config: Ns3MultiAgentEnv(env_config))


def renew_graph(config, item):
    # change the topology in a master thread
    if config["env_config"]["topology"] == "random":
        exp_name = config["env_config"].get("exp_name", "default")
        topology_file = config["env_config"]["topology"] + "-" + exp_name
        graph.gen_link_graph(topology_file, sigma=config["env_config"].get("sigma", 0))

    return item


def ns3_execution_plan(workers, config):
    rollouts = ParallelRollouts(workers, mode="bulk_sync")

    # Collect large batches of relevant experiences & standardize.
    rollouts = rollouts.for_each(
        SelectExperiences(workers.trainable_policies()))
    rollouts = rollouts.combine(
        ConcatBatches(min_batch_size=config["train_batch_size"]))
    rollouts = rollouts.for_each(StandardizeFields(["advantages"]))

    if config["simple_optimizer"]:
        train_op = rollouts.for_each(
            TrainOneStep(
                workers,
                num_sgd_iter=config["num_sgd_iter"],
                sgd_minibatch_size=config["sgd_minibatch_size"]))
    else:
        train_op = rollouts.for_each(
            TrainTFMultiGPU(
                workers,
                sgd_minibatch_size=config["sgd_minibatch_size"],
                num_sgd_iter=config["num_sgd_iter"],
                num_gpus=config["num_gpus"],
                rollout_fragment_length=config["rollout_fragment_length"],
                num_envs_per_worker=config["num_envs_per_worker"],
                train_batch_size=config["train_batch_size"],
                shuffle_sequences=config["shuffle_sequences"],
                _fake_gpus=config["_fake_gpus"]))

    # change the graph topology
    train_op = train_op.for_each(partial(renew_graph, config))

    # Update KL after each round of training.
    train_op = train_op.for_each(lambda t: t[1]).for_each(UpdateKL(workers))

    return StandardMetricsReporting(train_op, workers, config) \
        .for_each(lambda result: warn_about_bad_reward_scales(config, result))
