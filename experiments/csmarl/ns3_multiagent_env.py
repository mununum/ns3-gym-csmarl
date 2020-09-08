import os
import gym
from collections import defaultdict
import numpy as np

from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.registry import register_env
from ns3gym import ns3env

from ns3_import import import_graph_module

class Ns3MultiAgentEnv(MultiAgentEnv):

    def __init__(self, env_config):

        graph = import_graph_module()

        self.topology = env_config.get("topology", None)
        assert self.topology, "topology is needed to be specified"
        _, self.n_agents = graph.read_graph(self.topology)
        port = 0

        d = os.path.dirname(os.path.abspath(__file__))
        cwd = os.path.join(d, "../../scratch/csmarl2")

        simTime = env_config.get("simTime", None)
        stepTime = env_config.get("stepTime", None)
        intensity = env_config.get("intensity", None)
        self.debug = env_config.get("debug", False)

        # random topology generation

        simArgs = {
            "--topology": self.topology,
            "--simTime": simTime,
            "--stepTime": stepTime,
            "--intensity": intensity,
            "--debug": self.debug,
            "--algorithm": "rl",
        }
        for k in list(simArgs):
            if simArgs[k] is None:
                del simArgs[k]

        seed = env_config.get("seed", 0)

        self._env = ns3env.Ns3Env(port=port, startSim=True, simSeed=seed, simArgs=simArgs, debug=self.debug, cwd=cwd)

        self.multi_obs_space = self._env.observation_space
        self.multi_act_space = self._env.action_space

        self.observation_space = gym.spaces.Box(
            low=self.multi_obs_space.low[0], high=self.multi_obs_space.high[0], dtype=self.multi_obs_space.dtype)
        self.action_space = self.multi_act_space.spaces[0]

    def obs_to_dict(self, obs):
        multi_obs = np.array(obs).reshape(self.multi_obs_space.shape)
        return {i: multi_obs[i] for  i in range(self.n_agents)}

    def info_to_rew(self, info):
        pass

    def parse_info(self, info, obs):
        ret = {}
        for i in range(self.n_agents):
            o = obs[i]
            ret[i] = {
                "rate": o[0],
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
        episode.user_data["graph"], _ = graph.read_graph(env.topology)
        episode.user_data["stat"] = defaultdict(list)

    def on_episode_step(self, worker, base_env, episode, **kwargs):
        env = base_env.envs[0]
        for i in range(env.n_agents):
            for k, v in episode.last_info_for(i).items():
                episode.user_data["stat"][k+"_"+str(i)].append(v)

    def on_episode_end(self, worker, base_env, policies, episode, **kwargs):
        for k, v in episode.user_data["stat"].items():
            episode.custom_metrics[k] = np.sum(v)

register_env("ns3_multiagent_env", lambda env_config: Ns3MultiAgentEnv(env_config))