import os
import gym
import numpy as np
import ray
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ns3gym import ns3env


class Ns3MultiAgentEnv(MultiAgentEnv):

    def __init__(self, env_config):
        self.worker_index = env_config.worker_index
        self.n_agents = env_config.get("n_agents", 3)
        topology = env_config.get("topology", "fim")
        # MYTODO add topology
        assert topology in ["fc", "fim"], "invalid topology configuration"
        port = 0
        simTime = env_config.get("simTime", 20)
        stepTime = env_config.get("stepTime", 0.02)
        seed = 0
        cwd = env_config.get("cwd", None)

        self.debug = env_config.get("debug", False)

        self.reward = env_config.get("reward", "shared")
        assert self.reward in ["indiv", "shared"], 'self.reward must be either "indiv" or "shared"'

        simArgs = {"--simTime": simTime,
                   "--stepTime": stepTime,
                   "--nFlows": self.n_agents,
                   "--topology": topology,
                   "--debug": self.debug}

        print("worker {} start".format(self.worker_index))

        self._env = ns3env.Ns3Env(port=port, stepTime=stepTime, startSim=True,
                                  simSeed=seed, simArgs=simArgs, debug=self.debug, cwd=cwd)

        self.multi_obs_space = self._env.observation_space
        self.multi_act_space = self._env.action_space

        self.observation_space = gym.spaces.Box(
            low=self.multi_obs_space.low[0], high=self.multi_obs_space.high[0], dtype=self.multi_obs_space.dtype)
        self.action_space = self.multi_act_space.spaces[0]

    def obs_to_dict(self, obs):
        multi_obs = np.array(obs).reshape(self.multi_obs_space.shape)
        return {i: multi_obs[i] for i in range(self.n_agents)}

    def info_to_rew(self, info):
        return {int(kv.split('=')[0]): float(kv.split('=')[1])
                for kv in info.split()}

    def parse_info(self, info):
        return {int(kv.split('=')[0]): {"r_ind": float(kv.split('=')[1])}
                for kv in info.split()}

    def reset(self):
        print("worker {} reset".format(self.worker_index))
        obs = self._env.reset()
        return self.obs_to_dict(obs)

    def step(self, action_dict):
        action = [action_dict[i] for i in range(self.n_agents)]
        o, r, d, i = self._env.step(action)
        obs = self.obs_to_dict(o)
        if self.reward == "indiv":
            rew = self.info_to_rew(i) # individual rewards
        elif self.reward == "shared":
            rew = {i: r for i in range(self.n_agents)}  # shared rewards
        else:
            raise AssertionError
        done = {"__all__": d}
        info = self.parse_info(i)
        return obs, rew, done, info

    def close(self):
        print("worker {} close".format(self.worker_index))
        self._env.close()


def on_episode_start(info):
    episode = info["episode"]
    n_agents = info["env"].envs[0].n_agents
    for i in range(n_agents):
        episode.user_data["r_ind_"+str(i)] = []


def on_episode_step(info):
    episode = info["episode"]
    n_agents = info["env"].envs[0].n_agents
    for i in range(n_agents):
        if "r_ind" in episode.last_info_for(i):
            episode.user_data["r_ind_"+str(i)].append(
                episode.last_info_for(i)["r_ind"]
            )


def on_episode_end(info):
    episode = info["episode"]
    r_ind_sum = {k: np.sum(v) for k, v in episode.user_data.items()}
    r_total = 0
    for k, v in r_ind_sum.items():
        episode.custom_metrics[k] = v
        r_total += v
    episode.custom_metrics["r_total"] = r_total


if __name__ == "__main__":

    # register_env("dummy_multiagent_env",
    #              lambda env_config: DummyMultiAgentEnv(3, env_config))

    ray.init(log_to_driver=False)
    cwd = os.path.dirname(os.path.abspath(__file__))

    # with DummyEnv(None) as single_env:
    #     obs_space = single_env.observation_space
    #     act_space = single_env.action_space

    # temporary single environment for extracting single obs/act dim
    # with ns3env.Ns3Env(port=0, startSim=True, simArgs={
    #         "--nFlows": 1}, cwd=cwd) as single_env:
    #     multi_obs_space = single_env.observation_space
    #     multi_act_space = single_env.action_space
    #     obs_space = gym.spaces.Box(
    #         low=multi_obs_space.low[0], high=multi_obs_space.high[0], dtype=multi_obs_space.dtype)
    #     act_space = multi_act_space.spaces[0]

    # print(obs_space)
    # print(act_space)
    # exit()

    tune.run(
        "PPO",
        stop={"training_iteration": 3000},
        config={
            # "env": "ns3_multiagent_env",
            "env": Ns3MultiAgentEnv,
            "batch_mode": "complete_episodes",
            "log_level": "DEBUG",
            # "lr": 1e-4,
            "num_workers": 16,
            # "multiagent": {
            #     "policies": {
            #         "policy_0": (None, obs_space, act_space, {}),
            #         # "policy_1": (None, obs_space, act_space, {}),
            #         # "policy_2": (None, obs_space, act_space, {})
            #     },
            #     # "policy_mapping_fn": lambda i: "policy_"+str(i)
            #     "policy_mapping_fn": lambda _: "policy_0"
            # },
            "env_config": {
                "n_agents": 12,
                "cwd": cwd,
                "debug": True,
                "reward": "shared",
                "topology": "fc"
            },
            "callbacks": {
                "on_episode_start": on_episode_start,
                "on_episode_step": on_episode_step,
                "on_episode_end": on_episode_end
            },
        }
    )
