import os
import gym
import numpy as np
import ray
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ns3gym import ns3env


# class DummyEnv(gym.Env):

#     def __init__(self, env_config):
#         self.observation_space = gym.spaces.Discrete(2)
#         self.action_space = gym.spaces.Discrete(2)

#     def reset(self):
#         return 0

#     def step(self, action):
#         return 0, 0, False, {}


# class DummyMultiAgentEnv(MultiAgentEnv):

#     def __init__(self, num, env_config):
#         self.agents = [DummyEnv(env_config) for _ in range(num)]
#         self.dones = set()
#         self.observation_space = self.agents[0].observation_space
#         self.action_space = self.agents[0].action_space

#     def reset(self):
#         self.dones = set()
#         return {i: a.reset() for i, a in enumerate(self.agents)}

#     def step(self, action_dict):
#         obs, rew, done, info = {}, {}, {}, {}
#         for i, action in action_dict.items():
#             obs[i], rew[i], done[i], info[i] = self.agents[i].step(action)
#             if done[i]:
#                 self.dones.add(i)
#         done["__all__"] = len(self.dones) == len(self.agents)
#         return obs, rew, done, info

#     def close(self):
#         [a.close() for a in self.agents]


class Ns3MultiAgentEnv(MultiAgentEnv):

    def __init__(self, env_config):
        self.worker_index = env_config.worker_index
        self.n_agents = env_config.get("n_agents", 3)
        port = 0
        simTime = env_config.get("simTime", 20)
        stepTime = env_config.get("stepTime", 0.02)
        seed = 0
        cwd = env_config.get("cwd", None)

        self.debug = env_config.get("debug", False)

        simArgs = {"--simTime": simTime,
                   "--stepTime": stepTime,
                   "--nFlows": self.n_agents,
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
        return dict([(int(kv.split('=')[0]), float(kv.split('=')[1]))
                     for kv in info.split()])

    def reset(self):
        print("worker {} reset".format(self.worker_index))
        obs = self._env.reset()
        return self.obs_to_dict(obs)

    def step(self, action_dict):
        action = [action_dict[i] for i in range(self.n_agents)]
        o, _, d, i = self._env.step(action)
        obs = self.obs_to_dict(o)
        rew = self.info_to_rew(i)
        done = {"__all__": d}
        info = {}
        return obs, rew, done, info

    def close(self):
        print("worker {} close".format(self.worker_index))
        self._env.close()


def on_episode_step(info):
    episode = info["episode"]
    # print(episode.agent_rewards)


def on_episode_end(info):
    episode = info["episode"]
    print(episode.agent_rewards)
    for (agent_id, policy), reward in episode.agent_rewards.items():
        episode.custom_metrics["agent_"+str(agent_id)] = reward


if __name__ == "__main__":

    # register_env("dummy_multiagent_env",
    #              lambda env_config: DummyMultiAgentEnv(3, env_config))

    ray.init(log_to_driver=False)
    cwd = os.path.dirname(os.path.abspath(__file__))

    # with DummyEnv(None) as single_env:
    #     obs_space = single_env.observation_space
    #     act_space = single_env.action_space

    # temporary single environment for extracting single obs/act dim
    with ns3env.Ns3Env(port=0, startSim=True, simArgs={
            "--nFlows": 1}, cwd=cwd) as single_env:
        multi_obs_space = single_env.observation_space
        multi_act_space = single_env.action_space
        obs_space = gym.spaces.Box(
            low=multi_obs_space.low[0], high=multi_obs_space.high[0], dtype=multi_obs_space.dtype)
        act_space = multi_act_space.spaces[0]

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
            "lr": 1e-4,
            "num_workers": 0,
            "multiagent": {
                "policies": {
                    "policy_0": (None, obs_space, act_space, {}),
                    # "policy_1": (None, obs_space, act_space, {}),
                    # "policy_2": (None, obs_space, act_space, {})
                },
                # "policy_mapping_fn": lambda i: "policy_"+str(i)
                "policy_mapping_fn": lambda _: "policy_0"
            },
            "env_config": {
                "n_agents": 3,
                "cwd": cwd,
                "my_conf": 42,
                "debug": True
            },
            "callbacks": {
                "on_episode_step": on_episode_step,
                "on_episode_end": on_episode_end
            }
        }
    )
