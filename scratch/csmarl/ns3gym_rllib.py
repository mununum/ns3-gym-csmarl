import logging
import os
import gym
from gym.spaces import Tuple
from ns3gym import ns3env

import numpy as np
import ray
from ray import tune
# from ray.rllib.utils import try_import_tf

# tf = try_import_tf()

# NOTE unused
class Ns3EnvInherit(ns3env.Ns3Env):

    def __init__(self, env_config):
        port = 0
        simTime = 20
        stepTime = 0.1
        seed = 0
        simArgs = {"--simTime": simTime,
                   "--stepTime": stepTime}
        cwd = env_config.get("cwd", None)
        super(Ns3EnvInherit, self).__init__(port=port, stepTime=stepTime, startSim=True,
                                            simSeed=seed, simArgs=simArgs, debug=True, cwd=cwd)

class Ns3EnvWrapper(gym.Env):

    def __init__(self, env_config):
        self._worker_index = env_config.worker_index
        port = 5555 + self._worker_index
        simTime = 20
        stepTime = 0.1
        seed = 0
        simArgs = {"--simTime": simTime,
                   "--stepTime": stepTime,
                   "--continuous": True}
        cwd = env_config.get("cwd", None)
        self.debug = env_config.get("debug", False)
        print("worker {} start".format(self._worker_index)) if self.debug else None
        self._env = ns3env.Ns3Env(port=port, stepTime=stepTime, startSim=True,
                                  simSeed=seed, simArgs=simArgs, debug=False, cwd=cwd)
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space

    def reset(self):
        print("worker {} reset".format(self._worker_index)) if self.debug else None
        obs = self._env.reset()
        return np.array(obs).reshape(self.observation_space.shape)

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        info = dict([(kv.split('=')[0], kv.split('=')[1]) for kv in info.split()])
        return np.array(obs).reshape(self.observation_space.shape), reward, done, info

    def close(self):
        print("worker {} finish".format(self._worker_index)) if self.debug else None
        self._env.close()


if __name__ == "__main__":

    # ray.init(local_mode=False, logging_level=logging.DEBUG)
    ray.init()

    cwd = os.path.dirname(os.path.abspath(__file__))

    tune.run(
        "PPO",
        stop={"timesteps_total": 1000000},
        config={
            "env": Ns3EnvWrapper,
            # "vf_share_layers": True,
            "lr": 1e-4,
            "num_workers": 8,
            "env_config": {
                "cwd": cwd,
                "debug": False
            },
            # "log_level": "DEBUG"
        }
    )