from tf_agents.environments import py_environment
from tf_agents.environments import gym_wrapper

from ns3gym import ns3env

class Ns3PyEnv(py_environment.PyEnvironment):

    def __init__(self):
        port = 5555
        simTime = 20
        stepTime = 0.1
        seed = 0
        simArgs = {"--simTime": simTime,
                   "--stepTime": stepTime}
        gym_env = ns3env.Ns3Env(port=port, stepTime=stepTime, startSim=True,
                                simSeed=seed, simArgs=simArgs, debug=False)
        self._env = gym_wrapper.GymWrapper(gym_env)

    def close(self):
        self._env.close()

    def action_spec(self):
        return self._env.action_spec()

    def observation_spec(self):
        return self._env.observation_spec()

    def _reset(self):
        return self._env.reset()

    def _step(self, action):
        return self._env.step(action)


if __name__ == '__main__':

    with Ns3PyEnv() as env:
        print(env.observation_spec())
        print(env.action_spec())