import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from ns3gym import ns3env

# will not use for a while
class DqnAgent(object):

    def __init__(self, inNum, outNum):
        super(DqnAgent, self).__init__()
        self.model = keras.Sequential()
        self.model.add(keras.layers.Dense(inNum, input_shape=(inNum,), activation='relu'))
        self.model.add(keras.layers.Dense(outNum, activation='softmax'))
        self.model.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

    def get_action(self, state):
        return np.argmax(self.model.predict(state)[0])

    def predict(self, next_state):
        return self.model.predict(next_state)[0]
    
    def fit(self, state, target, action):
        target_f = self.model.predict(state)
        target_f[0][action] = target
        self.model.fit(state, target_f, epochs=1, verbose=0)


port = 5555
simTime = 20 # seconds
startSim = True
stepTime = 0.1 # seconds
seed = 132
simArgs = {"--simTime": simTime,
           "--stepTime": stepTime}
debug = False

env = ns3env.Ns3Env(port=port, stepTime=stepTime, startSim=startSim, simSeed=seed, simArgs=simArgs, debug=debug)

ob_space = env.observation_space
ac_space = env.action_space
print("Observation space: ", ob_space, ob_space.dtype)
print("Action space: ", ac_space, ac_space.dtype)
s_size = ob_space.shape[0] # TODO fix shaping
a_size = ac_space.shape[0]

obs_dim = ob_space.shape[1]
act_dim = ac_space.high[0] - ac_space.low[0] + 1 # 0 to 100

agent = DqnAgent(obs_dim, act_dim)

total_episodes = 10
max_env_steps = 100
env._max_episode_steps = max_env_steps

epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.999

rew_history = []

try:

    for e in range(total_episodes):

        obs = env.reset()
        rewardsum = 0
        obs = np.reshape(obs, [1, obs_dim])

        for time in range(max_env_steps):

            # choose action
            if np.random.rand(1) < epsilon:
                action = env.action_space.sample()
            else:
                action = agent.get_action(obs)
            # print("--action: ", action)

            # print("Step: ", time)
            next_obs, reward, done, _ = env.step([action])
            next_obs = np.reshape(next_obs, [1, obs_dim])
            # print("--next_obs, reward, done", next_obs, reward, done)

            target = reward + 0.95 * np.amax(agent.predict(next_obs))
            agent.fit(obs, target, action)

            obs = next_obs
            rewardsum += reward
            if epsilon > epsilon_min: epsilon *= epsilon_decay

        print("episode: {}/{}, time: {}, rew: {}, eps: {:.2}"
              .format(e, total_episodes, time, rewardsum, epsilon))
        rew_history.append(rewardsum)

    plt.plot(range(len(rew_history)), rew_history)
    plt.show()

except KeyboardInterrupt:
    print("Crtl-C -> Exit")
finally:
    env.close()