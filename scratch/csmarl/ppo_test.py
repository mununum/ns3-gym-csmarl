from datetime import datetime
import tensorflow as tf

from tf_agents.agents.ppo import ppo_agent
from tf_agents.environments import tf_py_environment
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import value_network
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.specs import tensor_spec

from ns3pyenv import Ns3PyEnv

# evaluation
def compute_avg_return(environment, policy, num_episodes=10):

    total_return = 0.0
    for _ in range(num_episodes):

        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return
    
    avg_return = total_return / num_episodes
    return avg_return.numpy()

# data collection
def collect_episode(environment, policy, num_episodes, replay_buffer):

    episode_counter = 0
    environment.reset()

    while episode_counter < num_episodes:
        time_step = environment.current_time_step()
        action_step = policy.action(time_step)
        next_time_step = environment.step(action_step.action)
        traj = trajectory.from_transition(time_step, action_step, next_time_step)

        # Add trajectory to the replay buffer
        replay_buffer.add_batch(traj)

        if traj.is_boundary():
            episode_counter += 1

if __name__ == '__main__':

    actor_fc_layers = (32,32)
    value_fc_layers = (32,32)

    learning_rate = 1e-4
    num_epochs = 25
    num_iterations = 5000
    replay_buffer_capacity = 2000
    num_eval_episodes = 1
    collect_episodes_per_iteration = 1

    log_tensorboard = True

    eval_interval = 10

    with Ns3PyEnv() as env:

        tf_env = tf_py_environment.TFPyEnvironment(env)

        actor_net = actor_distribution_network.ActorDistributionNetwork(
            tf_env.observation_spec(),
            tf_env.action_spec(),
            fc_layer_params=actor_fc_layers)
        value_net = value_network.ValueNetwork(
            tf_env.observation_spec(),
            fc_layer_params=value_fc_layers)

        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

        # for tensorboard logging
        if log_tensorboard:
            current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
            writer = tf.summary.create_file_writer('tensorboard_logs/' + current_time)
            writer.set_as_default()

        train_step_counter = tf.Variable(0, dtype=tf.int64)

        tf_agent = ppo_agent.PPOAgent(
            tf_env.time_step_spec(),
            tf_env.action_spec(),
            optimizer,
            actor_net=actor_net,
            value_net=value_net,
            num_epochs=num_epochs,
            normalize_rewards=False,
            # normalize_observations=False,
            importance_ratio_clipping=0.2,
            initial_adaptive_kl_beta=0.0,
            kl_cutoff_factor=0.0,
            debug_summaries=log_tensorboard,
            summarize_grads_and_vars=log_tensorboard,
            train_step_counter=train_step_counter)

        replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=tf_agent.collect_data_spec,
            batch_size=tf_env.batch_size,
            max_length=replay_buffer_capacity)

        tf_agent.train = common.function(tf_agent.train)
        tf_agent.train_step_counter.assign(0)

        # avg_return = compute_avg_return(tf_env, tf_agent.policy, num_eval_episodes)
        # print('{0}: step = {1} - Average return = {2}'.format(datetime.now(), 0, avg_return))
        # exit()

        for episode_step in range(num_iterations):
            
            collect_episode(tf_env, tf_agent.collect_policy, collect_episodes_per_iteration, replay_buffer)

            # on-policy
            experience = replay_buffer.gather_all()
            train_loss = tf_agent.train(experience)
            replay_buffer.clear()

            iter_step = tf_agent.train_step_counter.numpy()

            print('{0}: episode_step = {1} - loss = {2}'.format(datetime.now(), episode_step, train_loss.loss))
            if log_tensorboard:
                tf.summary.scalar('loss', train_loss.loss, step=iter_step)

            if episode_step % eval_interval == 0:
                avg_return = compute_avg_return(tf_env, tf_agent.policy, num_eval_episodes)
                print('{0}: episode_step = {1} - Average return = {2}'.format(datetime.now(), episode_step, avg_return))
                if log_tensorboard:
                    tf.summary.scalar('return', avg_return, step=iter_step)