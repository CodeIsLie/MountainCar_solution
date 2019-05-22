import gym
import numpy as np
from policy_learning import init_policy, policy_train, read_policy_from_file


velocity_bins = 100
position_bins = 30
env = gym.make('MountainCar-v0')

# policy = init_policy(velocity_bins, position_bins)

# для воспроизведения результата обучение надо использовать policy_train, но обучение занимает некоторое время
# Для использования уже обученной модели стоит использовать read_policy_from_file
# policy = policy_train(env)
policy = read_policy_from_file('policy.txt')
velocity_state_array = np.linspace(-0.08, 0.08, velocity_bins-1)
position_state_array = np.linspace(-1.2, 0.5, position_bins-1)


def episode(env, render=False):
    observation = env.reset()
    cumulative_reward = 0
    previous_pose = observation

    for t in range(201):
        if render:
            env.render()
        velocity = observation[1]
        vel_ind = np.digitize(velocity, velocity_state_array)
        pose_ind = np.digitize(previous_pose[0], position_state_array)
        action = policy[vel_ind, pose_ind]
        previous_pose = observation

        observation, reward, done, info = env.step(action)
        cumulative_reward += reward
        if done:
            break

    return cumulative_reward


def main():
    episodes = 10
    for _ in range(episodes):
        result = episode(env, True)
        print('episode reward is {}'.format(result))
    env.close()


if __name__ == "__main__":
    main()
