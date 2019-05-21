import gym
import numpy as np
from policy_learning import init_policy


velocity_bins = 100
position_bins = 30
# используется стандартная policy, потому что как бы я не обучал модель, результаты в среднем не улучшались
policy = init_policy(velocity_bins, position_bins)
# utility = init_utility(velocity_bins, position_bins, -200)
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
        if done:
            break
        cumulative_reward += reward
    return cumulative_reward


def main():
    env = gym.make('MountainCar-v0')
    result = episode(env, True)
    print('episode reward is {}'.format(result))


if __name__ == "__main__":
    main()
