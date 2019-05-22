import gym
import numpy as np


def init_policy(vel_bins, state_bins):
    # rows - velocity, cols - coordinate
    policy = np.zeros((vel_bins, state_bins))
    for vel in range(vel_bins // 2):
        policy[vel] = np.full(state_bins, 0)

    for vel in range(vel_bins // 2, vel_bins):
        policy[vel] = np.full(state_bins, 2)

    return policy.astype(np.int)


def init_utility(vel_bins, state_bins, min_value, count_actions=3):
    utility = np.full((vel_bins, state_bins, count_actions), min_value)
    return utility


def save_policy_to_file(filename, policy):
    f = open(filename, 'w')
    f.write('\n'.join([' '.join([str(x) for x in l]) for l in policy.tolist()]))
    f.close()


def read_policy_from_file(filename):
    f = open(filename, 'r')
    string_policy = [lines.split() for lines in f.readlines()]
    policy = np.array([[int(x) for x in line] for line in string_policy]).astype(np.int)
    f.close()
    return policy


def stochastic_improvement(utility, velocity_state_array, position_state_array, env, changed_action):
    observation = env.reset()
    cumulative_reward = 0
    alpha = 0.1

    changed_state = None
    random_action = None
    left_edge = len(position_state_array) * 0.3
    right_edge = len(position_state_array) * 0.4
    trace = []

    for t in range(1000):
        velocity = observation[1]
        vel_ind = np.digitize(velocity, velocity_state_array)
        pose_ind = np.digitize(observation[0], position_state_array)

        # usually stock in those moments
        if t == changed_action and (pose_ind < left_edge or pose_ind > right_edge):
            changed_state = vel_ind, pose_ind
            # if policy[vel_ind, pose_ind] == 0:
            #     action = 1 + np.random.randint(0, 2)
            # elif policy[vel_ind, pose_ind] == 2:
            #     action = np.random.randint(0, 2)
            # else:
            action = np.random.randint(0, 3)
            random_action = action
        else:
            action = np.argmax(utility[vel_ind, pose_ind, :])
        trace.append((vel_ind, pose_ind, action))

        observation, reward, done, info = env.step(action)
        cumulative_reward += reward
        if done:
            if changed_state is None:
                break
            old_utility = utility[changed_state[0], changed_state[1], random_action]

            if old_utility < cumulative_reward:
                # print('improvement finded, improve by {}'.format(cumulative_reward - old_utility))
                # print('before: {}\nafter: {}'.format(old_utility, cumulative_reward))
                utility[changed_state[0], changed_state[1], random_action] = cumulative_reward
                for row, col, action in trace:
                    utility[row, col, action] = max(utility[row, col, action], cumulative_reward)
            else:
                pass
            break
    return cumulative_reward


def policy_train(env):
    velocity_bins = 100
    position_bins = 30
    policy = init_policy(velocity_bins, position_bins)
    utility = init_utility(velocity_bins, position_bins, -200)
    velocity_state_array = np.linspace(-0.08, 0.08, velocity_bins - 1)
    position_state_array = np.linspace(-1.2, 0.5, position_bins - 1)

    epochs = 1000
    old_rewards = []
    trace = []
    alpha = 0.1

    print('start initiate filling of utility')
    for t in range(epochs):
        observation = env.reset()
        cumulative_reward = 0
        if t % 100 == 0:
            print('epoch {}'.format(t))

        for _ in range(1000):
            velocity = observation[1]
            vel_ind = np.digitize(velocity, velocity_state_array)
            pose_ind = np.digitize(observation[0], position_state_array)
            action = policy[vel_ind, pose_ind]

            trace.append((int(vel_ind), int(pose_ind), action))

            observation, reward, done, info = env.step(action)
            cumulative_reward += reward
            if done:
                for x, y, action in trace:
                    if utility[x, y, action] == -200:
                        utility[x, y, action] = cumulative_reward
                    else:
                        utility[x, y, action] = (1-alpha) * utility[x, y, action] + alpha * cumulative_reward
                break
        old_rewards.append(cumulative_reward)

    print('start training')
    for t in range(100 * epochs):
        changed_action = np.random.randint(10, 150)
        stochastic_improvement(utility, velocity_state_array, position_state_array, env, changed_action)
        if t % 500 == 0:
            print('epoch {} / {}'.format(t, 100 * epochs))

    for x in range(len(velocity_state_array)):
        for y in range(len(position_state_array)):
            policy[x, y] = policy[x, y] if np.max(utility[x, y]) < -125 else np.argmax(utility[x, y])

    rewards = []
    for t in range(epochs):
        observation = env.reset()
        cumulative_reward = 0

        for _ in range(1000):
            if t < 10:
                env.render()
            velocity = observation[1]
            vel_ind = np.digitize(velocity, velocity_state_array)
            pose_ind = np.digitize(observation[0], position_state_array)
            action = policy[vel_ind, pose_ind]

            observation, reward, done, info = env.step(action)
            cumulative_reward += reward
            if done:
                break
        rewards.append(cumulative_reward)
        if t < 10:
            print('reward is {}'.format(cumulative_reward))

    print('average reward before learning: {}'.format(sum(old_rewards) / len(old_rewards)))
    print('average reward after learning: {}'.format(sum(rewards) / len(rewards)))
    save_policy_to_file('utility.txt', utility)
    save_policy_to_file('policy.txt', policy)
    env.close()

    return policy
