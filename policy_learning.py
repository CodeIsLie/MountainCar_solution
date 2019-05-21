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


def init_utility(vel_bins, state_bins, min_value):
    utility = np.full((vel_bins, state_bins), min_value)
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


# def update_utility(utility,)


def stochastic_improvement(policy, utility, velocity_state_array, position_state_array, env, changed_action):
    observation = env.reset()
    cumulative_reward = 0
    # previous_action = 1
    previous_pose = observation
    alpha = 0.1

    changed_state = None
    random_action = None
    left_edge = len(position_state_array) * 0.25
    right_edge = len(position_state_array) * 0.5
    trace = []

    # chaned_action =
    for t in range(1000):
        # env.render()
        velocity = calc_velocity(previous_pose, observation)
        vel_ind = np.digitize(velocity, velocity_state_array)
        pose_ind = np.digitize(previous_pose[0], position_state_array)
        trace.append((vel_ind, pose_ind))

        if t == changed_action and (pose_ind < left_edge or pose_ind > right_edge):
            changed_state = vel_ind, pose_ind
            if policy[vel_ind, pose_ind] == 0:
                action = 1 + np.random.randint(0, 2)
            elif policy[vel_ind, pose_ind] == 2:
                action = np.random.randint(0, 2)
            else:
                action = np.random.randint(0, 3)
            random_action = action
        else:
            action = policy[vel_ind, pose_ind]
        # previous_action = action
        previous_pose = observation

        observation, reward, done, info = env.step(action)
        cumulative_reward += reward
        if done:
            if changed_state is None:
                break
            old_utility = utility[changed_state[0], changed_state[1]]
            # if (changed_state[0] - len(velocity_state_array) // 2) < 2:
            #     break
            if old_utility == -200:
                for row, col in trace:
                    utility[row, col] = max(cumulative_reward, utility[row, col])
                break
            if old_utility < cumulative_reward:
                print('improvement finded, improve by {}'
                      .format(cumulative_reward - old_utility))
                print('before: {}\nafter: {}'.format(old_utility, cumulative_reward))
                policy[changed_state[0], changed_state[1]] = random_action
                for row, col in trace:
                    utility[row, col] = cumulative_reward
            else:
                pass
                # for row, col in trace:
                #     utility[row, col] = (1 - alpha) * utility[row, col] + alpha * cumulative_reward
            break
    return cumulative_reward


def policy_learn(env):
    velocity_bins = 100
    position_bins = 30
    policy = init_policy(velocity_bins, position_bins)
    utility = init_utility(velocity_bins, position_bins, -200)
    velocity_state_array = np.linspace(-0.8, 0.8, velocity_bins - 1)
    position_state_array = np.linspace(-1.2, 0.5, position_bins - 1)

    epochs = 1000
    old_rewards = []
    trace = []
    # print('start initiate filling of utility')
    alpha = 0.1
    for _ in range(epochs):
        observation = env.reset()
        cumulative_reward = 0
        previous_action = 1
        previous_pose = observation

        for t in range(1000):
            velocity = calc_velocity(previous_pose, observation)
            vel_ind = np.digitize(velocity, velocity_state_array)
            pose_ind = np.digitize(previous_pose[0], position_state_array)
            trace.append((int(vel_ind), int(pose_ind)))
            # policy[vel_ind, pose_ind] = action
            action = policy[vel_ind, pose_ind]
            # previous_action = action
            previous_pose = observation

            observation, reward, done, info = env.step(action)
            cumulative_reward += reward
            if done:
                # for x, y in trace:
                #     utility[x, y] = (1-alpha) * utility[x, y] + alpha * cumulative_reward
                break
        old_rewards.append(cumulative_reward)

    for t in range(10 * epochs):
        changed_action = np.random.randint(10, 150)
        stochastic_improvement(policy, utility, velocity_state_array, position_state_array, env, changed_action)
        if t % 500 == 0:
            print('epoch {}'.format(t))

    rewards = []
    for t in range(epochs):
        observation = env.reset()
        cumulative_reward = 0
        previous_pose = observation

        for _ in range(1000):
            if t < 15:
                env.render()
            velocity = calc_velocity(previous_pose, observation)
            vel_ind = np.digitize(velocity, velocity_state_array)
            pose_ind = np.digitize(previous_pose[0], position_state_array)
            # trace.append((int(vel_ind), int(pose_ind)))
            # policy[vel_ind, pose_ind] = action
            action = policy[vel_ind, pose_ind]
            # previous_action = action
            previous_pose = observation

            observation, reward, done, info = env.step(action)
            cumulative_reward += reward
            if done:
                break
        rewards.append(cumulative_reward)
        if t < 15:
            print('reward is {}'.format(cumulative_reward))

    print('average reward before learning: {}'.format(sum(old_rewards) / len(old_rewards)))
    print('average reward after learning: {}'.format(sum(rewards) / len(rewards)))
    # print('rewards is {}'.format(rewards))
    # print('end policy is \n{}'.format(policy))
    # print('utility is \n{}'.format(utility))
    save_policy_to_file('utility.txt', utility)
    save_policy_to_file('policy.txt', policy)
    env.close()
