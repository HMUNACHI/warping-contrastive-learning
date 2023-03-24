from environment import *
import numpy as np
from tabular_model import *


def sarsa(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)

    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)

    q = np.zeros((env.n_states, env.n_actions))

    for i in range(max_episodes):
        s = env.reset()
        p0 = epsilon[i]  # Probability of taking a random action
        p1 = 1 - epsilon[i]  # Probability of using the argmax action of q[s,a]
        # Compute decision to choose either random action or agrmax action
        d = np.random.choice(a=[0, 1], p=[p0, p1])
        # Select action on the basis of the computed decision
        if d == 0:
            act = np.random.choice(a=[0, 1, 2, 3])
        else:
            act = q[s, :].argmax()

        done = False
        while not done:
            # Perform step in the environment and retrieve the new state and reward
            new_s, r, done = env.step(act)
            # Select action for the new state
            # Compute decision to choose either random action or agrmax action
            d = np.random.choice(a=[0, 1], p=[p0, p1])
            # Select action on the basis of the computed decision
            if d == 0:
                new_act = np.random.choice(a=[0, 1, 2, 3])
            else:
                new_act = q[s, :].argmax()

            # Compute new Q for current state and action
            q[s, act] += eta[i]*(r + gamma*q[new_s, new_act] - q[s, act])

            # Save new state and action as current ones
            s = new_s
            act = new_act

    policy = q.argmax(axis=1)
    value = q.max(axis=1)

    return policy, value


def q_learning(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)

    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)

    q = np.zeros((env.n_states, env.n_actions))

    for i in range(max_episodes):
        s = env.reset()

        done = False
        while not done:
            p0 = epsilon[i]  # Probability of taking a random action
            p1 = 1 - epsilon[i]  # Probability of using the argmax action of q[s,a]
            # Compute decision to choose either random action or agrmax action
            d = np.random.choice(a=[0, 1], p=[p0, p1])
            # Select action on the basis of the computed decision
            if d == 0:
                act = np.random.choice(a=[0, 1, 2, 3])
            else:
                act = q[s, :].argmax()

            # Perform step in the environment and retrieve the new state and reward
            new_s, r, done = env.step(act)

            # Compute new Q for current state and action
            q[s, act] += eta[i] * (r + gamma * (q[new_s, :].max()) - q[s, act])

            # Save new state and action as current ones
            s = new_s

    policy = q.argmax(axis=1)
    value = q.max(axis=1)

    return policy, value


def sarsa_compare(env, max_episodes, eta, gamma, epsilon, optimal_value, theta, max_iterations, seed=None):
    random_state = np.random.RandomState(seed)

    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)

    q = np.zeros((env.n_states, env.n_actions))

    for i in range(max_episodes):
        s = env.reset()
        p0 = epsilon[i]  # Probability of taking a random action
        p1 = 1 - epsilon[i]  # Probability of using the argmax action of q[s,a]
        # Compute decision to choose either random action or agrmax action
        d = np.random.choice(a=[0, 1], p=[p0, p1])
        # Select action on the basis of the computed decision
        if d == 0:
            act = np.random.choice(a=[0, 1, 2, 3])
        else:
            act = q[s, :].argmax()

        done = False
        while not done:
            # Perform step in the environment and retrieve the new state and reward
            new_s, r, done = env.step(act)
            # Select action for the new state
            # Compute decision to choose either random action or agrmax action
            d = np.random.choice(a=[0, 1], p=[p0, p1])
            # Select action on the basis of the computed decision
            if d == 0:
                new_act = np.random.choice(a=[0, 1, 2, 3])
            else:
                new_act = q[s, :].argmax()

            # Compute new Q for current state and action
            q[s, act] += eta[i]*(r + gamma*q[new_s, new_act] - q[s, act])

            # Save new state and action as current ones
            s = new_s
            act = new_act

        # Compare with optimal
        policy = q.argmax(axis=1)
        v_pi = policy_evaluation(env, policy, gamma, theta, max_iterations)
        delta = 0
        for s in range(env.n_states - 1):
            delta = max(delta, np.abs(v_pi[s] - optimal_value[s]))

        if delta < theta:
            print("Sarsa completed in ", i, " iterations")
            break

    policy = q.argmax(axis=1)
    value = q.max(axis=1)

    return policy, value


def q_learning_compare(env, max_episodes, eta, gamma, epsilon, optimal_value, theta, max_iterations, seed=None):
    random_state = np.random.RandomState(seed)

    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)

    q = np.zeros((env.n_states, env.n_actions))

    for i in range(max_episodes):
        s = env.reset()

        done = False
        while not done:
            p0 = epsilon[i]  # Probability of taking a random action
            p1 = 1 - epsilon[i]  # Probability of using the argmax action of q[s,a]
            # Compute decision to choose either random action or agrmax action
            d = np.random.choice(a=[0, 1], p=[p0, p1])
            # Select action on the basis of the computed decision
            if d == 0:
                act = np.random.choice(a=[0, 1, 2, 3])
            else:
                act = q[s, :].argmax()

            # Perform step in the environment and retrieve the new state and reward
            new_s, r, done = env.step(act)

            # Compute new Q for current state and action
            q[s, act] += eta[i] * (r + gamma * (q[new_s, :].max()) - q[s, act])

            # Save new state and action as current ones
            s = new_s

        # Compare with optimal
        policy = q.argmax(axis=1)
        v_pi = policy_evaluation(env, policy, gamma, theta, max_iterations)
        delta = 0
        for s in range(env.n_states - 1):
            delta = max(delta, np.abs(v_pi[s] - optimal_value[s]))

        if delta < theta:
            print("Q-Learning completed in ", i, " iterations")
            break

    policy = q.argmax(axis=1)
    value = q.max(axis=1)

    return policy, value
