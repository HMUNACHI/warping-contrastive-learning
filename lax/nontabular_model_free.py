from environment import *
import numpy as np
from tabular_model import *


class LinearWrapper:
    def __init__(self, env):
        self.env = env

        self.n_actions = self.env.n_actions
        self.n_states = self.env.n_states
        self.n_features = self.n_actions * self.n_states

    def encode_state(self, s):
        features = np.zeros((self.n_actions, self.n_features))
        for a in range(self.n_actions):
            i = np.ravel_multi_index((s, a), (self.n_states, self.n_actions))
            features[a, i] = 1.0

        return features

    def decode_policy(self, theta):
        policy = np.zeros(self.env.n_states, dtype=int)
        value = np.zeros(self.env.n_states)

        for s in range(self.n_states):
            features = self.encode_state(s)
            q = features.dot(theta)

            policy[s] = np.argmax(q)
            value[s] = np.max(q)

        return policy, value

    def reset(self):
        return self.encode_state(self.env.reset())

    def step(self, action):
        state, reward, done = self.env.step(action)

        return self.encode_state(state), reward, done

    def render(self, policy=None, value=None):
        self.env.render(policy, value)


def linear_sarsa(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)

    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)

    theta = np.zeros(env.n_features)

    for i in range(max_episodes):
        features = env.reset()

        q = features.dot(theta)

        # Choose an action by epsilon greedy
        p0 = epsilon[i]
        p1 = 1 - p0
        d = np.random.choice(a=[0, 1], p=[p0, p1])
        if d == 0:
            # Choose random action
            act = np.random.choice(a=[0, 1, 2, 3])
        else:
            # Choose the max value action
            act = q.argmax()

        done = False
        while not done:
            # Do the action
            features_2, r, done = env.step(act)
            q_2 = features_2.dot(theta)

            # Choose next action by epsilon greedy
            p0 = epsilon[i]
            p1 = 1 - p0

            d = np.random.choice(a=[0, 1], p=[p0, p1])
            if d == 0:
                # Choose random action
                new_act = np.random.choice(a=[0, 1, 2, 3])
            else:
                # Choose the max value action
                new_act = q.argmax()

            theta += eta[i] * (r + gamma * q_2[new_act] - q[act]) * features[act]

            features = features_2
            q = q_2
            act = new_act

    return theta

def linear_q_learning(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)

    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)

    theta = np.zeros(env.n_features)

    for i in range(max_episodes):
        features = env.reset()
        q = features.dot(theta)

        done = False
        while not done:
            # Choose action by epsilon greedy
            p0 = epsilon[i]
            p1 = 1 - p0

            d = np.random.choice(a=[0, 1], p=[p0, p1])
            if d == 0:
                # Choose random action
                act = np.random.choice(a=[0, 1, 2, 3])
            else:
                # Choose the max value action
                act = q.argmax()

            features_2, r, done = env.step(act)

            delta = r - q[act]
            q_2 = features_2.dot(theta)
            delta += gamma * q_2.max()

            theta += eta[i] * delta * features[act]

            features = features_2
            q = q_2

    return theta


def linear_sarsa_compare(env, max_episodes, eta, gamma, epsilon, optimal_value, threshold, max_iterations, seed=None):
    random_state = np.random.RandomState(seed)

    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)

    theta = np.zeros(env.n_features)

    for i in range(max_episodes):
        features = env.reset()

        q = features.dot(theta)

        # Choose an action by epsilon greedy
        p0 = epsilon[i]
        p1 = 1 - p0
        d = np.random.choice(a=[0, 1], p=[p0, p1])
        if d == 0:
            # Choose random action
            act = np.random.choice(a=[0, 1, 2, 3])
        else:
            # Choose the max value action
            act = q.argmax()

        done = False
        while not done:
            # Do the action
            features_2, r, done = env.step(act)
            q_2 = features_2.dot(theta)

            # Choose next action by epsilon greedy
            p0 = epsilon[i]
            p1 = 1 - p0

            d = np.random.choice(a=[0, 1], p=[p0, p1])
            if d == 0:
                # Choose random action
                new_act = np.random.choice(a=[0, 1, 2, 3])
            else:
                # Choose the max value action
                new_act = q.argmax()

            theta += eta[i] * (r + gamma * q_2[new_act] - q[act]) * features[act]

            features = features_2
            q = q_2
            act = new_act

        # Compare with the optimal
        policy, _ = env.decode_policy(theta)
        v_pi = policy_evaluation(env.env, policy, gamma, threshold, max_iterations)
        delta = 0
        for s in range(env.n_states - 1):
            delta = max(delta, np.abs(v_pi[s] - optimal_value[s]))

        if delta < threshold:
            print("Q-Learning completed in ", i, " iterations")
            break

    return theta


def linear_q_learning_compare(env, max_episodes, eta, gamma, epsilon, optimal_value, threshold, max_iterations, seed=None):
    random_state = np.random.RandomState(seed)

    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)

    theta = np.zeros(env.n_features)

    for i in range(max_episodes):
        features = env.reset()
        q = features.dot(theta)

        done = False
        while not done:
            # Choose action by epsilon greedy
            p0 = epsilon[i]
            p1 = 1 - p0

            d = np.random.choice(a=[0, 1], p=[p0, p1])
            if d == 0:
                # Choose random action
                act = np.random.choice(a=[0, 1, 2, 3])
            else:
                # Choose the max value action
                act = q.argmax()

            features_2, r, done = env.step(act)

            delta = r - q[act]
            q_2 = features_2.dot(theta)
            delta += gamma * q_2.max()

            theta += eta[i] * delta * features[act]

            features = features_2
            q = q_2

        # Compare with the optimal
        policy, _ = env.decode_policy(theta)
        v_pi = policy_evaluation(env.env, policy, gamma, threshold, max_iterations)
        delta = 0
        for s in range(env.n_states - 1):
            delta = max(delta, np.abs(v_pi[s] - optimal_value[s]))

        if delta < threshold:
            print("Q-Learning completed in ", i, " iterations")
            break

    return theta
