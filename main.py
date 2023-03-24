import jax.numpy as np

from environment import *
from lax.tabular_model import *
from lax.tabular_model_free import *
from lax.nontabular_model_free import *
import time


if __name__ == '__main__':
    seed = 0
    # Small lake
    # lake = [['&', '.', '.', '.'],
    #         ['.', '#', '.', '#'],
    #         ['.', '.', '.', '#'],
    #         ['#', '.', '.', '$']]

    # Big lake
    lake = jnp.array([['&', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '#', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '#', '.', '.'],
            ['.', '.', '.', '#', '.', '.', '.', '.'],
            ['.', '#', '#', '.', '.', '.', '#', '.'],
            ['.', '#', '.', '.', '#', '.', '#', '.'],
            ['.', '.', '.', '#', '.', '.', '.', '$']])

    env = FrozenLake(lake, slip=0.1, max_steps=200, seed=seed)

    # play(env)

    print('# Model-based algorithms')
    gamma = 0.9
    theta = 0.001
    max_iterations = 100

    print('')

    print('## Policy iteration')
    # Check time consuming of policy_iteration
    start_time = time.time()
    policy, value = policy_iteration(env, gamma, theta, max_iterations)
    print("--- {} seconds ---".format(time.time() - start_time))
    env.render(policy, value)
    optimal_value = value

    print('')

    print('## Value iteration')
    # Check time consuming of value_iteration
    start_time = time.time()
    policy, value = value_iteration(env, gamma, theta, max_iterations)
    print("--- {} seconds ---".format(time.time() - start_time))
    env.render(policy, value)

    print('')

    print('# Model-free algorithms')
    max_episodes = 5000
    eta = 0.5
    epsilon = 0.5

    print('')

    print('## Sarsa')
    # policy, value = sarsa(env, max_episodes, eta, gamma, epsilon, seed=seed)
    policy, value = sarsa_compare(env, max_episodes, eta, gamma, epsilon, optimal_value, theta, max_iterations, seed=seed)
    env.render(policy, value)

    print('=====Value of policy evaluation======')
    v_pi = policy_evaluation(env, policy, gamma, theta, max_iterations)
    env.render(policy, v_pi)
    print('')

    print('## Q-learning')
    # policy, value = q_learning(env, max_episodes, eta, gamma, epsilon, seed=seed)
    policy, value = q_learning_compare(env, max_episodes, eta, gamma, epsilon, optimal_value, theta, max_iterations, seed=seed)
    env.render(policy, value)

    print('=====Value of policy evaluation======')
    v_pi = policy_evaluation(env, policy, gamma, theta, max_iterations)
    env.render(policy, v_pi)

    print('')

    linear_env = LinearWrapper(env)

    print('## Linear Sarsa')

    # parameters = linear_sarsa(linear_env, max_episodes, eta,
    #                           gamma, epsilon, seed=seed)
    parameters = linear_sarsa_compare(linear_env, max_episodes, eta,
                                      gamma, epsilon, optimal_value, theta, max_iterations, seed=seed)
    policy, value = linear_env.decode_policy(parameters)
    linear_env.render(policy, value)

    print('=====Value of policy evaluation======')
    v_pi = policy_evaluation(env, policy, gamma, theta, max_iterations)
    env.render(policy, v_pi)

    print('')

    print('## Linear Q-learning')
    # parameters = linear_q_learning_compare(linear_env, max_episodes, eta,
    #                                        gamma, epsilon, seed=seed)
    parameters = linear_q_learning_compare(linear_env, max_episodes, eta,
                                   gamma, epsilon, optimal_value, theta, max_iterations, seed=seed)
    policy, value = linear_env.decode_policy(parameters)
    linear_env.render(policy, value)

    print('=====Value of policy evaluation======')
    v_pi = policy_evaluation(env, policy, gamma, theta, max_iterations)
    env.render(policy, v_pi)
