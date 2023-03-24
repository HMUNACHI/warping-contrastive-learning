import numpy as np
from abc import ABC
import operator
import contextlib
from itertools import product

ACTION_UP = 0
ACTION_LEFT = 1
ACTION_DOWN = 2
ACTION_RIGHT = 3


# Configures numpy print options
@contextlib.contextmanager
def printOptions(*args , **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    try:
        yield
    finally:
        np.set_printoptions(**original)


class EnvironmentModel:
    def __init__(self, n_states, n_actions, seed=None):
        self.n_states = n_states
        self.n_actions = n_actions
        self.random_state = np.random.RandomState(seed)

    def p(self, next_state, state, action):
        raise NotImplementedError()

    def r(self, next_state, state, action):
        raise NotImplementedError()

    def draw(self, state, action):
        p = [self.p(ns, state, action) for ns in range(self.n_states)]
        next_state = self.random_state.choice(self.n_states, p=p)
        reward = self.r(next_state, state, action)
        return next_state, reward


class Environment(EnvironmentModel, ABC):
    def __init__(self, n_states, n_actions, max_steps, dist, seed=None):
        EnvironmentModel.__init__(self, n_states, n_actions, seed)
        self.state = 0
        self.n_steps = 0
        self.max_steps = max_steps
        self.dist = dist
        if self.dist is None:
            self.dist = np.full(n_states, 1. / n_states)

    def reset(self):
        self.n_steps = 0
        self.state = self.random_state.choice(self.n_states, p=self.dist)
        return self.state

    def step(self, action):
        if action < 0 or action >= self.n_actions:
            raise Exception('Invalid action.')

        self.n_steps += 1
        done = (self.n_steps >= self.max_steps)
        self.state, reward = self.draw(self.state, action)
        return self.state, reward, done

    def render(self, policy=None, value=None):
        raise NotImplementedError()


class FrozenLake(Environment):
    def __init__(self, lake, slip, max_steps, seed=None):
        """
        :param lake: A matrix that represents the lake. For example:
            lake = [['&', '.', '.', '.' ],
                    ['.', '#', '.', '#' ],
                    ['.', '.', '.', '#' ],
                    ['#', '.', '.', '$' ]]
        :param slip: The probability that the agent will slip
        :param max_steps: The maximum number of time steps in an episode
        :param seed: A seed to control the random number generator(optional)
        """
        # start(&), frozen(.), hole(#), goal($)
        self.lake = np.array(lake)
        self.lake_flat = self.lake.reshape(-1)
        self.slip = slip
        n_states = self.lake.size + 1
        n_actions = 4
        pi = np.zeros(n_states, dtype=float)
        pi[np.where(self.lake_flat == '&')[0]] = 1.0
        self.absorbing_state = n_states - 1
        Environment.__init__(self, n_states, n_actions, max_steps, pi, seed)

        # Up, left, down, right.
        self.actions = [(-1, 0), (0, -1), (1, 0), (0, 1)]

        # Indices to states (coordinates), states (coordinates) to indices
        self.itos = list(product(range(self.lake.shape[0]), range(self.lake.shape[1])))
        self.stoi = {s: i for (i, s) in enumerate(self.itos)}

        # Precomputed transition probabilities
        self._p = np.zeros((self.n_states, self.n_states, self.n_actions))
        for state_index, state in enumerate(self.itos):
            for action_index, action in enumerate(self.actions):
                if self.lake_flat[state_index] == '&':
                    # Start block, no slip
                    next_state = (state[0] + action[0], state[1] + action[1])
                    next_state_index = self.stoi.get(next_state, state_index)
                    self._p[next_state_index, state_index, action_index] = 1

                elif self.lake_flat[state_index] == '#':
                    # The next state is absorbing state if now in trap
                    next_state_index = self.absorbing_state
                    self._p[next_state_index, state_index, action_index] = 1.0

                elif self.lake_flat[state_index] == '.':
                    # Now at an ice block, so the probability to move in action direction is 1-self.slip + self.slip/4
                    # The probabilities of moving towards other directions are self.slip/4
                    for ai, a in enumerate(self.actions):
                        prob = 0
                        if ai == action_index:
                            prob = 1 - self.slip + self.slip / 4
                        else:
                            prob = self.slip / 4

                        next_state = (state[0] + a[0], state[1] + a[1])
                        next_state_index = self.stoi.get(next_state, state_index)
                        self._p[next_state_index, state_index, action_index] += prob

                elif self.lake_flat[state_index] == '$':
                    # Now at the end point, so the next state is absorbing state
                    next_state_index = self.absorbing_state
                    self._p[next_state_index, state_index, action_index] = 1.0

    def step(self, action):
        state, reward, done = Environment.step(self, action)
        done = (state == self.absorbing_state) or done
        return state, reward, done

    def p(self, next_state, state, action):
        return self._p[next_state, state, action]

    def r(self, next_state, state, action):
        if self.lake_flat[state] == '$':
            return 1

        return 0

    def render(self, policy=None, value=None):
        if policy is None:
            lake = np.array(self.lake_flat)
            if self.state < self.absorbing_state:
                lake[self.state] = '@'

            print(lake.reshape(self.lake.shape))

        else:
            actions = ['↑', '←', '↓', '→']

            print("Lake:")
            print(self.lake)
            print("Policy:")
            policy = np.array([actions[a] for a in policy[: -1]])
            print(policy.reshape(self.lake.shape))
            print("Value:")
            with printOptions(precision=3, suppress=True):
                print(value[:-1].reshape(self.lake.shape))


def play(env):
    actions = ['w', 'a', 's', 'd']
    state = env.reset()
    env.render()
    done = False
    while not done:
        c = input('\nMove: ')
        if c not in actions:
            raise Exception('Invalid action')

        state, r, done = env.step(actions.index(c))
        env.render()
        print('Reward: {0}.'.format(r))

