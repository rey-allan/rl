"""Implementation of different Monte Carlo methods"""
import argparse
import matplotlib.pyplot as plt
import random
import numpy as np

from abc import ABC, abstractmethod
from env import Action, Easy21, State
from collections import defaultdict, namedtuple
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from policy import GreedyPolicy, Policy
from typing import List
from tqdm import tqdm

import matplotlib
matplotlib.use('macosx')

# For reproducibility
random.seed(0)

Trajectory = namedtuple('Trajectory', ['state', 'action', 'reward'])


class MonteCarloControl(ABC):
    """A base class defining a Monte Carlo control algorithm"""

    def __init__(self):
        self._env = Easy21(seed=24)

    @abstractmethod
    def learn(self, epochs=200, l=1., verbose=False) -> np.ndarray:
        """
        Learns the optimal value function.

        :param int epochs: The number of epochs to take to learn the value function
        :param float l: The discount factor lambda
        :param bool verbose: Whether to use verbose mode or not
        :return: The optimal value function
        :rtype: np.ndarray
        """
        raise NotImplementedError

    def _sample_episode(self, pi: Policy, s_0: State = None, a_0: Action = None) -> List[Trajectory]:
        # Samples trajectories following policy `pi` with an optional starting state-action pair
        trajectories = []

        s = self._env.reset(start=s_0)
        a = a_0 or pi[s]

        while True:
            s_prime, r, done = self._env.step(a)
            trajectories.append(Trajectory(s, a, r))

            if done:
                break

            s = s_prime
            a = pi[s]

        return trajectories


class MonteCarloES(MonteCarloControl):
    """Monte Carlo (every visit) with Exploring Starts"""

    def learn(self, epochs=200, l=1., verbose=False) -> np.ndarray:
        Q = np.zeros((*self._env.state_space, self._env.action_space))
        pi = GreedyPolicy()
        returns = defaultdict(list)

        for _ in tqdm(range(epochs), disable=not verbose):
            # Select starting pair with equal probability (exploring starts)
            # We subtract 1 because `randint` ranges are inclusive
            s_0 = State(random.randint(1, self._env.state_space[0] - 1),
                        random.randint(1, self._env.state_space[1] - 1))
            a_0 = random.choice([Action.hit, Action.stick])
            # Sample episode from that pair
            trajectories = self._sample_episode(pi, s_0, a_0)
            # Reverse the list so we start backpropagating the return from the last episode
            trajectories.reverse()

            # Learn from the episode
            g = 0
            for t in trajectories:
                g = t.reward + l * g
                returns[(*t.state, t.action)].append(g)
                # Prediction
                Q[t.state.dealer_first_card, t.state.player_sum, t.action] = np.squeeze(
                    np.mean(returns[(*t.state, t.action)]))
                # Improvement
                pi[t.state] = np.argmax(Q[t.state.dealer_first_card, t.state.player_sum, :])

        # Compute the optimal value function which is simply the value of the best action (last dimension) in each state
        return np.max(Q, axis=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Monte Carlo methods')
    parser.add_argument('--es', action='store_true', help='Execute Monte Carlo with Exploring Starts')
    parser.add_argument('--epochs', type=int, default=200, help='Epochs to train')
    parser.add_argument('--verbose', action='store_true', help='Run in verbose mode')
    args = parser.parse_args()

    # The optimal value function obtained
    V = None

    if args.es:
        print('Running Monte Carlo with Exploring Starts')
        mc = MonteCarloES()
        V = mc.learn(epochs=args.epochs, verbose=args.verbose)

    if V is not None:
        # Plot the value function as a surface
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        x, y = np.meshgrid(range(Easy21.state_space[0]), range(Easy21.state_space[1]), indexing='ij')
        surf = ax.plot_surface(x, y, V, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        ax.set_xlabel('Dealer showing')
        ax.set_ylabel('Player sum')
        ax.set_zlabel('Value')
        plt.savefig('output/monte_carlo_es.png', bbox_inches='tight')
