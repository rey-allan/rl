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
from policy import EpsilonGreedyPolicy, GreedyPolicy, Policy
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


class OnPolicyMonteCarlo(MonteCarloControl):
    """On-policy Monte Carlo with epsilon-soft policy"""

    def learn(self, epochs=200, l=1., verbose=False) -> np.ndarray:
        Q = np.zeros((*self._env.state_space, self._env.action_space))
        pi = EpsilonGreedyPolicy(seed=24)
        returns = defaultdict(list)

        for _ in tqdm(range(epochs), disable=not verbose):
            trajectories = self._sample_episode(pi)
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


class OffPolicyMonteCarlo(MonteCarloControl):
    """
    "Off-policy Monte Carlo with weighted importance sampling.
    It also uses an iterative implementation for computing the averages.
    """

    def learn(self, epochs=200, l=1., verbose=False) -> np.ndarray:
        Q = np.zeros((*self._env.state_space, self._env.action_space))
        C = np.zeros_like(Q)
        # Target policy
        pi = GreedyPolicy()
        # Behavior policy
        b = EpsilonGreedyPolicy(seed=24)

        for _ in tqdm(range(epochs), disable=not verbose):
            # Sample episode using the behavior policy
            trajectories = self._sample_episode(b)
            # Reverse the list so we start backpropagating the return from the last episode
            trajectories.reverse()

            # Learn from the episode
            g = 0
            w = 1.
            for t in trajectories:
                g = t.reward + l * g
                # Cumulative sum of the weights (importance sampling ratio)
                C[t.state.dealer_first_card, t.state.player_sum, t.action] += w
                # Prediction with iterative implementation and importance sampling
                Q[t.state.dealer_first_card, t.state.player_sum, t.action] += (
                    w / C[t.state.dealer_first_card, t.state.player_sum, t.action] * (g - Q[t.state.dealer_first_card, t.state.player_sum, t.action]))
                # Improvement using the target policy
                pi[t.state] = np.argmax(Q[t.state.dealer_first_card, t.state.player_sum, :])

                # If the action the behavio policy sampled is not consistent with the target policy we exit early
                # The reason is that we only update the target policy when there is consistency because the importance
                # sampling weight becomes zero when the probability of seeing the action in the target policy is zero
                # Recall the sampling ratio: pi(a|s) / b(a|s)
                if t.action != pi[t.state]:
                    break

                # Update the importance sampling weight
                # Because we now that the action is consistent with the target policy, then said action was taken greedily
                # therefore, the probability b(a|s) is the probability of taking the greedy action under policy b for state s
                w += 1. / b.greedy_prob(t.state)

        # Compute the optimal value function which is simply the value of the best action (last dimension) in each state
        return np.max(Q, axis=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Monte Carlo methods')
    parser.add_argument('--es', action='store_true', help='Execute Monte Carlo with Exploring Starts')
    parser.add_argument('--on-policy', action='store_true',
                        help='Execute On-policy Monte Carlo with epsilon-soft policies')
    parser.add_argument('--off-policy', action='store_true',
                        help='Execute Off-policy Monte Carlo with weighted importance sampling')
    parser.add_argument('--epochs', type=int, default=200, help='Epochs to train')
    parser.add_argument('--verbose', action='store_true', help='Run in verbose mode')
    args = parser.parse_args()

    # The optimal value function obtained
    V = None
    # The algorithm to run
    mc = None
    # The title of the plot
    title = None

    if args.es:
        print('Running Monte Carlo with Exploring Starts')
        mc = MonteCarloES()
        title = 'monte_carlo_es'
    elif args.on_policy:
        print('Running On-policy Monte Carlo')
        mc = OnPolicyMonteCarlo()
        title = 'on_policy_monte_carlo'
    elif args.off_policy:
        print('Running Off-policy Monte Carlo')
        mc = OffPolicyMonteCarlo()
        title = 'off_policy_monte_carlo'

    if mc is not None:
        V = mc.learn(epochs=args.epochs, verbose=args.verbose)

    if V is not None:
        # Plot the value function as a surface
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        # Remove the state where the dealer's first card is 0 and the player's sum is 0 because these are not possible
        # They were kept in the value function to avoid having to deal with 0-index vs 1-index
        x, y = np.meshgrid(range(1, Easy21.state_space[0]), range(1, Easy21.state_space[1]), indexing='ij')
        surf = ax.plot_surface(x, y, V[1:, 1:], rstride=1, cstride=1, cmap=cm.viridis, linewidth=0, antialiased=False)
        ax.set_xlabel('Dealer showing')
        ax.set_ylabel('Player sum')
        ax.set_zlabel('Value')
        plt.savefig(f'output/{title}.png', bbox_inches='tight')
