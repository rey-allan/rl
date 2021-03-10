"""Implementation of different Temporal-Difference methods"""
import argparse
import numpy as np
import plot as plt
import random

from abc import ABC, abstractmethod
from env import Easy21
from policy import EpsilonGreedyPolicy
from tqdm import tqdm

# For reproducibility
random.seed(0)


class TemporalDifferenceControl(ABC):
    """A base class defining a TD control algorithm"""

    def __init__(self):
        self._env = Easy21(seed=24)

    @abstractmethod
    def learn(self, epochs=200, alpha=0.5, gamma=0.9, verbose=False) -> np.ndarray:
        """
        Learns the optimal value function.

        :param int epochs: The number of epochs to take to learn the value function
        :param float alpha: The learning rate
        :param float gamma: The discount factor
        :param bool verbose: Whether to use verbose mode or not
        :return: The optimal value function
        :rtype: np.ndarray
        """
        raise NotImplementedError


class Sarsa(TemporalDifferenceControl):
    """On-policy SARSA"""

    def learn(self, epochs=200, alpha=0.5, gamma=0.9, verbose=False) -> np.ndarray:
        Q = np.zeros((*self._env.state_space, self._env.action_space))
        pi = EpsilonGreedyPolicy(seed=24)

        for _ in tqdm(range(epochs), disable=not verbose):
            s = self._env.reset()
            a = pi[s]
            done = False

            while not done:
                # Generate S,A,R,S',A' trajectory
                s_prime, r, done = self._env.step(a)

                # Learn from the trajectory using the TD update
                td_target = None

                # If we're at a terminal state the TD target is simply the reward
                if done:
                    td_target = r
                else:
                    a_prime = pi[s_prime]
                    td_target = r + gamma * Q[s_prime.dealer_first_card, s_prime.player_sum, a_prime]

                td_error = td_target - Q[s.dealer_first_card, s.player_sum, a]

                # Prediction
                Q[s.dealer_first_card, s.player_sum, a] += alpha * td_error
                # Improvement
                pi[s] = np.argmax(Q[s.dealer_first_card, s.player_sum, :])

                s = s_prime
                a = a_prime

        # Compute the optimal value function which is simply the value of the best action (last dimension) in each state
        return np.max(Q, axis=2)


class QLearning(TemporalDifferenceControl):
    """Q-Learning (always off-policy)"""

    def learn(self, epochs=200, alpha=0.5, gamma=0.9, verbose=False) -> np.ndarray:
        Q = np.zeros((*self._env.state_space, self._env.action_space))
        pi = EpsilonGreedyPolicy(seed=24)

        for _ in tqdm(range(epochs), disable=not verbose):
            s = self._env.reset()
            done = False

            while not done:
                a = pi[s]
                s_prime, r, done = self._env.step(a)

                # Learn from the trajectory using the TD update
                td_target = None

                # If we're at a terminal state the TD target is simply the reward
                if done:
                    td_target = r
                else:
                    # Q-learning is off-policy; therefore, we greedily select the best value of the successor state
                    # In other words, we assume we will behave optimally thereafter (i.e. a strictly greedy policy)
                    td_target = r + gamma * np.max(Q[s_prime.dealer_first_card, s_prime.player_sum, :])

                td_error = td_target - Q[s.dealer_first_card, s.player_sum, a]

                # Prediction
                Q[s.dealer_first_card, s.player_sum, a] += alpha * td_error
                # Improvement
                pi[s] = np.argmax(Q[s.dealer_first_card, s.player_sum, :])

                s = s_prime

        # Compute the optimal value function which is simply the value of the best action (last dimension) in each state
        return np.max(Q, axis=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run TD methods')
    parser.add_argument('--sarsa', action='store_true', help='Execute On-policy SARSA')
    parser.add_argument('--qlearning', action='store_true', help='Execute Q-Learning')
    parser.add_argument('--epochs', type=int, default=200, help='Epochs to train')
    parser.add_argument('--gamma', type=float, default=0.9, help='Discount factor')
    parser.add_argument('--alpha', type=float, default=0.5, help='Learning rate')
    parser.add_argument('--verbose', action='store_true', help='Run in verbose mode')
    args = parser.parse_args()

    # The optimal value function obtained
    V = None
    # The algorithm to run
    td = None
    # The title of the plot
    title = None

    if args.sarsa:
        print('Running On-policy SARSA')
        td = Sarsa()
        title = 'sarsa'
    elif args.qlearning:
        print('Running Q-learning')
        td = QLearning()
        title = 'qlearning'

    if td is not None:
        V = td.learn(epochs=args.epochs, alpha=args.alpha, gamma=args.gamma, verbose=args.verbose)

    if V is not None:
        # Plot the value function as a surface
        # Remove the state where the dealer's first card is 0 and the player's sum is 0 because these are not possible
        # They were kept in the value function to avoid having to deal with 0-index vs 1-index
        plt.plot_value_function(range(1, Easy21.state_space[0]), range(1, Easy21.state_space[1]), V[1:, 1:], title)
