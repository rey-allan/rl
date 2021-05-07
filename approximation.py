"""Implementation of different methods using function approximation"""
import argparse
import numpy as np
import plot as plt
import random

from env import Action, Easy21, State
from mc import MonteCarloControl
from policy import EpsilonGreedyApproximationPolicy
from tqdm import tqdm

# For reproducibility
random.seed(0)
np.random.seed(0)


def encode(s: State, a: Action) -> np.ndarray:
    """
    Encodes the given state-action pair using coarse coding as specified in the Easy21 assignment:

    A binary feature vector rho(s, a) with 3 ∗ 6 ∗ 2 = 36 features. Each binary feature
    has a value of 1 iff (s, a) lies within the cuboid of state-space corresponding to
    that feature, and the action corresponding to that feature. The cuboids have the
    following overlapping intervals:

    - dealer(s) = {[1, 4], [4, 7], [7, 10]}
    - player(s) = {[1, 6], [4, 9], [7, 12], [10, 15], [13, 18], [16, 21]}
    - a = {hit, stick}

    :param State s: The state to encode
    :param Action a: The action to encode
    :return: A binary feature vector representing the encoded state-action pair
    :rtype: np.ndarray
    """
    # `range` is end-exclusive so we add a 1 to make sure we capture the intervals inclusive ends
    dealer = [range(1, 5), range(4, 8), range(7, 11)]
    player = [range(1, 7), range(4, 10), range(7, 13), range(10, 16), range(13, 19), range(16, 22)]
    encoded = np.zeros((3, 6, 2))

    for i, d in enumerate(dealer):
        for j, p in enumerate(player):
            for k, action in enumerate([Action.hit, Action.stick]):
                if s.dealer_first_card in d and s.player_sum in p and a == action:
                    encoded[i, j, k] = 1

    return encoded.flatten()


class OnPolicyGradientMonteCarlo(MonteCarloControl):
    """On-policy gradient Monte Carlo with epsilon-soft policy"""

    def learn(self, epochs=200, alpha=0.01, l=1.0, verbose=False, **kwargs) -> np.ndarray:
        w = np.random.rand(36)
        approximator = lambda s: [np.dot(encode(s, a), w) for a in [Action.hit, Action.stick]]
        # Constant exploration as in the Easy21 assignment
        pi = EpsilonGreedyApproximationPolicy(epsilon=0.05, approximator=approximator, seed=24)

        for _ in tqdm(range(epochs), disable=not verbose):
            trajectories = self._sample_episode(pi)
            # Reverse the list so we start backpropagating the return from the last episode
            trajectories.reverse()

            # Learn from the episode
            g = 0
            for t in trajectories:
                g = t.reward + l * g
                # SGD update
                x = encode(t.state, t.action)
                w += alpha * (g - np.dot(x, w)) * x

        # Compute the optimal value function which is simply the value of the best action in each state
        values = np.zeros(self._env.state_space)
        for d in range(self._env.state_space[0]):
            for p in range(self._env.state_space[1]):
                values[d, p] = np.max(approximator(State(d, p)))

        return np.array(values)


class SemiGradientTDZero:
    """Semi-gradient TD(0) with epsilon-soft policy"""

    def __init__(self):
        self._env = Easy21(seed=24)

    def learn(self, epochs=200, alpha=0.01, gamma=0.9, verbose=False, **kwargs) -> np.ndarray:
        w = np.random.rand(36)
        approximator = lambda s: [np.dot(encode(s, a), w) for a in [Action.hit, Action.stick]]
        # Constant exploration as in the Easy21 assignment
        pi = EpsilonGreedyApproximationPolicy(epsilon=0.05, approximator=approximator, seed=24)

        for _ in tqdm(range(epochs), disable=not verbose):
            s = self._env.reset()
            done = False

            while not done:
                a = pi[s]
                s_prime, r, done = self._env.step(a)

                # Compute the TD target
                if done:
                    td_target = r
                else:
                    td_target = r + gamma * np.max(approximator(s_prime))

                # SGD update
                x = encode(s, a)
                w += alpha * (td_target - np.dot(x, w)) * x

                s = s_prime

        # Compute the optimal value function which is simply the value of the best action in each state
        values = np.zeros(self._env.state_space)
        for d in range(self._env.state_space[0]):
            for p in range(self._env.state_space[1]):
                values[d, p] = np.max(approximator(State(d, p)))

        return np.array(values)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run approximation methods")
    parser.add_argument(
        "--on-policy-mc", action="store_true", help="Execute On-policy gradient Monte Carlo with epsilon-soft policy"
    )
    parser.add_argument("--td-zero", action="store_true", help="Execute Semi-gradient TD(0) with epsilon-soft policy")
    parser.add_argument("--epochs", type=int, default=200, help="Epochs to train")
    parser.add_argument("--alpha", type=float, default=0.01, help="Learning rate to use")
    parser.add_argument("--gamma", type=float, default=0.9, help="Discount factor")
    parser.add_argument("--verbose", action="store_true", help="Run in verbose mode")
    args = parser.parse_args()

    # The optimal value function obtained
    V = None
    # The algorithm to run
    approx = None
    # The title of the plot
    title = None

    if args.on_policy_mc:
        print("Running On-policy Gradient Monte Carlo")
        approx = OnPolicyGradientMonteCarlo()
        title = "grad_on_policy_monte_carlo"
    elif args.td_zero:
        print("Running Semi-gradient TD(0)")
        approx = SemiGradientTDZero()
        title = "semi_grad_td_zero"

    if approx is not None:
        V = approx.learn(epochs=args.epochs, alpha=args.alpha, gamma=args.gamma, verbose=args.verbose)

    if V is not None:
        # Plot the value function as a surface
        # Remove the state where the dealer's first card is 0 and the player's sum is 0 because these are not possible
        # They were kept in the value function to avoid having to deal with 0-index vs 1-index
        plt.plot_value_function(range(1, Easy21.state_space[0]), range(1, Easy21.state_space[1]), V[1:, 1:], title)
