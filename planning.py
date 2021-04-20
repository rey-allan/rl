"""Implementation of different methods that integrate planning with learning"""
import argparse
import numpy as np
import plot as plt
import random

from env import Easy21, State
from policy import EpsilonGreedyPolicy
from tqdm import tqdm

# For reproducibility
random.seed(0)


class DynaQ:
    """Dyna-Q algorithm"""

    def __init__(self):
        self._env = Easy21(seed=24)

    def learn(self, epochs=200, n=100, alpha=0.5, gamma=0.9, verbose=False) -> np.ndarray:
        """
        Learns the optimal value function.

        :param int epochs: The number of epochs to take to learn the value function
        :param int n: The planning iterations to use
        :param float alpha: The learning rate
        :param float gamma: The discount factor
        :param bool verbose: Whether to use verbose mode or not
        :return: The optimal value function
        :rtype: np.ndarray
        """
        Q = np.zeros((*self._env.state_space, self._env.action_space))
        # Model of the world represented by approximations of the transition and reward functions
        # We initialize the transition function with a very small value greater than 0 to avoid
        # division by zero when computing transition probabilities for trajectories that haven't been encountered
        T = np.full((*self._env.state_space, self._env.action_space, *self._env.state_space), 0.00001)
        R = np.zeros((*self._env.state_space, self._env.action_space))
        pi = EpsilonGreedyPolicy(seed=24)

        for _ in tqdm(range(epochs), disable=not verbose):
            done = False
            s = self._env.reset()

            while not done:
                a = pi[s]
                s_prime, r, done = self._env.step(a)

                # Learning phase
                self._learn(Q, pi, s, a, r, s_prime, done, alpha, gamma)
                # Planning phase
                if n > 0:
                    self._update_model(T, R, s, a, r, s_prime, done, alpha)
                    self._plan(Q, T, R, pi, done, n, alpha, gamma)

                s = s_prime

        # Compute the optimal value function which is simply the value of the best action (last dimension) in each state
        return np.max(Q, axis=2)

    def _learn(self, Q, pi, s, a, r, s_prime, done, alpha, gamma):
        if done:
            td_target = r
        else:
            td_target = r + gamma * np.max(Q[s_prime.dealer_first_card, s_prime.player_sum, :])

        td_error = td_target - Q[s.dealer_first_card, s.player_sum, a]

        # Prediction
        Q[s.dealer_first_card, s.player_sum, a] += alpha * td_error
        # Improvement
        pi[s] = np.argmax(Q[s.dealer_first_card, s.player_sum, :])

    def _update_model(self, T, R, s, a, r, s_prime, done, alpha):
        if not done:
            # We only update the transition model for non-terminal states since the terminal state
            # is most likely an "invalid" state for this environment, e.g. a player's sum over 21
            T[s.dealer_first_card, s.player_sum, a, s_prime.dealer_first_card, s_prime.player_sum] += 1
        # We update the model of the reward by learning it similarly to the action values
        # This is not specified in Sutton & Barto's book but I learned it during my Master's
        R[s.dealer_first_card, s.player_sum, a] += 0.2 * (r - R[s.dealer_first_card, s.player_sum, a])

    def _plan(self, Q, T, R, pi, done, n, alpha, gamma):
        # Compute the probabilities of each s,a -> s' transition over all possible transitions from each s,a
        transition_probs = T / np.sum(T, axis=(0, 1, 2))
        dealer_state_space = list(range(self._env.state_space[0]))
        player_state_space = list(range(self._env.state_space[1]))
        action_space = list(range(self._env.action_space))

        for _ in range(n):
            # Select a random (s,a) pair uniformly from the _full_ state/action space as opposed to only
            # the ones that have been previously observed. This matches the enhancements of Dyna-Q+ as
            # described by Sutton & Barto. It was also what I did for one of my projects during the Master's.
            s = State(random.choice(dealer_state_space), random.choice(player_state_space))
            a = random.choice(action_space)
            r = R[s.dealer_first_card, s.player_sum, a]
            # Infer s' by using the state with the highest probability
            # This is different from what Sutton & Barto describe. They mention that expected updates
            # should be used for stochastic environments. However, I've used this kind of update before
            # and it works much better in practice
            # We use `unravel_index` to get the state as a pair of (dealer_first_card,player_sum)
            s_prime = np.unravel_index(np.argmax(
                transition_probs[s.dealer_first_card, s.player_sum, a]), transition_probs[s.dealer_first_card, s.player_sum, a].shape)
            self._learn(Q, pi, s, a, r, State(*s_prime), done, alpha, gamma)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run planning methods')
    parser.add_argument('--dynaq', action='store_true', help='Execute Dyna-Q')
    parser.add_argument('--epochs', type=int, default=200, help='Epochs to train')
    parser.add_argument('--gamma', type=float, default=0.9, help='Discount factor')
    parser.add_argument('--alpha', type=float, default=0.5, help='Learning rate')
    parser.add_argument('--n', type=int, default=100, help='Planning steps to use')
    parser.add_argument('--verbose', action='store_true', help='Run in verbose mode')
    args = parser.parse_args()

    # The optimal value function obtained
    V = None
    # The algorithm to run
    planner = None
    # The title of the plot
    title = None

    if args.dynaq:
        print('Running Dyna-Q')
        planner = DynaQ()
        title = 'dynaq'

    if planner is not None:
        V = planner.learn(epochs=args.epochs, n=args.n, alpha=args.alpha, gamma=args.gamma, verbose=args.verbose)

    if V is not None:
        # Plot the value function as a surface
        # Remove the state where the dealer's first card is 0 and the player's sum is 0 because these are not possible
        # They were kept in the value function to avoid having to deal with 0-index vs 1-index
        plt.plot_value_function(range(1, Easy21.state_space[0]), range(1, Easy21.state_space[1]), V[1:, 1:], title)
