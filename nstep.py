"""Implementation of different n-step Bootstrapping methods"""
import argparse
import numpy as np
import plot as plt
import random

from abc import ABC, abstractmethod
from env import Easy21
from policy import EpsilonGreedyPolicy, GreedyPolicy, Policy
from tqdm import tqdm

# For reproducibility
random.seed(0)


class NStepBootstrapping(ABC):
    """A base class defining n-step bootstrapping control algorithm"""

    def __init__(self):
        self._env = Easy21(seed=24)

    @abstractmethod
    def learn(self, epochs=200, n=10, alpha=0.5, gamma=0.9, verbose=False) -> np.ndarray:
        """
        Learns the optimal value function.

        :param int epochs: The number of epochs to take to learn the value function
        :param int n: The n-steps to use
        :param float alpha: The learning rate
        :param float gamma: The discount factor
        :param bool verbose: Whether to use verbose mode or not
        :return: The optimal value function
        :rtype: np.ndarray
        """
        raise NotImplementedError


class NStepSarsa(NStepBootstrapping):
    """A base class defining n-step SARSA"""

    def __init__(self, pi: Policy, b: Policy):
        """
        If `pi` is the same as `b` then this behaves as on-policy; otherwise, it follows an off-policy method with importance sampling.

        :param Policy pi: The target policy to use
        :param Policy b: The behavior policy to use
        """
        super().__init__()
        self._pi = pi
        self._b = b

    def learn(self, epochs=200, n=10, alpha=0.5, gamma=0.9, verbose=False) -> np.ndarray:
        Q = np.zeros((*self._env.state_space, self._env.action_space))

        for _ in tqdm(range(epochs), disable=not verbose):
            states = []
            actions = []
            rewards = []

            s = self._env.reset()
            states.append(s)

            # Take actions according to the behavior policy
            a = self._b[s]
            actions.append(a)

            # T controls the end of the episode
            T = np.inf
            # t is the current time step
            t = 0

            while True:
                if t < T:
                    s_prime, r, done = self._env.step(actions[t])

                    states.append(s_prime)
                    rewards.append(r)

                    if done:
                        # Stop in the next step
                        T = t + 1
                    else:
                        # Use again the behavior policy for decision-making
                        a_prime = self._b[s_prime]
                        actions.append(a_prime)

                # tau is the step whose estimate is being updated
                tau = t - n + 1
                if tau >= 0:
                    # Compute importance sampling ratio
                    # Note that the importance sampling ratio starts one step later (tau + 1) because we don't include the action taken at
                    # the current time step (tau); since it has already been taken, we want to fully learn from it and only weight subsequent ones
                    rho = np.prod(
                        [
                            self._pi.greedy_prob(states[i]) / self._b.greedy_prob(states[i])
                            for i in range(tau + 1, min(tau + n, T))
                        ]
                    )

                    # Compute approximate reward from the current step to n-steps later or the end of the episode (if tau + n goes beyond)
                    # Note that in the pseudocode presented by Sutton and Barto, they use (i - tau - 1) and (tau + 1) because they index the
                    # current reward as R_t+1; in this implementation, the reward is considered to be part of the current step R_t and hence
                    # we used tau instead of tau + 1
                    G = sum([gamma ** (i - tau) * rewards[i] for i in range(tau, min(tau + n, T))])

                    # Bootstrap the missing values if the we're not at the end of the episode
                    if tau + n < T:
                        s = states[tau + n]
                        a = actions[tau + n]
                        G += gamma ** n * Q[s.dealer_first_card, s.player_sum, a]

                    # Prediction (policy evaluation) of the *current* time step
                    s = states[tau]
                    a = actions[tau]
                    Q[s.dealer_first_card, s.player_sum, a] += (
                        alpha * rho * (G - Q[s.dealer_first_card, s.player_sum, a])
                    )

                    # Improvement of the *current* time step
                    # We use the target policy for learning
                    self._pi[s] = np.argmax(Q[s.dealer_first_card, s.player_sum, :])

                # Stop when we have reached the end of the episode
                if tau == T - 1:
                    break

                t += 1

        # Compute the optimal value function which is simply the value of the best action (last dimension) in each state
        return np.max(Q, axis=2)


class OnPolicyNStepSarsa(NStepSarsa):
    """On-policy n-step SARSA"""

    def __init__(self):
        # We use the same policy for target and behavior
        policy = EpsilonGreedyPolicy(seed=24)
        super().__init__(pi=policy, b=policy)


class OffPolicyNStepSarsa(NStepSarsa):
    """Off-policy n-step SARSA"""

    def __init__(self):
        # Use a greedy policy for the target while following a different, epsilon-soft policy for behavior
        super().__init__(pi=GreedyPolicy(), b=EpsilonGreedyPolicy(seed=24))


class NStepTreeBackup(NStepBootstrapping):
    """n-step Tree Backup: off-policy without importance sampling"""

    def learn(self, epochs=200, n=10, alpha=0.5, gamma=0.9, verbose=False) -> np.ndarray:
        Q = np.zeros((*self._env.state_space, self._env.action_space))
        pi = GreedyPolicy()
        b = EpsilonGreedyPolicy(seed=24)

        for _ in tqdm(range(epochs), disable=not verbose):
            states = []
            actions = []
            rewards = []

            s = self._env.reset()
            states.append(s)

            # Take actions according to the behavior policy
            a = b[s]
            actions.append(a)

            # T controls the end of the episode
            T = np.inf
            # t is the current time step
            t = 0

            while True:
                if t < T:
                    s_prime, r, done = self._env.step(actions[t])

                    states.append(s_prime)
                    rewards.append(r)

                    if done:
                        # Stop in the next step
                        T = t + 1
                    else:
                        # Use again the behavior policy for decision-making
                        a_prime = b[s_prime]
                        actions.append(a_prime)

                # tau is the step whose estimate is being updated
                tau = t - n + 1
                if tau >= 0:
                    # If we are at or go beyond the end of the episode, then we initialize the approximate return using the last observed reward
                    if tau + n >= T:
                        G = rewards[-1]
                    else:
                        # We initialize the approximate reward using the expected approximate value of the last step
                        s = states[t]
                        G = rewards[t] + gamma * sum(
                            [
                                pi.prob(a, s) * Q[s.dealer_first_card, s.player_sum, a]
                                for a in range(self._env.action_space)
                            ]
                        )

                    # Now compute the rest of the approximate reward by perfoming the tree backup starting from the last step down through our current step tau
                    for k in range(min(t, T - 1), tau, -1):
                        s = states[k]
                        G = (
                            rewards[k]
                            + gamma
                            * sum(
                                [
                                    pi.prob(a, s) * Q[s.dealer_first_card, s.player_sum, a]
                                    for a in range(self._env.action_space)
                                    if a != actions[k]
                                ]
                            )
                            + gamma * pi.prob(actions[k], s) * G
                        )

                    # Prediction (policy evaluation) of the *current* time step
                    s = states[tau]
                    a = actions[tau]
                    Q[s.dealer_first_card, s.player_sum, a] += alpha * (G - Q[s.dealer_first_card, s.player_sum, a])

                    # Improvement of the *current* time step
                    # We use the target policy for learning
                    pi[s] = np.argmax(Q[s.dealer_first_card, s.player_sum, :])

                # Stop when we have reached the end of the episode
                if tau == T - 1:
                    break

                t += 1

        # Compute the optimal value function which is simply the value of the best action (last dimension) in each state
        return np.max(Q, axis=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run n-step methods")
    parser.add_argument("--on-policy-sarsa", action="store_true", help="Execute On-policy n-step SARSA")
    parser.add_argument("--off-policy-sarsa", action="store_true", help="Execute Off-policy n-step SARSA")
    parser.add_argument("--tree-backup", action="store_true", help="Execute n-step Tree Backup")
    parser.add_argument("--epochs", type=int, default=200, help="Epochs to train")
    parser.add_argument("-n", type=int, default=10, help="n-steps to use")
    parser.add_argument("--gamma", type=float, default=0.9, help="Discount factor")
    parser.add_argument("--alpha", type=float, default=0.5, help="Learning rate")
    parser.add_argument("--verbose", action="store_true", help="Run in verbose mode")
    args = parser.parse_args()

    # The optimal value function obtained
    V = None
    # The algorithm to run
    n_step = None
    # The title of the plot
    title = None

    if args.on_policy_sarsa:
        print("Running On-policy n-step SARSA")
        n_step = OnPolicyNStepSarsa()
        title = "on_policy_n_step_sarsa"
    elif args.off_policy_sarsa:
        print("Running Off-policy n-step SARSA")
        n_step = OffPolicyNStepSarsa()
        title = "off_policy_n_step_sarsa"
    elif args.tree_backup:
        print("Running n-step Tree Backup")
        n_step = NStepTreeBackup()
        title = "n_step_tree_backup"

    if n_step is not None:
        V = n_step.learn(epochs=args.epochs, n=args.n, alpha=args.alpha, gamma=args.gamma, verbose=args.verbose)

    if V is not None:
        # Plot the value function as a surface
        # Remove the state where the dealer's first card is 0 and the player's sum is 0 because these are not possible
        # They were kept in the value function to avoid having to deal with 0-index vs 1-index
        plt.plot_value_function(range(1, Easy21.state_space[0]), range(1, Easy21.state_space[1]), V[1:, 1:], title)
