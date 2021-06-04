"""Implementation of different policy gradient methods"""
import argparse
import numpy as np
import plot as plt
import random

from collections import namedtuple
from env import Action, Easy21, State
from tqdm import tqdm
from typing import Callable, List

# For reproducibility
random.seed(0)
np.random.seed(0)

Trajectory = namedtuple("Trajectory", ["state", "action", "reward"])


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


def softmax(x: np.ndarray) -> np.ndarray:
    """
    Computes the softmax of the given array

    :param np.ndarray x: The input array
    :return: The softmax of each element of the input array
    :rtype: np.ndarray
    """
    return np.exp(x) / np.sum(np.exp(x))


class REINFORCEWithBaseline:
    """
    REINFORCE algorithm with baseline

    Uses softmax on linear action preferences for the policy, and
    linear approximation for the value function. Feature vectors
    are computed using coarse coding as described in the Easy21
    assignment.
    """

    def __init__(self):
        self._env = Easy21(seed=24)

    def learn(self, epochs=200, alpha_policy=0.01, alpha_value=0.01, gamma=0.9, verbose=False, **kwargs) -> np.ndarray:
        """
        Learns the optimal value function.

        :param int epochs: The number of epochs to take to learn the value function
        :param float alpha_policy: The learning rate for the policy approximation
        :param float alpha_value: The learning rate for the value approximation
        :param float gamma: The discount factor
        :param bool verbose: Whether to use verbose mode or not
        :return: The optimal value function
        :rtype: np.ndarray
        """
        # Value function
        w = np.random.rand(36)
        value_approximator = lambda s: [np.dot(w, encode(s, a)) for a in [Action.hit, Action.stick]]
        # Policy function
        theta = np.random.rand(36)
        pi = lambda s, theta: softmax(np.array([np.dot(theta, encode(s, a)) for a in [Action.hit, Action.stick]]))

        for _ in tqdm(range(epochs), disable=not verbose):
            trajectories = self._sample_episode(pi, theta)
            # Reverse the list so we start backpropagating the return from the last episode
            trajectories.reverse()

            # Learn from the episode
            g = 0
            for i, t in enumerate(trajectories):
                g = t.reward + gamma * g
                x = encode(t.state, t.action)

                # Baseline
                v = np.dot(w, x)
                delta = g - v

                # SGD update of the value function
                w += alpha_value * delta * x

                # SGD update of the policy function
                probs = pi(t.state, theta)
                eligibility_vector = x - np.sum([p * encode(t.state, a) for a, p in enumerate(probs)])
                theta += alpha_policy * gamma ** i * delta * eligibility_vector

        # Compute the optimal value function which is simply the value of the best action in each state
        values = np.zeros(self._env.state_space)
        for d in range(self._env.state_space[0]):
            for p in range(self._env.state_space[1]):
                values[d, p] = np.max(value_approximator(State(d, p)))

        return values

    def _sample_episode(self, pi: Callable[[State, Action, np.ndarray], float], theta: np.ndarray) -> List[Trajectory]:
        # Samples trajectories following policy `pi` with an optional starting state-action pair
        trajectories = []

        s = self._env.reset()
        # The policy selects the action with some constant exploration as in the Easy21 assignment
        policy = (
            lambda s: random.choice([Action.hit, Action.stick]) if random.random() < 0.05 else np.argmax(pi(s, theta))
        )
        a = policy(s)

        while True:
            s_prime, r, done = self._env.step(a)
            trajectories.append(Trajectory(s, a, r))

            if done:
                break

            s = s_prime
            a = policy(s)

        return trajectories


class OneStepActorCritic:
    """
    One-step Actor-Critic

    Uses softmax on linear action preferences for the policy, and
    linear approximation for the value function. Feature vectors
    are computed using coarse coding as described in the Easy21
    assignment.
    """

    def __init__(self):
        self._env = Easy21(seed=24)

    def learn(self, epochs=200, alpha_policy=0.01, alpha_value=0.01, gamma=0.9, verbose=False, **kwargs) -> np.ndarray:
        """
        Learns the optimal value function.

        :param int epochs: The number of epochs to take to learn the value function
        :param float alpha_policy: The learning rate for the policy approximation
        :param float alpha_value: The learning rate for the value approximation
        :param float gamma: The discount factor
        :param bool verbose: Whether to use verbose mode or not
        :return: The optimal value function
        :rtype: np.ndarray
        """
        # Value function
        w = np.random.rand(36)
        value_approximator = lambda s: [np.dot(w, encode(s, a)) for a in [Action.hit, Action.stick]]
        # Policy function
        theta = np.random.rand(36)
        pi = lambda s, theta: softmax(np.array([np.dot(theta, encode(s, a)) for a in [Action.hit, Action.stick]]))
        # The policy selects the action with some constant exploration as in the Easy21 assignment
        policy = (
            lambda s: random.choice([Action.hit, Action.stick]) if random.random() < 0.05 else np.argmax(pi(s, theta))
        )

        for _ in tqdm(range(epochs), disable=not verbose):
            I = 1
            s = self._env.reset()
            done = False

            while not done:
                a = policy(s)
                s_prime, r, done = self._env.step(a)

                # Compute the delta
                if done:
                    delta = r - np.dot(w, encode(s, a))
                else:
                    delta = r + gamma * np.max(value_approximator(s_prime)) - np.dot(w, encode(s, a))

                # SGD update of the value function
                x = encode(s, a)
                w += alpha_value * delta * x

                # SGD update of the policy function
                probs = pi(s, theta)
                eligibility_vector = x - np.sum([p * encode(s, a) for a, p in enumerate(probs)])
                theta += alpha_policy * I * delta * eligibility_vector

                I *= gamma
                s = s_prime

        # Compute the optimal value function which is simply the value of the best action in each state
        values = np.zeros(self._env.state_space)
        for d in range(self._env.state_space[0]):
            for p in range(self._env.state_space[1]):
                values[d, p] = np.max(value_approximator(State(d, p)))

        return values


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run policy gradient methods")
    parser.add_argument("--reinforce-with-baseline", action="store_true", help="Execute REINFORCE with Baseline")
    parser.add_argument("--one-step-ac", action="store_true", help="Execute One-step Actor-Critic")
    parser.add_argument("--epochs", type=int, default=200, help="Epochs to train")
    parser.add_argument(
        "--alpha-value", type=float, default=0.01, help="Learning rate to use for the value function approximation"
    )
    parser.add_argument(
        "--alpha-policy", type=float, default=0.01, help="Learning rate to use for the policy function approximation"
    )
    parser.add_argument("--gamma", type=float, default=0.9, help="Discount factor")
    parser.add_argument("--verbose", action="store_true", help="Run in verbose mode")
    args = parser.parse_args()

    # The optimal value function obtained
    V = None
    # The algorithm to run
    policy_grad = None
    # The title of the plot
    title = None

    if args.reinforce_with_baseline:
        print("Running REINFORCE with Baseline")
        policy_grad = REINFORCEWithBaseline()
        title = "reinforce_with_baseline"
    elif args.one_step_ac:
        print("Running One-step Actor-Critic")
        policy_grad = OneStepActorCritic()
        title = "one_step_actor_critic"

    if policy_grad is not None:
        V = policy_grad.learn(
            epochs=args.epochs,
            alpha_value=args.alpha_value,
            alpha_policy=args.alpha_policy,
            gamma=args.gamma,
            verbose=args.verbose,
        )

    if V is not None:
        # Plot the value function as a surface
        # Remove the state where the dealer's first card is 0 and the player's sum is 0 because these are not possible
        # They were kept in the value function to avoid having to deal with 0-index vs 1-index
        plt.plot_value_function(range(1, Easy21.state_space[0]), range(1, Easy21.state_space[1]), V[1:, 1:], title)
