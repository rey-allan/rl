"""Implementation of different methods that integrate planning with learning"""
import argparse
import numpy as np
import plot as plt
import random

from abc import ABC
from collections import defaultdict, namedtuple
from env import Action, Easy21, State
from heapq import heapify, heappop, heappush
from policy import EpsilonGreedyPolicy, Policy, RandomPolicy
from tqdm import tqdm
from typing import List

# For reproducibility
random.seed(0)

Trajectory = namedtuple("Trajectory", ["state", "action", "reward"])


class Planning(ABC):
    """A base class defining a planning algorithm"""

    def __init__(self):
        self._env = Easy21(seed=24)
        self.Q = np.zeros((*self._env.state_space, self._env.action_space))
        # Model of the world represented by approximations of the transition and reward functions
        # We initialize the transition function with a very small value greater than 0 to avoid
        # division by zero when computing transition probabilities for trajectories that haven't been encountered
        self.T = np.full((*self._env.state_space, self._env.action_space, *self._env.state_space), 0.00001)
        self.R = np.zeros((*self._env.state_space, self._env.action_space))

    def _update_model(self, s, a, r, s_prime, done):
        if not done:
            # We only update the transition model for non-terminal states since the terminal state
            # is most likely an "invalid" state for this environment, e.g. a player's sum over 21
            self.T[s.dealer_first_card, s.player_sum, a, s_prime.dealer_first_card, s_prime.player_sum] += 1
        # We update the model of the reward by learning it similarly to the action values
        # This is not specified in Sutton & Barto's book but I learned it during my Master's
        self.R[s.dealer_first_card, s.player_sum, a] += 0.2 * (r - self.R[s.dealer_first_card, s.player_sum, a])

    def _learn(self, pi, s, a, r, s_prime, done, alpha, gamma):
        if done:
            td_target = r
        else:
            td_target = r + gamma * np.max(self.Q[s_prime.dealer_first_card, s_prime.player_sum, :])

        td_error = td_target - self.Q[s.dealer_first_card, s.player_sum, a]

        # Prediction
        self.Q[s.dealer_first_card, s.player_sum, a] += alpha * td_error
        # Improvement
        pi[s] = np.argmax(self.Q[s.dealer_first_card, s.player_sum, :])


class DynaQ(Planning):
    """Dyna-Q algorithm"""

    def learn(self, epochs=200, n=100, alpha=0.5, gamma=0.9, verbose=False, **kwargs) -> np.ndarray:
        """
        Learns the optimal value function.

        :param int epochs: The number of epochs to take to learn the value function
        :param int n: The planning iterations to use
        :param float alpha: The learning rate
        :param float gamma: The discount factor
        :param bool verbose: Whether to use verbose mode or not
        :param dict kwargs: Extra arguments, ignored
        :return: The optimal value function
        :rtype: np.ndarray
        """
        pi = EpsilonGreedyPolicy(seed=24)

        for _ in tqdm(range(epochs), disable=not verbose):
            done = False
            s = self._env.reset()

            while not done:
                a = pi[s]
                s_prime, r, done = self._env.step(a)

                # Learning phase
                self._learn(pi, s, a, r, s_prime, done, alpha, gamma)
                # Planning phase
                if n > 0:
                    self._update_model(s, a, r, s_prime, done)
                    self._plan(pi, done, n, alpha, gamma)

                s = s_prime

        # Compute the optimal value function which is simply the value of the best action (last dimension) in each state
        return np.max(self.Q, axis=2)

    def _plan(self, pi, done, n, alpha, gamma):
        # Compute the probabilities of each s,a -> s' transition over all possible transitions from each s,a
        transition_probs = self.T / np.sum(self.T, axis=(0, 1, 2))
        dealer_state_space = list(range(self._env.state_space[0]))
        player_state_space = list(range(self._env.state_space[1]))
        action_space = list(range(self._env.action_space))

        for _ in range(n):
            # Select a random (s,a) pair uniformly from the _full_ state/action space as opposed to only
            # the ones that have been previously observed. This matches the enhancements of Dyna-Q+ as
            # described by Sutton & Barto. It was also what I did for one of my projects during the Master's.
            s = State(random.choice(dealer_state_space), random.choice(player_state_space))
            a = random.choice(action_space)
            r = self.R[s.dealer_first_card, s.player_sum, a]
            # Infer s' by using the state with the highest probability
            # This is different from what Sutton & Barto describe. They mention that expected updates
            # should be used for stochastic environments. However, I've used this kind of update before
            # and it works much better in practice
            # We use `unravel_index` to get the state as a pair of (dealer_first_card,player_sum)
            s_prime = np.unravel_index(
                np.argmax(transition_probs[s.dealer_first_card, s.player_sum, a]),
                transition_probs[s.dealer_first_card, s.player_sum, a].shape,
            )
            self._learn(pi, s, a, r, State(*s_prime), done, alpha, gamma)


class PrioritizedSweeping(Planning):
    """Prioritized sweeping algorithm"""

    class PriorityQueue:
        """A max-heap based priority queue for state-action pairs"""

        def __init__(self):
            self._heap = []
            heapify(self._heap)

        def push(self, s, a, priority):
            # `heapq` is implemented as a min-heap to use it as a max-heap we negate the priority
            heappush(self._heap, (-1 * priority, (s, a)))

        def pop(self):
            return heappop(self._heap)

        def empty(self):
            return len(self._heap) == 0

    def learn(self, epochs=200, n=100, alpha=0.5, gamma=0.9, theta=0.5, verbose=False, **kwargs) -> np.ndarray:
        """
        Learns the optimal value function.

        :param int epochs: The number of epochs to take to learn the value function
        :param int n: The planning iterations to use
        :param float alpha: The learning rate
        :param float gamma: The discount factor
        :param float theta: The threshold that determines whether updates should be prioritized or not
        :param bool verbose: Whether to use verbose mode or not
        :param dict kwargs: Extra arguments, ignored
        :return: The optimal value function
        :rtype: np.ndarray
        """
        pi = EpsilonGreedyPolicy(seed=24)
        queue = self.PriorityQueue()

        for _ in tqdm(range(epochs), disable=not verbose):
            done = False
            s = self._env.reset()

            while not done:
                a = pi[s]
                s_prime, r, done = self._env.step(a)

                # Prioritization
                if done:
                    td_target = r
                else:
                    td_target = r + gamma * np.max(self.Q[s_prime.dealer_first_card, s_prime.player_sum, :])
                # The absolute `td_error` is the priority for this update
                td_error = abs(td_target - self.Q[s.dealer_first_card, s.player_sum, a])

                if td_error > theta:
                    queue.push(s, a, td_error)

                # Planning phase
                if n > 0:
                    self._update_model(s, a, r, s_prime, done)
                    self._plan(queue, pi, done, n, alpha, gamma, theta)

                s = s_prime

        # Compute the optimal value function which is simply the value of the best action (last dimension) in each state
        return np.max(self.Q, axis=2)

    def _plan(self, queue, pi, done, n, alpha, gamma, theta):
        # Compute the probabilities of each s,a -> s' transition over all possible transitions from each s,a
        transition_probs = self.T / np.sum(self.T, axis=(0, 1, 2))

        for _ in range(n):
            if queue.empty():
                break

            # Select the (s,a) pair with the most priority
            _, (s, a) = queue.pop()
            r = self.R[s.dealer_first_card, s.player_sum, a]
            # Infer s' by using the state with the highest probability
            # We use `unravel_index` to get the state as a pair of (dealer_first_card,player_sum)
            s_prime = np.unravel_index(
                np.argmax(transition_probs[s.dealer_first_card, s.player_sum, a]),
                transition_probs[s.dealer_first_card, s.player_sum, a].shape,
            )
            self._learn(pi, s, a, r, State(*s_prime), done, alpha, gamma)

            # Since we updated s, this update has an effect on its predecessors because its value is backed up
            # However, we don't want to update all predecessors, we want to update the (s-,a-) pair with the
            # highest probability to _lead_ to s
            dealer_first_card, player_sum, a_bar = np.unravel_index(
                np.argmax(transition_probs[:, :, :, s.dealer_first_card, s.player_sum]),
                transition_probs[:, :, :, s.dealer_first_card, s.player_sum].shape,
            )
            s_bar = State(dealer_first_card, player_sum)
            # Predict the reward
            r_bar = self.R[dealer_first_card, player_sum, a_bar]
            # Compute its priority again for when it should be updated
            td_target = r_bar + gamma * np.max(self.Q[s.dealer_first_card, s.player_sum, :])
            td_error = abs(td_target - self.Q[s_bar.dealer_first_card, s_bar.player_sum, a_bar])

            if td_error > theta:
                queue.push(s_bar, a_bar, td_error)


class MonteCarloTreeSearch(Planning):
    """Monte Carlo Tree Search algorithm"""

    def learn(self, epochs=200, n=100, gamma=0.9, verbose=False, **kwargs) -> np.ndarray:
        """
        Learns the optimal value function.

        :param int epochs: The number of epochs to take to learn the value function
        :param int n: The planning iterations to use
        :param float gamma: The discount factor
        :param bool verbose: Whether to use verbose mode or not
        :param dict kwargs: Extra arguments, ignored
        :return: The optimal value function
        :rtype: np.ndarray
        """
        tree_policy = EpsilonGreedyPolicy(seed=24)
        rollout_policy = RandomPolicy(seed=24)

        for _ in tqdm(range(epochs), disable=not verbose):
            done = False
            current_state = self._env.reset()
            explored = set()
            # For Monte Carlo learning from the simulated experiences (averaging the returns)
            returns = defaultdict(list)

            while not done:
                # Run MCTS
                self._plan(n, gamma, tree_policy, rollout_policy, current_state, explored, returns)

                # Action selection for the current state
                self._env.reset(start=current_state)
                a = tree_policy[current_state]
                s_prime, _, done = self._env.step(a)
                current_state = s_prime

        # Compute the optimal value function which is simply the value of the best action (last dimension) in each state
        return np.max(self.Q, axis=2)

    def _plan(self, n, gamma, tree_policy, rollout_policy, current_state, explored, returns):
        for _ in range(n):
            tree = []
            s = current_state
            self._env.reset(start=s)

            # Selection (traverse the tree until finding a leaf node)
            # A leaf node is a node without explored children
            while s in explored:
                a = tree_policy[s]
                s_prime, r, done = self._env.step(a)
                tree.append(Trajectory(s, a, r))
                s = s_prime
                if done:
                    break

            # Expansion
            if not done:
                a = tree_policy[s]
                s_prime, r, done = self._env.step(a)
                tree.append(Trajectory(s, a, r))
                explored.add(s)
                s = s_prime

            # Simulation
            simulated = []
            if not done:
                simulated = self._sample_episode(rollout_policy, s_0=s)

            # Backup
            trajectories = tree + simulated
            # Learning from all trajectories only for those (s,a) pairs that are part of the tree
            to_learn_start = len(trajectories) - len(tree)
            # Reverse the list so we start backpropagating the return from the last episode
            trajectories.reverse()
            g = 0
            for i, t in enumerate(trajectories):
                g = t.reward + gamma * g
                returns[(*t.state, t.action)].append(g)

                if i >= to_learn_start:
                    # Prediction
                    self.Q[t.state.dealer_first_card, t.state.player_sum, t.action] = np.squeeze(
                        np.mean(returns[(*t.state, t.action)])
                    )
                    # Improvement using the tree policy
                    tree_policy[t.state] = np.argmax(self.Q[t.state.dealer_first_card, t.state.player_sum, :])

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run planning methods")
    parser.add_argument("--dynaq", action="store_true", help="Execute Dyna-Q")
    parser.add_argument("--priority", action="store_true", help="Execute Prioritized sweeping")
    parser.add_argument("--mcts", action="store_true", help="Executes Monte Carlo Tree Search")
    parser.add_argument("--epochs", type=int, default=200, help="Epochs to train")
    parser.add_argument("--gamma", type=float, default=0.9, help="Discount factor")
    parser.add_argument("--alpha", type=float, default=0.5, help="Learning rate")
    parser.add_argument("--n", type=int, default=100, help="Planning steps to use")
    parser.add_argument("--theta", type=float, default=0.5, help="Threshold for Prioritized sweeping")
    parser.add_argument("--verbose", action="store_true", help="Run in verbose mode")
    args = parser.parse_args()

    # The optimal value function obtained
    V = None
    # The algorithm to run
    planner = None
    # The title of the plot
    title = None

    if args.dynaq:
        print("Running Dyna-Q")
        planner = DynaQ()
        title = "dynaq"
    elif args.priority:
        print("Running Prioritized sweeping")
        planner = PrioritizedSweeping()
        title = "prioritized_sweeping"
    elif args.mcts:
        print("Running Monte Carlo Tree Search")
        planner = MonteCarloTreeSearch()
        title = "mcts"

    if planner is not None:
        V = planner.learn(
            epochs=args.epochs, n=args.n, alpha=args.alpha, gamma=args.gamma, theta=args.theta, verbose=args.verbose
        )

    if V is not None:
        # Plot the value function as a surface
        # Remove the state where the dealer's first card is 0 and the player's sum is 0 because these are not possible
        # They were kept in the value function to avoid having to deal with 0-index vs 1-index
        plt.plot_value_function(range(1, Easy21.state_space[0]), range(1, Easy21.state_space[1]), V[1:, 1:], title)
