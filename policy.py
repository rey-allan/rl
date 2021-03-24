"""Implementation of different types of policies"""
import numpy as np
import random

from abc import ABC, abstractmethod
from collections import defaultdict
from env import Action, Easy21, State


class Policy(ABC):
    """A base policy. This class should not be instantiated."""

    def __init__(self):
        self._pi = np.full(Easy21.state_space, fill_value=Action.hit)

    def __setitem__(self, s: State, a: Action) -> None:
        """
        Sets the given action for the given state.

        :param State s: The state to update
        :param Action a: The action to assign to the state
        """
        self._pi[s.dealer_first_card, s.player_sum] = a

    @abstractmethod
    def __getitem__(self, s: State) -> Action:
        """
        Retrieves the action for the given state.

        :param State s: The state to retrieve an action for
        :return: The action
        :rtype: Action
        """
        raise NotImplementedError

    @abstractmethod
    def greedy_prob(self, s: State) -> float:
        """
        Returns the probability of selecting a greedy action under this policy in state `s`.

        :param State s: The state to compute the greedy probability for
        :return: The greedy probability
        :rtype: float
        """
        raise NotImplementedError

    @abstractmethod
    def prob(self, a: Action, s: State) -> float:
        """
        Returns the probability of selecting action `a` in state `s` under this policy.

        :param Action a: The action
        :param State s: The state
        :return: The probability
        :rtype: float
        """
        raise NotImplementedError


class GreedyPolicy(Policy):
    """A greedy policy that selects actions based on its current mapping"""

    def __getitem__(self, s: State) -> Action:
        # Picks the action based on the current policy
        return self._pi[s.dealer_first_card, s.player_sum]

    def greedy_prob(self, s: State) -> float:
        return 1.

    def prob(self, a: Action, s: State) -> float:
        return 1. if a == self._pi[s.dealer_first_card, s.player_sum] else 0.


class EpsilonGreedyPolicy(Policy):
    """
    An epsilon greedy policy that selects random actions with probability epsilon.
    It follows the exploration strategy described in the Easy21 assignment instructions.
    """

    def __init__(self, seed: int = None):
        """
        :param int seed: The seed to use for the random number generator
        """
        super().__init__()

        random.seed(seed)
        self._n0 = 100.
        # Number of times a state has been visited
        self._n = defaultdict(int)

    def __getitem__(self, s: State) -> Action:
        # Compute epsilon following the strategy outlined in the assignment instructions
        self._n[s] += 1
        epsilon = self._epsilon(s)

        if random.random() < epsilon:
            return random.choice([Action.hit, Action.stick])

        return self._pi[s.dealer_first_card, s.player_sum]

    def greedy_prob(self, s: State) -> float:
        return 1. - self._epsilon(s)

    def prob(self, a: Action, s: State) -> float:
        eps = self._epsilon(s)
        return 1. - eps if a == self._pi[s.dealer_first_card, s.player_sum] else eps

    def _epsilon(self, s: State) -> float:
        return self._n0 / (self._n0 + self._n[s])
