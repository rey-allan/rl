"""Implementation of different types of policies"""
import numpy as np

from abc import ABC, abstractmethod
from env import Action, Easy21, State


class Policy(ABC):
    """A base policy. This class should not be instantiated."""

    def __init__(self):
        self._pi = np.full(Easy21.state_space, fill_value=Action.hit)

    @abstractmethod
    def __setitem__(self, s: State, a: Action) -> None:
        """
        Sets the given action for the given state.

        :param State s: The state to update
        :param Action a: The action to assign to the state
        """
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, s: State) -> Action:
        """
        Retrieves the action for the given state.

        :param State s: The state to retrieve an action for
        :return: The action
        :rtype: Action
        """
        raise NotImplementedError


class GreedyPolicy(Policy):
    """A greedy policy that selects actions based on its current mapping"""

    def __setitem__(self, s: State, a: Action) -> None:
        self._pi[s.dealer_first_card, s.player_sum] = a

    def __getitem__(self, s: State) -> Action:
        # Picks the action based on the current policy
        return self._pi[s.dealer_first_card, s.player_sum]
