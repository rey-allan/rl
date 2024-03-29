"""Environment implementing the Easy21 game"""
import random

from collections import namedtuple
from enum import Enum
from typing import Tuple

State = namedtuple("State", ["dealer_first_card", "player_sum"])
Color = Enum("Color", "red black")


class Action:
    """Represents all possible actions"""

    # We aren't using an enum because it's a pain having to convert
    # back and forth between numbers and the enum types.
    # Since actions are usually directly used with NumPy arrays in
    # learning algorithms, it's better to explicitly keep them as such.
    hit = 0
    stick = 1


class Easy21:
    """Easy21: A simplified Balckjack-like card game"""

    action_space = 2
    # Dealer’s first card 1–10 and the player’s sum 1–21
    # We add 1 to account for zero-based indexing
    state_space = (11, 22)

    def __init__(self, seed: int = None):
        """
        :param int seed: The seed to use for the random number generator
        """
        random.seed(seed)
        self._player_sum = 0
        self._dealer_sum = 0
        self._dealer_first_card = 0
        self._done = False

    def reset(self, start: State = None) -> State:
        """
        Resets the environment with an optional starting state

        :param State start: An optional starting state to reset the environment to
        :return: The initial state
        :rtype: State
        """
        self._player_sum = 0
        self._dealer_sum = 0
        self._done = False

        if start is not None:
            self._dealer_first_card = start.dealer_first_card
            self._dealer_sum += start.dealer_first_card
            self._player_sum = start.player_sum

            return State(start.dealer_first_card, start.player_sum)

        # By default, at the start of the game both the player and the dealer draw one black card
        dealer = Deck.draw(color=Color.black)
        player = Deck.draw(color=Color.black)

        self._player_sum += player
        self._dealer_sum += dealer
        self._dealer_first_card = dealer.value

        return State(dealer.value, player.value)

    def step(self, a: Action) -> Tuple[State, float, bool]:
        """
        Steps into the environment by taking action `a`

        :param Action a: The action to take
        :return: A tuple of next state, reward, and done
        :rtype: Tuple[State, float, bool]
        """
        if self._done:
            raise RuntimeError("Cannot step into terminated episode; call `reset()`!")

        # If the player hits then she draws another card from the deck
        if a == Action.hit:
            card = Deck.draw()
            self._player_sum += card
            s_prime = State(self._dealer_first_card, self._player_sum)
            # If the player’s sum exceeds 21, or becomes less than 1,
            # then she "goes bust" and loses the game (reward -1)
            bust = self._player_sum > 21 or self._player_sum < 1
            reward = -1 if bust else 0
            done = bust
        elif a == Action.stick:
            self._play_dealers_turn()
            s_prime = State(self._dealer_first_card, self._player_sum)
            # If the dealer goes bust, then the player wins; otherwise,
            # the outcome win (reward +1), lose (reward -1), or draw (reward 0)
            # is the player with the largest sum
            if self._dealer_sum > 21 or self._player_sum > self._dealer_sum:
                reward = 1
            elif self._player_sum < self._dealer_sum:
                reward = -1
            else:
                reward = 0
            done = True

        self._done = done
        return s_prime, reward, done

    def render(self):
        """
        Renders the environment's current state
        """
        print("===============")
        print(f"Player Sum: {self._player_sum}")
        print(f"Dealer Sum: {self._dealer_sum}")

    def _play_dealers_turn(self):
        #  The dealer always sticks on any sum of 17 or greater, and hits otherwise
        while self._dealer_sum < 17:
            card = Deck.draw()
            self._dealer_sum += card


class Card:
    """Represents a card in Easy21"""

    def __init__(self, value: int, color: Color):
        """
        :param int value: The value of the card [1-10]
        :param Color color: The color of the card [red|black]
        """
        self.value = value
        self.color = color
        # Black cards are added and red cards subtracted
        self._value = value * (1 if color == Color.black else -1)

    def __add__(self, other) -> int:
        return other.value + self._value

    def __radd__(self, other: int) -> int:
        return other + self._value


class Deck:
    """Represents a deck of Easy21 cards"""

    @staticmethod
    def draw(color: Color = None) -> Card:
        """
        Draws a card from the deck

        If `color` is passed then the card is assigned to it, else a random one is selected

        :param Color color: An optional color to assign the card to
        :return: A card
        :rtype: Card
        """
        value = random.randint(1, 10)
        color = color or random.choices([Color.red, Color.black], weights=[1.0 / 3.0, 2.0 / 3.0], k=1)[0]
        return Card(value, color)


if __name__ == "__main__":
    env = Easy21(seed=24)
    s = env.reset()

    while True:
        env.render()
        # Pick a random action
        a = random.choice([Action.hit, Action.stick])
        s_prime, r, done = env.step(a)

        print(f"Action: {a.name}")
        print(f"Reward: {r}")

        if done:
            env.render()
            break
