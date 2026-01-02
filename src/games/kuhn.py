"""
Kuhn Poker implementation.

Kuhn Poker is a simplified poker game:
- 3 cards: Jack (J), Queen (Q), King (K)
- 2 players, 1 card each
- Ante: 1 chip each
- Actions: Pass (check) or Bet (1 chip)
- Higher card wins at showdown
- ~12 information sets

Game flow:
1. Both players ante 1 chip
2. Each dealt 1 card from {J, Q, K}
3. P1 acts first: Pass or Bet
4. If P1 passes: P2 can Pass (showdown) or Bet
5. If P1 bets: P2 can Fold or Call
6. If P1 passes, P2 bets: P1 can Fold or Call
7. Higher card wins at showdown
"""

from typing import List, Tuple, Optional
from itertools import permutations
import numpy as np

from .base import Game, GameState, InfoSet


# Card constants
JACK = 0
QUEEN = 1
KING = 2
CARD_NAMES = ['J', 'Q', 'K']

# Action constants
PASS = 0   # Pass or Check
BET = 1    # Bet or Call


class KuhnPoker(Game):
    """Kuhn Poker game implementation."""

    def __init__(self):
        self._cards = [JACK, QUEEN, KING]
        self._ante = 1

    @property
    def num_players(self) -> int:
        return 2

    @property
    def num_actions(self) -> int:
        return 2  # Pass/Check, Bet/Call

    @property
    def name(self) -> str:
        return "kuhn"

    def initial_state(self) -> GameState:
        """
        Return initial state (chance node for dealing).
        At initial state, no cards dealt yet.
        """
        return GameState(
            player_cards=(),  # Empty - will be dealt at chance node
            public_card=None,
            pot=2 * self._ante,  # Both players ante
            current_player=-1,   # Chance node indicator
            bets=(self._ante, self._ante),
            stacks=(0, 0),       # Not used in Kuhn
            round_num=0,
            num_raises=0,
            history=(),
            is_terminal=False,
        )

    def is_chance_node(self, state: GameState) -> bool:
        """Chance node when cards not yet dealt."""
        return len(state.player_cards) == 0

    def chance_outcomes(self, state: GameState) -> List[Tuple[GameState, float]]:
        """
        All possible card dealings.
        Each of the 6 permutations has equal probability.
        """
        if not self.is_chance_node(state):
            return []

        outcomes = []
        # All permutations of dealing 2 cards from 3
        for cards in permutations(self._cards, 2):
            new_state = state.copy()
            new_state.player_cards = cards
            new_state.current_player = 0  # P1 acts first after deal
            # Probability: 1/6 for each of 6 permutations
            outcomes.append((new_state, 1.0 / 6.0))

        return outcomes

    def current_player(self, state: GameState) -> int:
        """Get current player to act."""
        if self.is_chance_node(state):
            return -1  # Nature/chance
        return state.current_player

    def get_infoset(self, state: GameState, player: int) -> InfoSet:
        """
        Get information set for a player.
        In Kuhn, player knows their own card and the action history.
        """
        return InfoSet(
            player=player,
            public_state=(state.pot, state.history),
            private_info=(state.player_cards[player],),
            history=state.history,
        )

    def get_actions(self, state: GameState) -> List[int]:
        """
        Get legal actions.
        In Kuhn, always 2 actions: Pass/Check (0) or Bet/Call (1)
        """
        return [PASS, BET]

    def apply_action(self, state: GameState, action: int) -> GameState:
        """
        Apply action and return new state.

        Game tree:
        - P1: Pass or Bet
        - If P1 Pass: P2 Pass (showdown) or Bet
        - If P1 Bet: P2 Fold (P1 wins) or Call (showdown)
        - If P1 Pass, P2 Bet: P1 Fold (P2 wins) or Call (showdown)
        """
        new_state = state.copy()
        history = list(state.history)
        current = state.current_player
        opponent = 1 - current

        action_name = 'p' if action == PASS else 'b'
        history.append(action_name)
        new_state.history = tuple(history)

        # Update state based on history
        h = new_state.history

        if h == ('p', 'p'):
            # P1 pass, P2 pass -> showdown
            new_state.is_terminal = True
            new_state.terminal_type = "showdown"
        elif h == ('p', 'b', 'p'):
            # P1 pass, P2 bet, P1 fold -> P2 wins
            new_state.is_terminal = True
            new_state.terminal_type = "fold"
            new_state.winner = 1  # P2 wins
        elif h == ('p', 'b', 'b'):
            # P1 pass, P2 bet, P1 call -> showdown
            new_state.is_terminal = True
            new_state.terminal_type = "showdown"
            new_state.pot = state.pot + 2  # Both add 1 for the bet and call
        elif h == ('b', 'p'):
            # P1 bet, P2 fold -> P1 wins
            new_state.is_terminal = True
            new_state.terminal_type = "fold"
            new_state.winner = 0  # P1 wins
        elif h == ('b', 'b'):
            # P1 bet, P2 call -> showdown
            new_state.is_terminal = True
            new_state.terminal_type = "showdown"
            new_state.pot = state.pot + 2  # Both add 1
        elif h == ('p', 'b'):
            # P1 pass, P2 bet -> P1 to respond
            new_state.current_player = 0
            new_state.pot = state.pot + 1  # P2 bet 1
        elif h == ('p',):
            # P1 pass -> P2 to act
            new_state.current_player = 1
        elif h == ('b',):
            # P1 bet -> P2 to act
            new_state.current_player = 1
            new_state.pot = state.pot + 1  # P1 bet 1
        else:
            raise ValueError(f"Unexpected history: {h}")

        return new_state

    def is_terminal(self, state: GameState) -> bool:
        """Check if game is over."""
        return state.is_terminal

    def get_payoffs(self, state: GameState) -> np.ndarray:
        """
        Get payoffs at terminal state.
        Payoffs are relative to initial state (ante already paid).
        """
        if not state.is_terminal:
            raise ValueError("Cannot get payoffs for non-terminal state")

        payoffs = np.zeros(2)

        if state.terminal_type == "fold":
            # Winner takes the pot
            winner = state.winner
            loser = 1 - winner

            # Calculate what each player put in
            # Pot size tells us total chips in
            if state.history == ('b', 'p'):
                # P1 bet (ante+1), P2 folded (ante only)
                payoffs[0] = self._ante  # P1 wins P2's ante
                payoffs[1] = -self._ante  # P2 loses ante
            elif state.history == ('p', 'b', 'p'):
                # P1 check, P2 bet, P1 fold
                payoffs[0] = -self._ante  # P1 loses ante
                payoffs[1] = self._ante   # P2 wins P1's ante
            else:
                # Generic fold handling
                payoffs[winner] = state.pot / 2
                payoffs[loser] = -state.pot / 2

        elif state.terminal_type == "showdown":
            # Higher card wins
            p1_card = state.player_cards[0]
            p2_card = state.player_cards[1]

            if p1_card > p2_card:
                winner = 0
            else:
                winner = 1
            loser = 1 - winner

            # Calculate chips each player put in
            h = state.history
            if h == ('p', 'p'):
                # Both just ante
                payoffs[winner] = self._ante
                payoffs[loser] = -self._ante
            elif h == ('b', 'b') or h == ('p', 'b', 'b'):
                # Both ante + bet
                payoffs[winner] = self._ante + 1
                payoffs[loser] = -(self._ante + 1)
            else:
                # Fallback
                payoffs[winner] = state.pot / 2
                payoffs[loser] = -state.pot / 2

        return payoffs

    def get_action_name(self, action: int) -> str:
        """Convert action to string."""
        return 'pass' if action == PASS else 'bet'

    def card_name(self, card: int) -> str:
        """Convert card index to name."""
        return CARD_NAMES[card]

    def __repr__(self) -> str:
        return "KuhnPoker()"
