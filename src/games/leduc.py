"""
Leduc Poker implementation.

Leduc Poker is a simplified poker game:
- 6 cards: 2 Jacks, 2 Queens, 2 Kings
- 2 players, 1 private card each
- 2 betting rounds (preflop + flop with 1 community card)
- Ante: 1 chip each
- Raise sizes: 2 chips (round 1), 4 chips (round 2)
- Max 2 raises per round
- ~936 information sets

Game flow:
1. Both players ante 1 chip
2. Each dealt 1 private card
3. Round 1 betting (preflop)
4. Community card dealt
5. Round 2 betting (flop)
6. Showdown: pair beats high card, else high card wins

Winning hands (highest to lowest):
1. Pair (private card matches community card)
2. High card (King > Queen > Jack)
"""

from typing import List, Tuple, Optional
from itertools import permutations
import numpy as np

from .base import Game, GameState, InfoSet


# Card constants - using (rank, suit) tuples
# Ranks: 0=Jack, 1=Queen, 2=King
# Suits: 0, 1 (two of each rank)
JACK = 0
QUEEN = 1
KING = 2
RANK_NAMES = ['J', 'Q', 'K']

# All 6 cards: (rank, suit)
ALL_CARDS = [(r, s) for r in range(3) for s in range(2)]

# Action constants
FOLD = 0
CALL = 1   # Also used for check
RAISE = 2


class LeducPoker(Game):
    """Leduc Poker game implementation."""

    def __init__(self):
        self._ante = 1
        self._raise_sizes = [2, 4]  # Round 1, Round 2
        self._max_raises = 2

    @property
    def num_players(self) -> int:
        return 2

    @property
    def num_actions(self) -> int:
        return 3  # Fold, Call/Check, Raise

    @property
    def name(self) -> str:
        return "leduc"

    def initial_state(self) -> GameState:
        """
        Return initial state (chance node for dealing private cards).
        """
        return GameState(
            player_cards=(),  # Empty - will be dealt at chance node
            public_card=None,
            pot=2 * self._ante,  # Both players ante
            current_player=-1,   # Chance node indicator
            bets=(self._ante, self._ante),
            stacks=(0, 0),
            round_num=0,
            num_raises=0,
            history=(),
            is_terminal=False,
        )

    def is_chance_node(self, state: GameState) -> bool:
        """
        Chance node when:
        1. Private cards not yet dealt (initial)
        2. Round 1 betting done, need to deal community card
        """
        # Initial deal
        if len(state.player_cards) == 0:
            return True

        # Between rounds - community card not dealt but round 1 complete
        if state.public_card is None and state.round_num == 1:
            return True

        return False

    def chance_outcomes(self, state: GameState) -> List[Tuple[GameState, float]]:
        """Get all possible chance outcomes."""
        if not self.is_chance_node(state):
            return []

        outcomes = []

        if len(state.player_cards) == 0:
            # Initial deal: deal 2 cards from 6
            for p1_card in ALL_CARDS:
                for p2_card in ALL_CARDS:
                    if p1_card == p2_card:
                        continue  # Can't deal same card twice

                    new_state = state.copy()
                    new_state.player_cards = (p1_card, p2_card)
                    new_state.current_player = 0  # P1 acts first

                    # Probability: 1/6 * 1/5 = 1/30 for each pair
                    outcomes.append((new_state, 1.0 / 30.0))

        elif state.public_card is None and state.round_num == 1:
            # Deal community card from remaining 4 cards
            dealt = set(state.player_cards)
            remaining = [c for c in ALL_CARDS if c not in dealt]

            for card in remaining:
                new_state = state.copy()
                new_state.public_card = card
                new_state.current_player = 0  # P1 acts first in new round
                new_state.num_raises = 0      # Reset raises for new round

                # Probability: 1/4 for each remaining card
                outcomes.append((new_state, 1.0 / 4.0))

        return outcomes

    def current_player(self, state: GameState) -> int:
        """Get current player to act."""
        if self.is_chance_node(state):
            return -1
        return state.current_player

    def get_infoset(self, state: GameState, player: int) -> InfoSet:
        """
        Get information set for a player.
        Player knows: own card, community card (if dealt), action history.
        """
        # Private info: own card (rank, suit) -> just rank for simplicity
        private_card = state.player_cards[player]
        private_rank = private_card[0]

        # Public info: community card rank (or None), pot, round
        public_rank = state.public_card[0] if state.public_card else None

        return InfoSet(
            player=player,
            public_state=(public_rank, state.pot, state.round_num),
            private_info=(private_rank,),
            history=state.history,
        )

    def get_actions(self, state: GameState) -> List[int]:
        """
        Get legal actions.
        - If facing a bet: Fold, Call, Raise (if raises left)
        - If not facing a bet: Check (Call), Raise
        """
        actions = []

        # Parse history to determine if facing a bet
        facing_bet = self._is_facing_bet(state)

        if facing_bet:
            actions.append(FOLD)
            actions.append(CALL)
            if state.num_raises < self._max_raises:
                actions.append(RAISE)
        else:
            # Check = Call action index
            actions.append(CALL)
            if state.num_raises < self._max_raises:
                actions.append(RAISE)

        return actions

    def _is_facing_bet(self, state: GameState) -> bool:
        """Check if current player is facing a bet/raise."""
        # Look at current round's actions
        round_history = self._get_current_round_history(state)

        if len(round_history) == 0:
            return False

        # If last action was a raise, we're facing a bet
        last_action = round_history[-1]
        return last_action == 'r'

    def _get_current_round_history(self, state: GameState) -> Tuple[str, ...]:
        """Get actions from current betting round only."""
        if state.round_num == 0:
            return state.history

        # Find where round 2 starts (after '/')
        history = state.history
        for i, action in enumerate(history):
            if action == '/':
                return history[i+1:]

        return ()

    def apply_action(self, state: GameState, action: int) -> GameState:
        """Apply action and return new state."""
        new_state = state.copy()
        history = list(state.history)
        current = state.current_player
        opponent = 1 - current

        raise_size = self._raise_sizes[state.round_num]

        # Convert action to string
        if action == FOLD:
            action_str = 'f'
        elif action == CALL:
            action_str = 'c'
        else:  # RAISE
            action_str = 'r'

        history.append(action_str)
        new_state.history = tuple(history)

        if action == FOLD:
            # Opponent wins
            new_state.is_terminal = True
            new_state.terminal_type = "fold"
            new_state.winner = opponent

        elif action == CALL:
            # Match opponent's bet
            bets = list(state.bets)
            bets[current] = bets[opponent]  # Match the bet
            new_state.bets = tuple(bets)
            new_state.pot = sum(bets)

            # Check if round ends
            round_history = self._get_current_round_history(new_state)

            if self._is_round_complete(round_history):
                if state.round_num == 0:
                    # End of round 1 - deal community card
                    new_state.round_num = 1
                    new_state.num_raises = 0
                    history.append('/')  # Round separator
                    new_state.history = tuple(history)
                    # Next state is chance node (dealt in chance_outcomes)
                else:
                    # End of round 2 - showdown
                    new_state.is_terminal = True
                    new_state.terminal_type = "showdown"
            else:
                new_state.current_player = opponent

        elif action == RAISE:
            # Add raise to pot
            bets = list(state.bets)
            # First match opponent's bet, then raise
            bets[current] = bets[opponent] + raise_size
            new_state.bets = tuple(bets)
            new_state.pot = sum(bets)
            new_state.num_raises = state.num_raises + 1
            new_state.current_player = opponent

        return new_state

    def _is_round_complete(self, round_history: Tuple[str, ...]) -> bool:
        """Check if current betting round is complete."""
        if len(round_history) < 2:
            return False

        # Round complete when both players have acted and last action is call/check
        last = round_history[-1]
        if last == 'c':
            # Check/call ends the round if:
            # 1. cc (both check)
            # 2. rc (raise-call)
            # 3. rrc (raise-raise-call)
            return True

        return False

    def is_terminal(self, state: GameState) -> bool:
        """Check if game is over."""
        return state.is_terminal

    def get_payoffs(self, state: GameState) -> np.ndarray:
        """
        Get payoffs at terminal state.
        """
        if not state.is_terminal:
            raise ValueError("Cannot get payoffs for non-terminal state")

        payoffs = np.zeros(2)

        if state.terminal_type == "fold":
            winner = state.winner
            loser = 1 - winner

            # Winner gets opponent's contribution
            loser_bet = state.bets[loser]
            payoffs[winner] = loser_bet
            payoffs[loser] = -loser_bet

        elif state.terminal_type == "showdown":
            # Determine winner by hand strength
            winner = self._determine_winner(state)
            loser = 1 - winner

            bet_amount = state.bets[loser]
            payoffs[winner] = bet_amount
            payoffs[loser] = -bet_amount

        return payoffs

    def _determine_winner(self, state: GameState) -> int:
        """
        Determine winner at showdown.
        Pair beats high card, else higher card wins.
        """
        p1_rank = state.player_cards[0][0]
        p2_rank = state.player_cards[1][0]
        public_rank = state.public_card[0]

        p1_pair = (p1_rank == public_rank)
        p2_pair = (p2_rank == public_rank)

        if p1_pair and not p2_pair:
            return 0
        elif p2_pair and not p1_pair:
            return 1
        elif p1_pair and p2_pair:
            # Both have pair - impossible in Leduc (only 2 of each rank)
            # But if somehow: higher pair wins
            return 0 if p1_rank > p2_rank else 1
        else:
            # Neither has pair - high card wins
            return 0 if p1_rank > p2_rank else 1

    def get_action_name(self, action: int) -> str:
        """Convert action to string."""
        names = {FOLD: 'fold', CALL: 'call/check', RAISE: 'raise'}
        return names.get(action, f'action_{action}')

    def card_name(self, card: Tuple[int, int]) -> str:
        """Convert card to string."""
        rank, suit = card
        return f"{RANK_NAMES[rank]}{suit}"

    def __repr__(self) -> str:
        return "LeducPoker()"
