"""Abstract base classes for extensive-form games."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import numpy as np


@dataclass(frozen=True)
class InfoSet:
    """
    Information set representation.

    An information set contains all information available to a player
    at a decision point, which may not include hidden opponent cards.
    """
    player: int                      # Which player's perspective (0 or 1)
    public_state: Tuple              # Publicly visible info (pot, actions, community cards)
    private_info: Tuple              # This player's private info (hole cards)
    history: Tuple[str, ...] = ()    # Action history

    def to_key(self) -> str:
        """
        Unique string key for this infoset.
        Used for indexing in CFR regret/strategy tables.
        """
        return f"p{self.player}|{self.private_info}|{self.public_state}|{self.history}"

    def __hash__(self) -> int:
        return hash(self.to_key())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, InfoSet):
            return False
        return self.to_key() == other.to_key()


@dataclass
class GameState:
    """
    Full game state for simulation.

    Contains complete information (including hidden cards) needed
    to simulate the game. This is NOT visible to players.
    """
    # Players' private cards (e.g., hole cards in poker)
    player_cards: Tuple[Any, ...]

    # Public information
    public_card: Optional[Any] = None  # Community card (for Leduc)
    pot: int = 0
    current_player: int = 0

    # Betting state
    bets: Tuple[int, ...] = (0, 0)     # Current round bets per player
    stacks: Tuple[int, ...] = (0, 0)   # Remaining stacks
    round_num: int = 0                  # 0 = preflop, 1 = flop
    num_raises: int = 0                 # Raises this round

    # Action history
    history: Tuple[str, ...] = ()

    # Terminal state info
    is_terminal: bool = False
    terminal_type: Optional[str] = None  # "fold", "showdown", etc.
    winner: Optional[int] = None

    def copy(self) -> 'GameState':
        """Create a deep copy of the game state."""
        return GameState(
            player_cards=self.player_cards,
            public_card=self.public_card,
            pot=self.pot,
            current_player=self.current_player,
            bets=self.bets,
            stacks=self.stacks,
            round_num=self.round_num,
            num_raises=self.num_raises,
            history=self.history,
            is_terminal=self.is_terminal,
            terminal_type=self.terminal_type,
            winner=self.winner,
        )


class Game(ABC):
    """
    Abstract base class for extensive-form games.

    Defines the interface for games that CFR can solve.
    """

    @property
    @abstractmethod
    def num_players(self) -> int:
        """Number of players in the game."""
        ...

    @property
    @abstractmethod
    def num_actions(self) -> int:
        """Maximum number of actions available at any decision point."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the game (e.g., 'kuhn', 'leduc')."""
        ...

    @abstractmethod
    def initial_state(self) -> GameState:
        """
        Create and return the initial game state.
        This is a chance node where cards are dealt.
        """
        ...

    @abstractmethod
    def get_infoset(self, state: GameState, player: int) -> InfoSet:
        """
        Get the information set for a player at a given state.

        Args:
            state: Full game state
            player: Player whose perspective to use

        Returns:
            InfoSet visible to that player
        """
        ...

    @abstractmethod
    def get_actions(self, state: GameState) -> List[int]:
        """
        Get list of legal actions at this state.

        Actions are represented as integers:
        - 0: fold/pass/check
        - 1: call
        - 2: bet/raise

        Returns:
            List of legal action indices
        """
        ...

    @abstractmethod
    def apply_action(self, state: GameState, action: int) -> GameState:
        """
        Apply an action to the game state.

        Args:
            state: Current game state
            action: Action index to apply

        Returns:
            New game state after action
        """
        ...

    @abstractmethod
    def is_terminal(self, state: GameState) -> bool:
        """Check if the game state is terminal (game over)."""
        ...

    @abstractmethod
    def get_payoffs(self, state: GameState) -> np.ndarray:
        """
        Get payoffs for all players at a terminal state.

        Args:
            state: Terminal game state

        Returns:
            Array of payoffs [num_players]
        """
        ...

    @abstractmethod
    def is_chance_node(self, state: GameState) -> bool:
        """Check if this is a chance node (nature acts, e.g., dealing cards)."""
        ...

    @abstractmethod
    def chance_outcomes(self, state: GameState) -> List[Tuple[GameState, float]]:
        """
        Get possible outcomes at a chance node.

        Returns:
            List of (next_state, probability) tuples
        """
        ...

    @abstractmethod
    def current_player(self, state: GameState) -> int:
        """Get the player who acts at this state."""
        ...

    def get_action_name(self, action: int) -> str:
        """Convert action index to human-readable name."""
        action_names = {0: "fold/check", 1: "call", 2: "raise"}
        return action_names.get(action, f"action_{action}")

    def all_infosets(self) -> List[InfoSet]:
        """
        Enumerate all possible information sets in the game.

        This is useful for building complete strategy tables.
        Default implementation traverses the game tree.
        """
        infosets = set()
        self._enumerate_infosets(self.initial_state(), infosets)
        return list(infosets)

    def _enumerate_infosets(self, state: GameState, infosets: set) -> None:
        """Recursively enumerate all infosets."""
        if self.is_terminal(state):
            return

        if self.is_chance_node(state):
            for next_state, _ in self.chance_outcomes(state):
                self._enumerate_infosets(next_state, infosets)
        else:
            player = self.current_player(state)
            infoset = self.get_infoset(state, player)
            infosets.add(infoset)

            for action in self.get_actions(state):
                next_state = self.apply_action(state, action)
                self._enumerate_infosets(next_state, infosets)
