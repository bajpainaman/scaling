"""
Unit tests for game implementations.

Tests verify:
1. Game rules are correctly implemented
2. Terminal states and payoffs are correct
3. Information sets are properly constructed
4. Chance nodes enumerate all outcomes
"""

import pytest
import numpy as np
from src.games.kuhn import KuhnPoker, JACK, QUEEN, KING, PASS, BET
from src.games.leduc import LeducPoker, FOLD, CALL, RAISE


class TestKuhnPoker:
    """Tests for Kuhn Poker implementation."""

    @pytest.fixture
    def game(self):
        return KuhnPoker()

    def test_properties(self, game):
        """Test basic game properties."""
        assert game.num_players == 2
        assert game.num_actions == 2
        assert game.name == "kuhn"

    def test_initial_state(self, game):
        """Test initial state is a chance node."""
        state = game.initial_state()
        assert game.is_chance_node(state)
        assert not game.is_terminal(state)
        assert len(state.player_cards) == 0

    def test_chance_outcomes(self, game):
        """Test all 6 card dealing permutations."""
        state = game.initial_state()
        outcomes = game.chance_outcomes(state)

        # 6 permutations of dealing 2 cards from 3
        assert len(outcomes) == 6

        # Probabilities sum to 1
        total_prob = sum(prob for _, prob in outcomes)
        assert abs(total_prob - 1.0) < 1e-10

        # Each outcome has 2 cards dealt
        for next_state, prob in outcomes:
            assert len(next_state.player_cards) == 2
            assert prob == pytest.approx(1/6)
            assert not game.is_chance_node(next_state)
            assert next_state.current_player == 0

    def test_pass_pass_showdown(self, game):
        """Test P1 pass, P2 pass -> showdown."""
        state = game.initial_state()
        outcomes = game.chance_outcomes(state)

        for dealt_state, _ in outcomes:
            # P1 passes
            state1 = game.apply_action(dealt_state, PASS)
            assert not game.is_terminal(state1)
            assert state1.current_player == 1

            # P2 passes -> showdown
            state2 = game.apply_action(state1, PASS)
            assert game.is_terminal(state2)
            assert state2.terminal_type == "showdown"

    def test_pass_bet_fold(self, game):
        """Test P1 pass, P2 bet, P1 fold -> P2 wins."""
        state = game.initial_state()
        outcomes = game.chance_outcomes(state)
        dealt_state, _ = outcomes[0]

        state1 = game.apply_action(dealt_state, PASS)  # P1 pass
        state2 = game.apply_action(state1, BET)         # P2 bet
        assert state2.current_player == 0               # P1 to respond

        state3 = game.apply_action(state2, PASS)        # P1 fold
        assert game.is_terminal(state3)
        assert state3.terminal_type == "fold"
        assert state3.winner == 1  # P2 wins

    def test_bet_call_showdown(self, game):
        """Test P1 bet, P2 call -> showdown."""
        state = game.initial_state()
        outcomes = game.chance_outcomes(state)
        dealt_state, _ = outcomes[0]

        state1 = game.apply_action(dealt_state, BET)    # P1 bet
        state2 = game.apply_action(state1, BET)         # P2 call
        assert game.is_terminal(state2)
        assert state2.terminal_type == "showdown"

    def test_bet_fold(self, game):
        """Test P1 bet, P2 fold -> P1 wins."""
        state = game.initial_state()
        outcomes = game.chance_outcomes(state)
        dealt_state, _ = outcomes[0]

        state1 = game.apply_action(dealt_state, BET)    # P1 bet
        state2 = game.apply_action(state1, PASS)        # P2 fold
        assert game.is_terminal(state2)
        assert state2.terminal_type == "fold"
        assert state2.winner == 0  # P1 wins

    def test_payoffs_higher_card_wins(self, game):
        """Test that higher card wins at showdown."""
        state = game.initial_state()
        outcomes = game.chance_outcomes(state)

        for dealt_state, _ in outcomes:
            p1_card, p2_card = dealt_state.player_cards

            # Pass-pass showdown
            state1 = game.apply_action(dealt_state, PASS)
            state2 = game.apply_action(state1, PASS)

            payoffs = game.get_payoffs(state2)

            if p1_card > p2_card:
                assert payoffs[0] > 0  # P1 wins
                assert payoffs[1] < 0  # P2 loses
            else:
                assert payoffs[0] < 0  # P1 loses
                assert payoffs[1] > 0  # P2 wins

            # Payoffs are zero-sum
            assert abs(payoffs[0] + payoffs[1]) < 1e-10

    def test_infoset_keys(self, game):
        """Test that infosets are unique and properly structured."""
        state = game.initial_state()
        outcomes = game.chance_outcomes(state)
        dealt_state, _ = outcomes[0]

        # P1's infoset
        infoset_p1 = game.get_infoset(dealt_state, 0)
        assert infoset_p1.player == 0
        assert infoset_p1.private_info[0] == dealt_state.player_cards[0]

        # P2's infoset
        infoset_p2 = game.get_infoset(dealt_state, 1)
        assert infoset_p2.player == 1
        assert infoset_p2.private_info[0] == dealt_state.player_cards[1]

        # Keys should be different
        assert infoset_p1.to_key() != infoset_p2.to_key()

    def test_all_infosets_count(self, game):
        """Test enumeration of all information sets."""
        infosets = game.all_infosets()

        # Kuhn has 12 information sets
        # 3 cards × 4 histories for P1 and P2 combined
        assert len(infosets) == 12


class TestLeducPoker:
    """Tests for Leduc Poker implementation."""

    @pytest.fixture
    def game(self):
        return LeducPoker()

    def test_properties(self, game):
        """Test basic game properties."""
        assert game.num_players == 2
        assert game.num_actions == 3
        assert game.name == "leduc"

    def test_initial_state(self, game):
        """Test initial state is a chance node."""
        state = game.initial_state()
        assert game.is_chance_node(state)
        assert not game.is_terminal(state)

    def test_chance_outcomes_dealing(self, game):
        """Test card dealing outcomes."""
        state = game.initial_state()
        outcomes = game.chance_outcomes(state)

        # 6 cards, deal 2 different: 6 × 5 = 30 combinations
        assert len(outcomes) == 30

        # Probabilities sum to 1
        total_prob = sum(prob for _, prob in outcomes)
        assert abs(total_prob - 1.0) < 1e-10

    def test_community_card_dealing(self, game):
        """Test community card is dealt after round 1."""
        state = game.initial_state()
        outcomes = game.chance_outcomes(state)
        dealt_state, _ = outcomes[0]

        # Both players check round 1
        state1 = game.apply_action(dealt_state, CALL)  # P1 check
        state2 = game.apply_action(state1, CALL)       # P2 check

        # Should be chance node for community card
        assert game.is_chance_node(state2)

        comm_outcomes = game.chance_outcomes(state2)
        # 4 remaining cards
        assert len(comm_outcomes) == 4

    def test_fold_terminates(self, game):
        """Test that folding ends the game."""
        state = game.initial_state()
        outcomes = game.chance_outcomes(state)
        dealt_state, _ = outcomes[0]

        # P1 raises, P2 folds
        state1 = game.apply_action(dealt_state, RAISE)
        state2 = game.apply_action(state1, FOLD)

        assert game.is_terminal(state2)
        assert state2.winner == 0  # P1 wins

    def test_showdown_pair_beats_high_card(self, game):
        """Test hand rankings: pair beats high card."""
        state = game.initial_state()
        outcomes = game.chance_outcomes(state)

        # Find a deal where P1 has Jack, P2 has Queen
        for dealt_state, _ in outcomes:
            p1_rank = dealt_state.player_cards[0][0]
            p2_rank = dealt_state.player_cards[1][0]

            if p1_rank == 0 and p2_rank == 1:  # Jack vs Queen
                # Play to showdown
                state1 = game.apply_action(dealt_state, CALL)
                state2 = game.apply_action(state1, CALL)

                # Deal community card
                comm_outcomes = game.chance_outcomes(state2)

                for comm_state, _ in comm_outcomes:
                    comm_rank = comm_state.public_card[0]

                    # Round 2: both check
                    state3 = game.apply_action(comm_state, CALL)
                    state4 = game.apply_action(state3, CALL)

                    assert game.is_terminal(state4)
                    payoffs = game.get_payoffs(state4)

                    # Determine expected winner
                    p1_pair = (p1_rank == comm_rank)
                    p2_pair = (p2_rank == comm_rank)

                    if p1_pair and not p2_pair:
                        assert payoffs[0] > 0  # P1 wins with pair
                    elif p2_pair and not p1_pair:
                        assert payoffs[1] > 0  # P2 wins with pair
                    else:
                        # High card: Queen beats Jack
                        assert payoffs[1] > 0

                return

    def test_max_raises_respected(self, game):
        """Test that only 2 raises allowed per round."""
        state = game.initial_state()
        outcomes = game.chance_outcomes(state)
        dealt_state, _ = outcomes[0]

        # R1: raise, raise
        state1 = game.apply_action(dealt_state, RAISE)  # P1 raise
        assert RAISE in game.get_actions(state1)        # Can still raise

        state2 = game.apply_action(state1, RAISE)       # P2 raise
        # After 2 raises, no more raises allowed
        actions = game.get_actions(state2)
        assert RAISE not in actions
        assert FOLD in actions
        assert CALL in actions

    def test_payoffs_zero_sum(self, game):
        """Test that payoffs are always zero-sum."""
        state = game.initial_state()
        outcomes = game.chance_outcomes(state)

        for dealt_state, _ in outcomes[:5]:  # Test first 5 deals
            # Play some games to terminal
            # Try fold
            state1 = game.apply_action(dealt_state, RAISE)
            state2 = game.apply_action(state1, FOLD)
            payoffs = game.get_payoffs(state2)
            assert abs(payoffs[0] + payoffs[1]) < 1e-10

    def test_infoset_hides_opponent_card(self, game):
        """Test that infoset doesn't reveal opponent's card."""
        state = game.initial_state()
        outcomes = game.chance_outcomes(state)

        # Find two deals with same P1 card but different P2 cards
        first_deal = None
        for dealt_state, _ in outcomes:
            if first_deal is None:
                first_deal = dealt_state
            elif dealt_state.player_cards[0] == first_deal.player_cards[0]:
                if dealt_state.player_cards[1] != first_deal.player_cards[1]:
                    # P1 has same card, P2 has different card
                    infoset1 = game.get_infoset(first_deal, 0)
                    infoset2 = game.get_infoset(dealt_state, 0)

                    # P1's infosets should be identical
                    assert infoset1.to_key() == infoset2.to_key()
                    return

        pytest.fail("Could not find suitable test deals")


class TestInfoSet:
    """Tests for InfoSet functionality."""

    def test_infoset_equality(self):
        """Test infoset equality and hashing."""
        from src.games.base import InfoSet

        info1 = InfoSet(
            player=0,
            public_state=(2, ('p',)),
            private_info=(1,),
            history=('p',)
        )

        info2 = InfoSet(
            player=0,
            public_state=(2, ('p',)),
            private_info=(1,),
            history=('p',)
        )

        info3 = InfoSet(
            player=1,
            public_state=(2, ('p',)),
            private_info=(1,),
            history=('p',)
        )

        assert info1 == info2
        assert info1 != info3
        assert info1.to_key() == info2.to_key()
        assert info1.to_key() != info3.to_key()

        # Test hashability
        infoset_dict = {info1: "value1", info3: "value3"}
        assert infoset_dict[info2] == "value1"
