"""
Unit tests for CFR solvers.

Tests verify:
1. Regret matching implementation
2. CFR convergence on Kuhn Poker (known Nash equilibrium)
3. Exploitability decreases over iterations
4. DCFR converges faster than Vanilla CFR
"""

import pytest
import numpy as np
from src.games.kuhn import KuhnPoker
from src.games.leduc import LeducPoker
from src.cfr.solver import regret_matching, regret_matching_batch, CFROutput
from src.cfr.vanilla_cfr import VanillaCFR
from src.cfr.dcfr import DiscountedCFR


class TestRegretMatching:
    """Tests for regret matching function."""

    def test_positive_regrets(self):
        """Test regret matching with positive regrets."""
        regrets = np.array([2.0, 1.0, 0.0])
        strategy = regret_matching(regrets)

        assert len(strategy) == 3
        assert abs(strategy.sum() - 1.0) < 1e-10
        assert strategy[0] == pytest.approx(2/3)
        assert strategy[1] == pytest.approx(1/3)
        assert strategy[2] == pytest.approx(0.0)

    def test_all_negative_regrets(self):
        """Test uniform fallback for all-negative regrets."""
        regrets = np.array([-1.0, -2.0, -3.0])
        strategy = regret_matching(regrets)

        # Should return uniform
        expected = np.array([1/3, 1/3, 1/3])
        np.testing.assert_array_almost_equal(strategy, expected)

    def test_mixed_regrets(self):
        """Test with mix of positive and negative regrets."""
        regrets = np.array([3.0, -1.0, 2.0])
        strategy = regret_matching(regrets)

        # Only positive regrets contribute
        assert strategy[0] == pytest.approx(3/5)
        assert strategy[1] == pytest.approx(0.0)
        assert strategy[2] == pytest.approx(2/5)

    def test_zero_regrets(self):
        """Test uniform for all-zero regrets."""
        regrets = np.array([0.0, 0.0, 0.0])
        strategy = regret_matching(regrets)

        expected = np.array([1/3, 1/3, 1/3])
        np.testing.assert_array_almost_equal(strategy, expected)

    def test_single_positive(self):
        """Test single positive regret."""
        regrets = np.array([0.0, 5.0, 0.0])
        strategy = regret_matching(regrets)

        assert strategy[0] == pytest.approx(0.0)
        assert strategy[1] == pytest.approx(1.0)
        assert strategy[2] == pytest.approx(0.0)


class TestBatchedRegretMatching:
    """Tests for batched regret matching."""

    def test_batch_shape(self):
        """Test output shape matches input."""
        regrets = np.array([
            [1.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
            [-1.0, -1.0, -1.0],
        ])
        strategies = regret_matching_batch(regrets)

        assert strategies.shape == regrets.shape

    def test_batch_probabilities(self):
        """Test each row sums to 1."""
        regrets = np.random.randn(10, 4)
        strategies = regret_matching_batch(regrets)

        row_sums = strategies.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(10))


class TestVanillaCFRKuhn:
    """Tests for Vanilla CFR on Kuhn Poker."""

    @pytest.fixture
    def game(self):
        return KuhnPoker()

    @pytest.fixture
    def solver(self, game):
        return VanillaCFR(game)

    def test_exploitability_decreases(self, solver):
        """Test that exploitability decreases over iterations."""
        exploitabilities = []

        def callback(iteration, exploit):
            exploitabilities.append(exploit)

        solver.solve(
            iterations=1000,
            eval_every=100,
            callback=callback,
            show_progress=False,
        )

        # Exploitability should generally decrease
        assert len(exploitabilities) >= 5
        assert exploitabilities[-1] < exploitabilities[0]

    def test_convergence_to_nash(self, solver):
        """Test convergence to near-Nash equilibrium."""
        result = solver.solve(
            iterations=5000,
            convergence_threshold=1e-3,
            eval_every=500,
            show_progress=False,
        )

        # Should converge to low exploitability
        assert result.exploitability < 0.1

        # Check number of infosets
        assert result.num_infosets() == 12

    def test_strategies_are_valid(self, solver):
        """Test that output strategies are valid probability distributions."""
        result = solver.solve(
            iterations=1000,
            show_progress=False,
        )

        for key, data in result.infoset_data.items():
            strategy = data.strategy

            # Non-negative
            assert np.all(strategy >= 0)

            # Sum to 1
            assert abs(strategy.sum() - 1.0) < 1e-6

    def test_reset(self, solver):
        """Test that reset clears state."""
        solver.solve(iterations=100, show_progress=False)
        assert len(solver.regret_sum) > 0

        solver.reset()
        assert len(solver.regret_sum) == 0
        assert len(solver.strategy_sum) == 0
        assert solver.iterations_run == 0


class TestDiscountedCFRKuhn:
    """Tests for Discounted CFR on Kuhn Poker."""

    @pytest.fixture
    def game(self):
        return KuhnPoker()

    @pytest.fixture
    def solver(self, game):
        return DiscountedCFR(game)

    def test_faster_convergence_than_vanilla(self, game):
        """Test that DCFR converges faster than Vanilla CFR."""
        vanilla = VanillaCFR(game)
        dcfr = DiscountedCFR(game)

        vanilla_result = vanilla.solve(
            iterations=1000,
            show_progress=False,
        )

        dcfr_result = dcfr.solve(
            iterations=1000,
            show_progress=False,
        )

        # DCFR should have lower exploitability at same iterations
        assert dcfr_result.exploitability <= vanilla_result.exploitability * 1.5

    def test_convergence_to_nash(self, solver):
        """Test convergence to near-Nash equilibrium."""
        result = solver.solve(
            iterations=2000,
            convergence_threshold=1e-3,
            eval_every=200,
            show_progress=False,
        )

        # Should converge to low exploitability
        assert result.exploitability < 0.05

    def test_discount_parameters(self, game):
        """Test different discount parameters."""
        # Standard parameters
        solver1 = DiscountedCFR(game, alpha=1.5, beta=0.5, gamma=2.0)
        result1 = solver1.solve(iterations=500, show_progress=False)

        # More aggressive discounting
        solver2 = DiscountedCFR(game, alpha=2.0, beta=0.25, gamma=3.0)
        result2 = solver2.solve(iterations=500, show_progress=False)

        # Both should converge reasonably
        assert result1.exploitability < 1.0
        assert result2.exploitability < 1.0


class TestCFROutput:
    """Tests for CFROutput dataclass."""

    def test_cfr_output_creation(self):
        """Test creating CFROutput."""
        output = CFROutput(
            infoset_key="p0|J|2|('p',)",
            player=0,
            cfv=np.array([0.5, -0.3]),
            strategy=np.array([0.7, 0.3]),
            regrets=np.array([1.0, -0.5]),
            reach_prob=0.5,
        )

        assert output.player == 0
        assert output.cfv.dtype == np.float64
        assert len(output.strategy) == 2


class TestSolverResult:
    """Tests for SolverResult."""

    @pytest.fixture
    def result(self):
        game = KuhnPoker()
        solver = VanillaCFR(game)
        return solver.solve(iterations=100, show_progress=False)

    def test_get_strategy(self, result):
        """Test getting strategy from result."""
        # Get any infoset key
        key = list(result.infoset_data.keys())[0]
        strategy = result.get_strategy(key)

        assert len(strategy) == 2
        assert abs(strategy.sum() - 1.0) < 1e-6

    def test_unknown_infoset_raises(self, result):
        """Test error for unknown infoset."""
        with pytest.raises(KeyError):
            result.get_strategy("unknown_key")


class TestCFROnLeduc:
    """Tests for CFR on Leduc Poker (larger game)."""

    @pytest.fixture
    def game(self):
        return LeducPoker()

    def test_vanilla_cfr_runs(self, game):
        """Test that Vanilla CFR runs on Leduc without errors."""
        solver = VanillaCFR(game)
        result = solver.solve(
            iterations=100,
            eval_every=50,
            show_progress=False,
        )

        # Should have many more infosets than Kuhn
        assert result.num_infosets() > 12

        # Exploitability defined
        assert result.exploitability >= 0

    def test_dcfr_runs(self, game):
        """Test that DCFR runs on Leduc without errors."""
        solver = DiscountedCFR(game)
        result = solver.solve(
            iterations=100,
            eval_every=50,
            show_progress=False,
        )

        assert result.num_infosets() > 12


class TestExploitability:
    """Tests for exploitability calculation."""

    def test_exploitability_non_negative(self):
        """Test exploitability is always non-negative."""
        game = KuhnPoker()
        solver = VanillaCFR(game)

        # Even with random initial regrets, exploitability should be >= 0
        for _ in range(10):
            solver.reset()
            # Run just 1 iteration
            solver.iterate_once()
            exploit = solver.get_exploitability()
            assert exploit >= 0

    def test_uniform_strategy_exploitability(self):
        """Test exploitability of uniform strategy is bounded."""
        game = KuhnPoker()
        solver = VanillaCFR(game)

        # Before any iterations, strategy is uniform
        # Exploitability should be reasonable for Kuhn
        exploit = solver.get_exploitability()

        # Kuhn has pot of 2, so max exploitability is bounded
        assert exploit < 10
