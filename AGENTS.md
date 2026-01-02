# AGENTS.MD - Sigma-Go

## What is this?
Research codebase testing scaling laws for equilibrium reasoning in imperfect-information games.

## Quick Context
- **Hypothesis**: CFV prediction loss scales predictably with model size (N), oracle data (D), and online search compute (C)
- **Testbed**: Leduc Poker (controlled, tractable, known equilibria)
- **Oracle**: CFR solver generates training labels
- **Student**: Neural network predicts counterfactual values

## File You Must Read First
**`BUILD_CONTEXT.md`** - Contains complete architecture, abstractions, implementation order, and all context.

## Implementation Order
1. Games (kuhn.py → leduc.py)
2. CFR solver (vanilla → dcfr)
3. Data pipeline (oracle generation → dataset)
4. Models (MLP family with size configs)
5. Training (losses, trainer, evaluation)
6. Experiments (scaling curves, threshold analysis)

## Key Abstractions
- `Game` - Abstract game interface
- `CFRSolver` - Oracle solver
- `CFVPredictor` - Neural model base class
- `OracleDataPoint` - Training example schema

## Critical Implementation Notes
- CFR: Use float64 for accumulators, default to uniform on negative regrets
- Training: Weight by reach probability, normalize CFVs
- Evaluation: Regret matching (not softmax) to extract strategy from CFV
- Metrics: CFV MSE is primary, exploitability for validation

## Tests to Write First
1. CFR converges on Kuhn poker (known Nash)
2. Exploitability calculation is correct
3. Infoset encoding is deterministic
4. Model forward pass shapes are correct

## Don't
- Don't optimize for win-rate (use CFV loss)
- Don't skip exploitability validation
- Don't use only 1 seed (need 3+ for error bars)
- Don't compute exploitability every epoch (expensive)

## Success = 
- Power-law scaling curves with R² > 0.9
- Observable search threshold phenomenon
