# Sigma-Go: Scaling Laws for Strategic Reasoning

> Can equilibrium reasoning be amortized and scaled, rather than hand-engineered?

## Research Question

We empirically test whether **counterfactual value (CFV) prediction** in imperfect-information games follows predictable scaling laws:

```
L(N, D, C) ≈ A/N^α + B/D^β + f(C, N)
```

Where:
- **N** = Model parameters
- **D** = Oracle data (CFR-labeled states)  
- **C** = Online search compute

## Hypothesis

1. CFV loss scales as a power law with N and D
2. Online search (C) only helps when N > N_critical ("competence threshold")
3. Optimal N/D allocation follows Chinchilla-style tradeoffs

## Testbed

**Leduc Poker** - Simple enough for tractable oracle generation, complex enough to be interesting:
- ~936 information sets
- Well-defined Nash equilibrium
- Measurable exploitability

## Quick Start

```bash
# Install
uv sync  # or pip install -e .

# Run CFR sanity check
python -m pytest tests/test_cfr.py

# Generate oracle data
python -m src.data.oracle_generator --game leduc --iterations 10000

# Run scaling experiment
python experiments/run_scaling.py --config experiments/configs/scaling_N.yaml
```

## Project Structure

```
src/
├── games/          # Game implementations (Kuhn, Leduc)
├── cfr/            # CFR solver variants
├── data/           # Oracle generation, datasets
├── models/         # Neural architectures
├── training/       # Training loop, losses
└── evaluation/     # Metrics, exploitability

experiments/        # Experiment configs and scripts
results/           # Output plots and data
```

## Key Results (TODO)

- [ ] Scaling N curves
- [ ] Scaling D curves  
- [ ] Search threshold analysis
- [ ] Generalization to Liar's Dice

## References

- Brown et al., 2019 - Deep CFR
- Brown et al., 2020 - ReBeL
- Kaplan et al., 2020 - Scaling Laws for Neural Language Models
- Snell et al., 2024 - Scaling LLM Test-Time Compute

## License

MIT
