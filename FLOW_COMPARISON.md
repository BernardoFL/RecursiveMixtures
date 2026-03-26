# Flow comparison experiment (LLM-oriented reference)

This document describes [`flow_comparison.py`](flow_comparison.py): a comparison
of **three recursive flow families** on the same simulated cat-paw dataset.

1. **`NewtonHellingerFlow`** — Fisher-Rao / Hellinger weight-only updates
2. **`NewtonFlow`** — recursive Bayesian mixing weight updates with a decaying schedule
3. **`HellingerKantorovichFlow`** — joint weight + atom updates (HK / WFR)

## Reference figures in Git

The script writes **`flow_comparison.pdf`**. That file may be committed to the
repository as a reference run (it is exempt from the repo-wide `*.pdf` ignore
via [`.gitignore`](.gitignore)). Treat any committed copy as produced by this
code, not an external illustration.

After you change the experiment or plotting code, run
`python flow_comparison.py` and commit the refreshed figure to keep the
repository snapshot aligned.

## Goal

On the same fixed dataset (n i.i.d. draws from the cat-paw mixture) and the
same initial particle measure (drawn from a Pitman–Yor prior), compare how each
flow family places particles after consuming all n observations.

## Target and data

- **Truth**: The 7-component 2D Gaussian **cat-paw** distribution from
  [`paw_distribution.PawDistribution`](paw_distribution.py) (implemented directly
  in `flow_comparison.py`).
- **Data**: `n` i.i.d. draws generated via `PawDistribution().sample(...)`.
- **Prior**: `PitmanYorProcessPrior(discount=py_discount, strength=py_strength)`
  centered at an isotropic Gaussian `GaussianPrior`.
- **Plot background**: `PawDistribution().pdf(...)` on the configured grid.

## Flows compared

| Panel | Flow class | Update type |
|-------|-----------|-------------|
| (a) | `NewtonHellingerFlow` | Fisher-Rao: weight-only, with ESS-triggered resampling |
| (b) | `NewtonFlow` | Recursive Bayesian: weight-only, α_n = (n+1)^(-γ) schedule |
| (c) | `HellingerKantorovichFlow` | WFR: joint weight + atom (Hellinger + Sinkhorn drift) |

All three share `initial_measure` and `data`. Only the update rule differs.

## Implementation notes

- `run_newton_hellinger`, `run_newton`, `run_hk` in
  [`flow_comparison.py`](flow_comparison.py) each call `flow.run` with the same
  `n_steps = n_data` and `bootstrap_after_data = False`.
- `prior_particles` is only used by `HellingerKantorovichFlow` (Sinkhorn drift).

## Output

| File | Content |
|------|---------|
| `flow_comparison.pdf` | Three panels (one per flow), true density heatmap + data scatter + particles (size ∝ weight). |

## How to run

```bash
python flow_comparison.py
python flow_comparison.py --n-data 500
```

- **`--n-data`** — dataset size n (default: 1000).

## Related code

- [`recursive_mixtures/flows.py`](recursive_mixtures/flows.py)
- [`recursive_mixtures/functionals.py`](recursive_mixtures/functionals.py)
- [`paw_distribution.py`](paw_distribution.py)
