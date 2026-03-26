# Metastability experiment (LLM-oriented reference)

This document describes [`metastability_experiment.py`](metastability_experiment.py)
as a comparison between **three flow families** on the same simulated dataset:

1. `NewtonHellingerFlow`
2. `NewtonFlow`
3. `HellingerKantorovichFlow`

## Reference figures in Git

The script writes **`paw_hk_comparison.pdf`**. That file may be **committed to the repository** as a reference run (it is exempt from the repo-wide `*.pdf` ignore via [`.gitignore`](.gitignore)). Treat any committed copy as **produced by this code**, not an external illustration.

After you change the experiment or plotting code, run
`python metastability_experiment.py` and commit the refreshed figure if you
want the repository snapshot to stay aligned.

## Goal

On fixed simulated data, compare how the three update rules behave under the
same target distribution, initialization strategy, and visualization pipeline.

## Target and data

- **Truth**: The experiment's configured target mixture in
  [`metastability_experiment.py`](metastability_experiment.py).
- **Data**: `n` i.i.d. draws via `generate_mixture_data`.
- **Plot background**: `true_mixture_density` on the configured grid.

## Flows compared

- **NewtonHellingerFlow**: recursive Newton-Hellinger update.
- **NewtonFlow**: Newton-style recursive mixing measure update.
- **HellingerKantorovichFlow**: HK/WFR flow with transport + Hellinger terms.

## Implementation notes

- All three flows are run on the same experiment setup for direct visual
  comparison.
- The figure compares final behavior of each flow family side by side.

## Output

| File | Content |
|------|---------|
| `paw_hk_comparison.pdf` | Three panels, one per flow family (NewtonHellinger, Newton, HK), with shared visual style. |

## How to run

```bash
python metastability_experiment.py
```

## Related code

- [`recursive_mixtures/flows.py`](recursive_mixtures/flows.py)
- [`recursive_mixtures/functionals.py`](recursive_mixtures/functionals.py)
