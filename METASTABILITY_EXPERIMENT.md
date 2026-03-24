# Metastability experiment (LLM-oriented reference)

This document describes [`metastability_experiment.py`](metastability_experiment.py) as of the **paw HK** redesign: one simulated dataset from the cat-paw mixture and **three** `HellingerKantorovichFlow` runs, then either a **three-panel** figure (default) or, when sweeping sample sizes, **one overlay figure per n**.

## Goal

On **fixed** paw-Gaussian-mixture data of size **n** and **fixed** initial particles (DP prior on atom locations), compare HK behaviour under:

1. **Case (a)** — **n** flow steps, data consumed **in order** (`x_0, \ldots, x_{n-1}`), Fisher–Rao **prior regularization on** (`use_prior_regularization=True`).
2. **Case (b)** — **n** steps, same ordering, prior regularization **off** (`use_prior_regularization=False`). Atom-level Sinkhorn drift (`use_sinkhorn=True`) is **unchanged**; only the Hellinger prior functional term is disabled.
3. **Case (c)** — **n + k** steps with prior **on**: first **n** steps use the data in order, then **k** additional steps use observations drawn **uniformly with replacement** from the same n points (index continuation via `flow.run(..., bootstrap_after_data=True)`; see [`_prepare_run`](recursive_mixtures/flows.py)).

Default **k = 1000** (large; reduce with `--k` for testing).

## Target and data

- **Truth**: Seven-component 2D Gaussian **cat paw** from [`paw_distribution.PawDistribution`](paw_distribution.py); parameters stored in config as `dumbbell_means`, `dumbbell_stds`, `dumbbell_weights`.
- **Data**: `n` i.i.d. draws via `generate_mixture_data`.
- **Plot background**: `true_mixture_density` on the same grid as before (`grid_min` / `grid_max` / `grid_size`).

## Initialization

- `GaussianPrior` + `DirichletProcessPrior` for `prior.sample` (initial atoms).
- `prior.to_particle_measure` builds `prior_particles` for the HK constructor.
- **Same** `ParticleMeasure.initialize(initial_atoms)` is passed into all three `flow.run` calls.

## Implementation notes

- [`make_hk_flow_for_metastability`](metastability_experiment.py) passes `use_prior_regularization` into `HellingerKantorovichFlow`.
- [`run_hk_case`](metastability_experiment.py) uses `GradientFlow.run` with explicit `n_steps` and `bootstrap_after_data` (no hand-rolled index loops).

## Output

| File | Content |
|------|---------|
| `paw_hk_comparison.pdf` | **Single-n mode:** three panels — true density heatmap, **data** scatter (small markers), final **particles** (size ∝ weight) for (a), (b), (c). |
| `paw_hk_overlay_n{n}.pdf` | **`--n-data-list` mode:** one axes per requested **n** — same heatmap and data, all three particle sets overlaid (teal / royalblue / crimson) with a legend. |

If both `--n-data` and `--n-data-list` are passed, **`--n-data-list` takes precedence** (the single-`n` options are ignored).

## How to run

```bash
python metastability_experiment.py
```

Optional:

```bash
python metastability_experiment.py --n-data 500 --k 100
```

Sweep several sample sizes (one overlay PDF per **n**):

```bash
python metastability_experiment.py --n-data-list 200,500,1000 --k 200
```

- **`--n-data`** — sample size **n** (ignored when `--n-data-list` is set).
- **`--n-data-list`** — comma-separated **n** values; repeats the full (a)(b)(c) procedure for each and writes `paw_hk_overlay_n{n}.pdf`.
- **`--k`** — extra resampled steps after the ordered pass (case **c** uses **n + k** total steps).

Case **(c)** with default **k = 1000** performs many Sinkhorn-heavy steps; use smaller `--k` for interactive runs.

## Related code

- [`paw_distribution.py`](paw_distribution.py)
- [`recursive_mixtures/flows.py`](recursive_mixtures/flows.py) (`HellingerKantorovichFlow`, `use_prior_regularization`)
