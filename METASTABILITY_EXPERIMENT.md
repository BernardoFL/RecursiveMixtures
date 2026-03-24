# Metastability experiment (LLM-oriented reference)

This document describes the **mode recovery / metastability** script [`metastability_experiment.py`](metastability_experiment.py): target law, algorithms compared, data schedule, diagnostics, outputs, and how to run it.

## Goal

Compare how three **particle gradient flows** behave on **multi-modal 2D data**, starting from the **same** initial particle measure, to see which scheme better **spreads mass across modes** (Wasserstein–Fisher-Rao style transport vs weight-only updates). The experiment reports timings, simple atom-displacement diagnostics, and two figures.

## Target distribution (data)

- **Ground truth**: A **seven-component Gaussian mixture** in \(\mathbb{R}^2\) arranged as a **cat paw** (four toe pads on an arc + three palm pads).
- **Parameter source**: [`paw_distribution.PawDistribution`](paw_distribution.py) defaults; means, diagonal standard deviations, and weights are copied into config as `dumbbell_means`, `dumbbell_stds`, `dumbbell_weights` (historical key names).
- **Observed data**: `n_data` i.i.d. samples from the mixture (`generate_mixture_data` in [`recursive_mixtures/utils.py`](recursive_mixtures/utils.py)).

## Initialization

- **Base prior**: `GaussianPrior` with mean `prior_mean`, isotropic scale `prior_std`.
- **Sampling prior for atoms**: `DirichletProcessPrior` with concentration `dp_concentration`, centered on that base prior. Atoms for the **initial** particle measure are drawn with `prior.sample`.
- **HK prior particles**: A fixed `ParticleMeasure` of the same size is built with `prior.to_particle_measure` and passed into `HellingerKantorovichFlow` as `prior_particles` (Sinkhorn / prior-leaning terms reuse this).

All three flows start from **`ParticleMeasure.initialize(initial_atoms)`** — the **same** initial atoms and weights.

## Flows compared (and figure titles)

The three columns in `metastability_density_comparison.pdf` are labeled:

| Panel title | Implementation in code | What updates |
|-------------|------------------------|--------------|
| **Wasserstein-Fisher-Rao** | `HellingerKantorovichFlow` via `make_hk_flow_for_metastability` | Fisher–Rao (Hellinger) **weights** + Wasserstein-like **atoms** (kernel mean map), with Sinkhorn regularization toward the prior (`use_sinkhorn=True`, `prior_flow_weight`, `prior_mc_samples`). |
| **Fisher-Rao** | `NewtonFlow` via `make_newton_weights_flow_for_metastability` | **Weights only** (recursive Newton update); atoms fixed except where resampling moves support internally. |
| **Newton** | `NewtonFlow` via `make_newton_wasserstein_flow_for_metastability` | Same class and hyperparameters as the middle panel (`NewtonFlow`, same `step_size`); **independent** RNG stream and data-index stream. Intended as a second weight-only run for comparison; it is **not** a separate “Newton–Wasserstein” atom-only flow despite the factory name. |

If you need **Hellinger weight flow without the Newton recursion**, that would require wiring `NewtonHellingerFlow` (not currently used in this script).

## Data stream schedule (all three flows)

Let `T = n_data` be the dataset length and `S = config["n_steps"]`.

- For each step index \(t \in \{0, \ldots, S-1\}\), the flows use `data[step_indices[t]]`, where:
  - If \(S \le T\): `step_indices` is `0, 1, \ldots, S-1` (prefix of the batch in order).
  - If \(S > T\): first `T` steps use `0..T-1`, then **uniform random** indices in `0..T-1` for the remainder (with-replacement **index** continuation on the fixed dataset, not Bayesian bootstrap).

The HK loop implements this explicitly; the two `NewtonFlow` loops follow the same rule.

Trajectory **recording** (HK only): atom snapshots are appended every `record_every` steps (plus the initial state in `hk_snaps`).

## Diagnostics

- **Atom displacement** (mean / max over particles): Euclidean norm of final atoms minus initial atoms, printed for HK vs both Newton runs.
- **Mode occupancy (HK only)**: At each stored snapshot, each particle is assigned to the **nearest** mixture component mean (`dumbbell_means`); histogram counts yield fractions over the **seven** paw modes. Plotted vs step index (time axis uses `step × record_every` in the occupancy plot).

## Plots written to disk

| File | Content |
|------|---------|
| `metastability_density_comparison.pdf` | Three panels: true mixture density (heatmap) + final particles (scatter, marker size ∝ weight). |
| `mode_occupancy_over_time.pdf` | HK only: fraction of particles nearest each of the 7 modes vs step. |

*(The script’s closing print may still mention `.png` in places; the `savefig` calls use `.pdf`.)*

## Configuration highlights (`setup_config`)

Stored in [`setup_config`](metastability_experiment.py): `n_data`, `n_particles`, HK hyperparameters (`hk_step_size`, `hk_kernel_bandwidth`, Sinkhorn knobs, `hk_wasserstein_weight`, `hk_prior_flow_weight`, `hk_prior_mc_samples`, …), DP `dp_concentration`, `n_steps`, `record_every`, density grid bounds (`grid_min` / `grid_max` / `grid_size`), `seed`.

## How to run

From the repository root:

```bash
python metastability_experiment.py
```

Override number of flow steps:

```bash
python metastability_experiment.py --n-steps 200
```

Requires a functioning JAX environment with dependencies from [`requirements.txt`](requirements.txt).

## Related code

- Cat-paw definition and reuse: [`paw_distribution.py`](paw_distribution.py)
- HK / Newton flows: [`recursive_mixtures/flows.py`](recursive_mixtures/flows.py)
- Mixture data and true density on a grid: [`recursive_mixtures/utils.py`](recursive_mixtures/utils.py)

## Legacy / unused code in the same file

The file still contains **banana-shaped mixture** helpers (`banana_*`) from an earlier experiment; the **active** `main()` path uses only the **Gaussian paw mixture** above.
