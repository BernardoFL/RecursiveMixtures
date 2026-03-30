# HK computational choices experiment (LLM-oriented reference)

This document describes [`hk_computational_choices.py`](hk_computational_choices.py)
as a comparison of **two computational switches** in the HK (WFR) flow (Studies A–B),
plus an optional **Study C** that tracks **2-Wasserstein distance vs sample size** over
a full **2×2 factorial** of those switches:

1. **Fisher-Rao prior regularization**: `use_prior_regularization` on/off
2. **Bootstrap continuation** after the ordered data pass: `bootstrap_after_data` on/off
   with a fixed continuation budget of **+50%** extra steps (`n_steps_on = ceil(1.5 n)`).

## Reference figures in Git

Selected PDF outputs are committed to the repository as reference artifacts
(see [`.gitignore`](.gitignore): general `*.pdf` is ignored, but these names
are negated). Acknowledge them as outputs of this codebase, not hand-drawn
figures.

When you change flow logic, defaults, or plotting code, **re-run** the matching
command below and **commit the refreshed PDFs**.

| PDF | Example command |
|-----|-----------------|
| [`bootstrap_truncation_vs_continuation.pdf`](bootstrap_truncation_vs_continuation.pdf) | `python hk_computational_choices.py --study truncation` |
| [`bootstrap_prior_regularization.pdf`](bootstrap_prior_regularization.pdf) | `python hk_computational_choices.py --study prior` |
| [`hk_wasserstein_vs_n.pdf`](hk_wasserstein_vs_n.pdf) | `python hk_computational_choices.py --study wasserstein` |

## Problem setup

- **Target**: A **four-petal clover** distribution: an equal-weight mixture of
  four anisotropic 2D Gaussians centered on the cardinal directions, with the
  long axis aligned radially. Implemented as [`CloverDistribution`](clover_distribution.py).
- **Observed data**: i.i.d. samples of size `n_data` (varies per study when
  using `n_data_list`).
- **Flow**: `HellingerKantorovichFlow` with a Pitman–Yor mixing prior
  PY(d, θ, G₀) on atom locations (G₀ = isotropic Gaussian; configured by
  `py_discount`, `py_strength`).
- **Per-replicate resampling**: Bayesian bootstrap weights over data indices
  produce a resampled stream `data_boot`.

## Compared computational choices

| Choice | Option 1 | Option 2 |
|--------|----------|----------|
| Fisher-Rao prior regularization | `use_prior_regularization=True` | `use_prior_regularization=False` |
| Continuation after ordered pass | `bootstrap_after_data=True` with extra steps | `bootstrap_after_data=False` (stop at `n_data`) |

This isolates algorithmic/computational effects while keeping the model and
data source fixed.

## Study C — Wasserstein vs \(n\)

- **Sweep**: `n ∈ {100, 200, …, 1000}` in steps of 100 (10 values).
- **Arms**: all four combinations of prior on/off and continuation off/on (same
  step schedules as Studies A–B: continuation off uses `n` steps; continuation on
  uses `n_steps = ceil(1.5 n)` with `bootstrap_after_data=True`).
- **Distance**: approximate **\(W_2\)** between the **final particle measure**
  and a **fixed empirical reference** — `w2_reference_size` i.i.d. samples from
  the clover target (default **2048**), uniform weights, computed with **POT**
  `ot.emd2` under **squared Euclidean** cost; the plot shows \(\sqrt{\text{OT}}\)
  (same units as coordinates).
- **Output**: line plot with four series — [`hk_wasserstein_vs_n.pdf`](hk_wasserstein_vs_n.pdf).

Requires the **POT** package (`pip install POT`).

## Output interpretation

- `bootstrap_truncation_vs_continuation.pdf` — Study A: continuation on vs off
  in a single **N×2** grid (rows = sample sizes `n`, cols = off/on).
- `bootstrap_prior_regularization.pdf` — Study B: prior regularization on vs
  off (N×2 grid, one row per `n`, no continuation in either arm).
- For HPC trajectory uncertainty quantification for Study B, use
  [`hk_prior_regularization_uq.py`](hk_prior_regularization_uq.py).
- `hk_wasserstein_vs_n.pdf` — Study C: \(W_2\) vs `n` for the four factorial arms.
- All panels: true-density heatmap + final HK particles (size ∝ weight); no
  training-data scatter.

## CLI

```bash
python hk_computational_choices.py --study truncation
python hk_computational_choices.py --study prior
python hk_computational_choices.py --study both
python hk_computational_choices.py --study wasserstein
python hk_prior_regularization_uq.py --niter 100 --n-data 1000 --store-every 10 --out-dir results/
```

HPC parallelization (array jobs) example: split `niter=1000` across 20 tasks:

```bash
# Each task writes its own shard .npz/.pdf
python hk_prior_regularization_uq.py --niter 1000 --num-shards 20 --shard-index ${SLURM_ARRAY_TASK_ID} --n-data 1000 --store-every 10 --out-dir results/ --no-plot

# Merge shards afterwards on a login node
python hk_prior_regularization_uq.py --merge-shards "$(ls results/prior_regularization_uq_*_shard*of20.npz | paste -sd, -)" --merge-out results/prior_regularization_uq_merged.npz
```

To override the automatic y-bounds (heatmap extent and panel y-axis match), use:

```bash
python hk_computational_choices.py --study both --y-min 0 --y-max 10
```

Optional: `--n-data-list 100,1000`, `--full`.
Default sample sizes are `n=100,1000`.

---

## Configuration quick reference

| Key / flag | Role |
|------------|------|
| `n_data_list` / `--n-data-list` | Sample sizes used in the comparison |
| `grid_size` | Heatmap resolution (default: **200**) |
| `--y-min`, `--y-max` | Optional `grid_y_min` / `grid_y_max` (heatmap extent and y-axis limits; default: auto from target) |
| Continuation schedule | Fixed `n_steps_on = ceil(1.5 * n_data)` |
| `n_bootstrap` | Replicates per cell (default **1**); PDFs show the first only |
| `use_prior_regularization` | Prior regularization switch (on/off arms) |
| `py_discount`, `py_strength` | Pitman–Yor parameters d and θ for PY(d, θ, G₀) |
| `prior_flow_weight`, `prior_mc_samples` | Strength and MC draws for HK prior term |
| `--full` | Heavier defaults (more data, Sinkhorn work) |
| `--study` | `truncation` \| `prior` \| `both` (default `both`) \| `wasserstein` (Study C) |
| `w2_reference_size` | Empirical reference sample size for Study C (default **2048**) |

---

## Files produced

| File | Content |
|------|---------|
| `bootstrap_truncation_vs_continuation.pdf` | HK comparison: continuation on vs off |
| `bootstrap_prior_regularization.pdf` | HK comparison: prior regularization on vs off |
| `hk_wasserstein_vs_n.pdf` | Study C: \(W_2\) vs `n` (2×2 factorial) |

---

## Implementation pointers

- Study runners: `run_study_truncation_vs_continuation`,
  `run_study_prior_regularization`, `run_study_wasserstein_sweep` in
  [`hk_computational_choices.py`](hk_computational_choices.py).
- Per-replicate overrides: `n_steps_override`, `bootstrap_after_data_override`,
  `use_prior_regularization` on `run_single_hk_replicate`.
- Index continuation logic: `_prepare_run` in
  [`recursive_mixtures/flows.py`](recursive_mixtures/flows.py).
- Prior term gate: `HellingerKantorovichFlow.use_prior_regularization` in
  [`recursive_mixtures/flows.py`](recursive_mixtures/flows.py).
