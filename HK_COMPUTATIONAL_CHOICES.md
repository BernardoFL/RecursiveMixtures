# HK computational choices experiment (LLM-oriented reference)

This document describes [`hk_computational_choices.py`](hk_computational_choices.py)
as a comparison of **two computational switches** in the HK (WFR) flow:

1. **Fisher-Rao prior regularization**: `use_prior_regularization` on/off
2. **Bootstrap continuation** after the ordered data pass: `bootstrap_after_data` on/off

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

## Problem setup

- **Target**: The **Rosenbrock distribution**. Parameters live in
  `setup_config()` (`rosen_a`, `rosen_b`, `rosen_sigma`).
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

## Output interpretation

- `bootstrap_truncation_vs_continuation.pdf` — Study A: continuation on vs off
  across several sample sizes (one 1×2 page per `n`).
- `bootstrap_prior_regularization.pdf` — Study B: prior regularization on vs
  off (N×2 grid, one row per `n`, no continuation in either arm).
- All panels: true-density heatmap + final HK particles (size ∝ weight); no
  training-data scatter.

## CLI

```bash
python hk_computational_choices.py --study truncation
python hk_computational_choices.py --study prior
python hk_computational_choices.py --study both
```

Optional: `--n-data-list 50,100,200`, `--continuation-factor 2.0`, `--full`.

---

## Configuration quick reference

| Key / flag | Role |
|------------|------|
| `n_data_list` / `--n-data-list` | Sample sizes used in the comparison |
| `continuation_factor` / `--continuation-factor` | Multiplier for continuation arm n_steps |
| `n_bootstrap` | Replicates per cell (default **1**); PDFs show the first only |
| `use_prior_regularization` | Prior regularization switch (on/off arms) |
| `py_discount`, `py_strength` | Pitman–Yor parameters d and θ for PY(d, θ, G₀) |
| `prior_flow_weight`, `prior_mc_samples` | Strength and MC draws for HK prior term |
| `--full` | Heavier defaults (more data, Sinkhorn work) |
| `--study` | `truncation` \| `prior` \| `both` (default `both`) |

---

## Files produced

| File | Content |
|------|---------|
| `bootstrap_truncation_vs_continuation.pdf` | HK comparison: continuation on vs off |
| `bootstrap_prior_regularization.pdf` | HK comparison: prior regularization on vs off |

---

## Implementation pointers

- Study runners: `run_study_truncation_vs_continuation`,
  `run_study_prior_regularization` in
  [`hk_computational_choices.py`](hk_computational_choices.py).
- Per-replicate overrides: `n_steps_override`, `bootstrap_after_data_override`,
  `use_prior_regularization` on `run_single_hk_replicate`.
- Index continuation logic: `_prepare_run` in
  [`recursive_mixtures/flows.py`](recursive_mixtures/flows.py).
- Prior term gate: `HellingerKantorovichFlow.use_prior_regularization` in
  [`recursive_mixtures/flows.py`](recursive_mixtures/flows.py).
