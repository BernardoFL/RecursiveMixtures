# HK computational choices experiment (LLM-oriented reference)

> **File renamed.** This document now tracks
> [`hk_computational_choices.py`](hk_computational_choices.py). See
> [`HK_COMPUTATIONAL_CHOICES.md`](HK_COMPUTATIONAL_CHOICES.md) for the full
> up-to-date reference.

This experiment compares two computational switches in the HK (WFR) flow:

1. Fisher-Rao prior regularization: **on/off**
2. Bootstrap continuation after the ordered pass: **on/off**

## Reference figures in Git

Selected PDF outputs are **committed to the repository** as reference artifacts (see [`.gitignore`](.gitignore): general `*.pdf` is ignored, but these names are negated). **Acknowledge them as outputs of this codebase**, not hand-drawn figures.

When you change flow logic, defaults, or plotting code, **re-run** the matching command below and **commit the refreshed PDFs** so the tracked figures stay consistent with the implementation.

| PDF | Example command |
|-----|-----------------|
| [`bootstrap_truncation_vs_continuation.pdf`](bootstrap_truncation_vs_continuation.pdf) | `python hk_computational_choices.py --study truncation` |
| [`bootstrap_prior_regularization.pdf`](bootstrap_prior_regularization.pdf) | `python hk_computational_choices.py --study prior` |

## Problem setup

- **Target**: The **Rosenbrock distribution**. Parameters live in `setup_config()`
  (`rosen_a`, `rosen_b`, `rosen_sigma`).
- **Observed data**: i.i.d. samples from that mixture of size **`n_data`** (varies per study iteration when using `n_data_list`).
- **Particle approximation**: **Hellinger–Kantorovich (HK)** flow with a **Pitman–Yor mixing prior** PY(\(d, \theta, G_0\)) on atom locations (\(G_0\) = isotropic Gaussian; `py_discount`, `py_strength` in `setup_config()`) and a fixed `n_particles`.
- **Bayesian bootstrap (per replicate)**: For each Monte Carlo replicate, Dirichlet-style weights are drawn over data indices; a **bootstrap dataset** `data_boot` of length `n_data` is built by sampling training indices **with replacement** according to those weights. That is independent from **index continuation** (below).

## Compared computational choices

The script studies a 2x2 factorial design:

| Choice | Option 1 | Option 2 |
|--------|----------|----------|
| Fisher-Rao prior regularization | `use_prior_regularization=True` | `use_prior_regularization=False` |
| Continuation after ordered pass | `bootstrap_after_data=True` with extra steps | `bootstrap_after_data=False` (stop at `n_data`) |

This isolates algorithmic/computational effects while keeping the model and
data source fixed.

## Output interpretation

- `bootstrap_truncation_vs_continuation.pdf`: slices emphasizing continuation on/off.
- `bootstrap_prior_regularization.pdf`: slices emphasizing prior regularization on/off.
- All panels show true-density heatmaps plus final HK particles (size ∝ weight),
  with no training-data scatter.

## CLI

```bash
python hk_computational_choices.py --study truncation
python hk_computational_choices.py --study prior
python hk_computational_choices.py --study both
```

---

## Configuration quick reference

| Key / flag | Role |
|------------|------|
| `n_data_list` / `--n-data-list` | Sample sizes used in the comparison |
| `continuation_factor` / `--continuation-factor` | Continuation length multiplier (A/B only) |
| `n_bootstrap` | Replicates per cell (default **1**); PDFs show the **first** replicate only |
| `use_prior_regularization` | Prior regularization switch (on/off arms) |
| `py_discount`, `py_strength` | Pitman–Yor parameters \(d\) and \(\theta\) for PY(\(d, \theta, G_0\)) |
| `prior_flow_weight`, `prior_mc_samples` | Strength and MC draws for HK prior term when enabled |
| `--full` | Heavier defaults (more data, replicates, Sinkhorn work) |
| `--study` | `truncation` \| `prior` \| `both` (default `both`) |

---

## Files produced

| File | Content |
|------|---------|
| `bootstrap_truncation_vs_continuation.pdf` | HK comparison focused on continuation on/off |
| `bootstrap_prior_regularization.pdf` | HK comparison focused on prior regularization on/off |

---

## Implementation pointers

- Study runners: `run_study_truncation_vs_continuation`, `run_study_prior_regularization` in [`hk_computational_choices.py`](hk_computational_choices.py).
- Per-replicate overrides: `n_steps_override`, `bootstrap_after_data_override`, `use_prior_regularization` on `run_single_hk_replicate`.
- Index continuation logic: `_prepare_run` in [`recursive_mixtures/flows.py`](recursive_mixtures/flows.py).
- Prior term gate: `HellingerKantorovichFlow.use_prior_regularization` and `need_mc` in `step`.
