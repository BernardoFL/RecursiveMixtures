# Bootstrap flow experiments (LLM-oriented reference)

This document describes the two structured studies implemented in [`bootstrap_experiment.py`](bootstrap_experiment.py). Use this as context when modifying code or interpreting results.

## Problem setup

- **Target**: A fixed **bivariate Gaussian mixture** (three components). True parameters live in `setup_config()` (`true_means`, `true_stds`, `true_weights`).
- **Observed data**: i.i.d. samples from that mixture of size **`n_data`** (varies per study iteration when using `n_data_list`).
- **Particle approximation**: HK, Newton–Hellinger, and Newton–Wasserstein flows (as in the script) with a **Gaussian prior** on atom locations and a **fixed particle count** `n_particles`.
- **Bayesian bootstrap (per replicate)**: For each Monte Carlo replicate, Dirichlet-style weights are drawn over data indices; a **bootstrap dataset** `data_boot` of length `n_data` is built by sampling training indices **with replacement** according to those weights. That is independent from **index continuation** (below).

## Two meanings of “bootstrap”

1. **Bayesian bootstrap (resampling)** — Happens **inside each replicate**: reweights and resamples the empirical measure to form `data_boot`. Used in all studies.
2. **Index continuation (flow runtime)** — After the first pass over `data_boot`, the flow can either **stop** or **draw more steps** by appending random indices into `data_boot` (uniformly). This is controlled by `n_steps` and `bootstrap_after_data` in `flow.run` / `_prepare_run` in [`recursive_mixtures/flows.py`](recursive_mixtures/flows.py).

Study **A** compares (1) one epoch through `data_boot` vs (2) extra steps with index continuation.  
Study **B** fixes continuation and toggles **Fisher–Rao prior regularization** in the HK flow only.

---

## Study A: Truncation vs index continuation

**Goal:** For several sample sizes `n ∈ n_data_list`, see whether **stopping after one pass** over the bootstrap stream differs from **running longer** with **index continuation** after the stream ends.

### Conditions

| Condition | `n_steps` | `bootstrap_after_data` | Interpretation |
|-----------|-----------|------------------------|----------------|
| **Truncated** | `n_steps = n_data` | `False` | Exactly one pass over indices `0 … n_data-1` of `data_boot`. |
| **Continuation** | `n_steps = ceil(continuation_factor × n_data)` | `True` | First epoch is sequential; remaining steps reuse random indices into `data_boot`. |

Default `continuation_factor` is **2.0** (configurable).

### Flows compared

For **each** `n` and **each** condition, the script runs **`n_bootstrap` replicates** and computes grid-based **95% credible interval coverage** of the **true** mixture density (same metric as elsewhere in the script):

- **HK** (Hellinger–Kantorovich: weight + atom updates)
- **Newton–H** (Fisher–Rao / Hellinger on weights, fixed atoms)
- **Newton–W** (Wasserstein on atoms, fixed weights)

### Data generation note

For each `n`, a **new full dataset** of size `n` is simulated from the same true mixture (independent draws per `n`; not a nested subsample from one super-population).

### Output

- **Plot**: `bootstrap_truncation_vs_continuation.pdf` — coverage vs `n` with **solid** = truncated, **dashed** = continuation, **colors**: HK teal, Newton–H royal blue, Newton–W crimson.
- **Console**: per-`n` coverage numbers for each flow and condition.

### CLI

```bash
python bootstrap_experiment.py --study truncation
python bootstrap_experiment.py --study both   # runs A then B
```

Optional: `--n-data-list 50,100,200`, `--continuation-factor 2.0`, `--full`.

---

## Study B: HK Fisher–Rao prior regularization ON vs OFF

**Goal:** For each `n ∈ n_data_list`, compare **HK only** with the **Sinkhorn-based prior term in the Fisher–Rao (Hellinger) weight update** turned **on** vs **off**, while **both** arms use **the same continuation schedule**.

### Schedule (both arms)

- `n_steps = ceil(continuation_factor × n_data)`
- `bootstrap_after_data = True`

### What is toggled

In [`HellingerKantorovichFlow`](recursive_mixtures/flows.py), `use_prior_regularization` gates whether the **Sinkhorn prior functional** contributes to the **Hellinger gradient** (`prior_flow_weight × h`).  

- **`True`**: Fisher–Rao update uses likelihood gradient **plus** weighted prior term (if `prior_flow_weight > 0` and `prior_mc_samples > 0`).
- **`False`**: That **weight-update** prior term is **skipped**, **even if** `prior_flow_weight` is positive.

**Not toggled:** Atom updates can still use **Sinkhorn drift toward the prior** when `use_sinkhorn=True` (separate from the Fisher–Rao prior regularization switch).

Newton–H and Newton–W **do not** implement this HK-specific term; Study B is **HK-only** by design.

### Output

- **Plot**: `bootstrap_prior_regularization.pdf` — coverage vs `n`, two curves (prior on / prior off).
- **Console**: per-`n` coverage for each arm.

### CLI

```bash
python bootstrap_experiment.py --study prior
```

---


---

## Configuration quick reference

| Key / flag | Role |
|------------|------|
| `n_data_list` / `--n-data-list` | Sample sizes for Studies A and B |
| `continuation_factor` / `--continuation-factor` | Continuation length multiplier |
| `n_bootstrap` | Replicates per cell (default **1** for speed; increase for stable intervals) |
| `use_prior_regularization` | Default `True` in config; Study B overrides per arm |
| `prior_flow_weight`, `prior_mc_samples` | Strength and MC draws for HK prior term when enabled |
| `--full` | Heavier defaults (more data, replicates, Sinkhorn work) |
| `--study` | `truncation` \| `prior` \| `both` (default `both`) |

---

## Files produced

| File | Content |
|------|---------|
| `bootstrap_truncation_vs_continuation.pdf` | Study A |
| `bootstrap_prior_regularization.pdf` | Study B |

---

## Implementation pointers

- Studies: `run_study_truncation_vs_continuation`, `run_study_prior_regularization` in [`bootstrap_experiment.py`](bootstrap_experiment.py).
- Per-replicate overrides: `n_steps_override`, `bootstrap_after_data_override`, `use_prior_regularization` on `run_single_hk_replicate`.
- Index continuation logic: `_prepare_run` in [`recursive_mixtures/flows.py`](recursive_mixtures/flows.py).
- Prior term gate: `HellingerKantorovichFlow.use_prior_regularization` and `need_mc` in `step`.
