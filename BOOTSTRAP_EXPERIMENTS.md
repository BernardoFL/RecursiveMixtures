# Bootstrap flow experiments (LLM-oriented reference)

This document describes the structured studies in [`bootstrap_experiment.py`](bootstrap_experiment.py): **A** and **B** use the bivariate Gaussian mixture with HK-only Bayesian-bootstrap replicates; **C (paw)** is a separate cat-paw HK comparison (same script, different target, no per-replicate bootstrap resampling).

## Reference figures in Git

Selected PDF outputs are **committed to the repository** as reference artifacts (see [`.gitignore`](.gitignore): general `*.pdf` is ignored, but these names are negated). **Acknowledge them as outputs of this codebase**, not hand-drawn figures.

When you change flow logic, defaults, or plotting code, **re-run** the matching command below and **commit the refreshed PDFs** so the tracked figures stay consistent with the implementation.

| PDF | Example command |
|-----|-----------------|
| [`bootstrap_truncation_vs_continuation.pdf`](bootstrap_truncation_vs_continuation.pdf) | `python bootstrap_experiment.py --study truncation` |
| [`bootstrap_prior_regularization.pdf`](bootstrap_prior_regularization.pdf) | `python bootstrap_experiment.py --study prior` |
| `paw_hk_comparison.pdf` | `python bootstrap_experiment.py --study paw` |
| `paw_hk_overlay_n{n}.pdf` | `python bootstrap_experiment.py --study paw --n-data-list 100,200 --k …` |

## Problem setup

- **Target**: A fixed **bivariate Gaussian mixture** (three components). True parameters live in `setup_config()` (`true_means`, `true_stds`, `true_weights`).
- **Observed data**: i.i.d. samples from that mixture of size **`n_data`** (varies per study iteration when using `n_data_list`).
- **Particle approximation**: **Hellinger–Kantorovich (HK)** flow with a **Gaussian prior** on atom locations and a **fixed particle count** `n_particles`.
- **Bayesian bootstrap (per replicate)**: For each Monte Carlo replicate, Dirichlet-style weights are drawn over data indices; a **bootstrap dataset** `data_boot` of length `n_data` is built by sampling training indices **with replacement** according to those weights. That is independent from **index continuation** (below).

## Two meanings of “bootstrap”

1. **Bayesian bootstrap (resampling)** — Happens **inside each replicate**: reweights and resamples the empirical measure to form `data_boot`. Used in all studies.
2. **Index continuation (flow runtime)** — After the first pass over `data_boot`, the flow can either **stop** or **draw more steps** by appending random indices into `data_boot` (uniformly). This is controlled by `n_steps` and `bootstrap_after_data` in `flow.run` / `_prepare_run` in [`recursive_mixtures/flows.py`](recursive_mixtures/flows.py).

Study **A** compares (1) one epoch through `data_boot` vs (2) extra steps with index continuation.  
Study **B** fixes continuation and toggles **Fisher–Rao prior regularization** on vs off (HK only).

---

## Study A: Truncation vs index continuation

**Goal:** For several sample sizes `n ∈ n_data_list`, visualize how **stopping after one pass** over the bootstrap stream compares to **running longer** with **index continuation** (final particle locations on the true density).

### Conditions

| Condition | `n_steps` | `bootstrap_after_data` | Interpretation |
|-----------|-----------|------------------------|----------------|
| **Truncated** | `n_steps = n_data` | `False` | Exactly one pass over indices `0 … n_data-1` of `data_boot`. |
| **Continuation** | `n_steps = ceil(continuation_factor × n_data)` | `True` | First epoch is sequential; remaining steps reuse random indices into `data_boot`. |

Default `continuation_factor` is **2.0** (configurable).

### Flow compared

Study A uses **HK only**. For **each** `n`, it runs **`n_bootstrap`** truncated HK flows and **`n_bootstrap`** continuation HK flows (independent bootstrap resamples per call); the PDF shows the **first** replicate of each arm (**teal** = stop after data, **crimson** = continuation).

### Data generation note

For each `n`, a **new full dataset** of size `n` is simulated from the same true mixture (independent draws per `n`; not a nested subsample from one super-population).

### Output

- **Plot**: `bootstrap_truncation_vs_continuation.pdf` — **multi-page PDF** (one page per `n`). Each page is **1×2**: HK **truncated** vs HK **continuation** — true density heatmap + final particles (size ∝ weight); **teal** / **crimson**.
- **Console**: progress only (sample sizes and step counts).

### CLI

```bash
python bootstrap_experiment.py --study truncation
python bootstrap_experiment.py --study both   # runs A then B
```

Optional: `--n-data-list 50,100,200`, `--continuation-factor 2.0`, `--full` (changes default list from `[50,100,200]` to `[200,500,1000]`, raises `grid_size`, `prior_mc_samples`, and `sinkhorn_num_iters`).

---

## Study B: HK Fisher–Rao prior regularization ON vs OFF

**Goal:** For each `n ∈ n_data_list`, visualize **HK only** with Fisher–Rao prior regularization **on** vs **off** with continuation explicitly disabled in both arms.

### Schedule (both arms)

- `n_steps = n_data`
- `bootstrap_after_data = False`

### What is toggled

In [`HellingerKantorovichFlow`](recursive_mixtures/flows.py), `use_prior_regularization` gates whether the **Sinkhorn prior functional** contributes to the **Hellinger gradient** (`prior_flow_weight × h`).  

- **`True`**: Fisher–Rao update uses likelihood gradient **plus** weighted prior term (if `prior_flow_weight > 0` and `prior_mc_samples > 0`).
- **`False`**: That **weight-update** prior term is **skipped**, **even if** `prior_flow_weight` is positive.

**Not toggled:** Atom updates can still use **Sinkhorn drift toward the prior** when `use_sinkhorn=True` (separate from the Fisher–Rao prior regularization switch).

### Output

- **Plot**: `bootstrap_prior_regularization.pdf` — **multi-page PDF** (one page per `n`). Each page matches Study A’s layout: **1×2** panels — left = Fisher–Rao prior regularization **on** (**teal**), right = **off** (**royalblue**); each panel has its own true-density heatmap + particles (size ∝ weight); both arms use the same non-continuation `n_steps = n_data`. **First** bootstrap replicate per arm.
- **Console**: progress only.

### CLI

```bash
python bootstrap_experiment.py --study prior
```

---

## Study C: Cat-paw HK triple (``--study paw``)

**Goal:** Same as the standalone [`metastability_experiment.py`](metastability_experiment.py) **paw HK** setup: **one** i.i.d. dataset from the **seven-component cat paw** mixture, **one** DP-based initial particle measure, then **three** HK runs on that fixed data:

1. **(a)** **n** ordered steps, Fisher–Rao prior regularization **on**.
2. **(b)** **n** ordered steps, prior regularization **off** (Sinkhorn atom drift unchanged).
3. **(c)** **n + k** steps, prior **on**, with **uniform resampling** from the data after the first **n** observations (`bootstrap_after_data=True`).

This path **does not** use the Bayesian bootstrap replicates from Studies A/B; it only shares the script entry point.

### Output

| File | When |
|------|------|
| `paw_hk_comparison.pdf` | Single **n** (default or `--n-data`): three panels (density + particles per case). |
| `paw_hk_overlay_n{n}.pdf` | `--n-data-list n1,n2,...`: one **overlay** figure per **n** (all three particle sets on one heatmap; no data scatter). |

If both `--n-data` and `--n-data-list` are passed, **`--n-data-list` wins** (overlay sweep only).

### CLI

```bash
python bootstrap_experiment.py --study paw
python bootstrap_experiment.py --study paw --n-data 500 --k 100
python bootstrap_experiment.py --study paw --n-data-list 200,500,1000 --k 200
```

- **`--n-data`** — sample size **n** (paw study only; ignored for `truncation`/`prior`/`both`).
- **`--k`** — extra resampled steps for case **(c)** (paw study only; ignored for `truncation`/`prior`/`both`).
- **`--n-data-list`** — for **paw**, comma-separated **n** values → overlay PDFs; for Studies A/B, sample sizes for each PDF page.

---

## Configuration quick reference

| Key / flag | Role |
|------------|------|
| `n_data_list` / `--n-data-list` | Sample sizes for Studies A and B; for **paw**, overlay sweep |
| `continuation_factor` / `--continuation-factor` | Continuation length multiplier (A/B only) |
| `n_bootstrap` | Replicates per cell (default **1**); PDFs show the **first** replicate only |
| `use_prior_regularization` | Default `True` in config; Study B overrides per arm |
| `prior_flow_weight`, `prior_mc_samples` | Strength and MC draws for HK prior term when enabled |
| `--full` | Heavier defaults (more data, replicates, Sinkhorn work) |
| `--study` | `truncation` \| `prior` \| `both` \| `paw` (default `both`) |
| `--n-data`, `--k` | Paw study only: **n** and continuation **k** for case (c) |

---

## Files produced

| File | Content |
|------|---------|
| `bootstrap_truncation_vs_continuation.pdf` | Study A — multi-page HK truncated vs continuation (1×2 per page) |
| `bootstrap_prior_regularization.pdf` | Study B — multi-page HK prior on vs off (1×2 per page, same style as A) |
| `paw_hk_comparison.pdf` / `paw_hk_overlay_n{n}.pdf` | Study C (paw) |

---

## Implementation pointers

- Studies: `run_study_truncation_vs_continuation`, `run_study_prior_regularization`, `run_study_paw_hk` in [`bootstrap_experiment.py`](bootstrap_experiment.py).
- Per-replicate overrides: `n_steps_override`, `bootstrap_after_data_override`, `use_prior_regularization` on `run_single_hk_replicate`.
- Index continuation logic: `_prepare_run` in [`recursive_mixtures/flows.py`](recursive_mixtures/flows.py).
- Prior term gate: `HellingerKantorovichFlow.use_prior_regularization` and `need_mc` in `step`.
