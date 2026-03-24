#!/usr/bin/env python3
"""
Uncertainty Quantification via Bootstrapping for Particle HK Flow.

This script runs the regularized Particle Hellinger–Kantorovich (HK) flow
on a bivariate Gaussian mixture using the Bayesian bootstrap to generate
multiple flow replicates. It then constructs pointwise credible intervals
for the density and (optionally) compares them against a NumPyro MCMC
baseline.

Studies (``--study``): **truncation** compares stopping after one data pass
vs index continuation across sample sizes; **prior** compares HK with
Fisher–Rao prior regularization on vs off (continuation for both).
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Optional, Tuple
import time

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np

from recursive_mixtures import (
    GaussianKernel,
    ParticleMeasure,
    GaussianPrior,
    HellingerKantorovichFlow,
)
from recursive_mixtures.flows import NewtonHellingerFlow, NewtonWassersteinFlow
from recursive_mixtures.utils import (
    bayesian_bootstrap,
    generate_mixture_data,
    true_mixture_density,
)


def setup_config(fast: bool = True) -> Dict:
    """Configuration dictionary for the bootstrap HK experiment.
    
    Use fast=True (default) for quicker runs: fewer data steps, bootstrap
    replicates, Sinkhorn iterations, and MCMC off by default.
    """
    # Fast defaults: fewer steps and Sinkhorn work so runs complete in minutes
    config = {
        # True bivariate mixture parameters
        "true_means": jnp.array(
            [
                [-2.0, -2.0],
                [0.0, 2.0],
                [2.5, -1.0],
            ]
        ),
        "true_stds": jnp.array([0.6, 0.8, 0.5]),
        "true_weights": jnp.array([0.3, 0.4, 0.3]),
        # Data
        "n_data": 200 if fast else 1000,
        # Flow parameters
        "n_particles": 50,
        "step_size": 0.05,
        "kernel_bandwidth": 1.0,
        "sinkhorn_reg": 0.05,
        "wasserstein_weight": 0.1,
        "prior_flow_weight": 0.1,  # lambda
        # Fisher–Rao Sinkhorn prior term in HK (see flows.HellingerKantorovichFlow)
        "use_prior_regularization": True,
        "prior_mc_samples": 1 if fast else 5,  # M (1 = much faster per step)
        "sinkhorn_num_iters": 25 if fast else 50,  # Sinkhorn iters per OT solve
        # Bootstrap replicates per experiment cell (B=1 for quick runs)
        "n_bootstrap": 1,
        # Prior
        "prior_mean": jnp.array([0.0, 0.0]),
        "prior_std": 3.0,
        # Density grid
        "grid_min": -5.0,
        "grid_max": 5.0,
        "grid_size": 35 if fast else 50,
        # Recording
        "store_every": 0,  # only final measures for bootstrap
        # Random seeds
        "seed": 123,
        # Optional override from CLI. If set and > n_data, flow.run uses
        # bootstrap continuation beyond data length.
        "n_steps": None,
        # Studies (truncation vs continuation; prior on/off): sample sizes and
        # continuation multiplier n_steps = ceil(continuation_factor * n_data).
        "n_data_list": [50, 100, 200] if fast else [200, 500, 1000],
        "continuation_factor": 2.0,
    }
    return config


def generate_bivariate_data(
    key: jax.Array,
    config: Dict,
) -> Tuple[jax.Array, jax.Array]:
    """Generate 2D Gaussian mixture data and assignments."""
    samples, assignments = generate_mixture_data(
        key,
        config["n_data"],
        config["true_means"],
        config["true_stds"],
        config["true_weights"],
    )
    # samples shape: (n_data, 2)
    return samples, assignments


def make_prior_and_kernel(config: Dict):
    """Create prior and kernel objects for the HK flow."""
    prior = GaussianPrior(
        mean=config["prior_mean"],
        std=config["prior_std"],
        dim=2,
    )
    kernel = GaussianKernel(bandwidth=config["kernel_bandwidth"])
    return prior, kernel


def make_hk_flow(
    prior,
    kernel,
    prior_particles: ParticleMeasure,
    config: Dict,
) -> HellingerKantorovichFlow:
    """Instantiate a Hellinger–Kantorovich flow with regularization."""
    flow = HellingerKantorovichFlow(
        kernel=kernel,
        prior=prior,
        step_size=config["step_size"],
        wasserstein_weight=config["wasserstein_weight"],
        sinkhorn_reg=config["sinkhorn_reg"],
        use_sinkhorn=True,
        prior_particles=prior_particles,
        prior_flow_weight=config["prior_flow_weight"],
        use_prior_regularization=config.get("use_prior_regularization", True),
        prior_mc_samples=config["prior_mc_samples"],
        sinkhorn_num_iters=config.get("sinkhorn_num_iters", 30),
    )
    return flow


def make_newton_hellinger_flow(
    prior,
    kernel,
    config: Dict,
) -> NewtonHellingerFlow:
    """Instantiate a Newton-Hellinger (weight-only) flow."""
    flow = NewtonHellingerFlow(
        kernel=kernel,
        prior=prior,
        step_size=config["step_size"],
    )
    return flow


def make_newton_wasserstein_flow(
    prior,
    kernel,
    prior_particles: ParticleMeasure,
    config: Dict,
) -> NewtonWassersteinFlow:
    """Instantiate a Newton-Wasserstein (location-only) flow."""
    flow = NewtonWassersteinFlow(
        kernel=kernel,
        prior=prior,
        step_size=config["step_size"],
        wasserstein_weight=config["wasserstein_weight"],
        sinkhorn_reg=config["sinkhorn_reg"],
        use_sinkhorn=True,
        prior_particles=prior_particles,
        sinkhorn_num_iters=config.get("sinkhorn_num_iters", 30),
    )
    return flow


def run_single_hk_replicate(
    key: jax.Array,
    data: jax.Array,
    prior,
    kernel,
    prior_particles: ParticleMeasure,
    config: Dict,
    *,
    n_steps_override: Optional[int] = None,
    bootstrap_after_data_override: Optional[bool] = None,
    use_prior_regularization: Optional[bool] = None,
) -> ParticleMeasure:
    """
    Run a single HK flow bootstrap replicate.

    Uses Bayesian bootstrap weights to resample the data indices and then
    runs the HK flow on that resampled stream.
    """
    n_data = data.shape[0]
    n_particles = config["n_particles"]

    # Split keys for bootstrap weights, resampling, prior initialization, and flow
    key_boot, key_resample, key_init, key_flow = jr.split(key, 4)

    # Bayesian bootstrap weights and resampling indices
    weights_boot = bayesian_bootstrap(key_boot, n_data)
    indices = jr.choice(
        key_resample,
        n_data,
        shape=(n_data,),
        p=weights_boot,
        replace=True,
    )
    data_boot = data[indices]

    # Initialize particles from the prior expectation measure E[P]
    atoms0 = prior.sample(key_init, n_particles)
    initial_measure = ParticleMeasure.initialize(atoms0)

    cfg = dict(config)
    if use_prior_regularization is not None:
        cfg["use_prior_regularization"] = use_prior_regularization
    flow = make_hk_flow(prior, kernel, prior_particles, cfg)

    n_steps = n_steps_override if n_steps_override is not None else cfg.get("n_steps")
    if bootstrap_after_data_override is not None:
        bootstrap_after_data = bootstrap_after_data_override
    else:
        bootstrap_after_data = bool(
            n_steps is not None and n_steps > int(data_boot.shape[0])
        )
    total_steps = int(n_steps) if n_steps is not None else int(data_boot.shape[0])

    if config["store_every"] and config["store_every"] > 0:
        final_measure, _ = flow.run(
            initial_measure,
            data_boot,
            key=key_flow,
            store_every=config["store_every"],
            n_steps=n_steps,
            bootstrap_after_data=bootstrap_after_data,
        )
    else:
        final_measure, _ = flow.run(
            initial_measure,
            data_boot,
            key=key_flow,
            store_every=total_steps,  # only store final state
            n_steps=n_steps,
            bootstrap_after_data=bootstrap_after_data,
        )
    return final_measure


def run_single_newton_h_replicate(
    key: jax.Array,
    data: jax.Array,
    prior,
    kernel,
    config: Dict,
    *,
    n_steps_override: Optional[int] = None,
    bootstrap_after_data_override: Optional[bool] = None,
) -> ParticleMeasure:
    """
    Run a single Newton-Hellinger bootstrap replicate (weights only, fixed atoms).
    """
    n_data = data.shape[0]
    n_particles = config["n_particles"]

    # Split keys for bootstrap weights, resampling, prior initialization, and flow
    key_boot, key_resample, key_init, key_flow = jr.split(key, 4)

    # Bayesian bootstrap weights and resampling indices
    weights_boot = bayesian_bootstrap(key_boot, n_data)
    indices = jr.choice(
        key_resample,
        n_data,
        shape=(n_data,),
        p=weights_boot,
        replace=True,
    )
    data_boot = data[indices]

    # Fixed atoms from the prior
    atoms0 = prior.sample(key_init, n_particles)
    initial_measure = ParticleMeasure.initialize(atoms0)

    flow = make_newton_hellinger_flow(prior, kernel, config)

    n_steps = n_steps_override if n_steps_override is not None else config.get("n_steps")
    if bootstrap_after_data_override is not None:
        bootstrap_after_data = bootstrap_after_data_override
    else:
        bootstrap_after_data = bool(
            n_steps is not None and n_steps > int(data_boot.shape[0])
        )
    total_steps = int(n_steps) if n_steps is not None else int(data_boot.shape[0])

    if config["store_every"] and config["store_every"] > 0:
        final_measure, _ = flow.run(
            initial_measure,
            data_boot,
            key=key_flow,
            store_every=config["store_every"],
            n_steps=n_steps,
            bootstrap_after_data=bootstrap_after_data,
        )
    else:
        final_measure, _ = flow.run(
            initial_measure,
            data_boot,
            key=key_flow,
            store_every=total_steps,
            n_steps=n_steps,
            bootstrap_after_data=bootstrap_after_data,
        )
    return final_measure


def run_single_newton_w_replicate(
    key: jax.Array,
    data: jax.Array,
    prior,
    kernel,
    prior_particles: ParticleMeasure,
    config: Dict,
    *,
    n_steps_override: Optional[int] = None,
    bootstrap_after_data_override: Optional[bool] = None,
) -> ParticleMeasure:
    """
    Run a single Newton-Wasserstein bootstrap replicate (locations only, fixed weights).
    """
    n_data = data.shape[0]
    n_particles = config["n_particles"]

    # Split keys for bootstrap weights, resampling, prior initialization, and flow
    key_boot, key_resample, key_init, key_flow = jr.split(key, 4)

    # Bayesian bootstrap weights and resampling indices
    weights_boot = bayesian_bootstrap(key_boot, n_data)
    indices = jr.choice(
        key_resample,
        n_data,
        shape=(n_data,),
        p=weights_boot,
        replace=True,
    )
    data_boot = data[indices]

    atoms0 = prior.sample(key_init, n_particles)
    initial_measure = ParticleMeasure.initialize(atoms0)

    flow = make_newton_wasserstein_flow(prior, kernel, prior_particles, config)

    n_steps = n_steps_override if n_steps_override is not None else config.get("n_steps")
    if bootstrap_after_data_override is not None:
        bootstrap_after_data = bootstrap_after_data_override
    else:
        bootstrap_after_data = bool(
            n_steps is not None and n_steps > int(data_boot.shape[0])
        )
    total_steps = int(n_steps) if n_steps is not None else int(data_boot.shape[0])

    if config["store_every"] and config["store_every"] > 0:
        final_measure, _ = flow.run(
            initial_measure,
            data_boot,
            key=key_flow,
            store_every=config["store_every"],
            n_steps=n_steps,
            bootstrap_after_data=bootstrap_after_data,
        )
    else:
        final_measure, _ = flow.run(
            initial_measure,
            data_boot,
            key=key_flow,
            store_every=total_steps,
            n_steps=n_steps,
            bootstrap_after_data=bootstrap_after_data,
        )
    return final_measure


def build_density_grid(config: Dict) -> jax.Array:
    """Construct a 2D grid of evaluation points."""
    gmin = config["grid_min"]
    gmax = config["grid_max"]
    n = config["grid_size"]
    xs = jnp.linspace(gmin, gmax, n)
    ys = jnp.linspace(gmin, gmax, n)
    X, Y = jnp.meshgrid(xs, ys)
    grid_points = jnp.stack([X.ravel(), Y.ravel()], axis=1)  # (n^2, 2)
    return grid_points


def hk_bootstrap_densities(
    measures: List[ParticleMeasure],
    kernel,
    grid_points: jax.Array,
) -> jax.Array:
    """
    Compute density estimates on the grid for each HK bootstrap replicate.

    Returns:
        Array of shape (B, G) where G = grid_points.shape[0].
    """
    densities = []
    for m in measures:
        dens = m.kernel_density(kernel, grid_points)
        densities.append(dens)
    return jnp.stack(densities, axis=0)


def credible_intervals(
    densities: jax.Array,
    alpha: float = 0.05,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """
    Compute pointwise credible intervals from bootstrap density samples.

    Args:
        densities: Array of shape (B, G)
        alpha: Credible level (e.g. 0.05 for 95% intervals)

    Returns:
        (mean, lower, upper) each of shape (G,)
    """
    mean = jnp.mean(densities, axis=0)
    lower = jnp.quantile(densities, alpha / 2.0, axis=0)
    upper = jnp.quantile(densities, 1.0 - alpha / 2.0, axis=0)
    return mean, lower, upper


def compute_coverage(
    true_density: jax.Array,
    lower: jax.Array,
    upper: jax.Array,
) -> float:
    """Compute coverage rate of true density within intervals."""
    inside = (true_density >= lower) & (true_density <= upper)
    return float(jnp.mean(inside))


def bootstrap_coverage_for_measures(
    measures: List[ParticleMeasure],
    kernel,
    grid_points: jax.Array,
    true_density_vals: jax.Array,
) -> float:
    """Grid-based 95% credible-interval coverage of the true density."""
    dens = hk_bootstrap_densities(measures, kernel, grid_points)
    _, lower, upper = credible_intervals(dens, alpha=0.05)
    return compute_coverage(true_density_vals, lower, upper)


def plot_truncation_vs_continuation(
    n_data_list: List[int],
    cov_hk_trunc: np.ndarray,
    cov_hk_cont: np.ndarray,
    cov_nh_trunc: np.ndarray,
    cov_nh_cont: np.ndarray,
    cov_nw_trunc: np.ndarray,
    cov_nw_cont: np.ndarray,
    out_path: str = "bootstrap_truncation_vs_continuation.pdf",
) -> None:
    """Line plot: coverage vs sample size for truncated vs continued runs."""
    xs = np.asarray(n_data_list, dtype=float)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, cov_hk_trunc, "o-", color="teal", label="HK (stop after data)")
    ax.plot(xs, cov_hk_cont, "s--", color="teal", alpha=0.85, label="HK (continuation)")
    ax.plot(xs, cov_nh_trunc, "o-", color="royalblue", label="Newton-H (stop after data)")
    ax.plot(xs, cov_nh_cont, "s--", color="royalblue", alpha=0.85, label="Newton-H (continuation)")
    ax.plot(xs, cov_nw_trunc, "o-", color="crimson", label="Newton-W (stop after data)")
    ax.plot(xs, cov_nw_cont, "s--", color="crimson", alpha=0.85, label="Newton-W (continuation)")
    ax.set_xlabel("Sample size $n$")
    ax.set_ylabel("Grid 95% CI coverage of true density")
    ax.set_title("Truncated run vs index continuation (Bayesian bootstrap per replicate)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_prior_regularization_study(
    n_data_list: List[int],
    cov_prior_on: np.ndarray,
    cov_prior_off: np.ndarray,
    out_path: str = "bootstrap_prior_regularization.pdf",
) -> None:
    """HK only: Fisher–Rao prior term on vs off, both with continuation."""
    xs = np.asarray(n_data_list, dtype=float)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(xs, cov_prior_on, "o-", color="teal", label="Prior regularization on")
    ax.plot(xs, cov_prior_off, "s--", color="dimgray", label="Prior regularization off")
    ax.set_xlabel("Sample size $n$")
    ax.set_ylabel("Grid 95% CI coverage of true density")
    ax.set_title(
        "HK flow: Fisher–Rao prior term (both runs use index continuation; "
        "atom Sinkhorn drift unchanged)"
    )
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def run_study_truncation_vs_continuation(config: Dict, key: jax.Array) -> jax.Array:
    """Compare stopping after one pass vs extra steps with index continuation."""
    print("=" * 80)
    print("Study A: stop after data vs index continuation (per sample size)")
    print("=" * 80)
    n_data_list = list(config["n_data_list"])
    continuation_factor = float(config["continuation_factor"])
    B = config["n_bootstrap"]
    prior, kernel = make_prior_and_kernel(config)
    grid_points = build_density_grid(config)
    true_density_vals = true_mixture_density(
        grid_points,
        config["true_means"],
        config["true_stds"],
        config["true_weights"],
    )

    n_sizes = len(n_data_list)
    cov_hk_trunc = np.zeros(n_sizes)
    cov_hk_cont = np.zeros(n_sizes)
    cov_nh_trunc = np.zeros(n_sizes)
    cov_nh_cont = np.zeros(n_sizes)
    cov_nw_trunc = np.zeros(n_sizes)
    cov_nw_cont = np.zeros(n_sizes)

    for i, nd in enumerate(n_data_list):
        cfg = dict(config)
        cfg["n_data"] = nd
        key, data_key, pp_key = jr.split(key, 3)
        data, _ = generate_bivariate_data(data_key, cfg)
        prior_particles = prior.to_particle_measure(pp_key, cfg["n_particles"])

        n_steps_trunc = nd
        n_steps_cont = int(np.ceil(continuation_factor * nd))
        print(
            f"  n_data={nd}: truncated n_steps={n_steps_trunc}, "
            f"continuation n_steps={n_steps_cont} (factor={continuation_factor})"
        )

        hk_t, nh_t, nw_t = [], [], []
        for _ in range(B):
            key, key_hk, key_nh, key_nw = jr.split(key, 4)
            hk_t.append(
                run_single_hk_replicate(
                    key_hk,
                    data,
                    prior,
                    kernel,
                    prior_particles,
                    cfg,
                    n_steps_override=n_steps_trunc,
                    bootstrap_after_data_override=False,
                )
            )
            nh_t.append(
                run_single_newton_h_replicate(
                    key_nh,
                    data,
                    prior,
                    kernel,
                    cfg,
                    n_steps_override=n_steps_trunc,
                    bootstrap_after_data_override=False,
                )
            )
            nw_t.append(
                run_single_newton_w_replicate(
                    key_nw,
                    data,
                    prior,
                    kernel,
                    prior_particles,
                    cfg,
                    n_steps_override=n_steps_trunc,
                    bootstrap_after_data_override=False,
                )
            )
        cov_hk_trunc[i] = bootstrap_coverage_for_measures(
            hk_t, kernel, grid_points, true_density_vals
        )
        cov_nh_trunc[i] = bootstrap_coverage_for_measures(
            nh_t, kernel, grid_points, true_density_vals
        )
        cov_nw_trunc[i] = bootstrap_coverage_for_measures(
            nw_t, kernel, grid_points, true_density_vals
        )

        hk_c, nh_c, nw_c = [], [], []
        for _ in range(B):
            key, key_hk, key_nh, key_nw = jr.split(key, 4)
            hk_c.append(
                run_single_hk_replicate(
                    key_hk,
                    data,
                    prior,
                    kernel,
                    prior_particles,
                    cfg,
                    n_steps_override=n_steps_cont,
                    bootstrap_after_data_override=True,
                )
            )
            nh_c.append(
                run_single_newton_h_replicate(
                    key_nh,
                    data,
                    prior,
                    kernel,
                    cfg,
                    n_steps_override=n_steps_cont,
                    bootstrap_after_data_override=True,
                )
            )
            nw_c.append(
                run_single_newton_w_replicate(
                    key_nw,
                    data,
                    prior,
                    kernel,
                    prior_particles,
                    cfg,
                    n_steps_override=n_steps_cont,
                    bootstrap_after_data_override=True,
                )
            )
        cov_hk_cont[i] = bootstrap_coverage_for_measures(
            hk_c, kernel, grid_points, true_density_vals
        )
        cov_nh_cont[i] = bootstrap_coverage_for_measures(
            nh_c, kernel, grid_points, true_density_vals
        )
        cov_nw_cont[i] = bootstrap_coverage_for_measures(
            nw_c, kernel, grid_points, true_density_vals
        )

        print(
            f"    coverage trunc: HK={cov_hk_trunc[i]:.3f} NH={cov_nh_trunc[i]:.3f} "
            f"NW={cov_nw_trunc[i]:.3f}"
        )
        print(
            f"    coverage cont:  HK={cov_hk_cont[i]:.3f} NH={cov_nh_cont[i]:.3f} "
            f"NW={cov_nw_cont[i]:.3f}"
        )

    plot_truncation_vs_continuation(
        n_data_list,
        cov_hk_trunc,
        cov_hk_cont,
        cov_nh_trunc,
        cov_nh_cont,
        cov_nw_trunc,
        cov_nw_cont,
    )
    print("\nSaved 'bootstrap_truncation_vs_continuation.pdf'.")
    return key


def run_study_prior_regularization(config: Dict, key: jax.Array) -> jax.Array:
    """HK only: Fisher–Rao prior on vs off; both arms use the same continuation schedule."""
    print("=" * 80)
    print("Study B: HK with Fisher–Rao prior regularization on vs off (continuation both)")
    print("=" * 80)
    n_data_list = list(config["n_data_list"])
    continuation_factor = float(config["continuation_factor"])
    B = config["n_bootstrap"]
    prior, kernel = make_prior_and_kernel(config)
    grid_points = build_density_grid(config)
    true_density_vals = true_mixture_density(
        grid_points,
        config["true_means"],
        config["true_stds"],
        config["true_weights"],
    )

    n_sizes = len(n_data_list)
    cov_on = np.zeros(n_sizes)
    cov_off = np.zeros(n_sizes)

    for i, nd in enumerate(n_data_list):
        cfg = dict(config)
        cfg["n_data"] = nd
        key, data_key, pp_key = jr.split(key, 3)
        data, _ = generate_bivariate_data(data_key, cfg)
        prior_particles = prior.to_particle_measure(pp_key, cfg["n_particles"])
        n_steps = int(np.ceil(continuation_factor * nd))
        print(f"  n_data={nd}, continuation n_steps={n_steps}")

        hk_on, hk_off = [], []
        for _ in range(B):
            key, key_on, key_off = jr.split(key, 3)
            hk_on.append(
                run_single_hk_replicate(
                    key_on,
                    data,
                    prior,
                    kernel,
                    prior_particles,
                    cfg,
                    n_steps_override=n_steps,
                    bootstrap_after_data_override=True,
                    use_prior_regularization=True,
                )
            )
            hk_off.append(
                run_single_hk_replicate(
                    key_off,
                    data,
                    prior,
                    kernel,
                    prior_particles,
                    cfg,
                    n_steps_override=n_steps,
                    bootstrap_after_data_override=True,
                    use_prior_regularization=False,
                )
            )
        cov_on[i] = bootstrap_coverage_for_measures(
            hk_on, kernel, grid_points, true_density_vals
        )
        cov_off[i] = bootstrap_coverage_for_measures(
            hk_off, kernel, grid_points, true_density_vals
        )
        print(f"    coverage prior on: {cov_on[i]:.3f}  prior off: {cov_off[i]:.3f}")

    plot_prior_regularization_study(n_data_list, cov_on, cov_off)
    print("\nSaved 'bootstrap_prior_regularization.pdf'.")
    return key




def main(
    fast: bool = True,
    study: str = "both",
    n_data_list: Optional[List[int]] = None,
    continuation_factor: float = 2.0,
) -> None:
    """Run truncation/prior bootstrap studies (or both)."""
    config = setup_config(fast=fast)
    if n_data_list is not None:
        config["n_data_list"] = n_data_list
    config["continuation_factor"] = float(continuation_factor)

    key = jr.PRNGKey(config["seed"])


    if study in ("truncation", "both"):
        key = run_study_truncation_vs_continuation(config, key)

    if study in ("prior", "both"):
        run_study_prior_regularization(config, key)


if __name__ == "__main__":
    # Enable 64-bit precision for numerical stability if available
    try:
        jax.config.update("jax_enable_x64", True)
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="Bootstrap flow comparison experiment")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full (slower) configuration instead of fast mode.",
    )
    parser.add_argument(
        "--study",
        type=str,
        choices=("truncation", "prior", "both"),
        default="both",
        help="truncation|prior|both sample-size studies.",
    )
    parser.add_argument(
        "--n-data-list",
        type=str,
        default=None,
        help="Comma-separated sample sizes for truncation/prior studies (default: config).",
    )
    parser.add_argument(
        "--continuation-factor",
        type=float,
        default=None,
        help="n_steps = ceil(factor * n_data) for continuation arms (default: config).",
    )
    args = parser.parse_args()

    nd_list: Optional[List[int]] = None
    if args.n_data_list:
        nd_list = [int(x.strip()) for x in args.n_data_list.split(",") if x.strip()]

    cont_factor = args.continuation_factor
    if cont_factor is None:
        sc = setup_config(fast=not args.full)
        cont_factor = float(sc["continuation_factor"])

    main(
        fast=not args.full,
        study=args.study,
        n_data_list=nd_list,
        continuation_factor=cont_factor,
    )

