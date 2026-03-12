#!/usr/bin/env python3
"""
Uncertainty Quantification via Bootstrapping for Particle HK Flow.

This script runs the regularized Particle Hellinger–Kantorovich (HK) flow
on a bivariate Gaussian mixture using the Bayesian bootstrap to generate
multiple flow replicates. It then constructs pointwise credible intervals
for the density and (optionally) compares them against a NumPyro MCMC
baseline.
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Tuple
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
        "prior_mc_samples": 1 if fast else 5,  # M (1 = much faster per step)
        "sinkhorn_num_iters": 25 if fast else 50,  # Sinkhorn iters per OT solve
        # Bootstrap
        "n_bootstrap": 8 if fast else 32,  # B
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

    # Configure flow
    flow = make_hk_flow(prior, kernel, prior_particles, config)

    # Use the built-in run method, which handles key splitting internally
    n_steps = config.get("n_steps")
    bootstrap_after_data = bool(n_steps is not None and n_steps > int(data_boot.shape[0]))
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

    # Deterministic flow, no key needed
    n_steps = config.get("n_steps")
    bootstrap_after_data = bool(n_steps is not None and n_steps > int(data_boot.shape[0]))
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

    n_steps = config.get("n_steps")
    bootstrap_after_data = bool(n_steps is not None and n_steps > int(data_boot.shape[0]))
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


def plot_bootstrap_results(
    config: Dict,
    grid_points: jax.Array,
    hk_mean: jax.Array,
    hk_lower: jax.Array,
    hk_upper: jax.Array,
    nh_mean: jax.Array,
    nh_lower: jax.Array,
    nh_upper: jax.Array,
    nw_mean: jax.Array,
    nw_lower: jax.Array,
    nw_upper: jax.Array,
    true_density_vals: jax.Array,
    data: jax.Array,
):
    """Three 2D plots: true density heatmaps with HK, Newton-H, and Newton-W contours.

    Each panel shows:
    - Background: true density as grayscale heatmap.
    - HK flow: mean and 95% interval as teal contours (solid / dashed / dotted).
    - Newton-H: mean and 95% interval as blue contours.
    - Newton-W: mean and 95% interval as red contours.
    """
    n = config["grid_size"]

    def reshape(field: jax.Array) -> np.ndarray:
        return np.asarray(field).reshape(n, n)

    X = np.linspace(config["grid_min"], config["grid_max"], n)
    Y = np.linspace(config["grid_min"], config["grid_max"], n)

    true_grid = reshape(true_density_vals)
    hk_mean_grid = reshape(hk_mean)
    hk_lower_grid = reshape(hk_lower)
    hk_upper_grid = reshape(hk_upper)
    nh_mean_grid = reshape(nh_mean)
    nh_lower_grid = reshape(nh_lower)
    nh_upper_grid = reshape(nh_upper)
    nw_mean_grid = reshape(nw_mean)
    nw_lower_grid = reshape(nw_lower)
    nw_upper_grid = reshape(nw_upper)

    extent = [config["grid_min"], config["grid_max"], config["grid_min"], config["grid_max"]]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Contour levels (use HK/NR grids for scale)
    levels_hk = np.linspace(hk_mean_grid.min(), hk_mean_grid.max(), 8)
    levels_nh = np.linspace(nh_mean_grid.min(), nh_mean_grid.max(), 8)
    levels_nw = np.linspace(nw_mean_grid.min(), nw_mean_grid.max(), 8)

    # Left: lower credible bands (HK vs Newton-H vs Newton-W)
    ax_left = axes[0]
    im_left = ax_left.imshow(
        true_grid,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap="gray_r",
    )
    # HK lower band (teal)
    ax_left.contour(
        X,
        Y,
        hk_lower_grid,
        levels=levels_hk,
        colors="teal",
        linewidths=1.2,
        linestyles="solid",
    )
    # Newton-H lower band (blue)
    ax_left.contour(
        X,
        Y,
        nh_lower_grid,
        levels=levels_nh,
        colors="royalblue",
        linewidths=1.2,
        linestyles="solid",
    )
    # Newton-W lower band (red)
    ax_left.contour(
        X,
        Y,
        nw_lower_grid,
        levels=levels_nw,
        colors="crimson",
        linewidths=1.2,
        linestyles="solid",
    )
    ax_left.set_title("Lower 95\\% bands")
    ax_left.set_xlabel("x")
    ax_left.set_ylabel("y")

    # Middle: posterior means (HK vs Newton-H vs Newton-W)
    ax_mid = axes[1]
    im_mid = ax_mid.imshow(
        true_grid,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap="gray_r",
    )
    # HK mean (teal)
    ax_mid.contour(
        X,
        Y,
        hk_mean_grid,
        levels=levels_hk,
        colors="teal",
        linewidths=1.4,
        linestyles="solid",
    )
    # Newton-H mean (blue)
    ax_mid.contour(
        X,
        Y,
        nh_mean_grid,
        levels=levels_nh,
        colors="royalblue",
        linewidths=1.4,
        linestyles="solid",
    )
    # Newton-W mean (red)
    ax_mid.contour(
        X,
        Y,
        nw_mean_grid,
        levels=levels_nw,
        colors="crimson",
        linewidths=1.4,
        linestyles="solid",
    )
    ax_mid.set_title("Posterior means")
    ax_mid.set_xlabel("x")
    ax_mid.set_ylabel("y")

    # Right: upper credible bands (HK vs Newton-H vs Newton-W)
    ax_right = axes[2]
    im_right = ax_right.imshow(
        true_grid,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap="gray_r",
    )
    # HK upper band (teal)
    ax_right.contour(
        X,
        Y,
        hk_upper_grid,
        levels=levels_hk,
        colors="teal",
        linewidths=1.2,
        linestyles="solid",
    )
    # Newton-H upper band (blue)
    ax_right.contour(
        X,
        Y,
        nh_upper_grid,
        levels=levels_nh,
        colors="royalblue",
        linewidths=1.2,
        linestyles="solid",
    )
    # Newton-W upper band (red)
    ax_right.contour(
        X,
        Y,
        nw_upper_grid,
        levels=levels_nw,
        colors="crimson",
        linewidths=1.2,
        linestyles="solid",
    )
    ax_right.set_title("Upper 95\\% bands")
    ax_right.set_xlabel("x")
    ax_right.set_ylabel("y")

    plt.tight_layout()
    plt.savefig("bootstrap_hk_coverage.png", dpi=200)
    plt.close(fig)


def main(fast: bool = True, n_steps: int | None = None):
    """Run bootstrap experiment. Use fast=False for full runs (more data, B, MCMC)."""
    config = setup_config(fast=fast)
    if n_steps is not None:
        if n_steps <= 0:
            raise ValueError("--n-steps must be positive")
        config["n_steps"] = int(n_steps)

    print("=" * 80)
    print("Bootstrap HK Flow Experiment (Bivariate Mixture)")
    if fast:
        print("(Fast mode: n_data=%d, B=%d, prior_mc_samples=%d, sinkhorn_num_iters=%d, MCMC off)"
              % (config["n_data"], config["n_bootstrap"], config["prior_mc_samples"],
                 config.get("sinkhorn_num_iters", 30)))
    if config.get("n_steps") is not None:
        print(f"(Flow run override: n_steps={config['n_steps']})")
    print("=" * 80)

    key = jr.PRNGKey(config["seed"])

    # Generate data
    key, data_key = jr.split(key)
    data, _ = generate_bivariate_data(data_key, config)
    print(f"Generated {config['n_data']} bivariate observations.")

    # Prior and kernel
    prior, kernel = make_prior_and_kernel(config)
    key, pp_key = jr.split(key)
    prior_particles = prior.to_particle_measure(pp_key, config["n_particles"])

    # Bootstrap flows
    B = config["n_bootstrap"]
    hk_measures: List[ParticleMeasure] = []
    nh_measures: List[ParticleMeasure] = []
    nw_measures: List[ParticleMeasure] = []

    print(f"\nRunning {B} bootstrap replicates for HK, Newton-H, and Newton-W flows...")
    t_hk_start = time.perf_counter()
    for b in range(B):
        key, key_hk, key_nh, key_nw = jr.split(key, 4)

        # HK replicate (weights + atoms)
        m_hk = run_single_hk_replicate(
            key_hk,
            data,
            prior,
            kernel,
            prior_particles,
            config,
        )
        hk_measures.append(m_hk)

        # Newton-H replicate (weights only)
        m_nh = run_single_newton_h_replicate(
            key_nh,
            data,
            prior,
            kernel,
            config,
        )
        nh_measures.append(m_nh)

        # Newton-W replicate (atoms only)
        m_nw = run_single_newton_w_replicate(
            key_nw,
            data,
            prior,
            kernel,
            prior_particles,
            config,
        )
        nw_measures.append(m_nw)

        if (b + 1) % max(1, B // 4) == 0:
            print(f"  Completed {b+1}/{B} replicates for all flows")
    t_hk_end = time.perf_counter()

    # Density grid
    grid_points = build_density_grid(config)
    hk_densities = hk_bootstrap_densities(hk_measures, kernel, grid_points)
    nh_densities = hk_bootstrap_densities(nh_measures, kernel, grid_points)
    nw_densities = hk_bootstrap_densities(nw_measures, kernel, grid_points)

    hk_mean, hk_lower, hk_upper = credible_intervals(hk_densities, alpha=0.05)
    nh_mean, nh_lower, nh_upper = credible_intervals(nh_densities, alpha=0.05)
    nw_mean, nw_lower, nw_upper = credible_intervals(nw_densities, alpha=0.05)

    # True density on grid
    true_density_vals = true_mixture_density(
        grid_points,
        config["true_means"],
        config["true_stds"],
        config["true_weights"],
    )

    hk_coverage = compute_coverage(true_density_vals, hk_lower, hk_upper)
    nh_coverage = compute_coverage(true_density_vals, nh_lower, nh_upper)
    nw_coverage = compute_coverage(true_density_vals, nw_lower, nw_upper)
    print(f"\nHK bootstrap 95% coverage (grid-based): {hk_coverage:.3f}")
    print(f"Newton-H bootstrap 95% coverage (grid-based): {nh_coverage:.3f}")
    print(f"Newton-W bootstrap 95% coverage (grid-based): {nw_coverage:.3f}")
    print(f"Bootstrap elapsed time (all flows): {t_hk_end - t_hk_start:.2f} s")

    # Plot results
    plot_bootstrap_results(
        config,
        grid_points,
        hk_mean,
        hk_lower,
        hk_upper,
        nh_mean,
        nh_lower,
        nh_upper,
        nw_mean,
        nw_lower,
        nw_upper,
        true_density_vals,
        data,
    )
    print("\nSaved figure 'bootstrap_hk_coverage.png'.")


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
        "--n-steps",
        type=int,
        default=None,
        help="Override flow run length; if > n_data, uses bootstrap continuation.",
    )
    args = parser.parse_args()

    # Default remains fast mode unless --full is passed.
    main(fast=not args.full, n_steps=args.n_steps)

