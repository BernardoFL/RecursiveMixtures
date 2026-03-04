#!/usr/bin/env python3
"""
Mode Recovery and Metastability Experiment.

This script compares the Hellinger–Kantorovich (HK) splitting scheme
against NumPyro NUTS on a mixture of banana-shaped densities. Single
long run per method (no bootstrap). Elapsed times are reported for
both methods; trajectory and mode-occupancy plots show HK particle
evolution and NUTS samples.
"""

from __future__ import annotations

import sys
import time
from typing import Dict, List, Tuple

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

try:
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS
    HAS_NUMPYRO = True
except ImportError:
    HAS_NUMPYRO = False


# -----------------------------------------------------------------------------
# Banana-shaped mixture: log-density, score, sampling
# -----------------------------------------------------------------------------

def _banana_component_log_density_single(
    x: jax.Array,
    c1: jax.Array,
    c2: jax.Array,
    b: jax.Array,
    s1: jax.Array,
    s2: jax.Array,
) -> jax.Array:
    """Log density of one banana component at a single 2D point x. Scalar in, scalar out."""
    x1, x2 = x[0], x[1]
    z1 = (x1 - c1) / (s1 + 1e-30)
    mean2 = c2 + b * (x1 - c1) ** 2
    z2 = (x2 - mean2) / (s2 + 1e-30)
    log_p = -0.5 * (z1 ** 2 + z2 ** 2) - jnp.log(2 * jnp.pi * s1 * s2 + 1e-30)
    return log_p


def banana_component_log_density(
    x: jax.Array,
    c1: jax.Array,
    c2: jax.Array,
    b: jax.Array,
    s1: jax.Array,
    s2: jax.Array,
) -> jax.Array:
    """Log density of one banana component. x shape (2,) or (N, 2); returns scalar or (N,)."""
    x = jnp.atleast_2d(x)
    return jax.vmap(
        lambda xi: _banana_component_log_density_single(xi, c1, c2, b, s1, s2)
    )(x).squeeze()


def banana_mixture_log_density(
    x: jax.Array,
    centers: jax.Array,
    curvatures: jax.Array,
    scales: jax.Array,
    weights: jax.Array,
) -> jax.Array:
    """
    Log of mixture density: log(sum_k w_k p_k(x)).
    centers (K, 2), curvatures (K,), scales (K, 2) or (K,), weights (K).
    """
    x = jnp.atleast_2d(x)
    K = centers.shape[0]
    if scales.ndim == 1:
        scales = jnp.tile(scales[:, None], (1, 2))
    log_component = jnp.stack([
        banana_component_log_density(
            x,
            centers[k, 0],
            centers[k, 1],
            curvatures[k],
            scales[k, 0],
            scales[k, 1],
        )
        for k in range(K)
    ], axis=0)
    # log_component is (K, N), weights (K,) -> broadcast via (K, 1)
    log_weighted = jnp.log(weights + 1e-30)[:, None] + log_component
    return jax.scipy.special.logsumexp(log_weighted, axis=0).squeeze()


def banana_mixture_sample(
    key: jax.Array,
    n: int,
    centers: jax.Array,
    curvatures: jax.Array,
    scales: jax.Array,
    weights: jax.Array,
) -> jax.Array:
    """Sample n points from the banana mixture."""
    K = centers.shape[0]
    if scales.ndim == 1:
        scales = jnp.tile(scales[:, None], (1, 2))
    key_comp, key_x1, key_x2 = jr.split(key, 3)
    comp = jr.choice(key_comp, K, shape=(n,), p=weights)
    u1 = jr.normal(key_x1, shape=(n,))
    u2 = jr.normal(key_x2, shape=(n,))
    c1 = centers[comp, 0]
    c2 = centers[comp, 1]
    b = curvatures[comp]
    s1 = scales[comp, 0]
    s2 = scales[comp, 1]
    x1 = c1 + s1 * u1
    mean2 = c2 + b * (x1 - c1) ** 2
    x2 = mean2 + s2 * u2
    return jnp.stack([x1, x2], axis=1)


def banana_mixture_score(
    x: jax.Array,
    centers: jax.Array,
    curvatures: jax.Array,
    scales: jax.Array,
    weights: jax.Array,
) -> jax.Array:
    """Gradient of log mixture density; x shape (2,) or (N, 2)."""
    def log_dens(xi):
        return banana_mixture_log_density(xi, centers, curvatures, scales, weights)
    grad_fn = jax.grad(log_dens)
    x = jnp.atleast_2d(x)
    return jax.vmap(grad_fn)(x).squeeze()


def setup_config() -> Dict:
    """Configuration dictionary for the metastability experiment (banana mixture)."""
    config = {
        # Banana mixture: K components with (center, curvature, scale)
        "banana_centers": jnp.array(
            [
                [-3.0, 0.0],
                [3.0, 0.0],
                [0.0, -3.0],
                [0.0, 3.0],
            ]
        ),
        "banana_curvatures": jnp.array([0.08, -0.08, 0.08, -0.08]),
        "banana_scales": jnp.array([[1.2, 0.8], [1.2, 0.8], [1.2, 0.8], [1.2, 0.8]]),
        "banana_weights": jnp.array([0.25, 0.25, 0.25, 0.25]),
        # Data
        "n_data": 1000,
        # Particles (match fast bootstrap settings)
        "n_particles": 50,
        # HK flow parameters (runtime: ~ n_steps * (prior_mc_samples + 1) * sinkhorn_num_iters Sinkhorn iters)
        "hk_step_size": 0.05,
        "hk_kernel_bandwidth": 1.0,
        "hk_sinkhorn_reg": 0.05,
        "hk_sinkhorn_num_iters": 25,
        "hk_wasserstein_weight": 0.1,
        "hk_prior_flow_weight": 0.1,
        "hk_prior_mc_samples": 1,
        # Prior for HK flow
        "prior_mean": jnp.array([0.0, 0.0]),
        "prior_std": 8.0,
        # Trajectory recording: single long run (reduce n_steps for faster run)
        "n_steps": 1000,
        "record_every": 20,
        # Density grid for background visualization (match fast bootstrap size)
        "grid_min": -8.0,
        "grid_max": 8.0,
        "grid_size": 35,
        # NumPyro NUTS
        "use_numpyro": True,
        "numpyro_num_warmup": 200,
        "numpyro_num_samples": 1000,
        "numpyro_num_chains": 1,
        "numpyro_seed": 2024,
        # Random seed
        "seed": 2024,
    }
    return config


def generate_banana_data(key: jax.Array, config: Dict) -> jax.Array:
    """Generate 2D data from the banana mixture."""
    return banana_mixture_sample(
        key,
        config["n_data"],
        config["banana_centers"],
        config["banana_curvatures"],
        config["banana_scales"],
        config["banana_weights"],
    )


def make_hk_flow_for_metastability(
    prior,
    kernel,
    prior_particles: ParticleMeasure,
    config: Dict,
) -> HellingerKantorovichFlow:
    """Configure an HK flow emphasizing teleportation between modes."""
    flow = HellingerKantorovichFlow(
        kernel=kernel,
        prior=prior,
        step_size=config["hk_step_size"],
        wasserstein_weight=config["hk_wasserstein_weight"],
        sinkhorn_reg=config["hk_sinkhorn_reg"],
        sinkhorn_num_iters=config.get("hk_sinkhorn_num_iters", 30),
        use_sinkhorn=True,
        prior_particles=prior_particles,
        prior_flow_weight=config["hk_prior_flow_weight"],
        prior_mc_samples=config["hk_prior_mc_samples"],
    )
    return flow


def run_hk_splitting(
    key: jax.Array,
    initial_measure: ParticleMeasure,
    data_stream: jax.Array,
    prior,
    kernel,
    prior_particles: ParticleMeasure,
    config: Dict,
) -> Tuple[ParticleMeasure, List[jax.Array]]:
    """
    Run the HK splitting flow on the galaxy data, recording trajectories.

    We use the existing HK flow implementation with the corrected
    Hellinger step and Sinkhorn regularization toward the prior.
    """
    flow = make_hk_flow_for_metastability(prior, kernel, prior_particles, config)

    measure = initial_measure
    traj_snapshots: List[jax.Array] = [measure.atoms]

    n_steps = config["n_steps"]
    record_every = config["record_every"]

    # For simplicity, cycle through the data stream
    data_len = data_stream.shape[0]
    keys = jr.split(key, n_steps)

    # Simple text progress bar for HK splitting
    bar_width = 30

    for t in range(n_steps):
        x = data_stream[t % data_len]
        measure = flow.step(measure, x, keys[t])
        if (t + 1) % record_every == 0:
            traj_snapshots.append(measure.atoms)

        # Update progress bar
        progress = (t + 1) / n_steps
        filled = int(bar_width * progress)
        bar = "#" * filled + "-" * (bar_width - filled)
        msg = f"\rHK splitting [{bar}] {t + 1}/{n_steps}"
        sys.stdout.write(msg)
        sys.stdout.flush()

    # Finish progress bar line
    sys.stdout.write("\n")
    sys.stdout.flush()

    return measure, traj_snapshots


# -----------------------------------------------------------------------------
# NumPyro: custom banana mixture distribution and NUTS
# -----------------------------------------------------------------------------

if HAS_NUMPYRO:

    class BananaMixtureDistribution(dist.Distribution):
        """NumPyro distribution for the banana mixture (fixed params)."""

        def __init__(
            self,
            centers: jax.Array,
            curvatures: jax.Array,
            scales: jax.Array,
            weights: jax.Array,
        ):
            super().__init__(batch_shape=(), event_shape=(2,))
            self._centers = centers
            self._curvatures = curvatures
            self._scales = scales
            self._weights = weights

        def sample(self, key, sample_shape=()):
            n = int(np.prod(sample_shape)) if sample_shape else 1
            keys = jr.split(key, 2)
            samples = banana_mixture_sample(
                keys[0],
                n,
                self._centers,
                self._curvatures,
                self._scales,
                self._weights,
            )
            if sample_shape:
                return samples.reshape(sample_shape + (2,))
            return samples.reshape(2)

        def log_prob(self, value):
            return banana_mixture_log_density(
                value,
                self._centers,
                self._curvatures,
                self._scales,
                self._weights,
            )


def _numpyro_banana_model(config: Dict):
    """NumPyro model: sample one 2D point from the banana mixture."""
    numpyro.sample(
        "x",
        BananaMixtureDistribution(
            config["banana_centers"],
            config["banana_curvatures"],
            config["banana_scales"],
            config["banana_weights"],
        ),
    )


def run_numpyro_nuts(config: Dict) -> Tuple[jax.Array, float]:
    """
    Run NUTS to sample from the banana mixture. Returns (samples, elapsed_seconds).
    samples shape (num_chains * num_samples, 2).
    """
    if not HAS_NUMPYRO or not config.get("use_numpyro", True):
        return jnp.zeros((0, 2)), 0.0
    kernel = NUTS(_numpyro_banana_model)
    mcmc = MCMC(
        kernel,
        num_warmup=config["numpyro_num_warmup"],
        num_samples=config["numpyro_num_samples"],
        num_chains=config["numpyro_num_chains"],
        progress_bar=True,
    )
    key = jr.PRNGKey(config["numpyro_seed"])
    t0 = time.perf_counter()
    mcmc.run(key, config=config)
    elapsed = time.perf_counter() - t0
    samples = mcmc.get_samples()
    x = samples["x"]
    if x.ndim == 2:
        return x, elapsed
    return x.reshape(-1, 2), elapsed


def assign_modes(points: jax.Array, means: jax.Array) -> jax.Array:
    """
    Assign each point to the nearest mixture component by Euclidean distance.

    Args:
        points: shape (N, 2)
        means: shape (K, 2)

    Returns:
        mode_indices: shape (N,) with values in {0, ..., K-1}
    """
    # points -> (N, 1, 2), means -> (1, K, 2)
    diff = points[:, None, :] - means[None, :, :]
    sq_dist = jnp.sum(diff ** 2, axis=-1)  # (N, K)
    return jnp.argmin(sq_dist, axis=1)


def mode_occupancy(
    snapshots: List[jax.Array],
    means: jax.Array,
) -> np.ndarray:
    """
    Compute mode occupancy fractions over time for a list of snapshots.

    Returns:
        occupancy: array of shape (T_snap, K)
    """
    K = means.shape[0]
    occ_list = []
    for atoms in snapshots:
        modes = assign_modes(atoms, means)
        counts = jnp.bincount(modes, length=K)
        occ = counts / jnp.sum(counts)
        occ_list.append(np.asarray(occ))
    return np.stack(occ_list, axis=0)


def build_density_background(config: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precompute banana mixture density on a grid for background plotting."""
    n = config["grid_size"]
    xs = jnp.linspace(config["grid_min"], config["grid_max"], n)
    ys = jnp.linspace(config["grid_min"], config["grid_max"], n)
    X, Y = jnp.meshgrid(xs, ys)
    grid_points = jnp.stack([X.ravel(), Y.ravel()], axis=1)
    log_dens = banana_mixture_log_density(
        grid_points,
        config["banana_centers"],
        config["banana_curvatures"],
        config["banana_scales"],
        config["banana_weights"],
    )
    dens = jnp.exp(log_dens)
    return (
        np.asarray(X),
        np.asarray(Y),
        np.asarray(dens.reshape(n, n)),
    )


def compute_hk_and_numpyro_densities(
    config: Dict,
    final_measure: ParticleMeasure,
    numpyro_samples: jax.Array | None,
    kernel: GaussianKernel,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Compute HK and NumPyro KDE densities on the same grid as the true background."""
    n = config["grid_size"]
    xs = jnp.linspace(config["grid_min"], config["grid_max"], n)
    ys = jnp.linspace(config["grid_min"], config["grid_max"], n)
    X, Y = jnp.meshgrid(xs, ys)
    grid_points = jnp.stack([X.ravel(), Y.ravel()], axis=1)

    # HK density from final particle measure
    hk_field = final_measure.kernel_density(kernel, grid_points)

    # NumPyro density via KDE with the same kernel (if samples are available)
    np_field = None
    if numpyro_samples is not None and numpyro_samples.size > 0:
        np_measure = ParticleMeasure.initialize(numpyro_samples)
        np_field = np_measure.kernel_density(kernel, grid_points)

    return (
        np.asarray(X),
        np.asarray(Y),
        np.asarray(hk_field),
        np.asarray(np_field) if np_field is not None else None,
    )


def plot_density_contours(
    config: Dict,
    true_grid: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    hk_field: np.ndarray,
    numpyro_field: np.ndarray | None,
):
    """Single contour plot: true density heatmap with HK and NumPyro contours."""
    n = config["grid_size"]
    hk_grid = hk_field.reshape(n, n)
    np_grid = numpyro_field.reshape(n, n) if numpyro_field is not None else None

    extent = [config["grid_min"], config["grid_max"], config["grid_min"], config["grid_max"]]

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    # Background: true density heatmap (grayscale, reversed as in bootstrap)
    ax.imshow(
        true_grid,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap="gray_r",
    )

    # Contour levels (ensure strictly increasing to avoid Matplotlib errors)
    hk_min, hk_max = float(hk_grid.min()), float(hk_grid.max())
    if hk_max > hk_min:
        levels_hk = np.linspace(hk_min, hk_max, 8)
    else:
        # Nearly constant field: create a tiny spread for contours
        levels_hk = np.linspace(hk_min, hk_min + 1e-6, 8)

    levels_np = None
    if np_grid is not None:
        np_min, np_max = float(np_grid.min()), float(np_grid.max())
        if np_max > np_min:
            levels_np = np.linspace(np_min, np_max, 8)
        else:
            levels_np = np.linspace(np_min, np_min + 1e-6, 8)

    burnt_orange = "#CC5500"

    # HK contours (teal)
    ax.contour(
        X,
        Y,
        hk_grid,
        levels=levels_hk,
        colors="teal",
        linewidths=1.4,
        linestyles="solid",
        label="HK",
    )

    # NumPyro contours (burnt orange)
    if np_grid is not None and levels_np is not None:
        ax.contour(
            X,
            Y,
            np_grid,
            levels=levels_np,
            colors=burnt_orange,
            linewidths=1.4,
            linestyles="solid",
        )

    ax.set_title("Banana mixture: HK vs NumPyro")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")

    plt.tight_layout()
    plt.savefig("metastability_density_comparison.png", dpi=200)
    plt.close(fig)


def plot_mode_occupancy(config: Dict, occ_hk: np.ndarray):
    """Plot mode occupancy over snapshots for HK flow."""
    T_snap, K = occ_hk.shape
    times = np.arange(T_snap) * config["record_every"]

    fig, ax = plt.subplots(figsize=(7, 5))
    for k in range(K):
        ax.plot(times, occ_hk[:, k], label=f"Mode {k+1}")
    ax.set_title("Mode occupancy - HK")
    ax.set_xlabel("Step")
    ax.set_ylabel("Fraction of particles")
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig("mode_occupancy_over_time.png", dpi=200)
    plt.close(fig)


def main():
    config = setup_config()

    print("=" * 80)
    print("Metastability Experiment: Banana Mixture — HK vs NumPyro NUTS (no bootstrap)")
    print("=" * 80)

    key = jr.PRNGKey(config["seed"])

    # Generate banana mixture data
    key, data_key = jr.split(key)
    data = generate_banana_data(data_key, config)
    print(f"Generated {config['n_data']} observations from banana mixture.")

    # Background density for visualization (true density grid)
    Xg, Yg, true_grid = build_density_background(config)

    # Initial particles from prior
    prior = GaussianPrior(
        mean=config["prior_mean"],
        std=config["prior_std"],
        dim=2,
    )
    key, init_key, hk_prior_key = jr.split(key, 3)
    initial_atoms = prior.sample(init_key, config["n_particles"])

    # HK splitting (single long run, with timing)
    print("\nRunning HK splitting scheme...")
    kernel = GaussianKernel(bandwidth=config["hk_kernel_bandwidth"])
    prior_particles = prior.to_particle_measure(hk_prior_key, config["n_particles"])
    initial_measure = ParticleMeasure.initialize(initial_atoms)
    key, hk_key = jr.split(key)
    t0_hk = time.perf_counter()
    final_measure, hk_snaps = run_hk_splitting(
        hk_key,
        initial_measure,
        data,
        prior,
        kernel,
        prior_particles,
        config,
    )
    time_hk = time.perf_counter() - t0_hk
    print(f"  HK splitting elapsed: {time_hk:.2f} s")

    # NumPyro NUTS (with timing)
    numpyro_samples = jnp.zeros((0, 2))
    time_numpyro = 0.0
    if config.get("use_numpyro", True) and HAS_NUMPYRO:
        print("Running NumPyro NUTS on banana mixture...")
        numpyro_samples, time_numpyro = run_numpyro_nuts(config)
        print(f"  NumPyro NUTS elapsed: {time_numpyro:.2f} s")
    else:
        print("NumPyro disabled or not available; skipping NUTS.")

    # Mode occupancy (banana centers) for HK
    banana_centers = config["banana_centers"]
    print("Computing mode occupancy over time (HK)...")
    occ_hk = mode_occupancy(hk_snaps, banana_centers)

    # Plots: density comparison contours and HK mode occupancy
    print("Creating density comparison and occupancy plots...")
    X_grid, Y_grid, hk_field, np_field = compute_hk_and_numpyro_densities(
        config,
        final_measure,
        numpyro_samples if numpyro_samples.size > 0 else None,
        kernel,
    )
    plot_density_contours(
        config,
        true_grid,
        X_grid,
        Y_grid,
        hk_field,
        np_field,
    )
    plot_mode_occupancy(config, occ_hk)

    # Timing summary
    print("\n--- Elapsed times ---")
    print(f"  HK splitting: {time_hk:.2f} s")
    print(f"  NumPyro NUTS: {time_numpyro:.2f} s")
    print("\nSaved figures 'metastability_density_comparison.png' and 'mode_occupancy_over_time.png'.")


if __name__ == "__main__":
    # Enable 64-bit precision for numerical stability if available
    try:
        jax.config.update("jax_enable_x64", True)
    except Exception:
        pass

    main()

