#!/usr/bin/env python3
"""
Mode Recovery and Metastability Experiment.

This script compares the Hellinger–Kantorovich (HK) splitting scheme,
Newton–Hellinger, and Newton flows on a dumbbell-like target: a weakly
connected Gaussian mixture with two main lobes linked by a low-density
bridge. Elapsed times are reported and density plots show how each flow
approximates the target.
"""

from __future__ import annotations

import argparse
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
    DirichletProcessPrior,
    HellingerKantorovichFlow,
    NewtonFlow,
)
from recursive_mixtures.flows import NewtonHellingerFlow
from recursive_mixtures.utils import generate_mixture_data, true_mixture_density


# -----------------------------------------------------------------------------
# Banana-shaped mixture: log-density, score, sampling
# Using base banana from https://tiao.io/post/building-probability-distributions-with-tensorflow-probability-bijector-api/
# -----------------------------------------------------------------------------

_BANANA_RHO = 0.95
_BANANA_DET = 1.0 - _BANANA_RHO**2
_BANANA_INV = (1.0 / _BANANA_DET) * jnp.array(
    [[1.0, -_BANANA_RHO], [-_BANANA_RHO, 1.0]]
)
_BANANA_LOG_NORM = -0.5 * (
    2.0 * jnp.log(2.0 * jnp.pi) + jnp.log(_BANANA_DET)
)


def _banana_base_log_density_x(x: jax.Array) -> jax.Array:
    """Log density of base correlated Gaussian N(0, Σ) with ρ=0.95, for a single 2D point x."""
    x = jnp.atleast_1d(x)
    dx = x  # mean zero
    quad = dx @ _BANANA_INV @ dx
    return _BANANA_LOG_NORM - 0.5 * quad


def _banana_component_log_density_single(y: jax.Array, center: jax.Array) -> jax.Array:
    """
    Log density of one banana component at a single 2D point y.

    Component is constructed as:
        x ~ N(0, Σ)
        z = G(x) with G(x) = [x1, x2 - x1^2 - 1]
        y = z + center
    Since G is volume-preserving, p_Y(y) = p_X(G^{-1}(y - center)).
    """
    # Shift by component center
    z = y - center
    # Inverse transform G^{-1}(z)
    x1 = z[0]
    x2 = z[1] + x1**2 + 1.0
    x = jnp.stack([x1, x2])
    return _banana_base_log_density_x(x)


def banana_component_log_density(
    y: jax.Array,
    c1: jax.Array,
    c2: jax.Array,
    b: jax.Array,
    s1: jax.Array,
    s2: jax.Array,
) -> jax.Array:
    """
    Log density of one banana component. y shape (2,) or (N, 2); returns scalar or (N,).

    The extra parameters (b, s1, s2) are unused here but kept for interface compatibility.
    """
    center = jnp.array([c1, c2])
    y = jnp.atleast_2d(y)
    return jax.vmap(lambda yi: _banana_component_log_density_single(yi, center))(y).squeeze()


def banana_mixture_log_density(
    y: jax.Array,
    centers: jax.Array,
    curvatures: jax.Array,
    scales: jax.Array,
    weights: jax.Array,
) -> jax.Array:
    """
    Log of mixture density: log(sum_k w_k p_k(y)).
    centers (K, 2), weights (K). Other arguments are ignored for this banana construction.
    """
    y = jnp.atleast_2d(y)
    K = centers.shape[0]
    log_components = jnp.stack(
        [
            jax.vmap(
                lambda yi: _banana_component_log_density_single(yi, centers[k])
            )(y)
            for k in range(K)
        ],
        axis=0,
    )  # (K, N)
    log_weighted = jnp.log(weights + 1e-30)[:, None] + log_components
    return jax.scipy.special.logsumexp(log_weighted, axis=0).squeeze()


def banana_mixture_sample(
    key: jax.Array,
    n: int,
    centers: jax.Array,
    curvatures: jax.Array,
    scales: jax.Array,
    weights: jax.Array,
) -> jax.Array:
    """
    Sample n points from the banana mixture.

    For each component:
        x ~ N(0, Σ) with ρ=0.95,
        z = G(x) = [x1, x2 - x1^2 - 1],
        y = z + center_k.
    """
    K = centers.shape[0]
    key_comp, key_base = jr.split(key, 2)
    # Component assignments
    comp = jr.choice(key_comp, K, shape=(n,), p=weights)
    # Base Gaussian samples
    Sigma = jnp.array([[1.0, _BANANA_RHO], [_BANANA_RHO, 1.0]])
    L = jnp.linalg.cholesky(Sigma)
    base = jr.normal(key_base, shape=(n, 2))
    x = base @ L.T  # (n, 2)
    x1 = x[:, 0]
    x2 = x[:, 1]
    z1 = x1
    z2 = x2 - x1**2 - 1.0
    z = jnp.stack([z1, z2], axis=1)
    # Shift by component centers
    c = centers[comp]
    y = z + c
    return y


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
    """Configuration dictionary for the metastability experiment (dumbbell mixture)."""
    config = {
        # Dumbbell-like Gaussian mixture: two main lobes plus a weak bridge (closer, larger variance)
        "dumbbell_means": jnp.array(
            [
                [-1.8, 0.0],  # left lobe
                [1.8, 0.0],   # right lobe
                [0.0, 0.0],   # bridge component
            ]
        ),
        "dumbbell_stds": jnp.array(
            [
                [1.2, 1.2],
                [1.2, 1.2],
                [2.2, 0.6],  # elongated along x, narrow in y
            ]
        ),
        "dumbbell_weights": jnp.array([0.45, 0.45, 0.10]),
        # Data
        "n_data": 1000,
        # Particles (match fast bootstrap settings)
        "n_particles": 50,
        # HK flow parameters (runtime: ~ n_steps * (prior_mc_samples + 1) * sinkhorn_num_iters Sinkhorn iters)
        # Smaller step/Wasserstein weight so atoms move more conservatively and stay closer together.
        # Kernel bandwidth kept small so KDE still tracks the data closely.
        "hk_step_size": 0.01,
        "hk_kernel_bandwidth": 0.8,
        "hk_sinkhorn_reg": 0.05,
        "hk_sinkhorn_num_iters": 25,
        "hk_wasserstein_weight": 0.03,
        "hk_prior_flow_weight": 0.1,
        "hk_prior_mc_samples": 1,
        # Prior for HK flow
        "prior_mean": jnp.array([0.0, 0.0]),
        "prior_std": 8.0,
        "dp_concentration": 10.0,
        # Trajectory recording: single long run (shorter, faster run)
        "n_steps": 100,
        "record_every": 20,
        # Density grid for background visualization (finer resolution heatmap)
        "grid_min": -8.0,
        "grid_max": 8.0,
        "grid_size": 80,
        # Random seed
        "seed": 2024,
    }
    return config


def generate_dumbbell_data(key: jax.Array, config: Dict) -> jax.Array:
    """Generate 2D data from the dumbbell Gaussian mixture."""
    samples, _ = generate_mixture_data(
        key,
        config["n_data"],
        config["dumbbell_means"],
        config["dumbbell_stds"],
        config["dumbbell_weights"],
    )
    return samples


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


def make_newton_hellinger_flow_for_metastability(
    prior,
    kernel,
    config: Dict,
) -> NewtonHellingerFlow:
    """Configure a Newton-Hellinger flow (weights only, fixed atoms)."""
    return NewtonHellingerFlow(
        kernel=kernel,
        prior=prior,
        step_size=config["hk_step_size"],
    )


def make_newton_wasserstein_flow_for_metastability(
    prior,
    kernel,
    prior_particles: ParticleMeasure,
    config: Dict,
) -> NewtonFlow:
    """Configure a Newton flow (weights only, fixed atoms)."""
    return NewtonFlow(
        kernel=kernel,
        prior=prior,
        step_size=config["hk_step_size"],
    )


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
    data_len = int(data_stream.shape[0])

    # First pass uses each point in order; beyond n, sample with replacement.
    if n_steps <= data_len:
        step_indices = jnp.arange(n_steps)
    else:
        n_extra = n_steps - data_len
        key, idx_key = jr.split(key)
        extra_idx = jr.randint(idx_key, shape=(n_extra,), minval=0, maxval=data_len)
        step_indices = jnp.concatenate([jnp.arange(data_len), extra_idx], axis=0)
    keys = jr.split(key, n_steps)

    # Simple text progress bar for HK splitting
    bar_width = 30

    for t in range(n_steps):
        x = data_stream[step_indices[t]]
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
    """Precompute dumbbell mixture density on a grid for background plotting."""
    n = config["grid_size"]
    xs = jnp.linspace(config["grid_min"], config["grid_max"], n)
    ys = jnp.linspace(config["grid_min"], config["grid_max"], n)
    X, Y = jnp.meshgrid(xs, ys)
    grid_points = jnp.stack([X.ravel(), Y.ravel()], axis=1)

    dens = true_mixture_density(
        grid_points,
        config["dumbbell_means"],
        config["dumbbell_stds"],
        config["dumbbell_weights"],
    )
    return (
        np.asarray(X),
        np.asarray(Y),
        np.asarray(dens.reshape(n, n)),
    )


def plot_particles(
    config: Dict,
    true_grid: np.ndarray,
    hk_measure: ParticleMeasure,
    nh_measure: ParticleMeasure,
    nw_measure: ParticleMeasure,
    initial_measure: ParticleMeasure,
):
    """
    Three-panel scatter plot: one panel per flow.
    Each panel shows the true density heatmap and final particle positions
    sized by weight.
    """
    extent = [config["grid_min"], config["grid_max"], config["grid_min"], config["grid_max"]]

    flows = [
        ("HK",           hk_measure, "teal"),
        ("Newton-H",     nh_measure, "royalblue"),
        ("Newton flow",  nw_measure, "crimson"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, (label, measure, color) in zip(axes, flows):
        ax.imshow(
            true_grid,
            origin="lower",
            extent=extent,
            aspect="auto",
            cmap="gray_r",
        )

        atoms = np.asarray(measure.atoms)
        weights = np.asarray(measure.weights)
        sizes = weights / weights.max() * 300

        ax.scatter(
            atoms[:, 0], atoms[:, 1],
            s=sizes, c=color,
            alpha=0.85,
            edgecolors="white", linewidths=0.4,
            zorder=3,
        )

        ax.set_title(label)
        ax.set_xlabel("x₁")
        ax.set_ylabel("x₂")
        ax.set_xlim(config["grid_min"], config["grid_max"])
        ax.set_ylim(config["grid_min"], config["grid_max"])

    plt.suptitle("Final particle positions (size ∝ weight)", y=1.02)
    plt.tight_layout()
    plt.savefig("metastability_density_comparison.pdf", bbox_inches="tight")
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
    plt.savefig("mode_occupancy_over_time.pdf", bbox_inches="tight")
    plt.close(fig)


def main(n_steps: int | None = None):
    config = setup_config()
    if n_steps is not None:
        if n_steps <= 0:
            raise ValueError("--n-steps must be positive")
        config["n_steps"] = int(n_steps)

    print("=" * 80)
    print("Metastability Experiment: Banana Mixture — HK vs Newton-H vs Newton-W")
    print("=" * 80)

    key = jr.PRNGKey(config["seed"])

    # Generate dumbbell mixture data
    key, data_key = jr.split(key)
    data = generate_dumbbell_data(data_key, config)
    print(f"Generated {config['n_data']} observations from dumbbell mixture.")

    # Background density for visualization (true density grid)
    Xg, Yg, true_grid = build_density_background(config)

    # Initial particles from prior
    base_prior = GaussianPrior(
        mean=config["prior_mean"],
        std=config["prior_std"],
        dim=2,
    )
    prior = DirichletProcessPrior(
        base_prior=base_prior,
        concentration=config["dp_concentration"],
    )
    key, init_key, hk_prior_key = jr.split(key, 3)
    initial_atoms = prior.sample(init_key, config["n_particles"])

    # HK splitting (single long run, with timing)
    print("\nRunning HK splitting scheme...")
    kernel = GaussianKernel(bandwidth=config["hk_kernel_bandwidth"])
    prior_particles = prior.to_particle_measure(hk_prior_key, config["n_particles"])
    initial_measure = ParticleMeasure.initialize(initial_atoms)
    key, hk_key, nh_key, nw_key = jr.split(key, 4)

    t0_hk = time.perf_counter()
    final_hk_measure, hk_snaps = run_hk_splitting(
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

    # Newton-Hellinger (weights only, fixed atoms)
    print("Running Newton-Hellinger flow (weights only)...")
    nh_flow = make_newton_hellinger_flow_for_metastability(prior, kernel, config)
    nh_measure = initial_measure
    n_steps = config["n_steps"]
    data_len = int(data.shape[0])
    # Same data-index policy: first pass in order, then bootstrap with replacement.
    if n_steps <= data_len:
        step_indices = jnp.arange(n_steps)
    else:
        n_extra = n_steps - data_len
        nh_key, nh_idx_key = jr.split(nh_key)
        nh_extra_idx = jr.randint(nh_idx_key, shape=(n_extra,), minval=0, maxval=data_len)
        step_indices = jnp.concatenate([jnp.arange(data_len), nh_extra_idx], axis=0)
    nh_step_keys = jr.split(nh_key, n_steps)
    t0_nh = time.perf_counter()
    for t in range(n_steps):
        x = data[step_indices[t]]
        nh_measure = nh_flow.step(nh_measure, x, key=nh_step_keys[t])
    time_nh = time.perf_counter() - t0_nh
    print(f"  Newton-Hellinger elapsed: {time_nh:.2f} s")

    # Newton flow (recursive weights-only update with fixed atoms)
    print("Running Newton flow (weights only)...")
    nw_flow = make_newton_wasserstein_flow_for_metastability(
        prior,
        kernel,
        prior_particles,
        config,
    )
    nw_measure = initial_measure
    if n_steps <= data_len:
        step_indices_nw = jnp.arange(n_steps)
    else:
        n_extra = n_steps - data_len
        nw_key, nw_idx_key = jr.split(nw_key)
        nw_extra_idx = jr.randint(nw_idx_key, shape=(n_extra,), minval=0, maxval=data_len)
        step_indices_nw = jnp.concatenate([jnp.arange(data_len), nw_extra_idx], axis=0)
    nw_step_keys = jr.split(nw_key, n_steps)
    t0_nw = time.perf_counter()
    for t in range(n_steps):
        x = data[step_indices_nw[t]]
        nw_measure = nw_flow.step(nw_measure, x, key=nw_step_keys[t])
    time_nw = time.perf_counter() - t0_nw
    print(f"  Newton flow elapsed: {time_nw:.2f} s")

    # Atom movement diagnostics: confirms whether particles physically moved.
    init_atoms_np = np.asarray(initial_measure.atoms)
    hk_atoms_np = np.asarray(final_hk_measure.atoms)
    nh_atoms_np = np.asarray(nh_measure.atoms)
    nw_atoms_np = np.asarray(nw_measure.atoms)
    hk_disp = np.linalg.norm(hk_atoms_np - init_atoms_np, axis=1)
    nh_disp = np.linalg.norm(nh_atoms_np - init_atoms_np, axis=1)
    nw_disp = np.linalg.norm(nw_atoms_np - init_atoms_np, axis=1)
    print(
        "Atom displacement (mean / max): "
        f"HK={hk_disp.mean():.4f}/{hk_disp.max():.4f}, "
        f"Newton-H={nh_disp.mean():.4f}/{nh_disp.max():.4f}, "
        f"NewtonFlow={nw_disp.mean():.4f}/{nw_disp.max():.4f}"
    )

    # Mode occupancy (dumbbell means) for HK
    dumbbell_means = config["dumbbell_means"]
    print("Computing mode occupancy over time (HK)...")
    occ_hk = mode_occupancy(hk_snaps, dumbbell_means)

    # Plots: particle scatter and HK mode occupancy
    print("Creating particle scatter and occupancy plots...")

    # Final HK weights summary (for debugging)
    log_w = np.asarray(final_hk_measure.log_weights)
    w = np.exp(log_w - log_w.max()); w /= w.sum()
    print(
        "Final HK weights: "
        f"min={w.min():.3e}, max={w.max():.3e}, "
        f"mean={w.mean():.3e}"
    )

    plot_particles(
        config,
        true_grid,
        final_hk_measure,
        nh_measure,
        nw_measure,
        initial_measure,
    )
    plot_mode_occupancy(config, occ_hk)

    # Timing summary
    print("\n--- Elapsed times ---")
    print(f"  HK splitting:     {time_hk:.2f} s")
    print(f"  Newton-Hellinger: {time_nh:.2f} s")
    print(f"  Newton flow:      {time_nw:.2f} s")
    print("\nSaved figures 'metastability_density_comparison.png' and 'mode_occupancy_over_time.png'.")


if __name__ == "__main__":
    # Enable 64-bit precision for numerical stability if available
    try:
        jax.config.update("jax_enable_x64", True)
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="Metastability flow comparison experiment")
    parser.add_argument(
        "--n-steps",
        type=int,
        default=None,
        help="Override number of flow steps (default uses config value).",
    )
    args = parser.parse_args()

    main(n_steps=args.n_steps)

