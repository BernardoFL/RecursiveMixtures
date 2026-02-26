#!/usr/bin/env python3
"""
Mode Recovery and Metastability Experiment.

This script compares standard Langevin dynamics (pure Wasserstein-type
updates on atom locations) against the Hellinger–Kantorovich (HK)
splitting scheme on a \"galaxy-like\" multimodal dataset. It visualizes
particle trajectories and mode occupancies over time to highlight the
ability of HK flows to \"teleport\" mass between modes.
"""

from __future__ import annotations

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
from recursive_mixtures.utils import generate_mixture_data, true_mixture_density


def setup_config() -> Dict:
    """Configuration dictionary for the metastability experiment."""
    config = {
        # Galaxy-like mixture (four well-separated modes)
        "true_means": jnp.array(
            [
                [-6.0, 0.0],
                [0.0, 6.0],
                [6.0, 0.0],
                [0.0, -6.0],
            ]
        ),
        "true_stds": jnp.array([0.6, 0.6, 0.6, 0.6]),
        "true_weights": jnp.array([0.25, 0.25, 0.25, 0.25]),
        # Data
        "n_data": 2000,
        # Particles
        "n_particles": 200,
        # Langevin parameters
        "langevin_step_size": 0.02,
        "langevin_noise_scale": 0.2,
        # HK flow parameters
        "hk_step_size": 0.05,
        "hk_kernel_bandwidth": 1.0,
        "hk_sinkhorn_reg": 0.05,
        "hk_wasserstein_weight": 0.2,
        "hk_prior_flow_weight": 0.1,
        "hk_prior_mc_samples": 5,
        # Prior for HK flow
        "prior_mean": jnp.array([0.0, 0.0]),
        "prior_std": 8.0,
        # Trajectory recording
        "n_steps": 400,
        "record_every": 10,
        # Density grid for background visualization
        "grid_min": -10.0,
        "grid_max": 10.0,
        "grid_size": 100,
        # Random seed
        "seed": 2024,
    }
    return config


def generate_galaxy_data(
    key: jax.Array,
    config: Dict,
) -> jax.Array:
    """Generate a galaxy-like 2D mixture dataset."""
    samples, _ = generate_mixture_data(
        key,
        config["n_data"],
        config["true_means"],
        config["true_stds"],
        config["true_weights"],
    )
    return samples


def log_target_density(
    x: jax.Array,
    config: Dict,
) -> jax.Array:
    """Log of the true galaxy mixture density at a point x (2D)."""
    # true_mixture_density accepts (M, D) input
    dens = true_mixture_density(
        x[None, :],
        config["true_means"],
        config["true_stds"],
        config["true_weights"],
    )[0]
    return jnp.log(dens + 1e-30)


def target_score(x: jax.Array, config: Dict) -> jax.Array:
    """Score ∇_x log p(x) of the true mixture at x."""
    return jax.grad(lambda x_: log_target_density(x_, config))(x)


def langevin_step(
    key: jax.Array,
    atoms: jax.Array,
    config: Dict,
) -> jax.Array:
    """
    Single overdamped Langevin step for all particles.

    θ^{k+1} = θ^k + ε ∇ log p(θ^k) + √(2ε) ξ
    """
    step_size = config["langevin_step_size"]
    noise_scale = config["langevin_noise_scale"]

    # Compute scores at all atoms
    score_fn = lambda theta: target_score(theta, config)
    scores = jax.vmap(score_fn)(atoms)

    noise = jr.normal(key, shape=atoms.shape)
    atoms_next = atoms + step_size * scores + jnp.sqrt(2.0 * step_size) * noise_scale * noise
    return atoms_next


def run_langevin_dynamics(
    key: jax.Array,
    initial_atoms: jax.Array,
    config: Dict,
) -> Tuple[jax.Array, List[jax.Array]]:
    """
    Run Langevin dynamics, recording particle positions at regular intervals.

    Returns:
        final_atoms: final particle locations, shape (N, 2)
        traj_snapshots: list of arrays of shape (N, 2) at recorded times
    """
    atoms = initial_atoms
    traj_snapshots: List[jax.Array] = [atoms]

    n_steps = config["n_steps"]
    record_every = config["record_every"]

    keys = jr.split(key, n_steps)
    for t in range(n_steps):
        atoms = langevin_step(keys[t], atoms, config)
        if (t + 1) % record_every == 0:
            traj_snapshots.append(atoms)
    return atoms, traj_snapshots


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

    for t in range(n_steps):
        x = data_stream[t % data_len]
        measure = flow.step(measure, x, keys[t])
        if (t + 1) % record_every == 0:
            traj_snapshots.append(measure.atoms)

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
    """Precompute true density on a grid for background plotting."""
    n = config["grid_size"]
    xs = jnp.linspace(config["grid_min"], config["grid_max"], n)
    ys = jnp.linspace(config["grid_min"], config["grid_max"], n)
    X, Y = jnp.meshgrid(xs, ys)
    grid_points = jnp.stack([X.ravel(), Y.ravel()], axis=1)

    dens = true_mixture_density(
        grid_points,
        config["true_means"],
        config["true_stds"],
        config["true_weights"],
    )
    return (
        np.asarray(X),
        np.asarray(Y),
        np.asarray(dens.reshape(n, n)),
    )


def plot_trajectories(
    config: Dict,
    background: Tuple[np.ndarray, np.ndarray, np.ndarray],
    langevin_snaps: List[jax.Array],
    hk_snaps: List[jax.Array],
):
    """Plot particle trajectories for Langevin and HK flows."""
    Xg, Yg, Zg = background

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    titles = ["Langevin Dynamics (pure Wasserstein)", "HK Splitting Scheme"]
    snapshots_list = [langevin_snaps, hk_snaps]

    for ax, snaps, title in zip(axes, snapshots_list, titles):
        ax.contourf(Xg, Yg, Zg, levels=30, cmap="Blues", alpha=0.8)

        # Plot trajectories for a subset of particles
        n_particles = snaps[0].shape[0]
        n_plot = min(40, n_particles)
        idx = np.linspace(0, n_particles - 1, n_plot, dtype=int)

        for i in idx:
            traj = np.stack([np.asarray(s)[i] for s in snaps], axis=0)
            ax.plot(traj[:, 0], traj[:, 1], "-o", markersize=2, linewidth=1, alpha=0.7)

        ax.set_title(title)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_aspect("equal", "box")

    plt.tight_layout()
    plt.savefig("metastability_trajectories.png", dpi=200)
    plt.close(fig)


def plot_mode_occupancy(
    config: Dict,
    occ_langevin: np.ndarray,
    occ_hk: np.ndarray,
):
    """Plot mode occupancy over snapshots for Langevin vs HK."""
    T_snap, K = occ_langevin.shape
    times = np.arange(T_snap) * config["record_every"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for k in range(K):
        axes[0].plot(times, occ_langevin[:, k], label=f"Mode {k+1}")
    axes[0].set_title("Mode occupancy - Langevin")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Fraction of particles")
    axes[0].legend(loc="best")

    for k in range(K):
        axes[1].plot(times, occ_hk[:, k], label=f"Mode {k+1}")
    axes[1].set_title("Mode occupancy - HK")
    axes[1].set_xlabel("Step")

    plt.tight_layout()
    plt.savefig("mode_occupancy_over_time.png", dpi=200)
    plt.close(fig)


def main():
    config = setup_config()

    print("=" * 80)
    print("Metastability Experiment: Langevin vs HK Splitting")
    print("=" * 80)

    key = jr.PRNGKey(config["seed"])

    # Generate galaxy-like data
    key, data_key = jr.split(key)
    data = generate_galaxy_data(data_key, config)
    print(f"Generated {config['n_data']} galaxy-like observations.")

    # Background density for visualization
    background = build_density_background(config)

    # Common initial particles: sample from a broad prior
    prior = GaussianPrior(
        mean=config["prior_mean"],
        std=config["prior_std"],
        dim=2,
    )
    key, init_key, hk_prior_key = jr.split(key, 3)
    initial_atoms = prior.sample(init_key, config["n_particles"])

    # Langevin dynamics
    print("\nRunning Langevin dynamics...")
    key, langevin_key = jr.split(key)
    _, langevin_snaps = run_langevin_dynamics(
        langevin_key,
        initial_atoms,
        config,
    )

    # HK splitting
    print("Running HK splitting scheme...")
    kernel = GaussianKernel(bandwidth=config["hk_kernel_bandwidth"])
    prior_particles = prior.to_particle_measure(hk_prior_key, config["n_particles"])
    initial_measure = ParticleMeasure.initialize(initial_atoms)

    key, hk_key = jr.split(key)
    final_measure, hk_snaps = run_hk_splitting(
        hk_key,
        initial_measure,
        data,
        prior,
        kernel,
        prior_particles,
        config,
    )

    # Mode occupancy
    print("Computing mode occupancy over time...")
    occ_langevin = mode_occupancy(langevin_snaps, config["true_means"])
    occ_hk = mode_occupancy(hk_snaps, config["true_means"])

    # Plots
    print("Creating trajectory and occupancy plots...")
    plot_trajectories(config, background, langevin_snaps, hk_snaps)
    plot_mode_occupancy(config, occ_langevin, occ_hk)

    print("\nSaved figures 'metastability_trajectories.png' and 'mode_occupancy_over_time.png'.")


if __name__ == "__main__":
    # Enable 64-bit precision for numerical stability if available
    try:
        jax.config.update("jax_enable_x64", True)
    except Exception:
        pass

    main()

