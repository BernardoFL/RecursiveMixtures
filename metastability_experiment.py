#!/usr/bin/env python3
"""
Cat-paw HK metastability experiment.

Simulates n i.i.d. points from the paw mixture, then runs three
Hellinger–Kantorovich (HK) flows from the same initial particles on the same
dataset: (a) n steps with Fisher–Rao prior regularization on, (b) n steps with
it off, (c) n+k steps with prior on and uniform resampling from the data after
the first n observations. Plots true density + data + final particles.
"""

from __future__ import annotations

import argparse
import time
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np

from recursive_mixtures import (
    GaussianKernel,
    ParticleMeasure,
    GaussianPrior,
    PitmanYorProcessPrior,
    HellingerKantorovichFlow,
)
from recursive_mixtures.utils import generate_mixture_data, true_mixture_density

from paw_distribution import PawDistribution


def setup_config() -> Dict:
    """Configuration for paw HK comparison (n samples, k continuation steps)."""
    paw = PawDistribution()
    paw_params = paw.to_dict()
    return {
        "dumbbell_means": jnp.array(paw_params["means"]),
        "dumbbell_stds": jnp.array(paw_params["stds"]),
        "dumbbell_weights": jnp.array(paw_params["weights"]),
        "n_data": 1000,
        "k": 1000,
        "n_particles": 50,
        "hk_step_size": 0.01,
        "hk_kernel_bandwidth": 0.4,
        "hk_sinkhorn_reg": 0.05,
        "hk_sinkhorn_num_iters": 25,
        "hk_wasserstein_weight": 0.03,
        "hk_prior_flow_weight": 0.1,
        "hk_prior_mc_samples": 1,
        "hk_use_prior_regularization": True,
        "prior_mean": jnp.array([0.0, 0.0]),
        "prior_std": 5.0,
        "py_discount": 0.2,
        "py_strength": 10.0,
        "grid_min": -2.5,
        "grid_max": 2.5,
        "grid_size": 80,
        "seed": 2024,
    }


def generate_dumbbell_data(key: jax.Array, config: Dict) -> jax.Array:
    """Generate 2D data from the cat-paw Gaussian mixture."""
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
    *,
    use_prior_regularization: bool | None = None,
) -> HellingerKantorovichFlow:
    """HK flow; Fisher–Rao prior term controlled by use_prior_regularization."""
    upr = (
        use_prior_regularization
        if use_prior_regularization is not None
        else config.get("hk_use_prior_regularization", True)
    )
    return HellingerKantorovichFlow(
        kernel=kernel,
        prior=prior,
        step_size=config["hk_step_size"],
        wasserstein_weight=config["hk_wasserstein_weight"],
        sinkhorn_reg=config["hk_sinkhorn_reg"],
        sinkhorn_num_iters=config.get("hk_sinkhorn_num_iters", 30),
        use_sinkhorn=True,
        prior_particles=prior_particles,
        prior_flow_weight=config["hk_prior_flow_weight"],
        use_prior_regularization=upr,
        prior_mc_samples=config["hk_prior_mc_samples"],
    )


def run_hk_case(
    key: jax.Array,
    initial_measure: ParticleMeasure,
    data: jax.Array,
    prior,
    kernel,
    prior_particles: ParticleMeasure,
    config: Dict,
    *,
    n_steps: int,
    bootstrap_after_data: bool,
    use_prior_regularization: bool,
) -> ParticleMeasure:
    """Run HK for n_steps using GradientFlow.run (ordered prefix + optional continuation)."""
    flow = make_hk_flow_for_metastability(
        prior,
        kernel,
        prior_particles,
        config,
        use_prior_regularization=use_prior_regularization,
    )
    final_measure, _ = flow.run(
        initial_measure,
        data,
        key=key,
        store_every=max(1, int(n_steps)),
        n_steps=int(n_steps),
        bootstrap_after_data=bootstrap_after_data,
    )
    return final_measure


def build_density_background(config: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """True paw mixture density on a grid for plotting."""
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


def plot_paw_hk_panels(
    config: Dict,
    true_grid: np.ndarray,
    data: jax.Array,
    measure_a: ParticleMeasure,
    measure_b: ParticleMeasure,
    measure_c: ParticleMeasure,
    out_path: str = "paw_hk_comparison.pdf",
) -> None:
    """Heatmap of true density + data scatter + particles (marker size ∝ weight)."""
    extent = [
        config["grid_min"],
        config["grid_max"],
        config["grid_min"],
        config["grid_max"],
    ]
    data_np = np.asarray(data)
    panels = [
        ("Prior on, stop at n", measure_a, "teal"),
        ("Prior off, stop at n", measure_b, "royalblue"),
        ("Prior on, n+k (resample)", measure_c, "crimson"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (title, measure, color) in zip(axes, panels):
        ax.imshow(
            true_grid,
            origin="lower",
            extent=extent,
            aspect="auto",
            cmap="gray_r",
        )
        ax.scatter(
            data_np[:, 0],
            data_np[:, 1],
            s=14,
            c="white",
            edgecolors="black",
            linewidths=0.35,
            alpha=0.75,
            zorder=2,
        )
        atoms = np.asarray(measure.atoms)
        weights = np.asarray(measure.weights)
        wmax = float(weights.max())
        sizes = weights / max(wmax, 1e-12) * 300
        ax.scatter(
            atoms[:, 0],
            atoms[:, 1],
            s=sizes,
            c=color,
            alpha=0.85,
            edgecolors="white",
            linewidths=0.4,
            zorder=3,
        )
        ax.set_title(title)
        ax.set_xlabel("x₁")
        ax.set_ylabel("x₂")
        ax.set_xlim(config["grid_min"], config["grid_max"])
        ax.set_ylim(config["grid_min"], config["grid_max"])

    plt.suptitle("Cat-paw HK: density, data, particles (size ∝ weight)", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main(n_data: int | None = None, k: int | None = None) -> None:
    config = setup_config()
    if n_data is not None:
        if n_data <= 0:
            raise ValueError("--n-data must be positive")
        config["n_data"] = int(n_data)
    if k is not None:
        if k < 0:
            raise ValueError("--k must be non-negative")
        config["k"] = int(k)

    n = int(config["n_data"])
    k_extra = int(config["k"])

    print("=" * 80)
    print("Cat-paw HK experiment: prior on/off and continuation")
    print(f"  n = {n} data points, k = {k_extra} extra resampled steps (case c total = {n + k_extra})")
    print("=" * 80)

    key = jr.PRNGKey(config["seed"])
    key, data_key = jr.split(key)
    data = generate_dumbbell_data(data_key, config)
    print(f"Simulated {n} observations from paw mixture.")

    _, _, true_grid = build_density_background(config)

    base_prior = GaussianPrior(
        mean=config["prior_mean"],
        std=config["prior_std"],
        dim=2,
    )
    prior = PitmanYorProcessPrior(
        base_prior=base_prior,
        discount=float(config["py_discount"]),
        strength=float(config["py_strength"]),
    )
    key, init_key, hk_prior_key = jr.split(key, 3)
    initial_atoms = prior.sample(init_key, config["n_particles"])
    kernel = GaussianKernel(bandwidth=config["hk_kernel_bandwidth"])
    prior_particles = prior.to_particle_measure(hk_prior_key, config["n_particles"])
    initial_measure = ParticleMeasure.initialize(initial_atoms)

    key, ka, kb, kc = jr.split(key, 4)

    print("\n(a) HK: prior on, n steps, ordered data...")
    t0 = time.perf_counter()
    m_a = run_hk_case(
        ka,
        initial_measure,
        data,
        prior,
        kernel,
        prior_particles,
        config,
        n_steps=n,
        bootstrap_after_data=False,
        use_prior_regularization=True,
    )
    print(f"    elapsed {time.perf_counter() - t0:.2f} s")

    print("(b) HK: prior off, n steps, ordered data...")
    t0 = time.perf_counter()
    m_b = run_hk_case(
        kb,
        initial_measure,
        data,
        prior,
        kernel,
        prior_particles,
        config,
        n_steps=n,
        bootstrap_after_data=False,
        use_prior_regularization=False,
    )
    print(f"    elapsed {time.perf_counter() - t0:.2f} s")

    print(f"(c) HK: prior on, n+k steps (then resample from data), k={k_extra}...")
    t0 = time.perf_counter()
    m_c = run_hk_case(
        kc,
        initial_measure,
        data,
        prior,
        kernel,
        prior_particles,
        config,
        n_steps=n + k_extra,
        bootstrap_after_data=True,
        use_prior_regularization=True,
    )
    print(f"    elapsed {time.perf_counter() - t0:.2f} s")

    init_atoms_np = np.asarray(initial_measure.atoms)
    for label, m in [("(a)", m_a), ("(b)", m_b), ("(c)", m_c)]:
        disp = np.linalg.norm(np.asarray(m.atoms) - init_atoms_np, axis=1)
        print(f"  Atom displacement {label}: mean={disp.mean():.4f}, max={disp.max():.4f}")

    print("\nCreating plot...")
    plot_paw_hk_panels(config, true_grid, data, m_a, m_b, m_c)
    print("Saved 'paw_hk_comparison.pdf'.")


if __name__ == "__main__":
    try:
        jax.config.update("jax_enable_x64", True)
    except Exception:
        pass

    parser = argparse.ArgumentParser(
        description="Cat-paw HK: prior on/off and n+k continuation with resampling"
    )
    parser.add_argument(
        "--n-data",
        type=int,
        default=None,
        help="Dataset size n (default: 1000).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Extra flow steps after ordered n (uniform resample from data); default 1000",
    )
    args = parser.parse_args()

    main(n_data=args.n_data, k=args.k)
