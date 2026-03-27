#!/usr/bin/env python3
"""
Flow comparison experiment on the cat-paw mixture.

Simulates n i.i.d. points from the cat-paw Gaussian mixture, then runs three
recursive update rules from the same initial particles on the same dataset:

    (a) NewtonHellingerFlow   — Fisher-Rao / Hellinger weight-only updates
    (b) NewtonFlow            — recursive Bayesian mixing weight updates
    (c) HellingerKantorovichFlow — HK/WFR: joint weight + atom updates

All three share the same initial particle measure and PY prior. The output is
a single three-panel PDF comparing the final particle configuration under each
flow family.
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
    NewtonHellingerFlow,
    NewtonFlow,
    HellingerKantorovichFlow,
)
from paw_distribution import PawDistribution


def setup_config() -> Dict:
    """Shared configuration for the three-flow comparison."""
    return {
        "n_data": 1000,
        "n_particles": 50,
        # Shared step / kernel params
        "step_size": 0.05,
        "kernel_bandwidth": 0.4,
        # HK-specific params
        "hk_sinkhorn_reg": 0.05,
        "hk_sinkhorn_num_iters": 25,
        "hk_wasserstein_weight": 0.03,
        "hk_prior_flow_weight": 0.1,
        "hk_prior_mc_samples": 1,
        # Newton-Hellinger params
        "nh_log_weight_clip": 5.0,
        "nh_ess_threshold": 0.5,
        "nh_resample_jitter": 0.1,
        # Newton params
        "newton_step_exponent": 0.6,
        # PY prior on atom locations
        "prior_mean": jnp.array([0.0, 0.0]),
        "prior_std": 5.0,
        "py_discount": 0.2,
        "py_strength": 10.0,
        # Plot grid
        "grid_min": -2.5,
        "grid_max": 2.5,
        "grid_size": 80,
        "seed": 2024,
    }


def generate_data(key: jax.Array, config: Dict) -> jax.Array:
    """Sample n i.i.d. points from the paw distribution."""
    paw = PawDistribution()
    return paw.sample(key, int(config["n_data"]))


def build_density_grid(config: Dict) -> np.ndarray:
    """True paw density on a 2-D grid for plotting."""
    n = config["grid_size"]
    xs = jnp.linspace(config["grid_min"], config["grid_max"], n)
    ys = jnp.linspace(config["grid_min"], config["grid_max"], n)
    X, Y = jnp.meshgrid(xs, ys)
    grid_points = jnp.stack([X.ravel(), Y.ravel()], axis=1)
    paw = PawDistribution()
    dens = paw.pdf(grid_points)
    return np.asarray(dens.reshape(n, n))


def make_prior(config: Dict) -> PitmanYorProcessPrior:
    base_prior = GaussianPrior(
        mean=config["prior_mean"],
        std=config["prior_std"],
        dim=2,
    )
    return PitmanYorProcessPrior(
        base_prior=base_prior,
        discount=float(config["py_discount"]),
        strength=float(config["py_strength"]),
    )


def run_newton_hellinger(
    key: jax.Array,
    initial_measure: ParticleMeasure,
    data: jax.Array,
    prior,
    kernel,
    config: Dict,
) -> ParticleMeasure:
    """NewtonHellingerFlow: Fisher-Rao weight-only updates."""
    flow = NewtonHellingerFlow(
        kernel=kernel,
        prior=prior,
        step_size=config["step_size"],
        resample=True,
        log_weight_clip=config["nh_log_weight_clip"],
        ess_threshold=config["nh_ess_threshold"],
        resample_jitter=config["nh_resample_jitter"],
    )
    n = int(config["n_data"])
    final_measure, _ = flow.run(
        initial_measure,
        data,
        key=key,
        store_every=n,
        n_steps=n,
        bootstrap_after_data=False,
    )
    return final_measure


def run_newton(
    key: jax.Array,
    initial_measure: ParticleMeasure,
    data: jax.Array,
    prior,
    kernel,
    config: Dict,
) -> ParticleMeasure:
    """NewtonFlow: recursive Bayesian weight updates with α_n = (n+1)^(-γ)."""
    gamma = float(config["newton_step_exponent"])

    def alpha_fn(n: int) -> float:
        return float((n + 1) ** (-gamma))

    flow = NewtonFlow(
        kernel=kernel,
        prior=prior,
        step_size=config["step_size"],
        alpha_fn=alpha_fn,
        resample=True,
        ess_threshold=config["nh_ess_threshold"],
        resample_jitter=config["nh_resample_jitter"],
    )
    n = int(config["n_data"])
    final_measure, _ = flow.run(
        initial_measure,
        data,
        key=key,
        store_every=n,
        n_steps=n,
        bootstrap_after_data=False,
    )
    return final_measure


def run_hk(
    key: jax.Array,
    initial_measure: ParticleMeasure,
    data: jax.Array,
    prior,
    kernel,
    prior_particles: ParticleMeasure,
    config: Dict,
) -> ParticleMeasure:
    """HellingerKantorovichFlow: joint weight + atom updates (WFR)."""
    flow = HellingerKantorovichFlow(
        kernel=kernel,
        prior=prior,
        step_size=config["step_size"],
        wasserstein_weight=config["hk_wasserstein_weight"],
        sinkhorn_reg=config["hk_sinkhorn_reg"],
        sinkhorn_num_iters=config["hk_sinkhorn_num_iters"],
        use_sinkhorn=True,
        prior_particles=prior_particles,
        prior_flow_weight=config["hk_prior_flow_weight"],
        use_prior_regularization=True,
        prior_mc_samples=config["hk_prior_mc_samples"],
    )
    n = int(config["n_data"])
    final_measure, _ = flow.run(
        initial_measure,
        data,
        key=key,
        store_every=n,
        n_steps=n,
        bootstrap_after_data=False,
    )
    return final_measure


def plot_flow_comparison(
    config: Dict,
    true_grid: np.ndarray,
    data: jax.Array,
    measure_nh: ParticleMeasure,
    measure_newton: ParticleMeasure,
    measure_hk: ParticleMeasure,
    out_path: str = "flow_comparison.pdf",
) -> None:
    """Three-panel figure: true density heatmap + data + final particles per flow."""
    extent = [
        config["grid_min"], config["grid_max"],
        config["grid_min"], config["grid_max"],
    ]
    data_np = np.asarray(data)
    panels = [
        ("Newton-FR flow", measure_nh, "darkorange"),
        ("Newton",           measure_newton, "royalblue"),
        ("WFR flow",         measure_hk, "teal"),
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

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main(n_data: int | None = None) -> None:
    config = setup_config()
    if n_data is not None:
        if n_data <= 0:
            raise ValueError("--n-data must be positive")
        config["n_data"] = int(n_data)

    n = int(config["n_data"])

    print("=" * 80)
    print("Flow comparison: NewtonHellinger vs Newton vs HK on cat-paw mixture")
    print(f"  n = {n} data points, n_particles = {config['n_particles']}")
    print("=" * 80)

    key = jr.PRNGKey(config["seed"])
    key, data_key = jr.split(key)
    data = generate_data(data_key, config)
    print(f"Simulated {n} observations from paw mixture.")

    true_grid = build_density_grid(config)
    prior = make_prior(config)
    kernel = GaussianKernel(bandwidth=config["kernel_bandwidth"])

    key, init_key, py_key = jr.split(key, 3)
    initial_atoms = prior.sample(init_key, config["n_particles"])
    initial_measure = ParticleMeasure.initialize(initial_atoms)
    prior_particles = prior.to_particle_measure(py_key, config["n_particles"])

    key, ka, kb, kc = jr.split(key, 4)

    print("\n(a) NewtonHellingerFlow ...")
    t0 = time.perf_counter()
    m_nh = run_newton_hellinger(ka, initial_measure, data, prior, kernel, config)
    print(f"    elapsed {time.perf_counter() - t0:.2f} s  "
          f"ESS={float(m_nh.effective_sample_size()):.1f}")

    print("(b) NewtonFlow ...")
    t0 = time.perf_counter()
    m_newton = run_newton(kb, initial_measure, data, prior, kernel, config)
    print(f"    elapsed {time.perf_counter() - t0:.2f} s  "
          f"ESS={float(m_newton.effective_sample_size()):.1f}")

    print("(c) HellingerKantorovichFlow ...")
    t0 = time.perf_counter()
    m_hk = run_hk(kc, initial_measure, data, prior, kernel, prior_particles, config)
    print(f"    elapsed {time.perf_counter() - t0:.2f} s  "
          f"ESS={float(m_hk.effective_sample_size()):.1f}")

    print("\nCreating plot...")
    plot_flow_comparison(config, true_grid, data, m_nh, m_newton, m_hk)
    print("Saved 'flow_comparison.pdf'.")


if __name__ == "__main__":
    try:
        jax.config.update("jax_enable_x64", True)
    except Exception:
        pass

    parser = argparse.ArgumentParser(
        description="Compare NewtonHellingerFlow, NewtonFlow, and HK on cat-paw mixture"
    )
    parser.add_argument(
        "--n-data",
        type=int,
        default=None,
        help="Dataset size n (default: 1000).",
    )
    args = parser.parse_args()
    main(n_data=args.n_data)
