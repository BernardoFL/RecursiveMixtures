#!/usr/bin/env python3
"""
Bootstrap resampling for the particle Hellinger–Kantorovich (HK) flow on a
bivariate Gaussian mixture.

Each replicate draws Bayesian bootstrap weights, builds a resampled data stream,
and runs HK. Study **truncation** saves a multi-page PDF (one 1×2 page per sample
size). Study **prior** saves a single-page **2×N** grid (rows: prior on / off;
columns: each `n` in `n_data_list`): true-density heatmaps with final particles
(marker size ∝ weight).
"""

from __future__ import annotations

import argparse
import time
from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from recursive_mixtures import (
    GaussianKernel,
    ParticleMeasure,
    GaussianPrior,
    HellingerKantorovichFlow,
)
from recursive_mixtures.utils import (
    bayesian_bootstrap,
    generate_mixture_data,
    true_mixture_density,
)



def setup_config(fast: bool = True) -> Dict:
    """Configuration dictionary for the bootstrap HK experiment.
    
    Use fast=True (default) for quicker runs: fewer data steps, bootstrap
    replicates, and Sinkhorn iterations.
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
        # Bootstrap replicates per cell; plots use the first replicate only
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


def build_bootstrap_true_density_grid(config: Dict) -> np.ndarray:
    """True mixture density on a 2D grid for imshow (same mixture as data generation)."""
    n = int(config["grid_size"])
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
    return np.asarray(dens.reshape(n, n))


def _extent_from_config(config: Dict) -> List[float]:
    return [
        float(config["grid_min"]),
        float(config["grid_max"]),
        float(config["grid_min"]),
        float(config["grid_max"]),
    ]


def _scatter_particles(
    ax,
    config: Dict,
    measure: ParticleMeasure,
    color: str,
    *,
    size_scale: float = 300.0,
    with_axis_labels: bool = True,
) -> None:
    """True-density axes: particles with marker size ∝ weight."""
    atoms = np.asarray(measure.atoms)
    weights = np.asarray(measure.weights)
    wmax = float(weights.max())
    sizes = weights / max(wmax, 1e-12) * size_scale
    ax.scatter(
        atoms[:, 0],
        atoms[:, 1],
        s=sizes,
        c=color,
        alpha=0.88,
        edgecolors="white",
        linewidths=0.4,
        zorder=3,
    )
    ax.set_xlim(config["grid_min"], config["grid_max"])
    ax.set_ylim(config["grid_min"], config["grid_max"])
    if with_axis_labels:
        ax.set_xlabel("x₁")
        ax.set_ylabel("x₂")


def plot_truncation_bootstrap_page(
    config: Dict,
    true_grid: np.ndarray,
    data: jax.Array,
    hk_trunc: ParticleMeasure,
    hk_cont: ParticleMeasure,
    n_data: int,
    n_steps_cont: int,
) -> plt.Figure:
    """HK only: 1×2 panels — truncated vs continuation."""
    extent = _extent_from_config(config)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)
    panels = [
        (hk_trunc, "teal", f"HK: stop after data (n_steps = {n_data})"),
        (hk_cont, "crimson", f"HK: continuation (n_steps = {n_steps_cont})"),
    ]
    for ax, (m, color, title) in zip(axes, panels):
        ax.imshow(
            true_grid,
            origin="lower",
            extent=extent,
            aspect="auto",
            cmap="gray_r",
        )
        _scatter_particles(ax, config, m, color)
        ax.set_title(title)
    fig.suptitle(
        f"HK — true density + particles (size ∝ weight), n = {n_data}",
        y=1.02,
    )
    plt.tight_layout()
    return fig


def plot_prior_regularization_grid(
    config: Dict,
    true_grid: np.ndarray,
    column_results: List[Tuple[jax.Array, ParticleMeasure, ParticleMeasure, int]],
) -> plt.Figure:
    """
    HK Study B: 2×N grid — columns = sample sizes, top row = prior on, bottom = prior off.
    Same panel style as Study A (true-density heatmap + particles, size ∝ weight).
    """
    ncols = len(column_results)
    extent = _extent_from_config(config)
    fig_w = max(12.0, 4.0 * ncols)
    fig, axes = plt.subplots(2, ncols, figsize=(fig_w, 9.0), sharex=True, sharey=True)
    axes = np.asarray(axes)
    if axes.ndim == 1:
        axes = axes.reshape(2, 1)
    for j, (_data, hk_on, hk_off, nd) in enumerate(column_results):
        n_steps = int(nd)
        for row, (measure, color, row_label) in enumerate(
            [
                (hk_on, "teal", "Fisher–Rao prior reg. on"),
                (hk_off, "royalblue", "Fisher–Rao prior reg. off"),
            ]
        ):
            ax = axes[row, j]
            ax.imshow(
                true_grid,
                origin="lower",
                extent=extent,
                aspect="auto",
                cmap="gray_r",
            )
            _scatter_particles(
                ax, config, measure, color, with_axis_labels=False
            )
            ax.set_title(f"n = {nd}\n{row_label}\n(n_steps = {n_steps})")
    for j in range(ncols):
        axes[1, j].set_xlabel("x₁")
    for row in range(2):
        axes[row, 0].set_ylabel("x₂")
    fig.suptitle(
        "HK — prior regularization (no continuation): true density + particles "
        "(size ∝ weight)",
        y=1.01,
    )
    plt.tight_layout()
    return fig




def run_study_truncation_vs_continuation(config: Dict, key: jax.Array) -> jax.Array:
    """Compare stopping after one pass vs extra steps with index continuation."""
    print("=" * 80)
    print("Study A: stop after data vs index continuation (per sample size)")
    print("=" * 80)
    n_data_list = list(config["n_data_list"])
    continuation_factor = float(config["continuation_factor"])
    B = config["n_bootstrap"]
    prior, kernel = make_prior_and_kernel(config)

    true_grid = build_bootstrap_true_density_grid(config)
    out_pdf = "bootstrap_truncation_vs_continuation.pdf"

    with PdfPages(out_pdf) as pdf:
        for nd in n_data_list:
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

            hk_t, hk_c = [], []
            for _ in range(B):
                key, key_trunc, key_cont = jr.split(key, 3)
                hk_t.append(
                    run_single_hk_replicate(
                        key_trunc,
                        data,
                        prior,
                        kernel,
                        prior_particles,
                        cfg,
                        n_steps_override=n_steps_trunc,
                        bootstrap_after_data_override=False,
                    )
                )
                hk_c.append(
                    run_single_hk_replicate(
                        key_cont,
                        data,
                        prior,
                        kernel,
                        prior_particles,
                        cfg,
                        n_steps_override=n_steps_cont,
                        bootstrap_after_data_override=True,
                    )
                )

            fig = plot_truncation_bootstrap_page(
                cfg,
                true_grid,
                data,
                hk_t[0],
                hk_c[0],
                nd,
                n_steps_cont,
            )
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print(f"\nSaved multi-page '{out_pdf}' (HK: truncation vs continuation, one page per n).")
    return key


def run_study_prior_regularization(config: Dict, key: jax.Array) -> jax.Array:
    """HK only: Fisher–Rao prior on vs off with continuation disabled in both arms."""
    print("=" * 80)
    print("Study B: HK with Fisher–Rao prior regularization on vs off (no continuation)")
    print("=" * 80)
    n_data_list = list(config["n_data_list"])
    B = config["n_bootstrap"]
    prior, kernel = make_prior_and_kernel(config)

    true_grid = build_bootstrap_true_density_grid(config)
    out_pdf = "bootstrap_prior_regularization.pdf"

    column_results: List[Tuple[jax.Array, ParticleMeasure, ParticleMeasure, int]] = []

    for nd in n_data_list:
        cfg = dict(config)
        cfg["n_data"] = nd
        key, data_key, pp_key = jr.split(key, 3)
        data, _ = generate_bivariate_data(data_key, cfg)
        prior_particles = prior.to_particle_measure(pp_key, cfg["n_particles"])
        n_steps = int(nd)
        print(f"  n_data={nd}, n_steps={n_steps} (continuation disabled)")

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
                    bootstrap_after_data_override=False,
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
                    bootstrap_after_data_override=False,
                    use_prior_regularization=False,
                )
            )

        column_results.append((data, hk_on[0], hk_off[0], nd))

    fig = plot_prior_regularization_grid(config, true_grid, column_results)
    with PdfPages(out_pdf) as pdf:
        pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    print(
        f"\nSaved '{out_pdf}' (HK: 2×{len(n_data_list)} grid, prior on/off × sample sizes)."
    )
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

