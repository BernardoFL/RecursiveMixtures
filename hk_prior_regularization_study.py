#!/usr/bin/env python3
"""
Study B (prior regularization on/off) extracted from hk_computational_choices.py.

This module keeps the original single-replicate PDF generation used for the
committed reference artifact `bootstrap_prior_regularization.pdf`.

For trajectory uncertainty quantification on HPC, see `hk_prior_regularization_uq.py`.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import jax
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from recursive_mixtures.measure import ParticleMeasure


def plot_prior_regularization_grid(
    config: Dict,
    true_grid: np.ndarray,
    row_results: List[Tuple[ParticleMeasure, ParticleMeasure, int]],
    *,
    extent,
    scatter_particles_fn,
) -> plt.Figure:
    """
    HK Study B: N×2 grid — one row per sample size; columns = prior on | prior off.
    Same panel style as Study A (true-density heatmap + HK particles only).
    """
    nrows = len(row_results)
    fig_h = max(9.0, 3.4 * nrows)
    fig, axes = plt.subplots(nrows, 2, figsize=(12.0, fig_h), sharex=True, sharey=True)
    axes = np.asarray(axes)
    if nrows == 1:
        axes = axes.reshape(1, 2)
    for i, (hk_on, hk_off, nd) in enumerate(row_results):
        n_steps = int(nd)
        for j, (measure, color, col_label) in enumerate(
            [
                (hk_on, "teal", "Prior on"),
                (hk_off, "royalblue", "Prior off"),
            ]
        ):
            ax = axes[i, j]
            ax.imshow(
                true_grid,
                origin="lower",
                extent=extent,
                aspect="auto",
                cmap="gray_r",
            )
            scatter_particles_fn(ax, config, measure, color, with_axis_labels=False)
            ax.set_title(f"n = {nd}, {col_label}\n(n_steps = {n_steps})")
    for i in range(nrows):
        axes[i, 0].set_ylabel("x₂")
    axes[-1, 0].set_xlabel("x₁")
    axes[-1, 1].set_xlabel("x₁")
    fig.suptitle(
        "WFR Flow — Prior Regularization: true density + particles (size ∝ weight)",
        y=1.01,
    )
    plt.tight_layout()
    return fig


def run_study_prior_regularization(config: Dict, key: jax.Array) -> jax.Array:
    """
    HK only: Fisher–Rao prior on vs off with continuation disabled in both arms.

    This function relies on helper functions defined in `hk_computational_choices.py`
    and imports them lazily to avoid circular imports at module import time.
    """
    from hk_computational_choices import (
        _extent_from_config,
        _scatter_particles,
        build_bootstrap_true_density_grid,
        generate_clover_data,
        make_prior_and_kernel,
        run_single_hk_replicate,
    )

    print("=" * 80)
    print("Study B: Fisher-Rao prior regularization on vs off (no continuation)")
    print("=" * 80)
    n_data_list = list(config["n_data_list"])
    B = int(config["n_bootstrap"])
    prior, kernel = make_prior_and_kernel(config)

    true_grid = build_bootstrap_true_density_grid(config)
    extent = _extent_from_config(config)
    out_pdf = "bootstrap_prior_regularization.pdf"

    row_results: List[Tuple[ParticleMeasure, ParticleMeasure, int]] = []

    for nd in n_data_list:
        cfg = dict(config)
        cfg["n_data"] = int(nd)
        key, data_key, pp_key = jr.split(key, 3)
        data = generate_clover_data(data_key, cfg)
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

        row_results.append((hk_on[0], hk_off[0], int(nd)))

    fig = plot_prior_regularization_grid(
        config,
        true_grid,
        row_results,
        extent=extent,
        scatter_particles_fn=_scatter_particles,
    )
    with PdfPages(out_pdf) as pdf:
        pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    print(
        f"\nSaved '{out_pdf}' (HK: {len(n_data_list)}×2 grid, "
        "rows = sample sizes, cols = prior on | off)."
    )
    return key

