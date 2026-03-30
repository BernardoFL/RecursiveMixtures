#!/usr/bin/env python3
"""
Computational choices experiment for the HK (WFR) flow on a four-petal clover
Gaussian mixture.

Isolates the effect of two algorithmic switches (Studies A/B) and optionally
Study C — Wasserstein distance vs sample size for a 2×2 factorial over those
switches:

    Study A — Bootstrap continuation on/off
        Compare stopping after exactly one ordered pass over the data vs running
        extra steps where data indices are resampled uniformly ("continuation").

    Study B — Fisher-Rao prior regularization on/off
        Compare HK with and without the Sinkhorn prior functional contributing
        to the Hellinger weight update (``use_prior_regularization``).

Both studies use a Pitman–Yor mixing prior on atom locations and produce
true-density heatmap panels with final HK particles (size ∝ weight).
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
    PitmanYorProcessPrior,
    GaussianKernel,
    ParticleMeasure,
    GaussianPrior,
    HellingerKantorovichFlow,
)
from recursive_mixtures.utils import bayesian_bootstrap

from clover_distribution import CloverDistribution


def _clover_plot_bounds(radius: float, radial_std: float, tangential_std: float, *, nsig: float = 4.0) -> dict:
    """Heuristic plot bounds that capture essentially all clover mass."""
    r = float(radius)
    rs = float(radial_std)
    ts = float(tangential_std)
    pad = nsig * max(rs, ts)
    x_min = -(r + pad)
    x_max = (r + pad)
    y_min = -(r + pad)
    y_max = (r + pad)
    return {
        "grid_x_min": float(x_min),
        "grid_x_max": float(x_max),
        "grid_y_min": float(y_min),
        "grid_y_max": float(y_max),
    }



def setup_config(fast: bool = True) -> Dict:
    """Configuration dictionary for the bootstrap HK experiment.
    
    Use fast=True (default) for quicker runs: fewer data steps, bootstrap
    replicates, and Sinkhorn iterations.
    """
    # Fast defaults: fewer steps and Sinkhorn work so runs complete in minutes
    config = {
        # Target distribution: four-petal clover Gaussian mixture parameters
        "clover_radius": 1.5,
        "clover_radial_std": 0.45,
        "clover_tangential_std": 0.18,
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
        # Mixing prior: DP(α, G0) with G0 = isotropic Gaussian on atom locations
        "prior_mean": jnp.array([0.0, 0.0]),
        "prior_std": 3.0,
        "py_discount": 0.2,
        "py_strength": 10.0,
        # Density grid / plot bounds
        "grid_size": 200,
        # Recording
        "store_every": 0,  # only final measures for bootstrap
        # Random seeds
        "seed": 123,
        # Optional override from CLI. If set and > n_data, flow.run uses
        # bootstrap continuation beyond data length.
        "n_steps": None,
        # Studies (truncation vs continuation; prior on/off): sample sizes and
        # fixed continuation scheme n_steps_on = ceil(1.5 * n_data).
        "n_data_list": [100, 1000],
        # Study C (Wasserstein vs n): empirical reference size for OT to clover
        # Study C: OT reference size (50×M cost matrix per curve point; keep M moderate)
        "w2_reference_size": 2048,
    }
    config.update(
        _clover_plot_bounds(
            radius=config["clover_radius"],
            radial_std=config["clover_radial_std"],
            tangential_std=config["clover_tangential_std"],
            nsig=4.0,
        )
    )
    return config


def generate_clover_data(
    key: jax.Array,
    config: Dict,
) -> jax.Array:
    """Sample n i.i.d. points from the clover Gaussian mixture."""
    clover = CloverDistribution(
        radius=float(config["clover_radius"]),
        radial_std=float(config["clover_radial_std"]),
        tangential_std=float(config["clover_tangential_std"]),
    )
    return clover.sample(key, int(config["n_data"]))


def make_prior_and_kernel(config: Dict):
    """Pitman–Yor mixing prior PY(d, θ, G0) with G0 Gaussian; Gaussian kernel for HK."""
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
    """True clover density on a 2D grid for imshow."""
    n = int(config["grid_size"])
    if "grid_x_min" in config:
        xmin = float(config["grid_x_min"])
        xmax = float(config["grid_x_max"])
        ymin = float(config["grid_y_min"])
        ymax = float(config["grid_y_max"])
    else:
        xmin = float(config["grid_min"])
        xmax = float(config["grid_max"])
        ymin = float(config["grid_min"])
        ymax = float(config["grid_max"])
    xs = jnp.linspace(xmin, xmax, n)
    ys = jnp.linspace(ymin, ymax, n)
    X, Y = jnp.meshgrid(xs, ys)
    grid_points = jnp.stack([X.ravel(), Y.ravel()], axis=1)
    clover = CloverDistribution(
        radius=float(config["clover_radius"]),
        radial_std=float(config["clover_radial_std"]),
        tangential_std=float(config["clover_tangential_std"]),
    )
    dens = clover.pdf(grid_points)
    return np.asarray(dens.reshape(n, n))


def _extent_from_config(config: Dict) -> List[float]:
    if "grid_x_min" in config:
        return [
            float(config["grid_x_min"]),
            float(config["grid_x_max"]),
            float(config["grid_y_min"]),
            float(config["grid_y_max"]),
        ]
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
    if "grid_x_min" in config:
        ax.set_xlim(float(config["grid_x_min"]), float(config["grid_x_max"]))
        ax.set_ylim(float(config["grid_y_min"]), float(config["grid_y_max"]))
    else:
        ax.set_xlim(config["grid_min"], config["grid_max"])
        ax.set_ylim(config["grid_min"], config["grid_max"])
    if with_axis_labels:
        ax.set_xlabel("x₁")
        ax.set_ylabel("x₂")


def plot_truncation_bootstrap_grid(
    config: Dict,
    true_grid: np.ndarray,
    row_results: List[Tuple[ParticleMeasure, ParticleMeasure, int, int]],
) -> plt.Figure:
    """
    Study A: N×2 grid — rows are sample sizes, columns are continuation off/on.
    """
    extent = _extent_from_config(config)
    nrows = len(row_results)
    fig_h = max(9.0, 3.4 * nrows)
    fig, axes = plt.subplots(nrows, 2, figsize=(12.0, fig_h), sharex=True, sharey=True)
    axes = np.asarray(axes)
    if nrows == 1:
        axes = axes.reshape(1, 2)

    for i, (hk_off, hk_on, nd, n_steps_on) in enumerate(row_results):
        for j, (measure, color, label, n_steps_used) in enumerate(
            [
                (hk_off, "teal", "Continuation off", int(nd)),
                (hk_on, "crimson", "Continuation on", int(n_steps_on)),
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
            _scatter_particles(ax, config, measure, color, with_axis_labels=False)
            ax.set_title(f"n = {nd}, {label}\n(n_steps = {n_steps_used})")
    for i in range(nrows):
        axes[i, 0].set_ylabel("x₂")
    axes[-1, 0].set_xlabel("x₁")
    axes[-1, 1].set_xlabel("x₁")
    fig.suptitle(
        "WFR Flow — Continuation off/on: true density + particles (size ∝ weight)",
        y=1.01,
    )
    plt.tight_layout()
    return fig


def plot_prior_regularization_grid(
    config: Dict,
    true_grid: np.ndarray,
    row_results: List[Tuple[ParticleMeasure, ParticleMeasure, int]],
) -> plt.Figure:
    # Study B plotting lives in hk_prior_regularization_study.py.
    from hk_prior_regularization_study import plot_prior_regularization_grid as _plot

    return _plot(
        config,
        true_grid,
        row_results,
        extent=_extent_from_config(config),
        scatter_particles_fn=_scatter_particles,
    )


def w2_particle_to_empirical_reference(
    measure: ParticleMeasure,
    ref_points: np.ndarray,
) -> float:
    """
    Approximate W_2 between a particle measure and a uniform empirical measure
    on ref_points using squared Euclidean cost (POT exact transport).
    """
    try:
        import ot
    except ImportError as e:
        raise ImportError(
            "The wasserstein study requires the POT package. Install with: pip install POT"
        ) from e
    a = np.asarray(measure.weights, dtype=np.float64).ravel()
    a = a / np.sum(a)
    m = int(ref_points.shape[0])
    b = np.full(m, 1.0 / m, dtype=np.float64)
    x = np.asarray(measure.atoms, dtype=np.float64)
    y = np.asarray(ref_points, dtype=np.float64)
    c = ot.dist(x, y, metric="sqeuclidean")
    w2_sq = float(ot.emd2(a, b, c))
    return float(np.sqrt(max(w2_sq, 0.0)))


def run_study_wasserstein_sweep(config: Dict, key: jax.Array) -> None:
    """
    Study C: for n in 100,200,...,1000, run HK on all four (prior × continuation)
    arms and plot W_2(final particles, empirical reference) vs n.
    """
    # Ensure POT is available before any heavy work
    try:
        import ot  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "The wasserstein study requires the POT package. Install with: pip install POT"
        ) from e

    print("=" * 80)
    print("Study C: Wasserstein distance vs n (2×2 factorial vs empirical target)")
    print("=" * 80)

    m_ref = int(config["w2_reference_size"])
    n_sweep = list(range(100, 1001, 100))

    key, kref = jr.split(key)
    clover = CloverDistribution(
        radius=float(config["clover_radius"]),
        radial_std=float(config["clover_radial_std"]),
        tangential_std=float(config["clover_tangential_std"]),
    )
    ref_points = np.asarray(clover.sample(kref, m_ref))
    print(
        f"Reference: {m_ref} i.i.d. samples from clover (fixed across all n and arms).",
        flush=True,
    )

    prior, kernel = make_prior_and_kernel(config)

    arms: List[Tuple[str, bool, bool]] = [
        ("Prior on, continuation off", True, False),
        ("Prior on, continuation on", True, True),
        ("Prior off, continuation off", False, False),
        ("Prior off, continuation on", False, True),
    ]
    series: Dict[str, List[float]] = {label: [] for label, _, _ in arms}

    for nd in n_sweep:
        cfg = dict(config)
        cfg["n_data"] = nd
        key, data_key, pp_key = jr.split(key, 3)
        data = generate_clover_data(data_key, cfg)
        prior_particles = prior.to_particle_measure(pp_key, cfg["n_particles"])
        n_steps_on = int(np.ceil(1.5 * nd))

        for label, use_prior, cont in arms:
            n_steps = n_steps_on if cont else int(nd)
            key, krun = jr.split(key)
            measure = run_single_hk_replicate(
                krun,
                data,
                prior,
                kernel,
                prior_particles,
                cfg,
                n_steps_override=n_steps,
                bootstrap_after_data_override=cont,
                use_prior_regularization=use_prior,
            )
            w2 = w2_particle_to_empirical_reference(measure, ref_points)
            series[label].append(w2)
            print(f"  n={nd:4d}  {label:32s}  W2 ≈ {w2:.5f}", flush=True)

    out_pdf = "hk_wasserstein_vs_n.pdf"
    fig, ax = plt.subplots(figsize=(10.0, 5.5))
    for label, _, _ in arms:
        ax.plot(n_sweep, series[label], marker="o", linewidth=1.8, label=label)
    ax.set_xlabel("n (data size)")
    ax.set_ylabel(r"$W_2(\hat\mu_n,\,\mu_{\mathrm{ref}})$")
    ax.set_title(
        "HK WFR: 2-Wasserstein vs n (particles vs fixed empirical reference; "
        "2×2 prior × continuation)"
    )
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved '{out_pdf}'.")


def run_study_truncation_vs_continuation(config: Dict, key: jax.Array) -> jax.Array:
    """Compare continuation off vs on with fixed +50% extra steps."""
    print("=" * 80)
    print("Study A: bootstrap continuation on vs off (per sample size)")
    print("=" * 80)
    n_data_list = list(config["n_data_list"])
    B = config["n_bootstrap"]
    prior, kernel = make_prior_and_kernel(config)

    true_grid = build_bootstrap_true_density_grid(config)
    out_pdf = "bootstrap_truncation_vs_continuation.pdf"
    row_results: List[Tuple[ParticleMeasure, ParticleMeasure, int, int]] = []

    for nd in n_data_list:
        cfg = dict(config)
        cfg["n_data"] = nd
        key, data_key, pp_key = jr.split(key, 3)
        data = generate_clover_data(data_key, cfg)
        prior_particles = prior.to_particle_measure(pp_key, cfg["n_particles"])

        n_steps_off = int(nd)
        n_steps_on = int(np.ceil(1.5 * nd))
        print(
            f"  n_data={nd}: continuation off n_steps={n_steps_off}, "
            f"continuation on n_steps={n_steps_on} (+50%)"
        )

        hk_off, hk_on = [], []
        for _ in range(B):
            key, key_off, key_on = jr.split(key, 3)
            hk_off.append(
                run_single_hk_replicate(
                    key_off,
                    data,
                    prior,
                    kernel,
                    prior_particles,
                    cfg,
                    n_steps_override=n_steps_off,
                    bootstrap_after_data_override=False,
                )
            )
            hk_on.append(
                run_single_hk_replicate(
                    key_on,
                    data,
                    prior,
                    kernel,
                    prior_particles,
                    cfg,
                    n_steps_override=n_steps_on,
                    bootstrap_after_data_override=True,
                )
            )
        row_results.append((hk_off[0], hk_on[0], int(nd), int(n_steps_on)))

    fig = plot_truncation_bootstrap_grid(config, true_grid, row_results)
    with PdfPages(out_pdf) as pdf:
        pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved '{out_pdf}' (HK: {len(n_data_list)}×2 grid, rows = n, cols = continuation off|on).")
    return key


def run_study_prior_regularization(config: Dict, key: jax.Array) -> jax.Array:
    """
    Study B runner (delegated).

    The uncertainty-quantification (niter trajectories) variant lives in
    `hk_prior_regularization_uq.py`. This function remains for backward-compatible
    PDF generation via `--study prior`.
    """
    # Import locally so hk_computational_choices.py stays lightweight unless needed.
    from hk_prior_regularization_study import run_study_prior_regularization as _run

    return _run(config, key)




def main(
    fast: bool = True,
    study: str = "both",
    n_data_list: Optional[List[int]] = None,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
) -> None:
    """Run truncation/prior bootstrap studies (or both)."""
    config = setup_config(fast=fast)
    if n_data_list is not None:
        config["n_data_list"] = n_data_list
    if y_min is not None:
        config["grid_y_min"] = float(y_min)
    if y_max is not None:
        config["grid_y_max"] = float(y_max)
    if config["grid_y_min"] >= config["grid_y_max"]:
        raise ValueError("Require grid_y_min < grid_y_max")

    key = jr.PRNGKey(config["seed"])

    if study == "wasserstein":
        run_study_wasserstein_sweep(config, key)
        return

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

    parser = argparse.ArgumentParser(
        description="HK computational choices: prior on/off and bootstrap continuation on/off"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full (slower) configuration instead of fast mode.",
    )
    parser.add_argument(
        "--study",
        type=str,
        choices=("truncation", "prior", "both", "wasserstein"),
        default="both",
        help="truncation|prior|both sample-size studies; wasserstein = Study C (W2 vs n).",
    )
    parser.add_argument(
        "--n-data-list",
        type=str,
        default=None,
        help="Comma-separated sample sizes for truncation/prior studies (default: config).",
    )
    parser.add_argument(
        "--y-min",
        type=float,
        default=None,
        help="Override y-axis lower bound for the density grid / extent (default: auto from target params).",
    )
    parser.add_argument(
        "--y-max",
        type=float,
        default=None,
        help="Override y-axis upper bound for the density grid / extent (default: auto from target params).",
    )
    args = parser.parse_args()

    nd_list: Optional[List[int]] = None
    if args.n_data_list:
        nd_list = [int(x.strip()) for x in args.n_data_list.split(",") if x.strip()]

    main(
        fast=not args.full,
        study=args.study,
        n_data_list=nd_list,
        y_min=args.y_min,
        y_max=args.y_max,
    )

