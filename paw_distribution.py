#!/usr/bin/env python3
"""
Cat-paw Gaussian mixture distribution.

A 7-component 2D GMM arranged as a cat paw: 4 elliptical toe pads on an arc
above 3 palm pads (central + two side pads). Designed for gradient-flow
experiments and visualization.

Usage:
    from paw_distribution import PawDistribution

    paw = PawDistribution()
    samples = paw.sample(key, 1000)
    dens = paw.pdf(grid_points)
    paw.plot(out_path="paw.png")

When run as script: generates a demo plot.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array

# Optional matplotlib for plotting
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ---------------------------------------------------------------------------
# Parameter selection guide (exposed for programmatic access)
# ---------------------------------------------------------------------------

PARAMETER_SELECTION_GUIDE = """
PARAMETER SELECTION GUIDE
========================

1. TOE ARC
   - arc_center: (cx, cy) — center of the semicircle on which toes sit.
     Typically place cy just above the central palm pad (e.g. cy = -0.55).
   - arc_radius: r — distance from arc center to each toe. Increase to
     separate toes from the palm; decrease to bring them closer.
   - arc_angles: (θ_min, θ_max) in degrees — angular span of the toe arc.
     (25, 155) gives a wide fan; (40, 140) a tighter cluster.
   - n_toes: number of toe pads (default 4).

2. TOE SHAPE (elliptical)
   - toe_std_x: σ_x — horizontal spread (smaller = narrower toe).
   - toe_std_y: σ_y — vertical spread (larger = taller toe).
   Typical paw: σ_x < σ_y (oval shape). Ratio 1:2 works well.

3. PALM PADS
   - pad_central: ((x,y), std) — large central pad. Usually above the side pads.
   - pad_left, pad_right: ((x,y), std) — two side pads below the central one.
   Place side pads at y < pad_central to form the palm base.

4. WEIGHTS
   - Proportional to desired sample counts per component, or uniform.
   - Central pad typically gets more mass (larger in real paws).
   Default: toes ~0.11 each, side pads ~0.15, central ~0.25.
"""


def print_parameter_guide() -> None:
    """Print the parameter selection guide to stdout."""
    print(PARAMETER_SELECTION_GUIDE)


def _toe_arc_means(
    center: Tuple[float, float],
    radius: float,
    n_toes: int,
    angle_deg: Tuple[float, float],
) -> np.ndarray:
    """Compute toe mean positions on a circular arc."""
    cx, cy = center
    angles = np.linspace(
        np.radians(angle_deg[0]),
        np.radians(angle_deg[1]),
        n_toes,
    )
    x = cx + radius * np.cos(angles)
    y = cy + radius * np.sin(angles)
    return np.column_stack([x, y])


class PawDistribution:
    """
    7-component cat-paw Gaussian mixture.

    Components: 4 toes (elliptical) on an arc + 3 palm pads.
    All components use diagonal covariance (axis-aligned ellipses).
    """

    def __init__(
        self,
        arc_center: Tuple[float, float] = (0.0, -0.55),
        arc_radius: float = 1.6,
        arc_angles: Tuple[float, float] = (30.0, 150.0),
        n_toes: int = 4,
        toe_std_x: float = 0.18,
        toe_std_y: float = 0.36,
        pad_central: Tuple[Tuple[float, float], float] = ((0.0, -0.55), 0.55),
        pad_left: Tuple[Tuple[float, float], float] = ((-0.85, -1.45), 0.38),
        pad_right: Tuple[Tuple[float, float], float] = ((0.85, -1.45), 0.38),
        weights: Optional[Array] = None,
    ):
        """
        Args:
            arc_center: (cx, cy) center of toe arc. Usually above central pad.
            arc_radius: radius of toe arc. Increase for more toe–palm separation.
            arc_angles: (θ_min, θ_max) in degrees for toe span.
            n_toes: number of toe pads.
            toe_std_x, toe_std_y: elliptical stds (narrow x, tall y).
            pad_central: ((x,y), std) central palm pad.
            pad_left, pad_right: ((x,y), std) side palm pads.
            weights: mixture weights, shape (7,). Default: proportional to
                typical sample counts (toes 0.1132 each, sides 0.1509, central 0.2453).
        """
        toe_means = _toe_arc_means(arc_center, arc_radius, n_toes, arc_angles)
        (pc_xy, pc_std) = pad_central
        (pl_xy, pl_std) = pad_left
        (pr_xy, pr_std) = pad_right

        self.means = jnp.array([
            *toe_means,
            list(pl_xy),
            list(pr_xy),
            list(pc_xy),
        ])
        self.stds = jnp.array([
            *([[toe_std_x, toe_std_y]] * n_toes),
            [pl_std, pl_std],
            [pr_std, pr_std],
            [pc_std, pc_std],
        ])
        if weights is not None:
            self.weights = jnp.asarray(weights) / jnp.sum(weights)
        else:
            # Default: toes ~0.113 each, sides ~0.151, central ~0.245
            w = jnp.array([
                0.1132, 0.1132, 0.1132, 0.1132,
                0.1509, 0.1509,
                0.2453,
            ])
            self.weights = w / jnp.sum(w)
        self.K = self.means.shape[0]
        self.D = self.means.shape[1]

    def sample(self, key: Array, n: int) -> Array:
        """Sample n points from the paw mixture. Returns shape (n, 2)."""
        key1, key2 = jr.split(key)
        assignments = jr.choice(key1, self.K, shape=(n,), p=self.weights)
        noise = jr.normal(key2, shape=(n, self.D))
        samples = self.means[assignments] + self.stds[assignments] * noise
        return samples

    def pdf(self, x: Array) -> Array:
        """Evaluate density at x. x: (M,) or (M, 2). Returns (M,)."""
        x = jnp.atleast_1d(x)
        if x.ndim == 1:
            x = x[:, None]
        means, stds, weights = self.means, self.stds, self.weights
        K, D = means.shape[0], means.shape[1]

        def comp_dens(k: int) -> Array:
            diff = x - means[k]
            s = stds[k]
            if jnp.ndim(s) == 0 or (jnp.size(s) == 1):
                sig = float(jnp.ravel(s)[0])
                ex = jnp.sum(diff ** 2, axis=1) / (sig ** 2)
                norm = (sig * jnp.sqrt(2 * jnp.pi)) ** D
            else:
                ex = jnp.sum(diff ** 2 / (s ** 2), axis=1)
                norm = jnp.prod(s * jnp.sqrt(2 * jnp.pi))
            return jnp.exp(-0.5 * ex) / norm

        dens = sum(weights[k] * comp_dens(k) for k in range(K))
        return dens

    def log_pdf(self, x: Array) -> Array:
        """Log density at x. x: (M,) or (M, 2). Returns (M,)."""
        return jnp.log(self.pdf(x) + 1e-30)

    def to_dict(self) -> dict:
        """Return (means, stds, weights) as dict for config compatibility."""
        return {
            "means": np.asarray(self.means),
            "stds": np.asarray(self.stds),
            "weights": np.asarray(self.weights),
        }

    def plot(
        self,
        ax=None,
        grid_min: float = -2.5,
        grid_max: float = 2.5,
        grid_resolution: int = 80,
        cmap: str = "viridis",
        out_path: Optional[str] = None,
        show: bool = True,
    ):
        """
        Plot the paw density as a heatmap.

        Args:
            ax: Matplotlib axes. If None, create new figure.
            grid_min, grid_max: plot bounds (square).
            grid_resolution: grid size for heatmap.
            cmap: colormap name.
            out_path: if set, save figure to this path.
            show: if True, call plt.show() when ax is None.
        """
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib is required for PawDistribution.plot()")
        xs = np.linspace(grid_min, grid_max, grid_resolution)
        ys = np.linspace(grid_min, grid_max, grid_resolution)
        X, Y = np.meshgrid(xs, ys)
        grid = np.stack([X.ravel(), Y.ravel()], axis=1)
        dens = np.asarray(self.pdf(grid)).reshape(grid_resolution, grid_resolution)
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(
            dens,
            origin="lower",
            extent=[grid_min, grid_max, grid_min, grid_max],
            aspect="auto",
            cmap=cmap,
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Cat-paw mixture density")
        if out_path:
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
        if show and ax is None:
            plt.show()

    def plot_scatter(
        self,
        key: Array,
        n: int = 500,
        ax=None,
        colors: Optional[list] = None,
        out_path: Optional[str] = None,
    ):
        """
        Plot a scatter of samples colored by component.

        Args:
            key: JAX random key.
            n: number of samples.
            ax: Matplotlib axes (optional).
            colors: list of colors per component (optional).
            out_path: save path (optional).
        """
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib is required for plot_scatter()")
        key1, key2 = jr.split(key)
        assigns = jr.choice(key1, self.K, shape=(n,), p=self.weights)
        noise = jr.normal(key2, shape=(n, 2))
        samples = self.means[assigns] + self.stds[assigns] * noise
        samples_np = np.asarray(samples)
        assigns_np = np.asarray(assigns)
        if colors is None:
            colors = ["#E07B54", "#A855F7", "#3B82F6", "#06B6D4",
                      "#F59E0B", "#10B981", "#EF4444"]
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))
        for k in range(self.K):
            mask = assigns_np == k
            ax.scatter(
                samples_np[mask, 0], samples_np[mask, 1],
                c=colors[k % len(colors)],
                s=20, alpha=0.7, label=f"C{k}",
            )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Cat-paw samples")
        ax.legend(title="Component")
        ax.set_aspect("equal")
        if out_path:
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
        return ax


# ---------------------------------------------------------------------------
# Convenience functions for drop-in use
# ---------------------------------------------------------------------------

def sample_paw(key: Array, n: int, **kwargs) -> Array:
    """Sample n points from the default paw. kwargs override PawDistribution params."""
    paw = PawDistribution(**kwargs)
    return paw.sample(key, n)


def pdf_paw(x: Array, **kwargs) -> Array:
    """Evaluate paw PDF at x. kwargs override PawDistribution params."""
    paw = PawDistribution(**kwargs)
    return paw.pdf(x)


def create_paw_config(
    n_data: int = 1000,
    grid_min: float = -2.5,
    grid_max: float = 2.5,
    grid_size: int = 80,
    **paw_kwargs,
) -> dict:
    """
    Build an experiment config dict with paw mixture parameters.

    Returns keys: dumbbell_means, dumbbell_stds, dumbbell_weights, n_data,
    grid_min, grid_max, grid_size (for drop-in use with metastability_experiment).

    Example:
        config = create_paw_config(n_data=500)
        config["dumbbell_means"]  # (7, 2) array
    """
    paw = PawDistribution(**paw_kwargs)
    d = paw.to_dict()
    return {
        "dumbbell_means": jnp.array(d["means"]),
        "dumbbell_stds": jnp.array(d["stds"]),
        "dumbbell_weights": jnp.array(d["weights"]),
        "n_data": n_data,
        "grid_min": grid_min,
        "grid_max": grid_max,
        "grid_size": grid_size,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Cat-paw GMM: sample, PDF, plot")
    ap.add_argument("--guide", action="store_true", help="Print parameter selection guide")
    ap.add_argument("--no-plot", action="store_true", help="Skip saving plots")
    args = ap.parse_args()

    if args.guide:
        print_parameter_guide()
        exit(0)

    jax.config.update("jax_enable_x64", True)
    key = jr.PRNGKey(42)
    paw = PawDistribution()
    print("Paw distribution: 7 components (4 toes + 3 palm pads)")
    print("  Means shape:", paw.means.shape)
    print("  Weights:", np.round(np.asarray(paw.weights), 4))
    if HAS_MATPLOTLIB and not args.no_plot:
        paw.plot(out_path="paw_density.png", show=False)
        paw.plot_scatter(key, n=600, out_path="paw_samples.png")
        print("Saved paw_density.png and paw_samples.png")
