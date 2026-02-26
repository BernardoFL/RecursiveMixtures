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
from recursive_mixtures.utils import (
    bayesian_bootstrap,
    generate_mixture_data,
    true_mixture_density,
)

try:
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS

    HAS_NUMPYRO = True
except ImportError:  # pragma: no cover - optional dependency
    HAS_NUMPYRO = False


def setup_config() -> Dict:
    """Configuration dictionary for the bootstrap HK experiment."""
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
        "n_data": 1000,
        # Flow parameters
        "n_particles": 200,
        "step_size": 0.05,
        "kernel_bandwidth": 1.0,
        "sinkhorn_reg": 0.05,
        "wasserstein_weight": 0.1,
        "prior_flow_weight": 0.1,  # lambda
        "prior_mc_samples": 5,  # M
        # Bootstrap
        "n_bootstrap": 32,  # B
        # Prior
        "prior_mean": jnp.array([0.0, 0.0]),
        "prior_std": 3.0,
        # Density grid
        "grid_min": -5.0,
        "grid_max": 5.0,
        "grid_size": 50,
        # Recording
        "store_every": 0,  # only final measures for bootstrap
        # NumPyro MCMC
        "use_numpyro": True,
        "mcmc_num_warmup": 500,
        "mcmc_num_samples": 1000,
        "mcmc_num_chains": 1,
        # Random seeds
        "seed": 123,
        "mcmc_seed": 321,
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
    if config["store_every"] and config["store_every"] > 0:
        final_measure, _ = flow.run(
            initial_measure,
            data_boot,
            key=key_flow,
            store_every=config["store_every"],
        )
    else:
        final_measure, _ = flow.run(
            initial_measure,
            data_boot,
            key=key_flow,
            store_every=len(data_boot),  # only store final state
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


def numpyro_mixture_model(
    data: jax.Array,
    K: int,
    prior_mean_scale: float = 3.0,
    prior_scale_scale: float = 1.0,
):
    """
    NumPyro model: K-component Gaussian mixture in 2D with shared scale.

    The prior is chosen to roughly match the flow prior.
    """
    if not HAS_NUMPYRO:
        raise RuntimeError("NumPyro is not available, cannot run MCMC baseline.")

    N, D = data.shape

    loc = numpyro.sample(
        "loc",
        dist.Normal(
            jnp.zeros((K, D)),
            prior_mean_scale * jnp.ones((K, D)),
        ),
    )  # shape (K, D)
    scale = numpyro.sample("scale", dist.HalfCauchy(prior_scale_scale))
    weights = numpyro.sample("weights", dist.Dirichlet(jnp.ones(K)))

    cov = jnp.eye(D) * (scale**2)
    covs = jnp.broadcast_to(cov, (K, D, D))
    components = dist.MultivariateNormal(loc, covariance_matrix=covs)
    mixture = dist.MixtureSameFamily(dist.Categorical(weights), components)

    with numpyro.plate("data", N):
        numpyro.sample("obs", mixture, obs=data)


def run_numpyro_mcmc(
    data: jax.Array,
    config: Dict,
) -> Dict[str, jax.Array]:
    """Run NumPyro MCMC for the mixture model and return posterior samples."""
    if not HAS_NUMPYRO or not config.get("use_numpyro", True):
        print("NumPyro not available or disabled; skipping MCMC baseline.")
        return {}

    K = config["true_means"].shape[0]

    kernel = NUTS(numpyro_mixture_model)
    mcmc = MCMC(
        kernel,
        num_warmup=config["mcmc_num_warmup"],
        num_samples=config["mcmc_num_samples"],
        num_chains=config["mcmc_num_chains"],
        progress_bar=True,
    )
    mcmc.run(
        jr.PRNGKey(config["mcmc_seed"]),
        data=data,
        K=K,
    )
    mcmc.print_summary()
    return mcmc.get_samples()


def mcmc_density_from_samples(
    samples: Dict[str, jax.Array],
    grid_points: jax.Array,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """
    Compute posterior predictive density and credible intervals from MCMC samples.

    Returns:
        (mean, lower, upper) each of shape (G,)
    """
    if not samples:
        return (
            jnp.zeros(grid_points.shape[0]),
            jnp.zeros(grid_points.shape[0]),
            jnp.zeros(grid_points.shape[0]),
        )

    loc_samples = samples["loc"]  # (S, K, D)
    scale_samples = samples["scale"]  # (S,)
    weight_samples = samples["weights"]  # (S, K)

    S, K, D = loc_samples.shape
    G = grid_points.shape[0]

    dens_list = []
    for s in range(S):
        loc = loc_samples[s]  # (K, D)
        scale = scale_samples[s]
        weights = weight_samples[s]

        cov = jnp.eye(D) * (scale**2)
        covs = jnp.broadcast_to(cov, (K, D, D))
        components = dist.MultivariateNormal(loc, covariance_matrix=covs)
        mixture = dist.MixtureSameFamily(dist.Categorical(weights), components)

        dens = jnp.exp(mixture.log_prob(grid_points))  # (G,)
        dens_list.append(dens)

    dens_array = jnp.stack(dens_list, axis=0)  # (S, G)
    return credible_intervals(dens_array)


def plot_bootstrap_results(
    config: Dict,
    grid_points: jax.Array,
    hk_mean: jax.Array,
    hk_lower: jax.Array,
    hk_upper: jax.Array,
    mcmc_mean: jax.Array | None,
    true_density_vals: jax.Array,
):
    """Create summary plots for HK bootstrap and (optional) MCMC baseline."""
    n = config["grid_size"]

    def reshape(field: jax.Array) -> np.ndarray:
        return np.asarray(field).reshape(n, n)

    X = np.linspace(config["grid_min"], config["grid_max"], n)
    Y = np.linspace(config["grid_min"], config["grid_max"], n)
    Xg, Yg = np.meshgrid(X, Y)

    true_grid = reshape(true_density_vals)
    hk_mean_grid = reshape(hk_mean)
    hk_lower_grid = reshape(hk_lower)
    hk_upper_grid = reshape(hk_upper)
    mcmc_mean_grid = reshape(mcmc_mean) if mcmc_mean is not None else None

    fig, axes = plt.subplots(2, 3 if mcmc_mean_grid is not None else 2, figsize=(18, 10))

    ax = axes[0, 0]
    im = ax.contourf(Xg, Yg, true_grid, levels=30, cmap="viridis")
    fig.colorbar(im, ax=ax)
    ax.set_title("True density")

    ax = axes[0, 1]
    im = ax.contourf(Xg, Yg, hk_mean_grid, levels=30, cmap="viridis")
    fig.colorbar(im, ax=ax)
    ax.set_title("HK bootstrap mean density")

    ax = axes[1, 0]
    im = ax.contourf(Xg, Yg, hk_lower_grid, levels=30, cmap="viridis")
    fig.colorbar(im, ax=ax)
    ax.set_title("HK 95% lower band")

    ax = axes[1, 1]
    im = ax.contourf(Xg, Yg, hk_upper_grid, levels=30, cmap="viridis")
    fig.colorbar(im, ax=ax)
    ax.set_title("HK 95% upper band")

    if mcmc_mean_grid is not None:
        ax = axes[0, 2]
        im = ax.contourf(Xg, Yg, mcmc_mean_grid, levels=30, cmap="viridis")
        fig.colorbar(im, ax=ax)
        ax.set_title("MCMC posterior mean density")

    plt.tight_layout()
    plt.savefig("bootstrap_hk_coverage.png", dpi=200)
    plt.close(fig)


def main():
    config = setup_config()

    print("=" * 80)
    print("Bootstrap HK Flow Experiment (Bivariate Mixture)")
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

    # Bootstrap HK flows
    B = config["n_bootstrap"]
    hk_measures: List[ParticleMeasure] = []

    print(f"\nRunning {B} bootstrap HK flow replicates...")
    for b in range(B):
        key, rep_key = jr.split(key)
        m_b = run_single_hk_replicate(
            rep_key,
            data,
            prior,
            kernel,
            prior_particles,
            config,
        )
        hk_measures.append(m_b)
        if (b + 1) % max(1, B // 4) == 0:
            print(f"  Completed {b+1}/{B} replicates")

    # Density grid
    grid_points = build_density_grid(config)
    hk_densities = hk_bootstrap_densities(hk_measures, kernel, grid_points)
    hk_mean, hk_lower, hk_upper = credible_intervals(hk_densities, alpha=0.05)

    # True density on grid
    true_density_vals = true_mixture_density(
        grid_points,
        config["true_means"],
        config["true_stds"],
        config["true_weights"],
    )

    hk_coverage = compute_coverage(true_density_vals, hk_lower, hk_upper)
    print(f"\nHK bootstrap 95% coverage (grid-based): {hk_coverage:.3f}")

    # NumPyro MCMC baseline
    mcmc_mean = mcmc_lower = mcmc_upper = None
    if HAS_NUMPYRO and config.get("use_numpyro", True):
        print("\nRunning NumPyro MCMC baseline...")
        samples = run_numpyro_mcmc(data, config)
        if samples:
            mcmc_mean, mcmc_lower, mcmc_upper = mcmc_density_from_samples(
                samples,
                grid_points,
            )
            mcmc_coverage = compute_coverage(true_density_vals, mcmc_lower, mcmc_upper)
            print(f"MCMC 95% coverage (grid-based): {mcmc_coverage:.3f}")
    else:
        print("\nNumPyro not available or disabled; skipping MCMC baseline.")

    # Plot results
    plot_bootstrap_results(
        config,
        grid_points,
        hk_mean,
        hk_lower,
        hk_upper,
        mcmc_mean,
        true_density_vals,
    )
    print("\nSaved figure 'bootstrap_hk_coverage.png'.")


if __name__ == "__main__":
    # Enable 64-bit precision for numerical stability if available
    try:
        jax.config.update("jax_enable_x64", True)
    except Exception:
        pass

    main()

