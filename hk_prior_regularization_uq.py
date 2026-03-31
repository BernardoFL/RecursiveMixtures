#!/usr/bin/env python3
"""
Study B (prior regularization on/off) with uncertainty quantification over trajectories.

Runs niter independent bootstrap replicates for both arms and stores a trajectory
summary (Sinkhorn divergence to a fixed empirical reference) at checkpoints.

This script is designed to be **single-process** and parallelize replicates
*inside JAX* (batched/vmap + lax.scan). This avoids multi-process JAX/XLA
compilation memory spikes on clusters.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
from jax.scipy.special import logsumexp

from clover_distribution import CloverDistribution
from recursive_mixtures import ParticleMeasure
from recursive_mixtures.utils import compute_cost_matrix, compute_sinkhorn_potentials, bayesian_bootstrap


def setup_config(*, fast: bool = True) -> Dict:
    """Defaults aligned with hk_computational_choices.py, plus HK stability params."""
    cfg = {
        # Target: clover
        "clover_radius": 1.5,
        "clover_radial_std": 0.45,
        "clover_tangential_std": 0.18,
        # Data / flow
        "n_data": 1000 if not fast else 200,
        "n_particles": 50,
        "step_size": 0.05,
        "kernel_bandwidth": 1.0,
        "sinkhorn_reg": 0.05,
        "wasserstein_weight": 0.1,
        "prior_flow_weight": 0.1,
        "sinkhorn_num_iters": 25 if fast else 50,
        # HK stability knobs (match defaults in HellingerKantorovichFlow)
        "atom_noise_std": 0.05,
        "log_weight_clip": 5.0,
        "ess_threshold": 0.5,
        "resample_jitter": 0.1,
        "weight_floor": 0.0,
        # PY prior
        "prior_mean": jnp.array([0.0, 0.0]),
        "prior_std": 3.0,
        "py_discount": 0.2,
        "py_strength": 10.0,
        # Trajectory checkpoints
        "store_every": 10,
        # Reference measure size (empirical clover samples)
        "ref_size": 2048,
        # Seed
        "seed": 123,
    }
    return cfg


def make_prior_and_kernel(config: Dict):
    # Kept for compatibility with other scripts; batched UQ runner does not
    # pass Python objects into JIT.
    from recursive_mixtures import GaussianKernel, GaussianPrior, PitmanYorProcessPrior

    base = GaussianPrior(mean=config["prior_mean"], std=config["prior_std"], dim=2)
    prior = PitmanYorProcessPrior(
        base_prior=base,
        discount=float(config["py_discount"]),
        strength=float(config["py_strength"]),
    )
    kernel = GaussianKernel(bandwidth=float(config["kernel_bandwidth"]))
    return prior, kernel


def _weights_from_logw(logw: jax.Array) -> jax.Array:
    return jnp.exp(logw - logsumexp(logw, axis=-1, keepdims=True))


def _normalize_logw(logw: jax.Array) -> jax.Array:
    return logw - logsumexp(logw, axis=-1, keepdims=True)


def _effective_sample_size(weights: jax.Array) -> jax.Array:
    return 1.0 / jnp.sum(weights**2, axis=-1)


def _weighted_mean(atoms: jax.Array, weights: jax.Array) -> jax.Array:
    return jnp.sum(weights[..., None] * atoms, axis=-2)


def _weighted_var(atoms: jax.Array, weights: jax.Array) -> jax.Array:
    mu = _weighted_mean(atoms, weights)
    diff = atoms - mu[..., None, :]
    return jnp.sum(weights[..., None] * diff**2, axis=-2)


def _resample_batched(key_b: jax.Array, atoms_b: jax.Array, weights_b: jax.Array, jitter_std_b: jax.Array) -> jax.Array:
    """
    Multinomial resampling per replicate.
    key_b: (B,2), atoms_b: (B,N,2), weights_b: (B,N), jitter_std_b: (B,)
    returns new_atoms: (B,N,2)
    """
    b, n, _ = atoms_b.shape

    def one(k, atoms, w, js):
        k1, k2 = jr.split(k, 2)
        idx = jr.choice(k1, n, shape=(n,), p=w, replace=True)
        out = atoms[idx]
        out = out + js * jr.normal(k2, shape=out.shape)
        return out

    return jax.vmap(one)(key_b, atoms_b, weights_b, jitter_std_b)


def _gaussian_kernel_eval(x: jax.Array, y: jax.Array, *, bandwidth: float) -> jax.Array:
    """k(x,y) for x shape (...,2), y shape (...,2)."""
    h2 = bandwidth**2
    d2 = jnp.sum((x - y) ** 2, axis=-1)
    return jnp.exp(-0.5 * d2 / h2)


def _ll_vd_single_x(*, bandwidth: float, x_b: jax.Array, atoms_b: jax.Array, weights_b: jax.Array) -> jax.Array:
    """Variational derivative of LogLikelihoodFunctional for a single x per replicate. Returns (B,N)."""

    def one(x, atoms, w):
        kx = _gaussian_kernel_eval(atoms, x[None, :], bandwidth=bandwidth)  # (N,)
        z = jnp.dot(kx, w)
        return -(kx / (z + 1e-30))

    return jax.vmap(one)(x_b, atoms_b, weights_b)


def _prior_potential_batched(
    atoms_b: jax.Array,
    weights_b: jax.Array,
    prior_atoms: jax.Array,
    prior_weights: jax.Array,
    *,
    reg: float,
    num_iters: int,
) -> jax.Array:
    """Source dual potentials f_{μ→prior} at μ atoms. Returns (B,N)."""

    def one(atoms, w):
        c = compute_cost_matrix(atoms, prior_atoms, metric="sqeuclidean")
        f, _ = compute_sinkhorn_potentials(w, prior_weights, c, reg=reg, num_iters=num_iters)
        return f

    return jax.vmap(one)(atoms_b, weights_b)


def _sinkhorn_drift_batched(
    atoms_b: jax.Array,
    weights_b: jax.Array,
    prior_atoms: jax.Array,
    prior_weights: jax.Array,
    *,
    reg: float,
    num_iters: int,
) -> jax.Array:
    """Barycentric Sinkhorn drift toward prior, shape (B,N,2)."""

    def one(atoms, w):
        c = compute_cost_matrix(atoms, prior_atoms, metric="sqeuclidean")
        f, g = compute_sinkhorn_potentials(w, prior_weights, c, reg=reg, num_iters=num_iters)
        k = jnp.exp(-c / reg)
        u = jnp.exp(f / reg)
        v = jnp.exp(g / reg)
        p = u[:, None] * k * v[None, :]
        p_norm = p / (jnp.sum(p, axis=1, keepdims=True) + 1e-30)
        bary = p_norm @ prior_atoms
        return bary - atoms

    return jax.vmap(one)(atoms_b, weights_b)


def _velocity_batched(*, bandwidth: float, x_b: jax.Array, atoms_b: jax.Array, weights_b: jax.Array) -> jax.Array:
    """Wasserstein velocity v_i for one data point per replicate. Returns (B,N,2)."""

    def one(x, atoms, w):
        # k(x,θ_i)
        kx = _gaussian_kernel_eval(atoms, x[None, :], bandwidth=bandwidth)  # (N,)
        z = jnp.dot(kx, w)
        # ∇_θ k(x,θ) = k(x,θ) * (x-θ)/h^2
        h2 = bandwidth**2
        grad = kx[:, None] * (x[None, :] - atoms) / h2
        return grad / (z + 1e-10)

    return jax.vmap(one)(x_b, atoms_b, weights_b)


def _sinkhorn_divergence_cost(
    atoms_a: jax.Array,
    w_a: jax.Array,
    atoms_b: jax.Array,
    w_b: jax.Array,
    *,
    reg: float,
    num_iters: int,
) -> jax.Array:
    """OT_ε cost via dual potentials, scalar."""
    c = compute_cost_matrix(atoms_a, atoms_b, metric="sqeuclidean")
    f, g = compute_sinkhorn_potentials(w_a, w_b, c, reg=reg, num_iters=num_iters)
    return jnp.sum(w_a * f) + jnp.sum(w_b * g)


def _sinkhorn_divergence_batched(
    atoms_b: jax.Array,
    weights_b: jax.Array,
    ref_atoms: jax.Array,
    ref_weights: jax.Array,
    *,
    reg: float,
    num_iters: int,
) -> jax.Array:
    """Sinkhorn divergence S_ε(μ,ν) for each replicate. Returns (B,)."""
    ot_vv = _sinkhorn_divergence_cost(ref_atoms, ref_weights, ref_atoms, ref_weights, reg=reg, num_iters=num_iters)

    def one(atoms, w):
        ot_uv = _sinkhorn_divergence_cost(atoms, w, ref_atoms, ref_weights, reg=reg, num_iters=num_iters)
        ot_uu = _sinkhorn_divergence_cost(atoms, w, atoms, w, reg=reg, num_iters=num_iters)
        return ot_uv - 0.5 * ot_uu - 0.5 * ot_vv

    return jax.vmap(one)(atoms_b, weights_b)


def run_batched_sd_trajectories(
    *,
    key: jax.Array,
    data: jax.Array,
    prior_mean: jax.Array,
    prior_std: float,
    bandwidth: float,
    prior_atoms: jax.Array,
    prior_weights: jax.Array,
    ref_atoms: jax.Array,
    ref_weights: jax.Array,
    n_steps: int,
    n_particles: int,
    store_every: int,
    step_size: float,
    sinkhorn_reg: float,
    sinkhorn_num_iters: int,
    wasserstein_weight: float,
    prior_flow_weight: float,
    atom_noise_std: float,
    log_weight_clip: float,
    ess_threshold: float,
    resample_jitter: float,
    weight_floor: float,
    niter: int,
    use_prior_regularization: bool,
) -> Tuple[jax.Array, jax.Array]:
    """
    Returns:
      times: (K,)
      sd: (B, K) where B=niter and K = number of checkpoints (including t=0).
    """
    n = n_steps
    store_every = store_every
    num_iters = sinkhorn_num_iters

    # Bootstrap resampling per replicate: data_boot (B,n,2)
    key_boot, key_choice, key_init = jr.split(key, 3)
    boot_keys = jr.split(key_boot, niter)
    choice_keys = jr.split(key_choice, niter)
    init_keys = jr.split(key_init, niter)

    w_boot = jax.vmap(lambda k: bayesian_bootstrap(k, n))(boot_keys)  # (B,n)
    idx = jax.vmap(lambda k, p: jr.choice(k, n, shape=(n,), p=p, replace=True))(choice_keys, w_boot)  # (B,n)
    data_boot = jax.vmap(lambda ii: data[ii])(idx)  # (B,n,2)

    def sample_init(k):
        return prior_mean[None, :] + prior_std * jr.normal(k, shape=(n_particles, 2))

    atoms0 = jax.vmap(sample_init)(init_keys)  # (B,N,2)
    logw0 = jnp.full((niter, n_particles), -jnp.log(n_particles))
    logw0 = _normalize_logw(logw0)

    prior_w = prior_weights

    # Per-replicate RNG stream for step stochasticity
    step_keys0 = jr.split(jr.fold_in(key, 999), niter)  # (B,2)

    def step_fn(carry, inp):
        atoms, logw, step_keys = carry
        x_t = inp  # (B,2)

        w = _weights_from_logw(logw)

        # Hellinger weight update
        g = _ll_vd_single_x(bandwidth=bandwidth, x_b=x_t, atoms_b=atoms, weights_b=w)  # (B,N)
        # Always compute h for JIT simplicity; gated by a static boolean.
        h = _prior_potential_batched(atoms, w, prior_atoms, prior_w, reg=sinkhorn_reg, num_iters=num_iters)
        prior_wt = prior_flow_weight * (1.0 if use_prior_regularization else 0.0)
        g = g + prior_wt * h
        vbar = jnp.sum(w * g, axis=-1, keepdims=True)  # (B,1)
        delta = step_size * (g - vbar)
        delta = jnp.clip(delta, -log_weight_clip, log_weight_clip)
        new_logw = _normalize_logw(logw - delta)
        w_new = _weights_from_logw(new_logw)

        w_min = weight_floor / n_particles
        w_f = jnp.clip(w_new, w_min, 1.0)
        w_f = w_f / jnp.sum(w_f, axis=-1, keepdims=True)
        new_logw = _normalize_logw(jnp.log(w_f + 1e-30))
        w_new = _weights_from_logw(new_logw)

        # Resample if ESS ratio drops
        ess_ratio = _effective_sample_size(w_new) / n_particles  # (B,)

        # split keys per replicate
        split = jax.vmap(lambda k: jr.split(k, 3))(step_keys)  # (B,3,2)
        k_resample = split[:, 0, :]
        k_noise = split[:, 1, :]
        k_next = split[:, 2, :]

        spread = jnp.mean(jnp.sqrt(_weighted_var(atoms, w_new) + 1e-8), axis=-1)  # (B,)
        jitter_std = resample_jitter * jnp.maximum(spread, 1e-3)

        do_resample = ess_ratio < ess_threshold
        atoms_r = _resample_batched(k_resample, atoms, w_new, jitter_std)
        atoms = jnp.where(do_resample[:, None, None], atoms_r, atoms)
        logw = jnp.where(do_resample[:, None], jnp.full_like(new_logw, -jnp.log(n_particles)), new_logw)
        logw = _normalize_logw(logw)
        w = _weights_from_logw(logw)

        # Atom update
        vel = _velocity_batched(bandwidth=bandwidth, x_b=x_t, atoms_b=atoms, weights_b=w)  # (B,N,2)
        drift = _sinkhorn_drift_batched(atoms, w, prior_atoms, prior_w, reg=sinkhorn_reg, num_iters=num_iters)
        vel = vel + sinkhorn_reg * drift
        atoms = atoms + step_size * wasserstein_weight * vel
        atoms = atoms + step_size * atom_noise_std * jax.vmap(
            lambda k: jr.normal(k, shape=(n_particles, 2))
        )(k_noise)

        return (atoms, logw, k_next), None

    # Scan over time with input x_t per replicate per step
    xs = jnp.transpose(data_boot, (1, 0, 2))  # (n,B,2)
    (atomsT, logwT, _), _ = jax.lax.scan(step_fn, (atoms0, logw0, step_keys0), xs)

    # Re-run scan but record metric each step including t=0 efficiently by recomputing from states:
    def step_with_metric(carry, inp):
        atoms, logw, step_keys = carry
        w = _weights_from_logw(logw)
        sd = _sinkhorn_divergence_batched(atoms, w, ref_atoms, ref_weights, reg=sinkhorn_reg, num_iters=num_iters)
        (atoms2, logw2, keys2), _ = step_fn((atoms, logw, step_keys), inp)
        return (atoms2, logw2, keys2), sd

    (atoms_end, logw_end, _), sd_steps = jax.lax.scan(step_with_metric, (atoms0, logw0, step_keys0), xs)
    # sd_steps: (n,B) for t=0..n-1; need include final at t=n as well
    w_end = _weights_from_logw(logw_end)
    sd_final = _sinkhorn_divergence_batched(atoms_end, w_end, ref_atoms, ref_weights, reg=sinkhorn_reg, num_iters=num_iters)  # (B,)
    sd_all = jnp.concatenate([sd_steps, sd_final[None, :]], axis=0)  # (n+1,B)

    times = jnp.arange(0, n + 1, store_every)
    idx_keep = jnp.clip(times, 0, n)
    sd_keep = sd_all[idx_keep, :].T  # (B,K)
    return times, sd_keep


def plot_uq(times: np.ndarray, arr: np.ndarray, label: str, ax) -> None:
    mean = np.mean(arr, axis=0)
    q10 = np.quantile(arr, 0.10, axis=0)
    q90 = np.quantile(arr, 0.90, axis=0)
    ax.plot(times, mean, linewidth=2.0, label=label)
    ax.fill_between(times, q10, q90, alpha=0.2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Study B UQ: prior regularization on/off with niter trajectories"
    )
    parser.add_argument("--niter", type=int, default=100, help="Number of bootstrap replicates.")
    parser.add_argument("--n-data", type=int, default=None, help="Override n_data (default: config).")
    parser.add_argument("--store-every", type=int, default=10, help="Store every k steps (trajectory checkpoints).")
    parser.add_argument("--ref-size", type=int, default=2048, help="Empirical reference size for Sinkhorn divergence.")
    parser.add_argument("--out-dir", type=str, default=".", help="Output directory for .npz and plot.")
    parser.add_argument("--seed", type=int, default=123, help="Base RNG seed.")
    parser.add_argument("--full", action="store_true", help="Use slower/full config.")
    parser.add_argument("--no-plot", action="store_true", help="Skip PDF plot generation.")
    parser.add_argument(
        "--platform",
        type=str,
        choices=("cpu", "gpu"),
        default="cpu",
        help="JAX backend to use for the run mode. Merge mode never uses JAX.",
    )
    parser.add_argument(
        "--compile-cache-dir",
        type=str,
        default=None,
        help="Optional JAX compilation cache directory (recommended on clusters).",
    )
    parser.add_argument(
        "--merge-out",
        type=str,
        default="prior_regularization_uq.npz",
        help="Output path for the result .npz.",
    )
    args = parser.parse_args()

    # Cluster: set backend before importing JAX. Do not inject XLA_FLAGS here:
    # flag names differ across jaxlib/XLA builds and can abort with "Unknown flag".
    if args.platform == "cpu":
        os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
    elif args.platform == "gpu":
        os.environ.setdefault("JAX_PLATFORM_NAME", "gpu")

    if args.compile_cache_dir:
        os.makedirs(args.compile_cache_dir, exist_ok=True)
        os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", args.compile_cache_dir)

    config = setup_config(fast=not args.full)
    config["seed"] = int(args.seed)
    config["store_every"] = int(args.store_every)
    config["ref_size"] = int(args.ref_size)
    if args.n_data is not None:
        config["n_data"] = int(args.n_data)

    niter = int(args.niter)
    if niter <= 0:
        raise ValueError("--niter must be positive")

    os.makedirs(args.out_dir, exist_ok=True)

    if args.compile_cache_dir:
        try:
            from jax.experimental import compilation_cache as _cc
            _cc.initialize_cache(args.compile_cache_dir)
        except Exception:
            pass

    base_seed = int(config["seed"])
    key = jr.PRNGKey(base_seed)
    key, data_key, ref_key, pp_key, run_key = jr.split(key, 5)

    clover = CloverDistribution(
        radius=float(config["clover_radius"]),
        radial_std=float(config["clover_radial_std"]),
        tangential_std=float(config["clover_tangential_std"]),
    )
    data = clover.sample(data_key, int(config["n_data"]))
    clover = CloverDistribution(
        radius=float(config["clover_radius"]),
        radial_std=float(config["clover_radial_std"]),
        tangential_std=float(config["clover_tangential_std"]),
    )
    ref_points = clover.sample(ref_key, int(config["ref_size"]))
    ref_atoms = ref_points
    ref_weights = jnp.full((int(config["ref_size"]),), 1.0 / float(config["ref_size"]))

    # Prior particles: fixed empirical prior measure used for Sinkhorn drift and prior term.
    n_particles = int(config["n_particles"])
    prior_atoms = config["prior_mean"][None, :] + float(config["prior_std"]) * jr.normal(
        pp_key, shape=(n_particles, 2)
    )
    prior_weights = jnp.full((n_particles,), 1.0 / float(n_particles))

    # Compile once per arm (use_prior_regularization is static per run)
    runner = jax.jit(
        run_batched_sd_trajectories,
        static_argnames=(
            "n_steps",
            "n_particles",
            "store_every",
            "sinkhorn_num_iters",
            "niter",
            "use_prior_regularization",
        ),
    )

    times_on, sd_on = runner(
        key=jr.fold_in(run_key, 1),
        data=data,
        prior_mean=config["prior_mean"],
        prior_std=float(config["prior_std"]),
        bandwidth=float(config["kernel_bandwidth"]),
        prior_atoms=prior_atoms,
        prior_weights=prior_weights,
        ref_atoms=ref_atoms,
        ref_weights=ref_weights,
        n_steps=int(config["n_data"]),
        n_particles=int(config["n_particles"]),
        store_every=int(config["store_every"]),
        step_size=float(config["step_size"]),
        sinkhorn_reg=float(config["sinkhorn_reg"]),
        sinkhorn_num_iters=int(config["sinkhorn_num_iters"]),
        wasserstein_weight=float(config["wasserstein_weight"]),
        prior_flow_weight=float(config["prior_flow_weight"]),
        atom_noise_std=float(config["atom_noise_std"]),
        log_weight_clip=float(config["log_weight_clip"]),
        ess_threshold=float(config["ess_threshold"]),
        resample_jitter=float(config["resample_jitter"]),
        weight_floor=float(config["weight_floor"]),
        niter=niter,
        use_prior_regularization=True,
    )
    times_off, sd_off = runner(
        key=jr.fold_in(run_key, 2),
        data=data,
        prior_mean=config["prior_mean"],
        prior_std=float(config["prior_std"]),
        bandwidth=float(config["kernel_bandwidth"]),
        prior_atoms=prior_atoms,
        prior_weights=prior_weights,
        ref_atoms=ref_atoms,
        ref_weights=ref_weights,
        n_steps=int(config["n_data"]),
        n_particles=int(config["n_particles"]),
        store_every=int(config["store_every"]),
        step_size=float(config["step_size"]),
        sinkhorn_reg=float(config["sinkhorn_reg"]),
        sinkhorn_num_iters=int(config["sinkhorn_num_iters"]),
        wasserstein_weight=float(config["wasserstein_weight"]),
        prior_flow_weight=float(config["prior_flow_weight"]),
        atom_noise_std=float(config["atom_noise_std"]),
        log_weight_clip=float(config["log_weight_clip"]),
        ess_threshold=float(config["ess_threshold"]),
        resample_jitter=float(config["resample_jitter"]),
        weight_floor=float(config["weight_floor"]),
        niter=niter,
        use_prior_regularization=False,
    )

    if not jnp.array_equal(times_on, times_off):
        raise ValueError("Time grids mismatch between arms")

    out_npz = os.path.join(args.out_dir, args.merge_out)
    np.savez(
        out_npz,
        times=np.asarray(times_on),
        sd_prior_on=np.asarray(sd_on),
        sd_prior_off=np.asarray(sd_off),
        config=config,
    )
    print(f"Saved {out_npz}")

    if not args.no_plot:
        out_pdf = os.path.join(
            args.out_dir,
            f"prior_regularization_uq_n{int(config['n_data'])}_niter{niter}.pdf",
        )
        fig, ax = plt.subplots(figsize=(10.0, 5.5))
        plot_uq(np.asarray(times_on), np.asarray(sd_on), "Prior on", ax)
        plot_uq(np.asarray(times_on), np.asarray(sd_off), "Prior off", ax)
        ax.set_xlabel("step")
        ax.set_ylabel(r"$S_\epsilon(\hat\mu_t,\,\mu_{\mathrm{ref}})$")
        ax.set_title("Study B UQ: Sinkhorn-divergence trajectories (mean with 10–90% band)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        plt.tight_layout()
        plt.savefig(out_pdf, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_pdf}")


if __name__ == "__main__":
    try:
        jax.config.update("jax_enable_x64", True)
    except Exception:
        pass
    main()

