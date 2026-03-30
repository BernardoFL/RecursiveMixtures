#!/usr/bin/env python3
"""
Study B (prior regularization on/off) with uncertainty quantification over trajectories.

Runs niter independent bootstrap replicates for both arms and stores a trajectory
summary (default: W2 to a fixed empirical reference) across the stored measures.

Intended for HPC usage: save results to disk as .npz plus an optional PDF plot.
"""

from __future__ import annotations

import argparse
import os
import glob
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np

from clover_distribution import CloverDistribution
from recursive_mixtures import ParticleMeasure


def setup_config(*, fast: bool = True) -> Dict:
    import jax.numpy as jnp
    # Keep consistent defaults with hk_computational_choices.py where relevant.
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
        "prior_mc_samples": 1 if fast else 5,
        "sinkhorn_num_iters": 25 if fast else 50,
        # PY prior
        "prior_mean": jnp.array([0.0, 0.0]),
        "prior_std": 3.0,
        "py_discount": 0.2,
        "py_strength": 10.0,
        # Trajectory storage
        "store_every": 10,
        # OT reference
        "w2_reference_size": 2048,
        # Seed
        "seed": 123,
    }
    return cfg


def make_prior_and_kernel(config: Dict):
    from recursive_mixtures import GaussianPrior, PitmanYorProcessPrior, GaussianKernel
    base = GaussianPrior(
        mean=config["prior_mean"],
        std=config["prior_std"],
        dim=2,
    )
    prior = PitmanYorProcessPrior(
        base_prior=base,
        discount=float(config["py_discount"]),
        strength=float(config["py_strength"]),
    )
    kernel = GaussianKernel(bandwidth=config["kernel_bandwidth"])
    return prior, kernel


def make_hk_flow(prior, kernel, prior_particles: ParticleMeasure, config: Dict) -> HellingerKantorovichFlow:
    from recursive_mixtures import HellingerKantorovichFlow
    return HellingerKantorovichFlow(
        kernel=kernel,
        prior=prior,
        step_size=config["step_size"],
        wasserstein_weight=config["wasserstein_weight"],
        sinkhorn_reg=config["sinkhorn_reg"],
        use_sinkhorn=True,
        prior_particles=prior_particles,
        prior_flow_weight=config["prior_flow_weight"],
        use_prior_regularization=bool(config.get("use_prior_regularization", True)),
        prior_mc_samples=int(config["prior_mc_samples"]),
        sinkhorn_num_iters=int(config["sinkhorn_num_iters"]),
    )


def generate_clover_data(key: jax.Array, config: Dict) -> jax.Array:
    import jax
    clover = CloverDistribution(
        radius=float(config["clover_radius"]),
        radial_std=float(config["clover_radial_std"]),
        tangential_std=float(config["clover_tangential_std"]),
    )
    return clover.sample(key, int(config["n_data"]))


def w2_particle_to_empirical_reference(measure: ParticleMeasure, ref_points: np.ndarray) -> float:
    try:
        import ot
    except ImportError as e:
        raise ImportError("This script requires POT. Install with: pip install POT") from e
    a = np.asarray(measure.weights, dtype=np.float64).ravel()
    a = a / np.sum(a)
    m = int(ref_points.shape[0])
    b = np.full(m, 1.0 / m, dtype=np.float64)
    x = np.asarray(measure.atoms, dtype=np.float64)
    y = np.asarray(ref_points, dtype=np.float64)
    c = ot.dist(x, y, metric="sqeuclidean")
    w2_sq = float(ot.emd2(a, b, c))
    return float(np.sqrt(max(w2_sq, 0.0)))


@dataclass(frozen=True)
class Arm:
    label: str
    use_prior_regularization: bool


def run_single_replicate_with_history(
    key,
    data,
    prior,
    kernel,
    prior_particles: ParticleMeasure,
    config: Dict,
    *,
    use_prior_regularization: bool,
) -> List[ParticleMeasure]:
    """
    Run a bootstrap replicate and return a stored trajectory (history of measures).
    """
    n_data = int(data.shape[0])
    n_particles = int(config["n_particles"])
    store_every = int(config["store_every"])
    if store_every <= 0:
        raise ValueError("store_every must be positive for trajectory UQ")

    import jax.random as jr
    from recursive_mixtures.utils import bayesian_bootstrap

    key_boot, key_resample, key_init, key_flow = jr.split(key, 4)
    weights_boot = bayesian_bootstrap(key_boot, n_data)
    indices = jr.choice(
        key_resample,
        n_data,
        shape=(n_data,),
        p=weights_boot,
        replace=True,
    )
    data_boot = data[indices]

    atoms0 = prior.sample(key_init, n_particles)
    initial = ParticleMeasure.initialize(atoms0)

    cfg = dict(config)
    cfg["use_prior_regularization"] = bool(use_prior_regularization)
    flow = make_hk_flow(prior, kernel, prior_particles, cfg)

    _, history = flow.run(
        initial,
        data_boot,
        key=key_flow,
        store_every=store_every,
        n_steps=n_data,
        bootstrap_after_data=False,
    )
    return history


def summarize_w2_trajectories(
    histories: List[List[ParticleMeasure]],
    ref_points: np.ndarray,
) -> np.ndarray:
    """
    Convert histories into an array W2[it, t].
    Assumes all histories share the same number of stored checkpoints.
    """
    niter = len(histories)
    tlen = len(histories[0])
    out = np.zeros((niter, tlen), dtype=np.float64)
    for i in range(niter):
        if len(histories[i]) != tlen:
            raise ValueError("All histories must have the same length")
        for t in range(tlen):
            out[i, t] = w2_particle_to_empirical_reference(histories[i][t], ref_points)
    return out


def plot_uq(times: np.ndarray, w2: np.ndarray, label: str, ax) -> None:
    mean = np.mean(w2, axis=0)
    q10 = np.quantile(w2, 0.10, axis=0)
    q90 = np.quantile(w2, 0.90, axis=0)
    ax.plot(times, mean, linewidth=2.0, label=label)
    ax.fill_between(times, q10, q90, alpha=0.2)


def _replicate_range(
    niter: int,
    *,
    shard_index: int,
    num_shards: int,
) -> range:
    if num_shards <= 0:
        raise ValueError("--num-shards must be positive")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard_index < num_shards")
    # Round-robin assignment for good load balance.
    return range(shard_index, niter, num_shards)


def merge_uq_shards(npz_paths: List[str], out_path: str) -> None:
    """
    Merge multiple shard .npz files produced by this script.

    Assumes consistent times/config across shards. Concatenates replicates.
    """
    if not npz_paths:
        raise ValueError("No shard paths provided")
    loaded = [np.load(p, allow_pickle=True) for p in npz_paths]
    times0 = loaded[0]["times"]
    def _cfg_to_comparable(obj):
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                try:
                    out[k] = np.asarray(v).tolist()
                except Exception:
                    out[k] = v
            return out
        try:
            return np.asarray(obj).tolist()
        except Exception:
            return obj

    cfg0 = _cfg_to_comparable(loaded[0]["config"].item())
    w2_on = [d["w2_prior_on"] for d in loaded]
    w2_off = [d["w2_prior_off"] for d in loaded]
    for d in loaded[1:]:
        if not np.array_equal(times0, d["times"]):
            raise ValueError("Shard times mismatch")
        if cfg0 != _cfg_to_comparable(d["config"].item()):
            raise ValueError("Shard configs mismatch")
    np.savez(
        out_path,
        times=times0,
        w2_prior_on=np.concatenate(w2_on, axis=0),
        w2_prior_off=np.concatenate(w2_off, axis=0),
        config=cfg0,
    )


def discover_shards(merge_dir: str, pattern: str) -> List[str]:
    paths = sorted(glob.glob(os.path.join(merge_dir, pattern)))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Study B UQ: prior regularization on/off with niter trajectories"
    )
    parser.add_argument("--niter", type=int, default=100, help="Number of bootstrap replicates.")
    parser.add_argument("--n-data", type=int, default=None, help="Override n_data (default: config).")
    parser.add_argument("--store-every", type=int, default=10, help="Store every k steps (trajectory checkpoints).")
    parser.add_argument("--ref-size", type=int, default=2048, help="Empirical reference size for W2.")
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
        "--shard-index",
        type=int,
        default=0,
        help="Shard index for embarrassingly-parallel runs (0..num_shards-1).",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Number of shards / parallel tasks. Each shard runs a disjoint subset of replicates.",
    )
    parser.add_argument(
        "--merge-shards",
        type=str,
        default=None,
        help="Comma-separated list of shard .npz paths to merge; when set, merges and exits.",
    )
    parser.add_argument(
        "--merge-dir",
        type=str,
        default=None,
        help="Directory to auto-discover shard .npz files for merging (alternative to --merge-shards).",
    )
    parser.add_argument(
        "--merge-pattern",
        type=str,
        default="prior_regularization_uq_*_shard*of*.npz",
        help="Glob pattern (within --merge-dir) for shard discovery.",
    )
    parser.add_argument(
        "--merge-out",
        type=str,
        default="prior_regularization_uq_merged.npz",
        help="Output path for --merge-shards.",
    )
    args = parser.parse_args()

    if args.merge_shards is not None or args.merge_dir is not None:
        if args.merge_dir is not None:
            paths = discover_shards(args.merge_dir, args.merge_pattern)
        else:
            paths = [p.strip() for p in str(args.merge_shards).split(",") if p.strip()]
        if not paths:
            raise ValueError(
                "No shard files found. Use --merge-dir/--merge-pattern or pass a non-empty --merge-shards list."
            )
        merge_uq_shards(paths, args.merge_out)
        print(f"Saved {args.merge_out}")
        return

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
    config["w2_reference_size"] = int(args.ref_size)
    if args.n_data is not None:
        config["n_data"] = int(args.n_data)

    niter = int(args.niter)
    if niter <= 0:
        raise ValueError("--niter must be positive")

    os.makedirs(args.out_dir, exist_ok=True)

    import jax
    import jax.random as jr

    # Enable compilation cache if available (JAX >= 0.4).
    if args.compile_cache_dir:
        try:
            from jax.experimental import compilation_cache as _cc
            _cc.initialize_cache(args.compile_cache_dir)
        except Exception:
            pass

    base_seed = int(config["seed"])
    key = jr.PRNGKey(base_seed)
    key, data_key, ref_key, pp_key = jr.split(key, 4)

    data = generate_clover_data(data_key, config)
    clover = CloverDistribution(
        radius=float(config["clover_radius"]),
        radial_std=float(config["clover_radial_std"]),
        tangential_std=float(config["clover_tangential_std"]),
    )
    ref_points = np.asarray(clover.sample(ref_key, int(config["w2_reference_size"])))

    prior, kernel = make_prior_and_kernel(config)
    prior_particles = prior.to_particle_measure(pp_key, int(config["n_particles"]))

    arms = [
        Arm("Prior on", True),
        Arm("Prior off", False),
    ]

    shard_index = int(args.shard_index)
    num_shards = int(args.num_shards)
    iters = list(_replicate_range(niter, shard_index=shard_index, num_shards=num_shards))
    if not iters:
        raise ValueError("This shard received an empty replicate set; check --niter/--num-shards.")

    # Run trajectories (sharded): deterministic per-replicate keys derived from base_seed + it.
    histories_by_arm: Dict[str, List[List[ParticleMeasure]]] = {a.label: [] for a in arms}
    for it in iters:
        for arm in arms:
            # Derive keys deterministically so shards are independent and reproducible.
            k = jr.PRNGKey(base_seed + 10_000 * it + (1 if arm.use_prior_regularization else 2))
            hist = run_single_replicate_with_history(
                k,
                data,
                prior,
                kernel,
                prior_particles,
                config,
                use_prior_regularization=arm.use_prior_regularization,
            )
            histories_by_arm[arm.label].append(hist)

    # Summarize to W2 arrays
    w2_by_arm = {}
    for arm in arms:
        w2_by_arm[arm.label] = summarize_w2_trajectories(histories_by_arm[arm.label], ref_points)

    n_steps = int(config["n_data"])
    store_every = int(config["store_every"])
    times = np.arange(0, n_steps + 1, store_every, dtype=np.int64)
    if times.shape[0] != next(iter(w2_by_arm.values())).shape[1]:
        # flow.run includes the initial measure at t=0; stored every k steps thereafter
        times = np.arange(0, store_every * next(iter(w2_by_arm.values())).shape[1], store_every, dtype=np.int64)

    out_npz = os.path.join(
        args.out_dir,
        f"prior_regularization_uq_n{int(config['n_data'])}"
        f"_niter{niter}_store{store_every}"
        f"_shard{shard_index}of{num_shards}.npz",
    )
    np.savez(
        out_npz,
        times=times,
        w2_prior_on=w2_by_arm["Prior on"],
        w2_prior_off=w2_by_arm["Prior off"],
        config=config,
        shard_index=shard_index,
        num_shards=num_shards,
        replicate_indices=np.asarray(iters, dtype=np.int64),
    )
    print(f"Saved {out_npz}")

    if not args.no_plot:
        out_pdf = os.path.join(
            args.out_dir,
            f"prior_regularization_uq_n{int(config['n_data'])}_niter{niter}"
            f"_shard{shard_index}of{num_shards}.pdf",
        )
        fig, ax = plt.subplots(figsize=(10.0, 5.5))
        plot_uq(times, w2_by_arm["Prior on"], "Prior on", ax)
        plot_uq(times, w2_by_arm["Prior off"], "Prior off", ax)
        ax.set_xlabel("step")
        ax.set_ylabel(r"$W_2(\hat\\mu_t,\\,\\mu_{\\mathrm{ref}})$")
        ax.set_title("Study B UQ: W2 trajectories (mean with 10–90% band)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        plt.tight_layout()
        plt.savefig(out_pdf, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_pdf}")


if __name__ == "__main__":
    try:
        import jax
        jax.config.update("jax_enable_x64", True)
    except Exception:
        pass
    main()

