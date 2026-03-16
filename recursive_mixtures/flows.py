"""
Gradient flow algorithms for mixture models.

- NewtonHellingerFlow (Algorithm A): weight-only updates via Hellinger/Fisher-Rao geometry.
- NewtonFlow: recursive Bayesian weight update (convex combination).
- HellingerKantorovichFlow (Algorithm B): weight + atom updates (Hellinger + Wasserstein).
- NewtonWassersteinFlow: atom-only updates (Wasserstein direction, fixed weights).
- RepulsiveFlow (Algorithm C): HK + MMD repulsion for particle diversity.
- CovariateDependentFlow (Algorithm D): regression parameters with Langevin diffusion.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import Array

from recursive_mixtures.kernels import Kernel
from recursive_mixtures.measure import ParticleMeasure, Prior
from recursive_mixtures.functionals import (
    LogLikelihoodFunctional,
    SinkhornPriorFunctional,
)


# ---------------------------------------------------------------------------
# Shared run-loop helper
# ---------------------------------------------------------------------------

def _prepare_run(
    data_stream: Array,
    n_steps: Optional[int],
    bootstrap_after_data: bool,
    key: Optional[Array],
) -> tuple[Array, Array, list]:
    """
    Normalise data_stream and build per-step (index, subkey) sequences.

    Returns:
        data_stream: shape (T, D)
        indices:     integer index array of length total_steps
        subkeys:     list of per-step JAX keys (or Nones when key is None)
    """
    data_stream = jnp.atleast_2d(data_stream)
    if data_stream.shape[0] == 1 and data_stream.shape[1] > 1:
        data_stream = data_stream.T

    n_data = int(data_stream.shape[0])
    total = n_data if n_steps is None else int(n_steps)

    if total < 0:
        raise ValueError("n_steps must be non-negative")
    if total > n_data and not bootstrap_after_data:
        raise ValueError(
            "n_steps exceeds data size; set bootstrap_after_data=True to resample"
        )

    if total <= n_data:
        indices = jnp.arange(total)
    else:
        if key is not None:
            key, idx_key = jr.split(key)
            extra = jr.randint(idx_key, shape=(total - n_data,), minval=0, maxval=n_data)
        else:
            extra = jnp.arange(n_data, total) % n_data
        indices = jnp.concatenate([jnp.arange(n_data), extra])

    subkeys = list(jr.split(key, total)) if key is not None else [None] * total
    return data_stream, indices, subkeys


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class GradientFlow(ABC):
    """Abstract base for gradient flows on measure spaces."""

    def __init__(self, kernel: Kernel, prior: Prior, step_size: float = 0.1):
        self.kernel = kernel
        self.prior = prior
        self.step_size = step_size

    @abstractmethod
    def step(
        self,
        measure: ParticleMeasure,
        data: Array,
        key: Optional[Array] = None,
    ) -> ParticleMeasure:
        """One step of the gradient flow."""

    def run(
        self,
        measure: ParticleMeasure,
        data_stream: Array,
        key: Optional[Array] = None,
        store_every: int = 1,
        n_steps: Optional[int] = None,
        bootstrap_after_data: bool = False,
    ) -> Tuple[ParticleMeasure, list[ParticleMeasure]]:
        """
        Run the flow over a data stream.

        Args:
            measure: Initial particle measure.
            data_stream: Shape (T, D).
            key: JAX random key (required for stochastic flows).
            store_every: Append to history every this many steps.
            n_steps: Total steps; if > T requires bootstrap_after_data=True.
            bootstrap_after_data: Resample data uniformly once T is exhausted.

        Returns:
            (final_measure, history)
        """
        data_stream, indices, subkeys = _prepare_run(
            data_stream, n_steps, bootstrap_after_data, key
        )
        history = [measure]
        for t, (idx, subkey) in enumerate(zip(indices, subkeys)):
            measure = self.step(measure, data_stream[idx], subkey)
            if (t + 1) % store_every == 0:
                history.append(measure)
        return measure, history


# ---------------------------------------------------------------------------
# Newton-Hellinger Flow (Algorithm A) — weight updates only
# ---------------------------------------------------------------------------

class NewtonHellingerFlow(GradientFlow):
    """
    Fisher-Rao gradient flow: updates weights, keeps atoms fixed.

    Step 1 – Hellinger:  w_i ← w_i · exp(−α (V_i − V̄))
    Step 2 – Resample:   atoms resampled by updated weights when ESS/N < ess_threshold
    """

    def __init__(
        self,
        kernel: Kernel,
        prior: Prior,
        step_size: float = 0.1,
        resample: bool = True,
        log_weight_clip: float = 5.0,
        ess_threshold: float = 0.5,
        resample_jitter: float = 0.1,
        weight_floor: float = 0.0,
    ):
        """
        Args:
            log_weight_clip: Max absolute per-step log-weight change (Fix 2).
            ess_threshold:   Resample only when ESS/N falls below this (Fix 1).
            resample_jitter: Gaussian noise σ added to resampled atoms as a
                             fraction of current particle spread (Fix 3).
            weight_floor:    Minimum weight per particle as a multiple of 1/N (Fix 4).
        """
        super().__init__(kernel, prior, step_size)
        self._ll = LogLikelihoodFunctional(kernel)
        self.resample = resample
        self.log_weight_clip = log_weight_clip
        self.ess_threshold = ess_threshold
        self.resample_jitter = resample_jitter
        self.weight_floor = weight_floor

    def step(
        self,
        measure: ParticleMeasure,
        data: Array,
        key: Optional[Array] = None,
    ) -> ParticleMeasure:
        if self.resample and key is None:
            raise ValueError("Resampling requires a PRNG key.")
        data = jnp.atleast_2d(jnp.atleast_1d(data))
        V = self._ll.set_data(data).variational_derivative(measure)
        V_bar = jnp.dot(measure.weights, V)

        # Fix 2: clip per-step log-weight change
        delta = self.step_size * (V - V_bar)
        if self.log_weight_clip > 0:
            delta = jnp.clip(delta, -self.log_weight_clip, self.log_weight_clip)
        new_log_w = measure.log_weights - delta

        updated = ParticleMeasure(atoms=measure.atoms, log_weights=new_log_w).normalize()

        # Fix 4: weight floor
        if self.weight_floor > 0:
            updated = updated.apply_weight_floor(self.weight_floor)

        # Fix 1 + 3: ESS-triggered resampling with auto-scaled jitter
        if self.resample:
            ess_ratio = float(updated.effective_sample_size()) / updated.n_particles
            if ess_ratio < self.ess_threshold:
                spread = float(jnp.mean(jnp.sqrt(measure.variance() + 1e-8)))
                jitter = self.resample_jitter * max(spread, 1e-3)
                return updated.resample(key, jitter_std=jitter)
        return updated


# ---------------------------------------------------------------------------
# Newton Flow — recursive Bayesian weight update
# ---------------------------------------------------------------------------

class NewtonFlow(GradientFlow):
    """
    Recursive Bayesian mixture update.

    w_i^{n+1} = (1−α_n) w_i^n + α_n · k(x, θ_i) w_i^n / Z

    Atom locations are fixed; only weights evolve.
    Includes the same stability safeguards as the Hellinger flows.
    """

    def __init__(
        self,
        kernel: Kernel,
        prior: Prior,
        step_size: float = 0.1,
        alpha_fn: Optional[Callable[[int], float]] = None,
        alpha_seq: Optional[Array] = None,
        resample: bool = True,
        ess_threshold: float = 0.5,
        resample_jitter: float = 0.1,
        weight_floor: float = 0.0,
    ):
        """
        Args:
            alpha_fn: Optional schedule α_n = alpha_fn(n).
            alpha_seq: Optional pre-computed array of α_n values.
            step_size: Constant α used when no schedule is given.
            resample: Resample atoms when ESS/N < ess_threshold.
            ess_threshold: ESS/N threshold below which resampling fires.
            resample_jitter: Noise σ added to resampled atoms as a fraction
                             of current particle spread.
            weight_floor: Minimum weight per particle as a multiple of 1/N.
        """
        super().__init__(kernel, prior, step_size)
        self.alpha_fn = alpha_fn
        self.alpha_seq = alpha_seq
        self.resample = resample
        self.ess_threshold = ess_threshold
        self.resample_jitter = resample_jitter
        self.weight_floor = weight_floor

    def _get_alpha(self, n: int) -> float:
        if self.alpha_seq is not None:
            if not (0 <= n < self.alpha_seq.shape[0]):
                raise IndexError(f"alpha_seq length {self.alpha_seq.shape[0]}, got n={n}")
            return float(self.alpha_seq[n])
        if self.alpha_fn is not None:
            return float(self.alpha_fn(n))
        return float(self.step_size)

    def _newton_update(self, measure: ParticleMeasure, data: Array, alpha: float) -> ParticleMeasure:
        x = jnp.atleast_2d(jnp.atleast_1d(data))
        k_vals = self.kernel.gram(x, measure.atoms)[0]   # (N,)
        w = measure.weights
        # Use jnp.maximum for a robust floor on Z — prevents explosion when
        # all atoms are far from the data point (near-zero kernel values).
        Z = jnp.maximum(jnp.dot(w, k_vals), 1e-10)
        w_post = w * k_vals / Z
        new_w = (1.0 - alpha) * w + alpha * w_post
        # Guard log with a meaningful minimum weight before normalizing
        result = ParticleMeasure(
            atoms=measure.atoms,
            log_weights=jnp.log(jnp.maximum(new_w, 1e-30)),
        ).normalize()
        # Fix 4: weight floor
        if self.weight_floor > 0:
            result = result.apply_weight_floor(self.weight_floor)
        return result

    def _maybe_resample(
        self,
        measure: ParticleMeasure,
        pre_resample_measure: ParticleMeasure,
        key: Optional[Array],
    ) -> ParticleMeasure:
        """Apply ESS-triggered resampling with auto-scaled jitter if key is available."""
        if not self.resample or key is None:
            return measure
        ess_ratio = float(measure.effective_sample_size()) / measure.n_particles
        if ess_ratio < self.ess_threshold:
            spread = float(jnp.mean(jnp.sqrt(pre_resample_measure.variance() + 1e-8)))
            jitter = self.resample_jitter * max(spread, 1e-3)
            return measure.resample(key, jitter_std=jitter)
        return measure

    def step(
        self,
        measure: ParticleMeasure,
        data: Array,
        key: Optional[Array] = None,
    ) -> ParticleMeasure:
        if self.resample and key is None:
            raise ValueError("NewtonFlow resampling requires a PRNG key.")
        pre = measure
        updated = self._newton_update(measure, data, float(self.step_size))
        return self._maybe_resample(updated, pre, key)

    def run(
        self,
        measure: ParticleMeasure,
        data_stream: Array,
        key: Optional[Array] = None,
        store_every: int = 1,
        n_steps: Optional[int] = None,
        bootstrap_after_data: bool = False,
    ) -> Tuple[ParticleMeasure, list[ParticleMeasure]]:
        """Like GradientFlow.run but uses the α_n schedule and stability safeguards."""
        data_stream, indices, subkeys = _prepare_run(
            data_stream, n_steps, bootstrap_after_data, key
        )
        history: list[ParticleMeasure] = [measure]
        for t, (idx, subkey) in enumerate(zip(indices, subkeys)):
            pre = measure
            measure = self._newton_update(measure, data_stream[idx], self._get_alpha(t))
            measure = self._maybe_resample(measure, pre, subkey)
            if (t + 1) % store_every == 0:
                history.append(measure)
        return measure, history


# ---------------------------------------------------------------------------
# Hellinger-Kantorovich Flow (Algorithm B) — weight + atom updates
# ---------------------------------------------------------------------------

class HellingerKantorovichFlow(GradientFlow):
    """
    Combines Fisher-Rao (weight) and Wasserstein (atom) gradient flows.

    Step 1 – Hellinger:   w_i ← w_i · exp(−α (V_i − V̄))
    Step 2 – Wasserstein: θ_i ← θ_i + α · λ_W · v_i  + noise
    """

    def __init__(
        self,
        kernel: Kernel,
        prior: Prior,
        step_size: float = 0.1,
        wasserstein_weight: float = 1.0,
        sinkhorn_reg: float = 0.1,
        use_sinkhorn: bool = True,
        prior_particles: Optional[ParticleMeasure] = None,
        prior_flow_weight: float = 0.0,
        prior_mc_samples: int = 1,
        sinkhorn_num_iters: int = 30,
        atom_noise_std: float = 0.05,
        resample: bool = True,
        log_weight_clip: float = 5.0,
        ess_threshold: float = 0.5,
        resample_jitter: float = 0.1,
        weight_floor: float = 0.0,
    ):
        """
        Args:
            wasserstein_weight: Scale factor λ_W for atom drift.
            sinkhorn_reg: Entropy regularisation ε for OT.
            use_sinkhorn: Add Sinkhorn drift toward prior in atom update.
            prior_particles: Fixed prior particle measure (re-used every step).
            prior_flow_weight: Weight λ for Sinkhorn prior force in the weight update.
            prior_mc_samples: Number M of Monte Carlo prior draws per step.
            sinkhorn_num_iters: Sinkhorn iterations per OT solve.
            atom_noise_std: σ for isotropic Gaussian noise added to atom updates.
            resample: If True, resample atoms by updated weights after Hellinger step.
            log_weight_clip: Max absolute per-step log-weight change (Fix 2).
            ess_threshold:   Resample only when ESS/N falls below this (Fix 1).
            resample_jitter: Gaussian noise σ added to resampled atoms as a
                             fraction of current particle spread (Fix 3).
            weight_floor:    Minimum weight per particle as a multiple of 1/N (Fix 4).
        """
        super().__init__(kernel, prior, step_size)
        self._ll = LogLikelihoodFunctional(kernel)
        self.wasserstein_weight = wasserstein_weight
        self.sinkhorn_reg = sinkhorn_reg
        self.use_sinkhorn = use_sinkhorn
        self._prior_particles = prior_particles
        self.prior_flow_weight = prior_flow_weight
        self.prior_mc_samples = prior_mc_samples
        self.sinkhorn_num_iters = sinkhorn_num_iters
        self.atom_noise_std = atom_noise_std
        self.resample = resample
        self.log_weight_clip = log_weight_clip
        self.ess_threshold = ess_threshold
        self.resample_jitter = resample_jitter
        self.weight_floor = weight_floor

    # --- private helpers ---

    def _get_prior_particles(self, key: Array, n: int) -> ParticleMeasure:
        return self._prior_particles if self._prior_particles is not None \
            else self.prior.to_particle_measure(key, n)

    def _compute_velocity(self, measure: ParticleMeasure, data: Array) -> Array:
        """Wasserstein velocity v_i = ∇_θ k(x, θ_i) / Z, averaged over data."""
        data = jnp.atleast_2d(data)
        K = self.kernel.gram(data, measure.atoms)                        # (M, N)
        Z = jnp.dot(K, measure.weights)                                  # (M,)
        # grad_y_gram[m, i, d] = ∇_θ k(x_m, θ_i)[d] — analytical for GaussianKernel
        grad_K = self.kernel.grad_y_gram(data, measure.atoms)            # (M, N, D)
        return jnp.mean(grad_K / (Z[:, None, None] + 1e-10), axis=0)    # (N, D)

    def _compute_sinkhorn_drift(
        self, measure: ParticleMeasure, prior_measure: ParticleMeasure
    ) -> Array:
        """Sinkhorn drift ∇W₂² toward prior, shape (N, D)."""
        from recursive_mixtures.utils import wasserstein_gradient
        return wasserstein_gradient(
            measure, prior_measure,
            reg=self.sinkhorn_reg, num_iters=self.sinkhorn_num_iters,
        )

    def _add_atom_noise(self, atoms: Array, key: Optional[Array]) -> Array:
        if self.atom_noise_std <= 0:
            return atoms
        if key is None:
            raise ValueError("atom_noise_std > 0 requires a PRNG key.")
        return atoms + self.step_size * self.atom_noise_std * jr.normal(key, shape=atoms.shape)

    # --- public step ---

    def step(
        self,
        measure: ParticleMeasure,
        data: Array,
        key: Optional[Array] = None,
    ) -> ParticleMeasure:
        data = jnp.atleast_2d(jnp.atleast_1d(data))

        need_mc = self.prior_flow_weight != 0.0 and self.prior_mc_samples > 0
        n_keys = (
            (self.prior_mc_samples if need_mc else 0)
            + int(self.use_sinkhorn)
            + int(self.atom_noise_std > 0)
            + int(self.resample)
        )
        if n_keys > 0 and key is None:
            raise ValueError(
                "A PRNG key is required (prior MC sampling / Sinkhorn drift / atom noise / resample)."
            )

        # Allocate sub-keys
        mc_keys, sinkhorn_key, noise_key, resample_key = [], None, None, None
        if key is not None and n_keys > 0:
            split = jr.split(key, n_keys)
            cur = 0
            if need_mc:
                mc_keys = split[cur:cur + self.prior_mc_samples]; cur += self.prior_mc_samples
            if self.use_sinkhorn:
                sinkhorn_key = split[cur]; cur += 1
            if self.atom_noise_std > 0:
                noise_key = split[cur]; cur += 1
            if self.resample:
                resample_key = split[cur]

        # Hellinger weight update
        g = self._ll.set_data(data).variational_derivative(measure)
        if need_mc:
            prior_ms = [self.prior.to_particle_measure(k, measure.n_particles) for k in mc_keys]
            h = SinkhornPriorFunctional(
                prior_ms, self.sinkhorn_reg, self.sinkhorn_num_iters
            ).variational_derivative(measure)
            g = g + self.prior_flow_weight * h
        V_bar = jnp.dot(measure.weights, g)

        # Fix 2: clip per-step log-weight change
        delta = self.step_size * (g - V_bar)
        if self.log_weight_clip > 0:
            delta = jnp.clip(delta, -self.log_weight_clip, self.log_weight_clip)
        new_log_w = measure.log_weights - delta

        updated = ParticleMeasure(atoms=measure.atoms, log_weights=new_log_w).normalize()

        # Fix 4: weight floor
        if self.weight_floor > 0:
            updated = updated.apply_weight_floor(self.weight_floor)

        # Fix 1 + 3: ESS-triggered resampling with auto-scaled jitter
        if self.resample:
            ess_ratio = float(updated.effective_sample_size()) / updated.n_particles
            if ess_ratio < self.ess_threshold:
                spread = float(jnp.mean(jnp.sqrt(measure.variance() + 1e-8)))
                jitter = self.resample_jitter * max(spread, 1e-3)
                measure = updated.resample(resample_key, jitter_std=jitter)
            else:
                measure = updated
        else:
            measure = updated
        new_log_w = measure.log_weights

        # Wasserstein atom update (on resampled atoms if resampling triggered)
        velocity = self._compute_velocity(measure, data)
        if self.use_sinkhorn and sinkhorn_key is not None:
            prior_m = self._get_prior_particles(sinkhorn_key, measure.n_particles)
            velocity = velocity + self.sinkhorn_reg * self._compute_sinkhorn_drift(measure, prior_m)
        new_atoms = measure.atoms + self.step_size * self.wasserstein_weight * velocity
        new_atoms = self._add_atom_noise(new_atoms, noise_key)

        return ParticleMeasure(atoms=new_atoms, log_weights=new_log_w).normalize()


# ---------------------------------------------------------------------------
# Newton-Wasserstein Flow — atom updates only
# ---------------------------------------------------------------------------

class NewtonWassersteinFlow(HellingerKantorovichFlow):
    """
    Wasserstein direction only: moves atoms, keeps weights fixed.

    Inherits HK infrastructure; weight update is disabled.
    """

    def __init__(
        self,
        kernel: Kernel,
        prior: Prior,
        step_size: float = 0.1,
        wasserstein_weight: float = 1.0,
        sinkhorn_reg: float = 0.1,
        use_sinkhorn: bool = True,
        prior_particles: Optional[ParticleMeasure] = None,
        sinkhorn_num_iters: int = 30,
        atom_noise_std: float = 0.05,
    ):
        super().__init__(
            kernel=kernel, prior=prior, step_size=step_size,
            wasserstein_weight=wasserstein_weight, sinkhorn_reg=sinkhorn_reg,
            use_sinkhorn=use_sinkhorn, prior_particles=prior_particles,
            prior_flow_weight=0.0, prior_mc_samples=0,
            sinkhorn_num_iters=sinkhorn_num_iters, atom_noise_std=atom_noise_std,
            resample=False,  # weights-fixed flow: no Hellinger step, no resampling
        )

    def step(
        self,
        measure: ParticleMeasure,
        data: Array,
        key: Optional[Array] = None,
    ) -> ParticleMeasure:
        data = jnp.atleast_2d(jnp.atleast_1d(data))

        n_keys = int(self.use_sinkhorn) + int(self.atom_noise_std > 0)
        if n_keys > 0 and key is None:
            raise ValueError("A PRNG key is required (Sinkhorn drift / atom noise).")

        sinkhorn_key, noise_key = None, None
        if key is not None and n_keys > 0:
            split = jr.split(key, n_keys)
            cur = 0
            if self.use_sinkhorn:
                sinkhorn_key = split[cur]; cur += 1
            if self.atom_noise_std > 0:
                noise_key = split[cur]

        velocity = self._compute_velocity(measure, data)
        if self.use_sinkhorn and sinkhorn_key is not None:
            prior_m = self._get_prior_particles(sinkhorn_key, measure.n_particles)
            velocity = velocity + self.sinkhorn_reg * self._compute_sinkhorn_drift(measure, prior_m)

        new_atoms = measure.atoms + self.step_size * self.wasserstein_weight * velocity
        new_atoms = self._add_atom_noise(new_atoms, noise_key)

        return ParticleMeasure(atoms=new_atoms, log_weights=measure.log_weights).normalize()


# ---------------------------------------------------------------------------
# Repulsive Flow (Algorithm C) — HK + MMD repulsion
# ---------------------------------------------------------------------------

class RepulsiveFlow(HellingerKantorovichFlow):
    """
    Extends HK with an MMD repulsive term to prevent particle collapse.

    Additional drift: 2 λ_rep (μ_ρ(θ) − μ_P(θ))
    """

    def __init__(
        self,
        kernel: Kernel,
        prior: Prior,
        step_size: float = 0.1,
        wasserstein_weight: float = 1.0,
        sinkhorn_reg: float = 0.1,
        use_sinkhorn: bool = True,
        repulsion_weight: float = 0.1,
        repulsion_kernel: Optional[Kernel] = None,
        prior_particles: Optional[ParticleMeasure] = None,
        atom_noise_std: float = 0.05,
        log_weight_clip: float = 5.0,
        ess_threshold: float = 0.5,
        resample_jitter: float = 0.1,
        weight_floor: float = 0.0,
    ):
        super().__init__(
            kernel, prior, step_size, wasserstein_weight,
            sinkhorn_reg, use_sinkhorn, prior_particles,
            atom_noise_std=atom_noise_std,
            log_weight_clip=log_weight_clip,
            ess_threshold=ess_threshold,
            resample_jitter=resample_jitter,
            weight_floor=weight_floor,
        )
        self.repulsion_weight = repulsion_weight
        self.repulsion_kernel = repulsion_kernel or kernel

    def _compute_repulsive_drift(
        self, measure: ParticleMeasure, prior_measure: ParticleMeasure
    ) -> Array:
        """
        Gradient of MMD²: 2 λ_rep (μ_ρ(θ_i) − μ_P(θ_i)), shape (N, D).

        Uses grad_x_batch to compute all pairwise kernel gradients in one
        vectorised operation, avoiding nested vmap loops.

        G_rho[i,j,:] = ∇_x k(θ_i, θ_j)
        grad_rho[i,:] = Σ_j w_j ∇_x k(θ_i, θ_j)
        """
        # (N, N, D) — analytical for GaussianKernel
        G_rho = self.repulsion_kernel.grad_x_batch(measure.atoms, measure.atoms)
        grad_rho = jnp.einsum('j,ijd->id', measure.weights, G_rho)       # (N, D)

        G_P = self.repulsion_kernel.grad_x_batch(measure.atoms, prior_measure.atoms)
        grad_P = jnp.einsum('j,ijd->id', prior_measure.weights, G_P)     # (N, D)

        return 2.0 * self.repulsion_weight * (grad_rho - grad_P)

    def step(
        self,
        measure: ParticleMeasure,
        data: Array,
        key: Optional[Array] = None,
    ) -> ParticleMeasure:
        data = jnp.atleast_2d(jnp.atleast_1d(data))

        n_keys = 1 + int(self.atom_noise_std > 0) + int(self.resample)
        if (self.atom_noise_std > 0 or self.resample) and key is None:
            raise ValueError("A PRNG key is required (atom noise / resample).")

        prior_key, noise_key, resample_key = key, None, None
        if key is not None and n_keys > 1:
            split = jr.split(key, n_keys)
            prior_key = split[0]
            cur = 1
            if self.atom_noise_std > 0:
                noise_key = split[cur]; cur += 1
            if self.resample:
                resample_key = split[cur]

        # Hellinger weight update
        V = self._ll.set_data(data).variational_derivative(measure)
        V_bar = jnp.dot(measure.weights, V)

        # Fix 2: clip per-step log-weight change
        delta = self.step_size * (V - V_bar)
        if self.log_weight_clip > 0:
            delta = jnp.clip(delta, -self.log_weight_clip, self.log_weight_clip)
        new_log_w = measure.log_weights - delta

        updated = ParticleMeasure(atoms=measure.atoms, log_weights=new_log_w).normalize()

        # Fix 4: weight floor
        if self.weight_floor > 0:
            updated = updated.apply_weight_floor(self.weight_floor)

        # Fix 1 + 3: ESS-triggered resampling with auto-scaled jitter
        if self.resample:
            ess_ratio = float(updated.effective_sample_size()) / updated.n_particles
            if ess_ratio < self.ess_threshold:
                spread = float(jnp.mean(jnp.sqrt(measure.variance() + 1e-8)))
                jitter = self.resample_jitter * max(spread, 1e-3)
                measure = updated.resample(resample_key, jitter_std=jitter)
            else:
                measure = updated
        else:
            measure = updated
        new_log_w = measure.log_weights

        # Prior particles
        if prior_key is not None:
            prior_m = self._get_prior_particles(prior_key, measure.n_particles)
        else:
            prior_m = self._prior_particles or ParticleMeasure.initialize(
                self.prior.sample(jr.PRNGKey(0), measure.n_particles)
            )

        # Atom update with Sinkhorn + repulsion (on resampled atoms if resampling is on)
        velocity = self._compute_velocity(measure, data)
        if self.use_sinkhorn:
            velocity = velocity + self.sinkhorn_reg * self._compute_sinkhorn_drift(measure, prior_m)
        velocity = velocity + self._compute_repulsive_drift(measure, prior_m)
        new_atoms = measure.atoms + self.step_size * self.wasserstein_weight * velocity
        new_atoms = self._add_atom_noise(new_atoms, noise_key)

        return ParticleMeasure(atoms=new_atoms, log_weights=new_log_w).normalize()


# ---------------------------------------------------------------------------
# Covariate-Dependent Flow (Algorithm D) — regression + Langevin
# ---------------------------------------------------------------------------

class CovariateDependentFlow(GradientFlow):
    """
    Atoms are regression parameters η; likelihood depends on covariates z.

    Prediction: Φ_η(z) = η · z
    Drift:      ∇_η log k(x | Φ_η(z))
    Diffusion:  √(2λ) · ξ  (Langevin noise)
    """

    def __init__(
        self,
        kernel: Kernel,
        prior: Prior,
        step_size: float = 0.1,
        diffusion_weight: float = 0.01,
        hellinger_weight: float = 1.0,
        resample: bool = True,
        log_weight_clip: float = 5.0,
        ess_threshold: float = 0.5,
        resample_jitter: float = 0.1,
        weight_floor: float = 0.0,
    ):
        super().__init__(kernel, prior, step_size)
        self.diffusion_weight = diffusion_weight
        self.hellinger_weight = hellinger_weight
        self.resample = resample
        self.log_weight_clip = log_weight_clip
        self.ess_threshold = ess_threshold
        self.resample_jitter = resample_jitter
        self.weight_floor = weight_floor
        self._ll = LogLikelihoodFunctional(kernel)

    def _predict(self, eta: Array, z: Array) -> Array:
        return jnp.dot(z, eta)

    def _drift(self, measure: ParticleMeasure, x: Array, z: Array) -> Array:
        """∇_η log k(x | Φ_η(z)) for each atom, shape (N, D)."""
        def log_lik(eta):
            return jnp.log(self.kernel(x, self._predict(eta, z)) + 1e-30)
        return jax.vmap(jax.grad(log_lik))(measure.atoms)

    def _var_deriv(self, measure: ParticleMeasure, x: Array, z: Array) -> Array:
        """Variational derivative V_i = −k(x, μ_i) / Z, shape (N,)."""
        preds = jax.vmap(lambda eta: self._predict(eta, z))(measure.atoms)
        K = jax.vmap(lambda mu: self.kernel(x, mu))(preds)
        Z = jnp.dot(measure.weights, K)
        return -K / (Z + 1e-30)

    def step(
        self,
        measure: ParticleMeasure,
        data: Tuple[Array, Array],
        key: Optional[Array] = None,
    ) -> ParticleMeasure:
        x, z = jnp.atleast_1d(data[0]), jnp.atleast_1d(data[1])

        if self.resample and key is None:
            raise ValueError("Resampling requires a PRNG key.")

        # Split key for resample and diffusion if both needed
        resample_key, diffusion_key = None, key
        if key is not None and self.resample:
            resample_key, diffusion_key = jr.split(key)

        # Hellinger weight update
        V = self._var_deriv(measure, x, z)
        V_bar = jnp.dot(measure.weights, V)

        # Fix 2: clip per-step log-weight change
        delta = self.step_size * self.hellinger_weight * (V - V_bar)
        if self.log_weight_clip > 0:
            delta = jnp.clip(delta, -self.log_weight_clip, self.log_weight_clip)
        new_log_w = measure.log_weights - delta

        updated = ParticleMeasure(atoms=measure.atoms, log_weights=new_log_w).normalize()

        # Fix 4: weight floor
        if self.weight_floor > 0:
            updated = updated.apply_weight_floor(self.weight_floor)

        # Fix 1 + 3: ESS-triggered resampling with auto-scaled jitter
        if self.resample:
            ess_ratio = float(updated.effective_sample_size()) / updated.n_particles
            if ess_ratio < self.ess_threshold:
                spread = float(jnp.mean(jnp.sqrt(measure.variance() + 1e-8)))
                jitter = self.resample_jitter * max(spread, 1e-3)
                measure = updated.resample(resample_key, jitter_std=jitter)
            else:
                measure = updated
        else:
            measure = updated
        new_log_w = measure.log_weights

        # Drift + Langevin diffusion (on resampled atoms if resampling triggered)
        new_atoms = measure.atoms + self.step_size * self._drift(measure, x, z)
        if diffusion_key is not None and self.diffusion_weight > 0:
            noise = jr.normal(diffusion_key, shape=measure.atoms.shape)
            new_atoms = new_atoms + self.step_size * jnp.sqrt(2 * self.diffusion_weight) * noise

        return ParticleMeasure(atoms=new_atoms, log_weights=new_log_w).normalize()

    def run_regression(
        self,
        measure: ParticleMeasure,
        X: Array,
        Z: Array,
        key: Optional[Array] = None,
        store_every: int = 1,
        n_steps: Optional[int] = None,
        bootstrap_after_data: bool = False,
    ) -> Tuple[ParticleMeasure, list[ParticleMeasure]]:
        """Run on paired (X, Z) regression data with optional bootstrap continuation."""
        X = jnp.atleast_2d(jnp.atleast_1d(X))
        Z = jnp.atleast_2d(Z)
        # Use X as the driver for _prepare_run (Z indexed identically)
        X, indices, subkeys = _prepare_run(X, n_steps, bootstrap_after_data, key)
        history = [measure]
        for t, (idx, subkey) in enumerate(zip(indices, subkeys)):
            measure = self.step(measure, (X[idx], Z[idx]), subkey)
            if (t + 1) % store_every == 0:
                history.append(measure)
        return measure, history


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_flow(algorithm: str, kernel: Kernel, prior: Prior, **kwargs) -> GradientFlow:
    """
    Instantiate a flow by name.

    algorithm: 'newton' | 'newton_hellinger' | 'hk' | 'repulsive' | 'covariate'
               (and common aliases: 'a', 'b', 'c', 'd', 'hellinger_kantorovich', …)
    """
    registry = {
        "newton":                NewtonFlow,
        "newton_flow":           NewtonFlow,
        "newton_hellinger":      NewtonHellingerFlow,
        "hellinger":             NewtonHellingerFlow,
        "a":                     NewtonHellingerFlow,
        "hk":                    HellingerKantorovichFlow,
        "hellinger_kantorovich": HellingerKantorovichFlow,
        "b":                     HellingerKantorovichFlow,
        "repulsive":             RepulsiveFlow,
        "mmd":                   RepulsiveFlow,
        "c":                     RepulsiveFlow,
        "covariate":             CovariateDependentFlow,
        "regression":            CovariateDependentFlow,
        "d":                     CovariateDependentFlow,
    }
    key = algorithm.lower()
    if key not in registry:
        raise ValueError(f"Unknown algorithm '{algorithm}'. Choose from: {sorted(registry)}")
    return registry[key](kernel, prior, **kwargs)
