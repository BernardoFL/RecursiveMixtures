"""
Particle measures and prior distributions for gradient flows.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Union, TYPE_CHECKING

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import Array
from jax.scipy.special import logsumexp

if TYPE_CHECKING:
    from recursive_mixtures.kernels import Kernel


# ---------------------------------------------------------------------------
# Particle measure
# ---------------------------------------------------------------------------

@dataclass
class ParticleMeasure:
    """
    Discrete probability measure  ρ = Σ_i w_i δ_{θ_i}.

    Weights are stored in log-domain for numerical stability.
    """

    atoms: Array        # (N, D)
    log_weights: Array  # (N,)

    @classmethod
    def initialize(
        cls,
        atoms: Array,
        log_weights: Optional[Array] = None,
    ) -> ParticleMeasure:
        """Create a ParticleMeasure with optional log-weights (uniform if None)."""
        if atoms.ndim == 1:
            atoms = atoms[:, None]
        n = atoms.shape[0]
        if log_weights is None:
            log_weights = jnp.full(n, -jnp.log(n))
        return cls(atoms=atoms, log_weights=log_weights).normalize()

    # --- properties ---

    @property
    def n_particles(self) -> int:
        return self.atoms.shape[0]

    @property
    def dim(self) -> int:
        return self.atoms.shape[1]

    @property
    def weights(self) -> Array:
        """Normalized weights (not in log-domain)."""
        return jnp.exp(self.log_weights - logsumexp(self.log_weights))

    # --- operations ---

    def normalize(self) -> ParticleMeasure:
        """Return a new measure with log-weights normalized to sum to 1."""
        return ParticleMeasure(
            atoms=self.atoms,
            log_weights=self.log_weights - logsumexp(self.log_weights),
        )

    def resample(
        self,
        key: Array,
        n_particles: Optional[int] = None,
        jitter_std: float = 0.0,
    ) -> ParticleMeasure:
        """
        Multinomial resampling; returns uniform-weight measure.

        Args:
            key: PRNG key.
            n_particles: Number of particles to draw (default: same as current).
            jitter_std: If > 0, add N(0, jitter_std²) noise to resampled atoms
                to break ties between duplicated particles.
        """
        n = n_particles or self.n_particles
        key, jitter_key = jr.split(key)
        indices = jr.choice(key, self.n_particles, shape=(n,), p=self.weights, replace=True)
        new_atoms = self.atoms[indices]
        if jitter_std > 0:
            new_atoms = new_atoms + jitter_std * jr.normal(jitter_key, shape=new_atoms.shape)
        return ParticleMeasure.initialize(new_atoms)

    def apply_weight_floor(self, floor: float) -> ParticleMeasure:
        """
        Enforce a minimum weight of floor/N per particle and renormalize.

        Prevents complete weight degeneracy by bringing near-zero weights up
        to a minimum fraction of the uniform weight 1/N.

        Args:
            floor: Minimum weight expressed as a multiple of 1/N.
                   E.g. floor=0.1 means no particle can have weight below 0.1/N.
        """
        if floor <= 0:
            return self
        w = self.weights
        w_min = floor / self.n_particles
        w_floored = jnp.clip(w, w_min, 1.0)
        w_floored = w_floored / w_floored.sum()
        return ParticleMeasure(
            atoms=self.atoms,
            log_weights=jnp.log(w_floored + 1e-30),
        ).normalize()

    def effective_sample_size(self) -> Array:
        """ESS = 1 / Σ w_i²."""
        return 1.0 / jnp.sum(self.weights ** 2)

    def mean(self) -> Array:
        """Weighted mean of atoms, shape (D,)."""
        return jnp.sum(self.weights[:, None] * self.atoms, axis=0)

    def variance(self) -> Array:
        """Weighted variance of atoms, shape (D,)."""
        diff = self.atoms - self.mean()
        return jnp.sum(self.weights[:, None] * diff ** 2, axis=0)

    def mean_embedding(self, kernel: Kernel, points: Array) -> Array:
        """Kernel mean embedding μ_ρ(x) = Σ_i w_i k(x, θ_i), shape (M,)."""
        if points.ndim == 1:
            points = points[:, None]
        return jnp.dot(kernel.gram(points, self.atoms), self.weights)

    def kernel_density(self, kernel: Kernel, points: Array) -> Array:
        """KDE at points via mean embedding."""
        return self.mean_embedding(kernel, points)


# ---------------------------------------------------------------------------
# Prior base class
# ---------------------------------------------------------------------------

class Prior(ABC):
    """Abstract base for prior distributions."""

    @abstractmethod
    def sample(self, key: Array, n: int) -> Array:
        """Draw n samples, shape (n, D)."""

    @abstractmethod
    def log_prob(self, x: Array) -> Array:
        """Log density at x (shape (D,) → scalar, or (N, D) → (N,))."""

    def score(self, x: Array) -> Array:
        """Score ∇_x log p(x), shape (D,). Uses autodiff by default."""
        return jax.grad(self.log_prob)(x)

    def score_batch(self, X: Array) -> Array:
        """Batch score, shape (N, D)."""
        return jax.vmap(self.score)(X)

    def to_particle_measure(self, key: Array, n_particles: int) -> ParticleMeasure:
        """Sample n_particles from the prior and return as a ParticleMeasure."""
        return ParticleMeasure.initialize(self.sample(key, n_particles))


# ---------------------------------------------------------------------------
# Concrete priors
# ---------------------------------------------------------------------------

class GaussianPrior(Prior):
    """Isotropic Gaussian prior  N(μ, σ²I)."""

    def __init__(
        self,
        mean: Union[float, Array] = 0.0,
        std: Union[float, Array] = 1.0,
        dim: int = 1,
    ):
        self.mean = jnp.atleast_1d(jnp.asarray(mean))
        self.std = jnp.atleast_1d(jnp.asarray(std))
        if self.mean.shape[0] == 1:
            self.mean = jnp.full(dim, self.mean[0])
        if self.std.shape[0] == 1:
            self.std = jnp.full(dim, self.std[0])
        self.dim = self.mean.shape[0]
        self.var = self.std ** 2

    def sample(self, key: Array, n: int) -> Array:
        return self.mean + self.std * jr.normal(key, shape=(n, self.dim))

    def log_prob(self, x: Array) -> Array:
        """Works for both a single point (D,) and a batch (N, D)."""
        x = jnp.atleast_1d(x)
        diff = x - self.mean
        log_norm = 0.5 * self.dim * jnp.log(2 * jnp.pi) + jnp.sum(jnp.log(self.std))
        return -0.5 * jnp.sum(diff ** 2 / self.var, axis=-1) - log_norm

    def score(self, x: Array) -> Array:
        """Analytical score ∇ log p(x) = −(x − μ) / σ²."""
        return -(jnp.atleast_1d(x) - self.mean) / self.var


class DirichletProcessPrior(Prior):
    """
    Dirichlet-process prior DP(α, G0) centered at a base prior G0.

    Sampling uses the Pólya-urn (Chinese restaurant) predictive:
      - with prob α / (α + i): draw a new atom from G0
      - otherwise: copy a previous atom uniformly.

    log_prob proxies to G0 (a DP draw is discrete, not Lebesgue-continuous).
    """

    def __init__(self, base_prior: Prior, concentration: float = 1.0):
        if concentration <= 0:
            raise ValueError("concentration must be positive")
        self.base_prior = base_prior
        self.concentration = float(concentration)

    def sample(self, key: Array, n: int) -> Array:
        if n <= 0:
            raise ValueError("n must be positive")
        atoms: list[Array] = []
        for i in range(n):
            key, k_decision, k_base, k_copy = jr.split(key, 4)
            p_new = self.concentration / (self.concentration + i) if i > 0 else 1.0
            if i == 0 or bool(jr.uniform(k_decision) < p_new):
                atoms.append(self.base_prior.sample(k_base, 1)[0])
            else:
                atoms.append(atoms[int(jr.randint(k_copy, shape=(), minval=0, maxval=i))])
        return jnp.stack(atoms, axis=0)

    def log_prob(self, x: Array) -> Array:
        return self.base_prior.log_prob(x)


class PitmanYorProcessPrior(Prior):
    """
    Pitman–Yor process prior PY(d, θ, G0) centered at a base prior G0.

    This generalizes the Dirichlet process (DP) by adding a discount d ∈ [0, 1).
    When d = 0, PY reduces to DP(θ, G0) (θ is the DP concentration).

    Sampling uses the Chinese restaurant predictive:
      - with prob (θ + d * k) / (θ + i): draw a new atom from G0
      - otherwise: choose an existing cluster j with prob (n_j - d) / (θ + i)
        and copy its atom.

    log_prob proxies to G0 (a PY draw is discrete, not Lebesgue-continuous).
    """

    def __init__(self, base_prior: Prior, discount: float = 0.0, strength: float = 1.0):
        d = float(discount)
        theta = float(strength)
        if not (0.0 <= d < 1.0):
            raise ValueError("discount must satisfy 0 <= discount < 1")
        if theta <= -d:
            raise ValueError("strength must satisfy strength > -discount")
        self.base_prior = base_prior
        self.discount = d
        self.strength = theta

    def sample(self, key: Array, n: int) -> Array:
        if n <= 0:
            raise ValueError("n must be positive")

        atoms: list[Array] = []
        cluster_counts: list[int] = []

        for i in range(n):
            key, k_u, k_base, k_choice = jr.split(key, 4)
            if i == 0:
                atoms.append(self.base_prior.sample(k_base, 1)[0])
                cluster_counts.append(1)
                continue

            k = len(cluster_counts)
            p_new = (self.strength + self.discount * k) / (self.strength + i)

            if bool(jr.uniform(k_u) < p_new):
                atoms.append(self.base_prior.sample(k_base, 1)[0])
                cluster_counts.append(1)
            else:
                # Choose a cluster with probability proportional to (n_j - d).
                weights = jnp.asarray(cluster_counts, dtype=jnp.float64) - self.discount
                weights = weights / jnp.sum(weights)
                j = int(jr.choice(k_choice, k, shape=(), p=weights))
                atoms.append(atoms[j])
                cluster_counts[j] += 1

        return jnp.stack(atoms, axis=0)

    def log_prob(self, x: Array) -> Array:
        return self.base_prior.log_prob(x)


class MixturePrior(Prior):
    """Mixture of priors  p(x) = Σ_k π_k p_k(x)."""

    def __init__(self, components: list[Prior], weights: Optional[Array] = None):
        self.components = components
        self.n_components = len(components)
        w = jnp.ones(self.n_components) if weights is None else jnp.asarray(weights)
        self.weights = w / w.sum()
        self.log_weights = jnp.log(self.weights)

    def sample(self, key: Array, n: int) -> Array:
        k1, k2 = jr.split(key)
        assignments = jr.choice(k1, self.n_components, shape=(n,), p=self.weights)
        keys = jr.split(k2, self.n_components)
        all_samples = jnp.stack(
            [comp.sample(keys[k], n) for k, comp in enumerate(self.components)], axis=0
        )  # (K, n, D)
        return all_samples[assignments, jnp.arange(n)]

    def log_prob(self, x: Array) -> Array:
        x = jnp.atleast_1d(x)
        log_probs = jnp.stack(
            [comp.log_prob(x) for comp in self.components], axis=0
        )  # (K,) or (K, N)
        if log_probs.ndim == 1:
            return logsumexp(self.log_weights + log_probs)
        return logsumexp(self.log_weights[:, None] + log_probs, axis=0)


class UniformPrior(Prior):
    """Uniform prior on [low, high]^D."""

    def __init__(
        self,
        low: Union[float, Array] = -1.0,
        high: Union[float, Array] = 1.0,
        dim: int = 1,
    ):
        self.low = jnp.atleast_1d(jnp.asarray(low))
        self.high = jnp.atleast_1d(jnp.asarray(high))
        if self.low.shape[0] == 1:
            self.low = jnp.full(dim, self.low[0])
        if self.high.shape[0] == 1:
            self.high = jnp.full(dim, self.high[0])
        self.dim = self.low.shape[0]
        self.log_volume = jnp.sum(jnp.log(self.high - self.low))

    def sample(self, key: Array, n: int) -> Array:
        return self.low + jr.uniform(key, shape=(n, self.dim)) * (self.high - self.low)

    def log_prob(self, x: Array) -> Array:
        x = jnp.atleast_1d(x)
        in_bounds = jnp.all((x >= self.low) & (x <= self.high), axis=-1)
        return jnp.where(in_bounds, -self.log_volume, -jnp.inf)
