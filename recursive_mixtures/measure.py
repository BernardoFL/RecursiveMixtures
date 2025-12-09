"""
Particle measures and prior distributions for gradient flows.

This module provides the ParticleMeasure class for representing
discrete probability measures on particle locations, along with
various prior distribution classes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple, Union, TYPE_CHECKING

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import Array
from jax.scipy.special import logsumexp

if TYPE_CHECKING:
    from recursive_mixtures.kernels import Kernel


@dataclass
class ParticleMeasure:
    """
    Discrete probability measure represented by weighted particles.
    
    The measure is ρ = Σ_i w_i δ_{θ_i} where:
    - θ_i are atom locations (particles)
    - w_i are weights (stored in log-domain for numerical stability)
    
    Attributes:
        atoms: Particle locations, shape (N, D)
        log_weights: Log of weights, shape (N,)
    """
    atoms: Array  # Shape (N, D)
    log_weights: Array  # Shape (N,)
    
    @classmethod
    def initialize(
        cls,
        atoms: Array,
        log_weights: Optional[Array] = None,
    ) -> ParticleMeasure:
        """
        Create a ParticleMeasure from atoms.
        
        Args:
            atoms: Particle locations, shape (N, D) or (N,) for 1D
            log_weights: Optional log weights. If None, uniform weights are used.
            
        Returns:
            Normalized ParticleMeasure
        """
        # Ensure 2D atoms
        if atoms.ndim == 1:
            atoms = atoms[:, None]
        
        n_particles = atoms.shape[0]
        
        if log_weights is None:
            # Uniform weights: log(1/N) = -log(N)
            log_weights = jnp.full(n_particles, -jnp.log(n_particles))
        
        return cls(atoms=atoms, log_weights=log_weights).normalize()
    
    @property
    def n_particles(self) -> int:
        """Number of particles."""
        return self.atoms.shape[0]
    
    @property
    def dim(self) -> int:
        """Dimension of particle space."""
        return self.atoms.shape[1]
    
    @property
    def weights(self) -> Array:
        """Get normalized weights (not in log domain)."""
        return jnp.exp(self.log_weights - logsumexp(self.log_weights))
    
    def normalize(self) -> ParticleMeasure:
        """
        Normalize weights to sum to 1.
        
        Returns:
            New ParticleMeasure with normalized weights
        """
        log_sum = logsumexp(self.log_weights)
        return ParticleMeasure(
            atoms=self.atoms,
            log_weights=self.log_weights - log_sum,
        )
    
    def resample(
        self,
        key: Array,
        n_particles: Optional[int] = None,
    ) -> ParticleMeasure:
        """
        Multinomial resampling of particles.
        
        Resamples particles according to their weights, returning
        a new measure with uniform weights.
        
        Args:
            key: JAX random key
            n_particles: Number of particles to sample (default: same as current)
            
        Returns:
            Resampled ParticleMeasure with uniform weights
        """
        if n_particles is None:
            n_particles = self.n_particles
        
        # Get normalized weights
        weights = self.weights
        
        # Sample indices according to weights
        indices = jr.choice(
            key,
            self.n_particles,
            shape=(n_particles,),
            p=weights,
            replace=True,
        )
        
        # Resample atoms
        new_atoms = self.atoms[indices]
        
        # Reset to uniform weights
        return ParticleMeasure.initialize(new_atoms)
    
    def effective_sample_size(self) -> Array:
        """
        Compute effective sample size (ESS).
        
        ESS = 1 / Σ_i w_i² is a measure of particle degeneracy.
        
        Returns:
            ESS value (between 1 and N)
        """
        weights = self.weights
        return 1.0 / jnp.sum(weights ** 2)
    
    def mean(self) -> Array:
        """
        Compute weighted mean of atoms.
        
        Returns:
            Mean, shape (D,)
        """
        return jnp.sum(self.weights[:, None] * self.atoms, axis=0)
    
    def variance(self) -> Array:
        """
        Compute weighted variance of atoms.
        
        Returns:
            Variance, shape (D,)
        """
        mean = self.mean()
        diff = self.atoms - mean
        return jnp.sum(self.weights[:, None] * diff ** 2, axis=0)
    
    def mean_embedding(self, kernel: Kernel, points: Array) -> Array:
        """
        Compute kernel mean embedding μ_ρ(x) = Σ_i w_i k(x, θ_i).
        
        Args:
            kernel: Kernel function
            points: Evaluation points, shape (M, D) or (M,) for 1D
            
        Returns:
            Embedding values, shape (M,)
        """
        # Ensure 2D points
        if points.ndim == 1:
            points = points[:, None]
        
        # Compute Gram matrix K[i, j] = k(points[i], atoms[j])
        K = kernel.gram(points, self.atoms)
        
        # Weight by particle weights
        return jnp.dot(K, self.weights)
    
    def kernel_density(
        self,
        kernel: Kernel,
        points: Array,
    ) -> Array:
        """
        Compute kernel density estimate at given points.
        
        This is essentially the mean embedding normalized appropriately.
        
        Args:
            kernel: Kernel function
            points: Evaluation points, shape (M, D) or (M,) for 1D
            
        Returns:
            Density values, shape (M,)
        """
        return self.mean_embedding(kernel, points)


class Prior(ABC):
    """
    Abstract base class for prior distributions.
    
    Priors must support sampling and log-probability evaluation.
    """
    
    @abstractmethod
    def sample(self, key: Array, n: int) -> Array:
        """
        Sample from the prior.
        
        Args:
            key: JAX random key
            n: Number of samples
            
        Returns:
            Samples, shape (n, D)
        """
        pass
    
    @abstractmethod
    def log_prob(self, x: Array) -> Array:
        """
        Evaluate log probability density.
        
        Args:
            x: Points to evaluate, shape (D,) or (N, D)
            
        Returns:
            Log probabilities, scalar or shape (N,)
        """
        pass
    
    def score(self, x: Array) -> Array:
        """
        Compute score function ∇_x log p(x).
        
        Uses JAX autodiff by default.
        
        Args:
            x: Point to evaluate, shape (D,)
            
        Returns:
            Score, shape (D,)
        """
        return jax.grad(self.log_prob)(x)
    
    def score_batch(self, X: Array) -> Array:
        """
        Compute score function for batch of points.
        
        Args:
            X: Points, shape (N, D)
            
        Returns:
            Scores, shape (N, D)
        """
        return jax.vmap(self.score)(X)


class GaussianPrior(Prior):
    """
    Multivariate Gaussian prior N(μ, σ²I).
    
    For simplicity, uses isotropic covariance.
    """
    
    def __init__(
        self,
        mean: Union[float, Array] = 0.0,
        std: Union[float, Array] = 1.0,
        dim: int = 1,
    ):
        """
        Initialize Gaussian prior.
        
        Args:
            mean: Mean (scalar or array of shape (D,))
            std: Standard deviation (scalar or array of shape (D,))
            dim: Dimension (used if mean/std are scalars)
        """
        self.mean = jnp.atleast_1d(jnp.asarray(mean))
        self.std = jnp.atleast_1d(jnp.asarray(std))
        
        # Broadcast to dimension
        if self.mean.shape[0] == 1:
            self.mean = jnp.full(dim, self.mean[0])
        if self.std.shape[0] == 1:
            self.std = jnp.full(dim, self.std[0])
        
        self.dim = self.mean.shape[0]
        self.var = self.std ** 2
    
    def sample(self, key: Array, n: int) -> Array:
        """Sample from N(μ, σ²I)."""
        samples = jr.normal(key, shape=(n, self.dim))
        return self.mean + self.std * samples
    
    def log_prob(self, x: Array) -> Array:
        """Evaluate log N(x; μ, σ²I)."""
        x = jnp.atleast_1d(x)
        if x.ndim == 1:
            diff = x - self.mean
            return -0.5 * jnp.sum(diff ** 2 / self.var) - \
                   0.5 * self.dim * jnp.log(2 * jnp.pi) - \
                   jnp.sum(jnp.log(self.std))
        else:
            # Batch evaluation
            diff = x - self.mean
            return -0.5 * jnp.sum(diff ** 2 / self.var, axis=1) - \
                   0.5 * self.dim * jnp.log(2 * jnp.pi) - \
                   jnp.sum(jnp.log(self.std))
    
    def score(self, x: Array) -> Array:
        """Analytical score: ∇ log p(x) = -(x - μ) / σ²."""
        x = jnp.atleast_1d(x)
        return -(x - self.mean) / self.var
    
    def to_particle_measure(self, key: Array, n_particles: int) -> ParticleMeasure:
        """
        Create a ParticleMeasure by sampling from this prior.
        
        Args:
            key: JAX random key
            n_particles: Number of particles
            
        Returns:
            ParticleMeasure initialized from prior samples
        """
        atoms = self.sample(key, n_particles)
        return ParticleMeasure.initialize(atoms)


class MixturePrior(Prior):
    """
    Mixture of prior distributions.
    
    p(x) = Σ_k π_k p_k(x) where p_k are component priors.
    """
    
    def __init__(
        self,
        components: list[Prior],
        weights: Optional[Array] = None,
    ):
        """
        Initialize mixture prior.
        
        Args:
            components: List of component Prior objects
            weights: Mixture weights (default: uniform)
        """
        self.components = components
        self.n_components = len(components)
        
        if weights is None:
            self.weights = jnp.ones(self.n_components) / self.n_components
        else:
            self.weights = jnp.asarray(weights)
            self.weights = self.weights / jnp.sum(self.weights)
        
        self.log_weights = jnp.log(self.weights)
    
    def sample(self, key: Array, n: int) -> Array:
        """Sample from mixture."""
        key1, key2 = jr.split(key)
        
        # Sample component assignments
        assignments = jr.choice(
            key1,
            self.n_components,
            shape=(n,),
            p=self.weights,
        )
        
        # Sample from each component
        keys = jr.split(key2, self.n_components)
        component_samples = [
            comp.sample(keys[k], n) for k, comp in enumerate(self.components)
        ]
        
        # Stack and select based on assignments
        all_samples = jnp.stack(component_samples, axis=0)  # (K, n, D)
        samples = all_samples[assignments, jnp.arange(n)]  # (n, D)
        
        return samples
    
    def log_prob(self, x: Array) -> Array:
        """Evaluate log mixture density using logsumexp."""
        x = jnp.atleast_1d(x)
        
        if x.ndim == 1:
            # Single point
            log_probs = jnp.array([
                comp.log_prob(x) for comp in self.components
            ])
            return logsumexp(self.log_weights + log_probs)
        else:
            # Batch of points
            log_probs = jnp.stack([
                comp.log_prob(x) for comp in self.components
            ], axis=0)  # (K, N)
            return logsumexp(self.log_weights[:, None] + log_probs, axis=0)


class UniformPrior(Prior):
    """
    Uniform prior on [low, high]^D.
    """
    
    def __init__(
        self,
        low: Union[float, Array] = -1.0,
        high: Union[float, Array] = 1.0,
        dim: int = 1,
    ):
        """
        Initialize uniform prior.
        
        Args:
            low: Lower bound (scalar or array)
            high: Upper bound (scalar or array)
            dim: Dimension
        """
        self.low = jnp.atleast_1d(jnp.asarray(low))
        self.high = jnp.atleast_1d(jnp.asarray(high))
        
        if self.low.shape[0] == 1:
            self.low = jnp.full(dim, self.low[0])
        if self.high.shape[0] == 1:
            self.high = jnp.full(dim, self.high[0])
        
        self.dim = self.low.shape[0]
        self.log_volume = jnp.sum(jnp.log(self.high - self.low))
    
    def sample(self, key: Array, n: int) -> Array:
        """Sample uniformly."""
        u = jr.uniform(key, shape=(n, self.dim))
        return self.low + u * (self.high - self.low)
    
    def log_prob(self, x: Array) -> Array:
        """Evaluate log uniform density."""
        x = jnp.atleast_1d(x)
        if x.ndim == 1:
            in_bounds = jnp.all((x >= self.low) & (x <= self.high))
            return jnp.where(in_bounds, -self.log_volume, -jnp.inf)
        else:
            in_bounds = jnp.all((x >= self.low) & (x <= self.high), axis=1)
            return jnp.where(in_bounds, -self.log_volume, -jnp.inf)

