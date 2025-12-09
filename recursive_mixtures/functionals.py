"""
Functionals on measure spaces for gradient flows.

This module provides an extensible interface for defining
energy functionals F(ρ) and their variational derivatives δF/δρ,
which drive the gradient flow dynamics.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.special import logsumexp

if TYPE_CHECKING:
    from recursive_mixtures.kernels import Kernel
    from recursive_mixtures.measure import ParticleMeasure, Prior


class Functional(ABC):
    """
    Abstract base class for energy functionals on measure spaces.
    
    A functional F: P(Θ) → R maps probability measures to real numbers.
    The variational derivative δF/δρ(θ) determines the gradient flow dynamics.
    """
    
    @abstractmethod
    def __call__(self, measure: ParticleMeasure) -> Array:
        """
        Evaluate the functional F(ρ).
        
        Args:
            measure: The probability measure (as ParticleMeasure)
            
        Returns:
            Functional value (scalar)
        """
        pass
    
    @abstractmethod
    def variational_derivative(
        self,
        measure: ParticleMeasure,
        x: Optional[Array] = None,
    ) -> Array:
        """
        Compute variational derivative δF/δρ at atom locations.
        
        The variational derivative satisfies:
        d/dt F(ρ_t)|_{t=0} = ∫ (δF/δρ) dρ̇
        
        Args:
            measure: The probability measure
            x: Optional evaluation points. If None, evaluate at measure.atoms.
            
        Returns:
            Variational derivative values, shape (N,) or (M,)
        """
        pass
    
    def gradient_at_atoms(self, measure: ParticleMeasure) -> Array:
        """
        Compute gradient of variational derivative at atom locations.
        
        This is ∇_θ (δF/δρ)(θ), used for the Wasserstein component of flows.
        
        Args:
            measure: The probability measure
            
        Returns:
            Gradients at atom locations, shape (N, D)
        """
        def vd_at_point(theta):
            """Variational derivative at a single point."""
            # Create temporary measure to get the derivative
            return self.variational_derivative(measure, theta[None])[0]
        
        return jax.vmap(jax.grad(vd_at_point))(measure.atoms)


class LogLikelihoodFunctional(Functional):
    """
    Log-likelihood functional for data fitting.
    
    Given data x, this represents F(ρ) = -log ∫ k(x, θ) dρ(θ)
    where k is a likelihood kernel.
    
    The variational derivative is:
    δF/δρ(θ) = -k(x, θ) / ∫ k(x, θ') dρ(θ')
    """
    
    def __init__(self, kernel: Kernel, data: Optional[Array] = None):
        """
        Initialize log-likelihood functional.
        
        Args:
            kernel: Likelihood kernel k(x, θ)
            data: Data point(s), shape (D,) or (M, D)
        """
        self.kernel = kernel
        self.data = data
    
    def set_data(self, data: Array) -> LogLikelihoodFunctional:
        """Set new data point for evaluation."""
        return LogLikelihoodFunctional(self.kernel, data)
    
    def __call__(self, measure: ParticleMeasure) -> Array:
        """
        Compute -log ∫ k(x, θ) dρ(θ).
        
        Uses logsumexp for numerical stability.
        """
        if self.data is None:
            raise ValueError("Data must be set before evaluation")
        
        data = jnp.atleast_2d(self.data)
        
        # Compute k(x, θ_i) for all atoms
        # K[i, j] = k(data[i], atoms[j])
        K = self.kernel.gram(data, measure.atoms)
        
        # Log of weighted sum: log Σ_j w_j k(x_i, θ_j)
        # = logsumexp(log_w + log k)
        log_K = jnp.log(K + 1e-30)  # Add small value for stability
        log_likelihood = logsumexp(measure.log_weights[None, :] + log_K, axis=1)
        
        # Average over data points
        return -jnp.mean(log_likelihood)
    
    def variational_derivative(
        self,
        measure: ParticleMeasure,
        x: Optional[Array] = None,
    ) -> Array:
        """
        Compute δF/δρ(θ) = -k(x, θ) / ∫ k(x, θ') dρ(θ').
        """
        if self.data is None:
            raise ValueError("Data must be set before evaluation")
        
        data = jnp.atleast_2d(self.data)
        
        if x is None:
            eval_points = measure.atoms
        else:
            eval_points = jnp.atleast_2d(x)
        
        # Compute normalization: Z = ∫ k(x, θ) dρ(θ) = Σ_j w_j k(x, θ_j)
        K_norm = self.kernel.gram(data, measure.atoms)  # (M_data, N_atoms)
        Z = jnp.dot(K_norm, measure.weights)  # (M_data,)
        
        # Compute kernel values at evaluation points
        K_eval = self.kernel.gram(data, eval_points)  # (M_data, M_eval)
        
        # Variational derivative (averaged over data points)
        vd = -jnp.mean(K_eval / (Z[:, None] + 1e-30), axis=0)  # (M_eval,)
        
        return vd


class KLFunctional(Functional):
    """
    KL divergence functional to a prior: F(ρ) = KL(ρ || P).
    
    For particle measures:
    KL(ρ || P) = Σ_i w_i (log w_i - log P(θ_i))
    
    The variational derivative is:
    δF/δρ(θ) = log ρ(θ) - log P(θ) + 1
    """
    
    def __init__(self, prior: Prior):
        """
        Initialize KL functional.
        
        Args:
            prior: Prior distribution P
        """
        self.prior = prior
    
    def __call__(self, measure: ParticleMeasure) -> Array:
        """
        Compute KL(ρ || P) for particle measure.
        """
        # Log weights (normalized)
        log_w = measure.log_weights - logsumexp(measure.log_weights)
        weights = jnp.exp(log_w)
        
        # Log prior at atoms
        log_prior = self.prior.log_prob(measure.atoms)
        if log_prior.ndim == 0:
            log_prior = jnp.full(measure.n_particles, log_prior)
        
        # KL = Σ w_i (log w_i - log P(θ_i))
        # Note: This is approximate for continuous KL
        return jnp.sum(weights * (log_w - log_prior))
    
    def variational_derivative(
        self,
        measure: ParticleMeasure,
        x: Optional[Array] = None,
    ) -> Array:
        """
        Compute δKL/δρ(θ) ≈ -log P(θ).
        
        For particle measures, we approximate this using the prior density.
        """
        if x is None:
            eval_points = measure.atoms
        else:
            eval_points = jnp.atleast_2d(x)
        
        # The variational derivative is approximately -log P(θ) + const
        log_prior = self.prior.log_prob(eval_points)
        if log_prior.ndim == 0:
            log_prior = jnp.full(eval_points.shape[0], log_prior)
        
        return -log_prior


class MMDFunctional(Functional):
    """
    Maximum Mean Discrepancy (MMD) functional: F(ρ) = MMD²(ρ, P).
    
    MMD²(ρ, P) = ∫∫ k(θ, θ') dρ(θ)dρ(θ') 
                 - 2 ∫∫ k(θ, θ') dρ(θ)dP(θ')
                 + ∫∫ k(θ, θ') dP(θ)dP(θ')
    
    The variational derivative is:
    δMMD²/δρ(θ) = 2(μ_ρ(θ) - μ_P(θ))
    
    where μ_ρ(θ) = ∫ k(θ, θ') dρ(θ') is the kernel mean embedding.
    """
    
    def __init__(
        self,
        kernel: Kernel,
        target: ParticleMeasure,
    ):
        """
        Initialize MMD functional.
        
        Args:
            kernel: Kernel for MMD computation
            target: Target measure P (as ParticleMeasure)
        """
        self.kernel = kernel
        self.target = target
    
    def __call__(self, measure: ParticleMeasure) -> Array:
        """
        Compute MMD²(ρ, P).
        """
        # Gram matrices
        K_rho = self.kernel.gram(measure.atoms, measure.atoms)
        K_P = self.kernel.gram(self.target.atoms, self.target.atoms)
        K_cross = self.kernel.gram(measure.atoms, self.target.atoms)
        
        # Weights
        w_rho = measure.weights
        w_P = self.target.weights
        
        # MMD² = w_ρ' K_ρ w_ρ - 2 w_ρ' K_cross w_P + w_P' K_P w_P
        term1 = jnp.dot(w_rho, jnp.dot(K_rho, w_rho))
        term2 = -2 * jnp.dot(w_rho, jnp.dot(K_cross, w_P))
        term3 = jnp.dot(w_P, jnp.dot(K_P, w_P))
        
        return term1 + term2 + term3
    
    def variational_derivative(
        self,
        measure: ParticleMeasure,
        x: Optional[Array] = None,
    ) -> Array:
        """
        Compute δMMD²/δρ(θ) = 2(μ_ρ(θ) - μ_P(θ)).
        """
        if x is None:
            eval_points = measure.atoms
        else:
            eval_points = jnp.atleast_2d(x)
        
        # Mean embeddings at evaluation points
        mu_rho = measure.mean_embedding(self.kernel, eval_points)
        mu_P = self.target.mean_embedding(self.kernel, eval_points)
        
        return 2.0 * (mu_rho - mu_P)


class EntropyFunctional(Functional):
    """
    Negative entropy functional: F(ρ) = ∫ ρ log ρ.
    
    This encourages spread in the particle distribution.
    """
    
    def __call__(self, measure: ParticleMeasure) -> Array:
        """
        Compute entropy H(ρ) = -Σ w_i log w_i.
        
        Returns negative entropy (so minimization increases entropy).
        """
        log_w = measure.log_weights - logsumexp(measure.log_weights)
        weights = jnp.exp(log_w)
        
        # Entropy (negative so that gradient flow increases it)
        return jnp.sum(weights * log_w)
    
    def variational_derivative(
        self,
        measure: ParticleMeasure,
        x: Optional[Array] = None,
    ) -> Array:
        """
        The variational derivative of entropy w.r.t. measure.
        
        For discrete measures, this involves the log weights.
        """
        if x is None:
            # At atoms, return log weights + 1
            log_w = measure.log_weights - logsumexp(measure.log_weights)
            return log_w + 1.0
        else:
            # For arbitrary points, return zeros (entropy only acts on weights)
            return jnp.zeros(x.shape[0] if x.ndim > 1 else 1)


class CompositeFunctional(Functional):
    """
    Composite functional: F(ρ) = Σ_i λ_i F_i(ρ).
    
    Allows combining multiple functionals with different weights.
    """
    
    def __init__(
        self,
        functionals: list[Functional],
        weights: Optional[list[float]] = None,
    ):
        """
        Initialize composite functional.
        
        Args:
            functionals: List of component functionals
            weights: Weights for each functional (default: all 1.0)
        """
        self.functionals = functionals
        if weights is None:
            self.weights = [1.0] * len(functionals)
        else:
            self.weights = weights
    
    def __call__(self, measure: ParticleMeasure) -> Array:
        """Evaluate weighted sum of functionals."""
        return sum(
            w * f(measure) 
            for w, f in zip(self.weights, self.functionals)
        )
    
    def variational_derivative(
        self,
        measure: ParticleMeasure,
        x: Optional[Array] = None,
    ) -> Array:
        """Compute weighted sum of variational derivatives."""
        return sum(
            w * f.variational_derivative(measure, x)
            for w, f in zip(self.weights, self.functionals)
        )


class InteractionFunctional(Functional):
    """
    Interaction energy functional: F(ρ) = ∫∫ V(θ, θ') dρ(θ)dρ(θ').
    
    This is used for repulsive interactions when V is a repulsive potential.
    """
    
    def __init__(self, kernel: Kernel, scale: float = 1.0):
        """
        Initialize interaction functional.
        
        Args:
            kernel: Interaction kernel V(θ, θ')
            scale: Scaling factor
        """
        self.kernel = kernel
        self.scale = scale
    
    def __call__(self, measure: ParticleMeasure) -> Array:
        """
        Compute ∫∫ V(θ, θ') dρ(θ)dρ(θ').
        """
        K = self.kernel.gram(measure.atoms, measure.atoms)
        weights = measure.weights
        return self.scale * jnp.dot(weights, jnp.dot(K, weights))
    
    def variational_derivative(
        self,
        measure: ParticleMeasure,
        x: Optional[Array] = None,
    ) -> Array:
        """
        Compute δF/δρ(θ) = 2 ∫ V(θ, θ') dρ(θ').
        """
        if x is None:
            eval_points = measure.atoms
        else:
            eval_points = jnp.atleast_2d(x)
        
        # Mean embedding gives ∫ V(θ, θ') dρ(θ')
        mu = measure.mean_embedding(self.kernel, eval_points)
        return 2.0 * self.scale * mu

