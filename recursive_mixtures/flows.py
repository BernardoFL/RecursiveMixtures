"""
Gradient flow algorithms for mixture models.

This module implements the core gradient flow algorithms:
- Algorithm A: Newton-Hellinger Flow (weight updates only)
- Algorithm B: Hellinger-Kantorovich Flow (weight + atom updates)
- Algorithm C: Repulsive Flow (HK + MMD interaction)
- Algorithm D: Covariate-Dependent Flow (regression parameters with Langevin)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import Array
from jax.scipy.special import logsumexp

from recursive_mixtures.kernels import Kernel, GaussianKernel
from recursive_mixtures.measure import ParticleMeasure, Prior, GaussianPrior
from recursive_mixtures.functionals import (
    Functional,
    LogLikelihoodFunctional,
    MMDFunctional,
    KLFunctional,
)


class GradientFlow(ABC):
    """
    Abstract base class for gradient flows on measure spaces.
    
    A gradient flow evolves a measure ρ_t according to the gradient
    of some energy functional F in an appropriate geometry.
    """
    
    def __init__(
        self,
        kernel: Kernel,
        prior: Prior,
        step_size: float = 0.1,
    ):
        """
        Initialize gradient flow.
        
        Args:
            kernel: Kernel for likelihood/interactions
            prior: Prior distribution
            step_size: Step size α for updates
        """
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
        """
        Perform one step of the gradient flow.
        
        Args:
            measure: Current particle measure ρ_t
            data: New data point(s), shape (D,) or (M, D)
            key: Optional JAX random key for stochastic methods
            
        Returns:
            Updated particle measure ρ_{t+1}
        """
        pass
    
    def run(
        self,
        measure: ParticleMeasure,
        data_stream: Array,
        key: Optional[Array] = None,
        store_every: int = 1,
    ) -> Tuple[ParticleMeasure, list[ParticleMeasure]]:
        """
        Run gradient flow on a stream of data.
        
        Args:
            measure: Initial particle measure
            data_stream: Data points, shape (T, D) or (T,) for 1D
            key: JAX random key
            store_every: Store history every N steps
            
        Returns:
            Tuple of (final_measure, history)
        """
        data_stream = jnp.atleast_2d(data_stream)
        if data_stream.shape[0] == 1 and data_stream.shape[1] > 1:
            # Handle 1D data passed as (1, T) instead of (T, 1)
            data_stream = data_stream.T
        
        history = [measure]
        
        if key is not None:
            keys = jr.split(key, len(data_stream))
        else:
            keys = [None] * len(data_stream)
        
        for t, (data_point, subkey) in enumerate(zip(data_stream, keys)):
            measure = self.step(measure, data_point, subkey)
            
            if (t + 1) % store_every == 0:
                history.append(measure)
        
        return measure, history


class NewtonHellingerFlow(GradientFlow):
    """
    Algorithm A: Newton-Hellinger Flow.
    
    Updates only weights while keeping atom locations fixed.
    This is the Fisher-Rao gradient flow on the space of measures.
    
    Weight update: w_i ← w_i * exp(-α * (V_i - V̄))
    
    where V_i is the variational derivative of the log-likelihood
    and V̄ is the weighted mean.
    """
    
    def __init__(
        self,
        kernel: Kernel,
        prior: Prior,
        step_size: float = 0.1,
    ):
        """
        Initialize Newton-Hellinger flow.
        
        Args:
            kernel: Likelihood kernel k(x, θ)
            prior: Prior distribution (not used in weight updates)
            step_size: Step size α
        """
        super().__init__(kernel, prior, step_size)
        self.likelihood_functional = LogLikelihoodFunctional(kernel)
    
    def step(
        self,
        measure: ParticleMeasure,
        data: Array,
        key: Optional[Array] = None,
    ) -> ParticleMeasure:
        """
        Perform Newton-Hellinger step (weight update only).
        
        Args:
            measure: Current particle measure
            data: New data point, shape (D,)
            key: Not used (deterministic)
            
        Returns:
            Updated measure with new weights
        """
        data = jnp.atleast_1d(data)
        if data.ndim == 1:
            data = data[None, :]  # Add batch dimension
        
        # Compute variational derivative at atom locations
        func = self.likelihood_functional.set_data(data)
        V = func.variational_derivative(measure)  # Shape (N,)
        
        # Compute weighted mean V̄ = Σ_i w_i V_i
        weights = measure.weights
        V_bar = jnp.sum(weights * V)
        
        # Weight update in log domain: log w_i += -α * (V_i - V̄)
        # This is the Hellinger gradient direction
        new_log_weights = measure.log_weights - self.step_size * (V - V_bar)
        
        # Return normalized measure
        return ParticleMeasure(
            atoms=measure.atoms,
            log_weights=new_log_weights,
        ).normalize()


class HellingerKantorovichFlow(GradientFlow):
    """
    Algorithm B: Hellinger-Kantorovich Flow.
    
    Updates both weights (Hellinger step) and atom locations (Wasserstein step).
    This combines the Fisher-Rao and Wasserstein geometries.
    
    Step 1 (Hellinger): w_i ← w_i * exp(-α * (V_i - V̄))
    Step 2 (Wasserstein): θ_i ← θ_i + α * v_i
    
    The velocity field v_i is computed from the kernel gradient,
    optionally regularized with Sinkhorn transport to the prior.
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
    ):
        """
        Initialize Hellinger-Kantorovich flow.
        
        Args:
            kernel: Likelihood kernel k(x, θ)
            prior: Prior distribution
            step_size: Step size α
            wasserstein_weight: Weight for atom updates (relative to Hellinger)
            sinkhorn_reg: Sinkhorn regularization parameter ε
            use_sinkhorn: Whether to add Sinkhorn regularization drift
            prior_particles: Particle representation of prior (computed if None)
        """
        super().__init__(kernel, prior, step_size)
        self.likelihood_functional = LogLikelihoodFunctional(kernel)
        self.wasserstein_weight = wasserstein_weight
        self.sinkhorn_reg = sinkhorn_reg
        self.use_sinkhorn = use_sinkhorn
        self._prior_particles = prior_particles
    
    def _get_prior_particles(
        self,
        key: Array,
        n_particles: int,
    ) -> ParticleMeasure:
        """Get or create prior particle measure."""
        if self._prior_particles is not None:
            return self._prior_particles
        return self.prior.to_particle_measure(key, n_particles)
    
    def _compute_velocity(
        self,
        measure: ParticleMeasure,
        data: Array,
    ) -> Array:
        """
        Compute velocity field v_i for atom updates.
        
        v_i = ∇_θ k(x, θ_i) / ∫ k(x, θ) dρ
        
        This is the Wasserstein gradient direction.
        
        Args:
            measure: Current particle measure
            data: Data point(s)
            
        Returns:
            Velocity at each atom, shape (N, D)
        """
        data = jnp.atleast_2d(data)
        
        # Compute normalization: Z = ∫ k(x, θ) dρ = Σ_j w_j k(x, θ_j)
        K = self.kernel.gram(data, measure.atoms)  # (M, N)
        Z = jnp.dot(K, measure.weights)  # (M,)
        
        # Compute gradient ∇_θ k(x, θ_i) for each atom
        # We need grad_y since θ_i is the second argument
        # Shape: (M, N, D)
        def grad_at_atom(theta_i):
            """Gradient w.r.t. atom location for all data points."""
            return jax.vmap(lambda x: self.kernel.grad_y(x, theta_i))(data)
        
        grad_K = jax.vmap(grad_at_atom)(measure.atoms)  # (N, M, D)
        grad_K = jnp.transpose(grad_K, (1, 0, 2))  # (M, N, D)
        
        # Velocity: average over data, normalized by Z
        # v_i = (1/M) Σ_m ∇_θ k(x_m, θ_i) / Z_m
        velocity = jnp.mean(grad_K / (Z[:, None, None] + 1e-10), axis=0)  # (N, D)
        
        return velocity
    
    def _compute_sinkhorn_drift(
        self,
        measure: ParticleMeasure,
        prior_measure: ParticleMeasure,
    ) -> Array:
        """
        Compute Sinkhorn-regularized drift toward prior.
        
        This adds a term that encourages the measure to stay close
        to the prior in Wasserstein distance.
        
        Args:
            measure: Current particle measure
            prior_measure: Target prior measure
            
        Returns:
            Drift velocities, shape (N, D)
        """
        from recursive_mixtures.utils import wasserstein_gradient
        
        return wasserstein_gradient(
            measure,
            prior_measure,
            reg=self.sinkhorn_reg,
        )
    
    def step(
        self,
        measure: ParticleMeasure,
        data: Array,
        key: Optional[Array] = None,
    ) -> ParticleMeasure:
        """
        Perform Hellinger-Kantorovich step.
        
        Args:
            measure: Current particle measure
            data: New data point, shape (D,)
            key: JAX random key (for prior sampling if needed)
            
        Returns:
            Updated measure with new weights and atom locations
        """
        data = jnp.atleast_1d(data)
        if data.ndim == 1:
            data = data[None, :]
        
        # Step 1: Hellinger (weight) update
        func = self.likelihood_functional.set_data(data)
        V = func.variational_derivative(measure)
        weights = measure.weights
        V_bar = jnp.sum(weights * V)
        
        new_log_weights = measure.log_weights - self.step_size * (V - V_bar)
        
        # Step 2: Wasserstein (atom) update
        velocity = self._compute_velocity(measure, data)
        
        # Add Sinkhorn regularization drift if enabled
        if self.use_sinkhorn and key is not None:
            key1, key2 = jr.split(key)
            prior_measure = self._get_prior_particles(key1, measure.n_particles)
            sinkhorn_drift = self._compute_sinkhorn_drift(measure, prior_measure)
            velocity = velocity + self.sinkhorn_reg * sinkhorn_drift
        
        new_atoms = measure.atoms + self.step_size * self.wasserstein_weight * velocity
        
        return ParticleMeasure(
            atoms=new_atoms,
            log_weights=new_log_weights,
        ).normalize()


class RepulsiveFlow(HellingerKantorovichFlow):
    """
    Algorithm C: Repulsive Flow with MMD interaction.
    
    Extends the Hellinger-Kantorovich flow with a repulsive term
    based on Maximum Mean Discrepancy. This encourages particle
    diversity and prevents collapse.
    
    Additional drift: 2λ_rep * (μ_ρ(θ) - μ_P(θ))
    
    where μ_ρ(θ) is the kernel mean embedding of the current measure
    and μ_P(θ) is the kernel mean embedding of the prior.
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
    ):
        """
        Initialize Repulsive flow.
        
        Args:
            kernel: Likelihood kernel
            prior: Prior distribution
            step_size: Step size α
            wasserstein_weight: Weight for atom updates
            sinkhorn_reg: Sinkhorn regularization parameter
            use_sinkhorn: Whether to use Sinkhorn regularization
            repulsion_weight: Weight λ_rep for repulsive term
            repulsion_kernel: Kernel for MMD (default: same as likelihood)
            prior_particles: Particle representation of prior
        """
        super().__init__(
            kernel, prior, step_size, wasserstein_weight,
            sinkhorn_reg, use_sinkhorn, prior_particles
        )
        self.repulsion_weight = repulsion_weight
        self.repulsion_kernel = repulsion_kernel or kernel
    
    def _compute_repulsive_drift(
        self,
        measure: ParticleMeasure,
        prior_measure: ParticleMeasure,
    ) -> Array:
        """
        Compute repulsive drift from MMD gradient.
        
        Drift = 2λ_rep * (μ_ρ(θ) - μ_P(θ))
        
        This is the gradient of MMD² with respect to atom locations.
        
        Args:
            measure: Current particle measure
            prior_measure: Prior particle measure
            
        Returns:
            Repulsive drift, shape (N, D)
        """
        # Mean embedding of current measure at atoms
        mu_rho = measure.mean_embedding(self.repulsion_kernel, measure.atoms)
        
        # Mean embedding of prior at atoms
        mu_P = prior_measure.mean_embedding(self.repulsion_kernel, measure.atoms)
        
        # The gradient w.r.t. θ_i involves kernel gradients
        # ∇_θ μ_ρ(θ) = Σ_j w_j ∇_θ k(θ, θ_j)
        def grad_embedding_at_atom(theta_i):
            """Gradient of mean embedding at theta_i."""
            # Gradient w.r.t. current measure
            grad_rho = jnp.sum(
                measure.weights[:, None] * 
                jax.vmap(lambda theta_j: self.repulsion_kernel.grad_x(theta_i, theta_j))(measure.atoms),
                axis=0
            )
            # Gradient w.r.t. prior
            grad_P = jnp.sum(
                prior_measure.weights[:, None] *
                jax.vmap(lambda theta_j: self.repulsion_kernel.grad_x(theta_i, theta_j))(prior_measure.atoms),
                axis=0
            )
            return grad_rho - grad_P
        
        # Compute for all atoms
        drift = jax.vmap(grad_embedding_at_atom)(measure.atoms)
        
        return 2.0 * self.repulsion_weight * drift
    
    def step(
        self,
        measure: ParticleMeasure,
        data: Array,
        key: Optional[Array] = None,
    ) -> ParticleMeasure:
        """
        Perform Repulsive flow step.
        
        Args:
            measure: Current particle measure
            data: New data point
            key: JAX random key
            
        Returns:
            Updated measure
        """
        data = jnp.atleast_1d(data)
        if data.ndim == 1:
            data = data[None, :]
        
        # Step 1: Hellinger (weight) update
        func = self.likelihood_functional.set_data(data)
        V = func.variational_derivative(measure)
        weights = measure.weights
        V_bar = jnp.sum(weights * V)
        
        new_log_weights = measure.log_weights - self.step_size * (V - V_bar)
        
        # Step 2: Wasserstein (atom) update with repulsion
        velocity = self._compute_velocity(measure, data)
        
        # Get prior particles
        if key is not None:
            key1, key2 = jr.split(key)
            prior_measure = self._get_prior_particles(key1, measure.n_particles)
        else:
            # Use cached or create deterministically
            prior_measure = self._prior_particles
            if prior_measure is None:
                prior_measure = ParticleMeasure.initialize(
                    self.prior.sample(jr.PRNGKey(0), measure.n_particles)
                )
        
        # Add Sinkhorn drift
        if self.use_sinkhorn:
            sinkhorn_drift = self._compute_sinkhorn_drift(measure, prior_measure)
            velocity = velocity + self.sinkhorn_reg * sinkhorn_drift
        
        # Add repulsive drift
        repulsive_drift = self._compute_repulsive_drift(measure, prior_measure)
        velocity = velocity + repulsive_drift
        
        new_atoms = measure.atoms + self.step_size * self.wasserstein_weight * velocity
        
        return ParticleMeasure(
            atoms=new_atoms,
            log_weights=new_log_weights,
        ).normalize()


class CovariateDependentFlow(GradientFlow):
    """
    Algorithm D: Covariate-Dependent Flow.
    
    Atoms are regression parameters η, and the likelihood depends
    on covariates z through a linear model: Φ_η(z) = η · z.
    
    Input: Tuple (x_t, z_t) of response and covariates.
    Drift: ∇_η log k(x | Φ_η(z)) computed via chain rule.
    Diffusion: Langevin noise √(2λ) * ξ for exploration.
    """
    
    def __init__(
        self,
        kernel: Kernel,
        prior: Prior,
        step_size: float = 0.1,
        diffusion_weight: float = 0.01,
        hellinger_weight: float = 1.0,
    ):
        """
        Initialize Covariate-Dependent flow.
        
        Args:
            kernel: Likelihood kernel k(x, μ) where μ = Φ_η(z)
            prior: Prior on regression parameters η
            step_size: Step size α
            diffusion_weight: Langevin diffusion weight λ
            hellinger_weight: Weight for Hellinger (weight) updates
        """
        super().__init__(kernel, prior, step_size)
        self.diffusion_weight = diffusion_weight
        self.hellinger_weight = hellinger_weight
        self.likelihood_functional = LogLikelihoodFunctional(kernel)
    
    def _linear_model(self, eta: Array, z: Array) -> Array:
        """
        Compute linear model Φ_η(z) = η · z.
        
        Args:
            eta: Regression parameters, shape (D_eta,)
            z: Covariates, shape (D_z,) or (D_z, D_eta) for matrix
            
        Returns:
            Prediction, shape depends on z
        """
        # For simplicity, assume z and eta have compatible shapes
        # η · z can be dot product or matrix-vector product
        return jnp.dot(z, eta)
    
    def _compute_drift(
        self,
        measure: ParticleMeasure,
        x: Array,
        z: Array,
    ) -> Array:
        """
        Compute gradient drift ∇_η log k(x | Φ_η(z)).
        
        Uses JAX autodiff with chain rule through the linear model.
        
        Args:
            measure: Current particle measure (atoms are η parameters)
            x: Response variable
            z: Covariates
            
        Returns:
            Drift velocities, shape (N, D)
        """
        def log_likelihood_at_eta(eta):
            """Log likelihood for single parameter eta."""
            mu = self._linear_model(eta, z)
            # k(x, mu) is the likelihood kernel
            return jnp.log(self.kernel(x, mu) + 1e-30)
        
        # Compute gradient for each atom
        grad_fn = jax.grad(log_likelihood_at_eta)
        drift = jax.vmap(grad_fn)(measure.atoms)
        
        return drift
    
    def _compute_variational_derivative(
        self,
        measure: ParticleMeasure,
        x: Array,
        z: Array,
    ) -> Array:
        """
        Compute variational derivative for weight updates.
        
        V_i = -k(x, Φ_{η_i}(z)) / Σ_j w_j k(x, Φ_{η_j}(z))
        """
        # Compute predictions for all atoms
        predictions = jax.vmap(lambda eta: self._linear_model(eta, z))(measure.atoms)
        
        # Compute kernel values k(x, μ_i)
        K = jax.vmap(lambda mu: self.kernel(x, mu))(predictions)  # (N,)
        
        # Normalization
        Z = jnp.sum(measure.weights * K)
        
        # Variational derivative
        V = -K / (Z + 1e-30)
        
        return V
    
    def step(
        self,
        measure: ParticleMeasure,
        data: Tuple[Array, Array],
        key: Optional[Array] = None,
    ) -> ParticleMeasure:
        """
        Perform Covariate-Dependent flow step.
        
        Args:
            measure: Current particle measure (atoms are η)
            data: Tuple (x, z) of response and covariates
            key: JAX random key for Langevin noise
            
        Returns:
            Updated measure
        """
        x, z = data
        x = jnp.atleast_1d(x)
        z = jnp.atleast_1d(z)
        
        # Step 1: Hellinger (weight) update
        V = self._compute_variational_derivative(measure, x, z)
        weights = measure.weights
        V_bar = jnp.sum(weights * V)
        
        new_log_weights = measure.log_weights - \
            self.step_size * self.hellinger_weight * (V - V_bar)
        
        # Step 2: Drift update
        drift = self._compute_drift(measure, x, z)
        
        new_atoms = measure.atoms + self.step_size * drift
        
        # Step 3: Langevin diffusion (if key provided)
        if key is not None and self.diffusion_weight > 0:
            noise = jr.normal(key, shape=measure.atoms.shape)
            diffusion = jnp.sqrt(2 * self.diffusion_weight) * noise
            new_atoms = new_atoms + self.step_size * diffusion
        
        return ParticleMeasure(
            atoms=new_atoms,
            log_weights=new_log_weights,
        ).normalize()
    
    def run_regression(
        self,
        measure: ParticleMeasure,
        X: Array,
        Z: Array,
        key: Optional[Array] = None,
        store_every: int = 1,
    ) -> Tuple[ParticleMeasure, list[ParticleMeasure]]:
        """
        Run flow on regression data (X responses, Z covariates).
        
        Args:
            measure: Initial particle measure
            X: Response variables, shape (T,) or (T, D_x)
            Z: Covariates, shape (T, D_z)
            key: JAX random key
            store_every: Store history every N steps
            
        Returns:
            Tuple of (final_measure, history)
        """
        X = jnp.atleast_1d(X)
        Z = jnp.atleast_2d(Z)
        
        if X.ndim == 1:
            X = X[:, None]
        
        history = [measure]
        
        if key is not None:
            keys = jr.split(key, len(X))
        else:
            keys = [None] * len(X)
        
        for t, (x, z, subkey) in enumerate(zip(X, Z, keys)):
            measure = self.step(measure, (x, z), subkey)
            
            if (t + 1) % store_every == 0:
                history.append(measure)
        
        return measure, history


# Convenience function for creating flows
def create_flow(
    algorithm: str,
    kernel: Kernel,
    prior: Prior,
    **kwargs,
) -> GradientFlow:
    """
    Factory function for creating gradient flows.
    
    Args:
        algorithm: One of 'newton_hellinger', 'hk', 'repulsive', 'covariate'
        kernel: Kernel function
        prior: Prior distribution
        **kwargs: Additional arguments for specific flow types
        
    Returns:
        Configured GradientFlow instance
    """
    flows = {
        'newton_hellinger': NewtonHellingerFlow,
        'hellinger': NewtonHellingerFlow,
        'a': NewtonHellingerFlow,
        'hk': HellingerKantorovichFlow,
        'hellinger_kantorovich': HellingerKantorovichFlow,
        'b': HellingerKantorovichFlow,
        'repulsive': RepulsiveFlow,
        'mmd': RepulsiveFlow,
        'c': RepulsiveFlow,
        'covariate': CovariateDependentFlow,
        'regression': CovariateDependentFlow,
        'd': CovariateDependentFlow,
    }
    
    algorithm = algorithm.lower()
    if algorithm not in flows:
        raise ValueError(f"Unknown algorithm: {algorithm}. "
                        f"Choose from: {list(flows.keys())}")
    
    return flows[algorithm](kernel, prior, **kwargs)

