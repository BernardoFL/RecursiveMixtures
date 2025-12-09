"""
Utility functions for gradient flows.

This module provides:
- Bayesian Bootstrap for data reweighting
- Optimal Transport solvers and wrappers
- Visualization utilities
- Diagnostic functions
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import Array
import numpy as np

# Import POT for optimal transport
try:
    import ot
    HAS_POT = True
except ImportError:
    HAS_POT = False
    
from recursive_mixtures.measure import ParticleMeasure


# =============================================================================
# Bayesian Bootstrap
# =============================================================================

def bayesian_bootstrap(
    key: Array,
    n_samples: int,
) -> Array:
    """
    Generate Bayesian Bootstrap weights.
    
    Returns Dirichlet(1, ..., 1) weights that can be used to reweight
    a data stream for the Continuous Bayesian Bootstrap.
    
    Args:
        key: JAX random key
        n_samples: Number of samples/weights to generate
        
    Returns:
        Weights summing to 1, shape (n_samples,)
    """
    # Dirichlet(1, ..., 1) = normalized Exponential(1) random variables
    # Equivalently, use gaps between order statistics of Uniform[0,1]
    exponentials = jr.exponential(key, shape=(n_samples,))
    return exponentials / jnp.sum(exponentials)


def weighted_data_stream(
    key: Array,
    data: Array,
    n_bootstrap: int = 1,
) -> Tuple[Array, Array]:
    """
    Create weighted data stream using Bayesian Bootstrap.
    
    Args:
        key: JAX random key
        data: Data array, shape (T, D) or (T,)
        n_bootstrap: Number of bootstrap samples
        
    Returns:
        Tuple of (data, weights) where weights shape is (n_bootstrap, T)
    """
    data = jnp.atleast_2d(data)
    n_data = data.shape[0]
    
    keys = jr.split(key, n_bootstrap)
    weights = jax.vmap(lambda k: bayesian_bootstrap(k, n_data))(keys)
    
    return data, weights


# =============================================================================
# Optimal Transport Utilities
# =============================================================================

def compute_cost_matrix(
    X: Array,
    Y: Array,
    metric: str = 'sqeuclidean',
) -> Array:
    """
    Compute cost matrix between two sets of points.
    
    Args:
        X: First set of points, shape (N, D)
        Y: Second set of points, shape (M, D)
        metric: Distance metric ('euclidean' or 'sqeuclidean')
        
    Returns:
        Cost matrix, shape (N, M)
    """
    X = jnp.atleast_2d(X)
    Y = jnp.atleast_2d(Y)
    
    # Squared Euclidean distance
    X_sqnorm = jnp.sum(X ** 2, axis=1, keepdims=True)
    Y_sqnorm = jnp.sum(Y ** 2, axis=1, keepdims=True)
    sq_dist = X_sqnorm + Y_sqnorm.T - 2 * jnp.dot(X, Y.T)
    sq_dist = jnp.maximum(sq_dist, 0)  # Numerical stability
    
    if metric == 'sqeuclidean':
        return sq_dist
    elif metric == 'euclidean':
        return jnp.sqrt(sq_dist + 1e-10)
    else:
        raise ValueError(f"Unknown metric: {metric}")


def compute_sinkhorn_potentials(
    a: Array,
    b: Array,
    M: Array,
    reg: float = 0.1,
    num_iters: int = 100,
    threshold: float = 1e-6,
) -> Tuple[Array, Array]:
    """
    Compute Sinkhorn dual potentials using JAX.
    
    Solves the entropy-regularized optimal transport problem:
    min_P <P, M> + reg * KL(P || a ⊗ b)
    
    Returns the dual potentials (f, g) such that the optimal
    transport plan is P* = diag(e^{f/reg}) K diag(e^{g/reg})
    where K = e^{-M/reg}.
    
    Args:
        a: Source distribution weights, shape (N,)
        b: Target distribution weights, shape (M,)
        M: Cost matrix, shape (N, M)
        reg: Regularization parameter ε
        num_iters: Maximum number of Sinkhorn iterations
        threshold: Convergence threshold
        
    Returns:
        Tuple of dual potentials (f, g) with shapes (N,) and (M,)
    """
    # Kernel matrix
    K = jnp.exp(-M / reg)
    
    # Initialize
    u = jnp.ones_like(a)
    v = jnp.ones_like(b)
    
    # Sinkhorn iterations
    def sinkhorn_step(carry, _):
        u, v = carry
        u_new = a / (jnp.dot(K, v) + 1e-30)
        v_new = b / (jnp.dot(K.T, u_new) + 1e-30)
        return (u_new, v_new), None
    
    (u, v), _ = jax.lax.scan(sinkhorn_step, (u, v), None, length=num_iters)
    
    # Convert to potentials
    f = reg * jnp.log(u + 1e-30)
    g = reg * jnp.log(v + 1e-30)
    
    return f, g


def compute_sinkhorn_potentials_pot(
    a: Array,
    b: Array,
    M: Array,
    reg: float = 0.1,
) -> Tuple[Array, Array]:
    """
    Compute Sinkhorn potentials using POT library.
    
    This is a wrapper around ot.sinkhorn that returns dual potentials.
    Falls back to JAX implementation if POT is not available.
    
    Args:
        a: Source distribution weights, shape (N,)
        b: Target distribution weights, shape (M,)
        M: Cost matrix, shape (N, M)
        reg: Regularization parameter
        
    Returns:
        Tuple of dual potentials (f, g)
    """
    if not HAS_POT:
        return compute_sinkhorn_potentials(a, b, M, reg)
    
    # Convert to numpy for POT
    a_np = np.asarray(a)
    b_np = np.asarray(b)
    M_np = np.asarray(M)
    
    # Compute transport plan and extract potentials
    # POT returns the plan directly, so we need to get potentials differently
    _, log = ot.sinkhorn(a_np, b_np, M_np, reg, log=True)
    
    f = jnp.array(log.get('u', np.zeros_like(a_np)))
    g = jnp.array(log.get('v', np.zeros_like(b_np)))
    
    # Convert from scaling factors to potentials
    f = reg * jnp.log(f + 1e-30)
    g = reg * jnp.log(g + 1e-30)
    
    return f, g


def compute_emd_1d(
    a: Array,
    b: Array,
    x_source: Array,
    x_target: Array,
) -> Array:
    """
    Compute 1D Earth Mover's Distance (exact).
    
    Uses the fact that in 1D, EMD can be computed via sorted distributions.
    
    Args:
        a: Source weights, shape (N,)
        b: Target weights, shape (M,)
        x_source: Source locations, shape (N,) or (N, 1)
        x_target: Target locations, shape (M,) or (M, 1)
        
    Returns:
        EMD value (scalar)
    """
    x_source = jnp.atleast_1d(x_source).ravel()
    x_target = jnp.atleast_1d(x_target).ravel()
    
    if HAS_POT:
        # Use POT's exact solver
        M = ot.dist(
            np.asarray(x_source)[:, None],
            np.asarray(x_target)[:, None],
            metric='euclidean'
        )
        emd_val = ot.emd2(np.asarray(a), np.asarray(b), M)
        return jnp.array(emd_val)
    else:
        # Fallback: compute via CDF comparison
        # Sort both distributions
        idx_s = jnp.argsort(x_source)
        idx_t = jnp.argsort(x_target)
        
        x_s = x_source[idx_s]
        a_s = a[idx_s]
        x_t = x_target[idx_t]
        b_t = b[idx_t]
        
        # Compute CDFs at all unique points
        all_x = jnp.sort(jnp.concatenate([x_s, x_t]))
        
        # This is a simplified version; full implementation would
        # interpolate CDFs properly
        cdf_s = jnp.cumsum(a_s)
        cdf_t = jnp.cumsum(b_t)
        
        # EMD = integral of |CDF_s - CDF_t|
        # Approximation using trapezoidal rule
        return jnp.sum(jnp.abs(cdf_s - cdf_t[:len(cdf_s)]))


def wasserstein_gradient(
    source: ParticleMeasure,
    target: ParticleMeasure,
    reg: float = 0.1,
) -> Array:
    """
    Compute Wasserstein gradient for atom drift toward target.
    
    The gradient is derived from the dual potentials of the
    entropy-regularized optimal transport problem.
    
    For the source measure ρ = Σ_i a_i δ_{x_i}, the gradient
    direction at x_i is approximately:
    
    v_i = -∇_x f(x_i) where f is the source dual potential
    
    This points toward mass in the target measure.
    
    Args:
        source: Source particle measure
        target: Target particle measure
        reg: Sinkhorn regularization parameter
        
    Returns:
        Gradient directions at source atoms, shape (N, D)
    """
    # Compute cost matrix
    M = compute_cost_matrix(source.atoms, target.atoms, 'sqeuclidean')
    
    # Get weights
    a = source.weights
    b = target.weights
    
    # Compute Sinkhorn potentials
    f, g = compute_sinkhorn_potentials(a, b, M, reg)
    
    # The gradient of f with respect to x_i can be approximated
    # using the transport plan
    K = jnp.exp(-M / reg)
    
    # Scaling factors
    u = jnp.exp(f / reg)
    v = jnp.exp(g / reg)
    
    # Transport plan P = diag(u) K diag(v)
    P = u[:, None] * K * v[None, :]  # (N, M)
    
    # Normalize rows to get transport weights for each source
    P_normalized = P / (jnp.sum(P, axis=1, keepdims=True) + 1e-30)
    
    # Gradient direction: barycentric projection toward target
    # v_i = Σ_j P_{ij} (y_j - x_i) / Σ_j P_{ij}
    target_barycenter = jnp.dot(P_normalized, target.atoms)  # (N, D)
    gradient = target_barycenter - source.atoms
    
    return gradient


def sinkhorn_divergence(
    source: ParticleMeasure,
    target: ParticleMeasure,
    reg: float = 0.1,
) -> Array:
    """
    Compute Sinkhorn divergence (debiased Sinkhorn loss).
    
    S(ρ, ν) = OT_ε(ρ, ν) - 0.5 * OT_ε(ρ, ρ) - 0.5 * OT_ε(ν, ν)
    
    This removes the entropic bias and gives a proper divergence.
    
    Args:
        source: Source particle measure
        target: Target particle measure
        reg: Regularization parameter
        
    Returns:
        Sinkhorn divergence (scalar)
    """
    def sinkhorn_cost(m1, m2):
        M = compute_cost_matrix(m1.atoms, m2.atoms, 'sqeuclidean')
        f, g = compute_sinkhorn_potentials(m1.weights, m2.weights, M, reg)
        return jnp.sum(m1.weights * f) + jnp.sum(m2.weights * g)
    
    # Cross term
    ot_cross = sinkhorn_cost(source, target)
    
    # Self terms
    ot_source = sinkhorn_cost(source, source)
    ot_target = sinkhorn_cost(target, target)
    
    return ot_cross - 0.5 * ot_source - 0.5 * ot_target


# =============================================================================
# Diagnostic Utilities
# =============================================================================

def effective_sample_size(log_weights: Array) -> Array:
    """
    Compute effective sample size from log weights.
    
    ESS = 1 / Σ w_i² = (Σ w_i)² / Σ w_i²
    
    For normalized weights, this is between 1 and N.
    
    Args:
        log_weights: Log of (potentially unnormalized) weights
        
    Returns:
        ESS value
    """
    from jax.scipy.special import logsumexp
    
    # Normalize
    log_w = log_weights - logsumexp(log_weights)
    w = jnp.exp(log_w)
    
    return 1.0 / jnp.sum(w ** 2)


def log_marginal_likelihood(
    measure: ParticleMeasure,
    kernel,
    data: Array,
) -> Array:
    """
    Estimate log marginal likelihood: log ∫ p(data | θ) dρ(θ).
    
    Uses importance sampling with the particle measure.
    
    Args:
        measure: Particle measure approximation to posterior
        kernel: Likelihood kernel k(x, θ)
        data: Data points, shape (M, D) or (M,)
        
    Returns:
        Log marginal likelihood estimate
    """
    from jax.scipy.special import logsumexp
    
    data = jnp.atleast_2d(data)
    
    # Compute log likelihood for each atom
    K = kernel.gram(data, measure.atoms)  # (M, N)
    
    # Log of weighted sum
    log_liks = logsumexp(measure.log_weights[None, :] + jnp.log(K + 1e-30), axis=1)
    
    # Average over data points
    return jnp.mean(log_liks)


def kernel_stein_discrepancy(
    measure: ParticleMeasure,
    target_score: callable,
    kernel,
) -> Array:
    """
    Compute Kernel Stein Discrepancy.
    
    KSD(ρ, P) measures how far ρ is from a target distribution P
    with known score function ∇ log P.
    
    Args:
        measure: Particle measure ρ
        target_score: Function computing ∇ log P(x)
        kernel: Kernel for KSD
        
    Returns:
        KSD value
    """
    atoms = measure.atoms
    weights = measure.weights
    
    # Score at atoms
    scores = jax.vmap(target_score)(atoms)  # (N, D)
    
    def ksd_kernel(x, y, sx, sy):
        """Stein kernel."""
        k = kernel(x, y)
        gx_k = kernel.grad_x(x, y)
        gy_k = kernel.grad_x(y, x)
        
        # Laplacian term (simplified for scalar output)
        hess_k = jax.hessian(lambda x_: kernel(x_, y))(x)
        lap_k = jnp.trace(hess_k)
        
        return (
            jnp.dot(sx, sy) * k +
            jnp.dot(sx, gy_k) +
            jnp.dot(sy, gx_k) +
            lap_k
        )
    
    # Compute U-statistic
    ksd_sum = 0.0
    for i in range(len(atoms)):
        for j in range(len(atoms)):
            if i != j:
                ksd_sum += weights[i] * weights[j] * ksd_kernel(
                    atoms[i], atoms[j], scores[i], scores[j]
                )
    
    return jnp.sqrt(jnp.maximum(ksd_sum, 0.0))


# =============================================================================
# Data Generation Utilities
# =============================================================================

def generate_mixture_data(
    key: Array,
    n_samples: int,
    means: Array,
    stds: Array,
    weights: Optional[Array] = None,
) -> Tuple[Array, Array]:
    """
    Generate data from a Gaussian mixture model.
    
    Args:
        key: JAX random key
        n_samples: Number of samples
        means: Component means, shape (K,) for 1D or (K, D) for multivariate
        stds: Component standard deviations, shape (K,) or (K, D)
        weights: Mixture weights (default: uniform)
        
    Returns:
        Tuple of (samples, component_assignments)
    """
    means = jnp.atleast_1d(means)
    
    # Handle 1D case: means is (K,) -> make it (K, 1)
    if means.ndim == 1:
        means = means[:, None]
    
    K = means.shape[0]
    D = means.shape[1]
    
    stds = jnp.atleast_1d(stds)
    # Expand stds to (K, D) if needed
    if stds.ndim == 1:
        if len(stds) == K:
            stds = jnp.tile(stds[:, None], (1, D))
        else:
            stds = jnp.full((K, D), stds[0])
    
    if weights is None:
        weights = jnp.ones(K) / K
    
    key1, key2 = jr.split(key)
    
    # Sample component assignments
    assignments = jr.choice(key1, K, shape=(n_samples,), p=weights)
    
    # Sample from components
    noise = jr.normal(key2, shape=(n_samples, D))
    samples = means[assignments] + stds[assignments] * noise
    
    return samples, assignments


def true_mixture_density(
    x: Array,
    means: Array,
    stds: Array,
    weights: Optional[Array] = None,
) -> Array:
    """
    Evaluate true Gaussian mixture density.
    
    Args:
        x: Evaluation points, shape (M,) or (M, D)
        means: Component means
        stds: Component standard deviations
        weights: Mixture weights
        
    Returns:
        Density values, shape (M,)
    """
    x = jnp.atleast_1d(x)
    if x.ndim == 1:
        x = x[:, None]
    
    means = jnp.atleast_1d(means)
    if means.ndim == 1:
        means = means[:, None]
    
    stds = jnp.atleast_1d(stds)
    
    K = means.shape[0]
    D = means.shape[1]
    
    if weights is None:
        weights = jnp.ones(K) / K
    
    # Expand stds if needed
    if stds.ndim == 1 and len(stds) == K:
        stds = stds[:, None]  # Make (K, 1) for broadcasting
    
    # Compute density for each component
    def component_density(mean, std):
        diff = x - mean  # (M, D)
        # For 1D, std is scalar or (1,)
        std_val = std[0] if std.ndim > 0 and len(std) == 1 else std
        return jnp.exp(-0.5 * jnp.sum(diff ** 2, axis=1) / std_val ** 2) / \
               (std_val * jnp.sqrt(2 * jnp.pi)) ** D
    
    densities = jnp.stack([
        component_density(means[k], stds[k] if stds.ndim > 0 else stds)
        for k in range(K)
    ], axis=0)  # (K, M)
    
    return jnp.sum(weights[:, None] * densities, axis=0)

