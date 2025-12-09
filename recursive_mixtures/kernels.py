"""
Kernel functions for gradient flows on measure spaces.

This module provides kernel implementations with JAX autodiff support
for computing gradients ∇_x k(x, y), which are essential for the
velocity field in particle-based gradient flows.
"""

from abc import ABC, abstractmethod
from functools import partial
from typing import Union

import jax
import jax.numpy as jnp
from jax import Array


class Kernel(ABC):
    """
    Abstract base class for positive definite kernels.
    
    All kernels must implement the __call__ method for evaluation.
    The grad_x method is provided via JAX autodiff by default.
    """
    
    @abstractmethod
    def __call__(self, x: Array, y: Array) -> Array:
        """
        Evaluate kernel k(x, y).
        
        Args:
            x: First argument, shape (D,) for single point or (N, D) for batch
            y: Second argument, shape (D,) for single point or (M, D) for batch
            
        Returns:
            Kernel value(s), scalar or array depending on input shapes
        """
        pass
    
    def grad_x(self, x: Array, y: Array) -> Array:
        """
        Compute gradient ∇_x k(x, y) with respect to the first argument.
        
        Uses JAX autodiff for automatic differentiation.
        
        Args:
            x: First argument, shape (D,)
            y: Second argument, shape (D,)
            
        Returns:
            Gradient with respect to x, shape (D,)
        """
        return jax.grad(lambda x_: self(x_, y))(x)
    
    def grad_y(self, x: Array, y: Array) -> Array:
        """
        Compute gradient ∇_y k(x, y) with respect to the second argument.
        
        This is the velocity field direction for gradient flows when
        we want to move the second argument (e.g., atom locations).
        
        Args:
            x: First argument, shape (D,)
            y: Second argument, shape (D,)
            
        Returns:
            Gradient with respect to y, shape (D,)
        """
        return jax.grad(lambda y_: self(x, y_))(y)
    
    def gram(self, X: Array, Y: Array) -> Array:
        """
        Compute Gram matrix K[i,j] = k(X[i], Y[j]).
        
        Fully vectorized using jax.vmap for efficiency.
        
        Args:
            X: First set of points, shape (N, D)
            Y: Second set of points, shape (M, D)
            
        Returns:
            Gram matrix, shape (N, M)
        """
        # Vectorize over rows of X, then over rows of Y
        return jax.vmap(lambda x: jax.vmap(lambda y: self(x, y))(Y))(X)
    
    def grad_x_batch(self, X: Array, Y: Array) -> Array:
        """
        Compute gradients ∇_x k(x, y) for all pairs.
        
        Args:
            X: First set of points, shape (N, D)
            Y: Second set of points, shape (M, D)
            
        Returns:
            Gradients, shape (N, M, D)
        """
        return jax.vmap(lambda x: jax.vmap(lambda y: self.grad_x(x, y))(Y))(X)


class GaussianKernel(Kernel):
    """
    Gaussian (RBF) kernel: k(x, y) = exp(-||x - y||² / (2σ²))
    
    The bandwidth parameter σ controls the kernel width.
    """
    
    def __init__(self, bandwidth: float = 1.0):
        """
        Initialize Gaussian kernel.
        
        Args:
            bandwidth: Kernel bandwidth σ (standard deviation)
        """
        self.bandwidth = bandwidth
        self.gamma = 1.0 / (2.0 * bandwidth ** 2)
    
    def __call__(self, x: Array, y: Array) -> Array:
        """Evaluate k(x, y) = exp(-||x - y||² / (2σ²))"""
        diff = x - y
        sq_dist = jnp.sum(diff ** 2)
        return jnp.exp(-self.gamma * sq_dist)
    
    def grad_x(self, x: Array, y: Array) -> Array:
        """
        Analytical gradient: ∇_x k(x, y) = -2γ(x - y)k(x, y)
        
        More efficient than autodiff for this kernel.
        """
        diff = x - y
        k_val = self(x, y)
        return -2.0 * self.gamma * diff * k_val
    
    def grad_y(self, x: Array, y: Array) -> Array:
        """
        Analytical gradient: ∇_y k(x, y) = +2γ(x - y)k(x, y) = -∇_x k(x, y)
        
        For symmetric kernels depending on (x - y), grad_y = -grad_x.
        """
        return -self.grad_x(x, y)


class MaternKernel(Kernel):
    """
    Matérn kernel family.
    
    Supports ν = 1/2 (exponential), ν = 3/2, and ν = 5/2.
    k(x, y) depends on r = ||x - y|| / ℓ where ℓ is the lengthscale.
    """
    
    def __init__(self, lengthscale: float = 1.0, nu: float = 2.5):
        """
        Initialize Matérn kernel.
        
        Args:
            lengthscale: Lengthscale parameter ℓ
            nu: Smoothness parameter, one of {0.5, 1.5, 2.5}
        """
        if nu not in [0.5, 1.5, 2.5]:
            raise ValueError(f"nu must be 0.5, 1.5, or 2.5, got {nu}")
        self.lengthscale = lengthscale
        self.nu = nu
    
    def __call__(self, x: Array, y: Array) -> Array:
        """Evaluate Matérn kernel."""
        diff = x - y
        # Add small epsilon for numerical stability in gradient
        r = jnp.sqrt(jnp.sum(diff ** 2) + 1e-12) / self.lengthscale
        
        if self.nu == 0.5:
            # Exponential kernel: k(r) = exp(-r)
            return jnp.exp(-r)
        elif self.nu == 1.5:
            # Matérn 3/2: k(r) = (1 + √3 r) exp(-√3 r)
            sqrt3_r = jnp.sqrt(3.0) * r
            return (1.0 + sqrt3_r) * jnp.exp(-sqrt3_r)
        else:  # nu == 2.5
            # Matérn 5/2: k(r) = (1 + √5 r + 5r²/3) exp(-√5 r)
            sqrt5_r = jnp.sqrt(5.0) * r
            return (1.0 + sqrt5_r + sqrt5_r ** 2 / 3.0) * jnp.exp(-sqrt5_r)
    
    def grad_x(self, x: Array, y: Array) -> Array:
        """
        Gradient computed via JAX autodiff.
        
        The Matérn kernel gradient is more complex, so we use autodiff.
        """
        return jax.grad(lambda x_: self(x_, y))(x)


class LaplacianKernel(Kernel):
    """
    Laplacian (exponential) kernel: k(x, y) = exp(-||x - y|| / σ)
    
    This is equivalent to Matérn with ν = 1/2.
    """
    
    def __init__(self, bandwidth: float = 1.0):
        """
        Initialize Laplacian kernel.
        
        Args:
            bandwidth: Kernel bandwidth σ
        """
        self.bandwidth = bandwidth
    
    def __call__(self, x: Array, y: Array) -> Array:
        """Evaluate k(x, y) = exp(-||x - y|| / σ)"""
        diff = x - y
        # Add small epsilon for numerical stability
        dist = jnp.sqrt(jnp.sum(diff ** 2) + 1e-12)
        return jnp.exp(-dist / self.bandwidth)


class IMQKernel(Kernel):
    """
    Inverse Multiquadric kernel: k(x, y) = (c² + ||x - y||²)^(-β)
    
    Common choice for MMD-based algorithms as it has heavier tails
    than the Gaussian kernel.
    """
    
    def __init__(self, c: float = 1.0, beta: float = 0.5):
        """
        Initialize IMQ kernel.
        
        Args:
            c: Scale parameter
            beta: Exponent (typically 0.5)
        """
        self.c = c
        self.beta = beta
    
    def __call__(self, x: Array, y: Array) -> Array:
        """Evaluate k(x, y) = (c² + ||x - y||²)^(-β)"""
        diff = x - y
        sq_dist = jnp.sum(diff ** 2)
        return (self.c ** 2 + sq_dist) ** (-self.beta)


# Convenience function for computing kernel mean embedding
def kernel_mean_embedding(
    kernel: Kernel,
    measure_atoms: Array,
    measure_weights: Array,
    eval_points: Array,
) -> Array:
    """
    Compute kernel mean embedding μ_ρ(x) = Σ_i w_i k(x, θ_i).
    
    Args:
        kernel: Kernel function
        measure_atoms: Atom locations, shape (N, D)
        measure_weights: Weights (not log), shape (N,)
        eval_points: Points to evaluate at, shape (M, D)
        
    Returns:
        Embedding values at eval_points, shape (M,)
    """
    # Gram matrix: K[i, j] = k(eval_points[i], atoms[j])
    K = kernel.gram(eval_points, measure_atoms)
    # Weighted sum over atoms
    return jnp.dot(K, measure_weights)


def kernel_mean_embedding_gradient(
    kernel: Kernel,
    measure_atoms: Array,
    measure_weights: Array,
    eval_points: Array,
) -> Array:
    """
    Compute gradient of kernel mean embedding ∇_x μ_ρ(x) = Σ_i w_i ∇_x k(x, θ_i).
    
    Args:
        kernel: Kernel function
        measure_atoms: Atom locations, shape (N, D)
        measure_weights: Weights (not log), shape (N,)
        eval_points: Points to evaluate at, shape (M, D)
        
    Returns:
        Gradient at eval_points, shape (M, D)
    """
    # Gradients: G[i, j, :] = ∇_x k(eval_points[i], atoms[j])
    G = kernel.grad_x_batch(eval_points, measure_atoms)
    # Weighted sum over atoms: shape (M, D)
    return jnp.einsum('ijd,j->id', G, measure_weights)

