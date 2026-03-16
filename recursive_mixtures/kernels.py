"""
Kernel functions for gradient flows on measure spaces.

Analytical gradients are provided for all built-in kernels.
Autodiff is used only as a fallback in the base class for custom kernels.
"""

from abc import ABC, abstractmethod
from typing import Union

import jax
import jax.numpy as jnp
from jax import Array


class Kernel(ABC):
    """
    Abstract base class for positive definite kernels.

    Subclasses should override `grad_x` (and by symmetry `grad_y`) with
    closed-form expressions where available. The base implementations
    fall back to JAX autodiff.
    """

    @abstractmethod
    def __call__(self, x: Array, y: Array) -> Array:
        """Evaluate k(x, y) for single points x, y of shape (D,)."""

    def grad_x(self, x: Array, y: Array) -> Array:
        """∇_x k(x, y), shape (D,). Falls back to JAX autodiff."""
        return jax.grad(lambda x_: self(x_, y))(x)

    def grad_y(self, x: Array, y: Array) -> Array:
        """∇_y k(x, y), shape (D,). Falls back to JAX autodiff."""
        return jax.grad(lambda y_: self(x, y_))(y)

    def gram(self, X: Array, Y: Array) -> Array:
        """K[i,j] = k(X[i], Y[j]), shape (N, M). Falls back to vmap."""
        return jax.vmap(lambda x: jax.vmap(lambda y: self(x, y))(Y))(X)

    def grad_x_batch(self, X: Array, Y: Array) -> Array:
        """
        ∇_x k(x, y) for all pairs, shape (N, M, D).

        G[i, j, :] = ∇_x k(X[i], Y[j])
        Falls back to vmap; override for vectorised analytical forms.
        """
        return jax.vmap(lambda x: jax.vmap(lambda y: self.grad_x(x, y))(Y))(X)

    def grad_y_gram(self, X: Array, Y: Array) -> Array:
        """
        ∇_y k(x, y) for all pairs, shape (N, M, D).

        G[i, j, :] = ∇_y k(X[i], Y[j])
        Falls back to vmap; override for vectorised analytical forms.
        """
        return jax.vmap(lambda x: jax.vmap(lambda y: self.grad_y(x, y))(Y))(X)


class GaussianKernel(Kernel):
    """
    Gaussian (RBF) kernel: k(x, y) = exp(−‖x − y‖² / (2σ²))

    All gradients and Gram matrices are computed analytically without autodiff.
    """

    def __init__(self, bandwidth: float = 1.0):
        self.bandwidth = bandwidth
        self.gamma = 1.0 / (2.0 * bandwidth ** 2)

    def __call__(self, x: Array, y: Array) -> Array:
        return jnp.exp(-self.gamma * jnp.sum((x - y) ** 2))

    # --- Analytical single-pair gradients ---

    def grad_x(self, x: Array, y: Array) -> Array:
        """∇_x k = −2γ (x − y) k(x, y)"""
        return -2.0 * self.gamma * (x - y) * self(x, y)

    def grad_y(self, x: Array, y: Array) -> Array:
        """∇_y k = +2γ (x − y) k(x, y)  [= −∇_x k]"""
        return -self.grad_x(x, y)

    # --- Fully vectorised batch operations (no Python-level vmap loop) ---

    def gram(self, X: Array, Y: Array) -> Array:
        """K[i,j] = exp(−γ ‖X[i] − Y[j]‖²), shape (N, M)."""
        diff = X[:, None, :] - Y[None, :, :]        # (N, M, D)
        return jnp.exp(-self.gamma * jnp.sum(diff ** 2, axis=-1))  # (N, M)

    def grad_x_batch(self, X: Array, Y: Array) -> Array:
        """
        G[i,j,:] = −2γ (X[i] − Y[j]) K[i,j], shape (N, M, D).
        Analytical; no autodiff or vmap loop.
        """
        K = self.gram(X, Y)                         # (N, M)
        diff = X[:, None, :] - Y[None, :, :]        # (N, M, D)
        return -2.0 * self.gamma * diff * K[:, :, None]

    def grad_y_gram(self, X: Array, Y: Array) -> Array:
        """
        G[i,j,:] = +2γ (X[i] − Y[j]) K[i,j], shape (N, M, D).
        Analytical; no autodiff or vmap loop.
        """
        return -self.grad_x_batch(X, Y)


class MaternKernel(Kernel):
    """
    Matérn kernel family with analytical gradients.

    Supported: ν ∈ {0.5, 1.5, 2.5}.
    Let r = ‖x − y‖ / ℓ  (with ε stabilisation).

    Gradients use the chain rule ∇_x k = (dk/dr) * (∂r/∂x):
      ∂r/∂x = (x − y) / (‖x − y‖ * ℓ)

    Key simplification:  r * ∂r/∂x = (x − y) / ℓ²
    """

    def __init__(self, lengthscale: float = 1.0, nu: float = 2.5):
        if nu not in [0.5, 1.5, 2.5]:
            raise ValueError(f"nu must be 0.5, 1.5, or 2.5, got {nu}")
        self.lengthscale = lengthscale
        self.nu = nu

    def __call__(self, x: Array, y: Array) -> Array:
        diff = x - y
        r = jnp.sqrt(jnp.sum(diff ** 2) + 1e-12) / self.lengthscale
        if self.nu == 0.5:
            return jnp.exp(-r)
        elif self.nu == 1.5:
            sqrt3_r = jnp.sqrt(3.0) * r
            return (1.0 + sqrt3_r) * jnp.exp(-sqrt3_r)
        else:  # 2.5
            sqrt5_r = jnp.sqrt(5.0) * r
            return (1.0 + sqrt5_r + sqrt5_r ** 2 / 3.0) * jnp.exp(-sqrt5_r)

    def grad_x(self, x: Array, y: Array) -> Array:
        """
        Closed-form ∇_x k(x, y).

        ν=0.5:  −k/(dist*ℓ) * (x−y)
        ν=1.5:  −3 exp(−√3 r)/ℓ² * (x−y)
        ν=2.5:  −(5/3)(1+√5 r) exp(−√5 r)/ℓ² * (x−y)
        """
        diff = x - y
        sq_dist = jnp.sum(diff ** 2)
        ell = self.lengthscale
        if self.nu == 0.5:
            dist = jnp.sqrt(sq_dist + 1e-12)
            k = self(x, y)
            return -(k / (dist * ell)) * diff
        elif self.nu == 1.5:
            r = jnp.sqrt(sq_dist + 1e-12) / ell
            sqrt3_r = jnp.sqrt(3.0) * r
            return -3.0 * jnp.exp(-sqrt3_r) / ell ** 2 * diff
        else:  # 2.5
            r = jnp.sqrt(sq_dist + 1e-12) / ell
            sqrt5_r = jnp.sqrt(5.0) * r
            return -(5.0 / 3.0) * (1.0 + sqrt5_r) * jnp.exp(-sqrt5_r) / ell ** 2 * diff

    def grad_y(self, x: Array, y: Array) -> Array:
        return -self.grad_x(x, y)


class LaplacianKernel(Kernel):
    """
    Laplacian kernel: k(x, y) = exp(−‖x − y‖ / σ)

    ∇_x k = −k / (σ * dist) * (x − y)   where dist = ‖x − y‖ (with ε).
    """

    def __init__(self, bandwidth: float = 1.0):
        self.bandwidth = bandwidth

    def __call__(self, x: Array, y: Array) -> Array:
        dist = jnp.sqrt(jnp.sum((x - y) ** 2) + 1e-12)
        return jnp.exp(-dist / self.bandwidth)

    def grad_x(self, x: Array, y: Array) -> Array:
        diff = x - y
        dist = jnp.sqrt(jnp.sum(diff ** 2) + 1e-12)
        k = self(x, y)
        return -(k / (self.bandwidth * dist)) * diff

    def grad_y(self, x: Array, y: Array) -> Array:
        return -self.grad_x(x, y)


class IMQKernel(Kernel):
    """
    Inverse Multiquadric kernel: k(x, y) = (c² + ‖x − y‖²)^(−β)

    ∇_x k = −2β * k / (c² + ‖x − y‖²) * (x − y)
    """

    def __init__(self, c: float = 1.0, beta: float = 0.5):
        self.c = c
        self.beta = beta

    def __call__(self, x: Array, y: Array) -> Array:
        sq_dist = jnp.sum((x - y) ** 2)
        return (self.c ** 2 + sq_dist) ** (-self.beta)

    def grad_x(self, x: Array, y: Array) -> Array:
        diff = x - y
        sq_dist = jnp.sum(diff ** 2)
        k = self(x, y)
        return -2.0 * self.beta * k / (self.c ** 2 + sq_dist) * diff

    def grad_y(self, x: Array, y: Array) -> Array:
        return -self.grad_x(x, y)


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

