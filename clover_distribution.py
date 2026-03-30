#!/usr/bin/env python3
"""
Four-petal clover distribution: an equal-weight mixture of 2D Gaussians.

The "petals" are anisotropic Gaussians centered on the four cardinal directions
on a circle of radius r, with the long axis aligned radially.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import Array


def _rot(theta: float) -> Array:
    c = jnp.cos(theta)
    s = jnp.sin(theta)
    return jnp.array([[c, -s], [s, c]])


def _mvnormal_logpdf(x: Array, mean: Array, cov: Array) -> Array:
    """Log N(x | mean, cov) for x shape (M,2)."""
    x = jnp.asarray(x)
    mean = jnp.asarray(mean)
    cov = jnp.asarray(cov)
    d = 2
    diff = x - mean[None, :]
    sign, logdet = jnp.linalg.slogdet(cov)
    # cov is PSD by construction; guard sign for numerical safety
    logdet = jnp.where(sign > 0, logdet, jnp.inf)
    sol = jnp.linalg.solve(cov, diff.T).T
    quad = jnp.sum(diff * sol, axis=1)
    return -0.5 * (d * jnp.log(2.0 * jnp.pi) + logdet + quad)


@dataclass(frozen=True)
class CloverDistribution:
    """
    Four-petal clover: 4-component Gaussian mixture in 2D.

    Params:
        radius: distance of petal centers from origin.
        radial_std: std along radial direction (petal length).
        tangential_std: std orthogonal to radial direction (petal thickness).
    """

    radius: float = 1.5
    radial_std: float = 0.45
    tangential_std: float = 0.18

    def _component_params(self) -> Tuple[Array, Array]:
        thetas = jnp.array([0.0, 0.5 * jnp.pi, jnp.pi, 1.5 * jnp.pi])
        means = self.radius * jnp.stack([jnp.cos(thetas), jnp.sin(thetas)], axis=1)

        diag = jnp.diag(jnp.array([self.radial_std**2, self.tangential_std**2]))
        covs = jax.vmap(lambda th: _rot(th) @ diag @ _rot(th).T)(thetas)
        return means, covs

    def sample(self, key: Array, n: int) -> Array:
        """Sample n points; returns shape (n, 2)."""
        if n <= 0:
            raise ValueError("n must be positive")
        means, covs = self._component_params()
        k_idx, k_z = jr.split(key, 2)
        idx = jr.randint(k_idx, shape=(n,), minval=0, maxval=4)
        z = jr.normal(k_z, shape=(n, 2))
        L = jax.vmap(jnp.linalg.cholesky)(covs)  # (4,2,2)
        samples = means[idx] + jax.vmap(lambda i, zi: L[i] @ zi)(idx, z)
        return samples

    def pdf(self, x: Array) -> Array:
        """
        Evaluate mixture density.

        Args:
            x: shape (2,) or (M, 2)
        Returns:
            density: shape (M,)
        """
        x = jnp.asarray(x)
        if x.ndim == 1:
            x = x[None, :]
        if x.shape[-1] != 2:
            raise ValueError("x must have shape (M, 2)")
        means, covs = self._component_params()
        logps = jax.vmap(lambda m, c: _mvnormal_logpdf(x, m, c))(means, covs)  # (4,M)
        # equal weights
        return jnp.exp(jax.scipy.special.logsumexp(logps, axis=0) - jnp.log(4.0))

