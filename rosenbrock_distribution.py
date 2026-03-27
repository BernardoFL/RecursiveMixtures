#!/usr/bin/env python3
"""
Rosenbrock-shaped probability distribution in 2D.

We use the standard Rosenbrock "valley" energy:
    f(x, y) = (a - x)^2 + b (y - x^2)^2

Define an (unnormalized) density:
    p(x, y) ∝ exp( - f(x, y) / 20 ).

This corresponds to the normalized model with σ² = 10 in the form
exp(-f/(2σ²)), so we can sample exactly via:
    x ~ Normal(a, 10)
    y | x ~ Normal(x^2, 10 / b)
"""

from __future__ import annotations

from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import Array


class RosenbrockDistribution:
    """A 2D distribution concentrated near y = x^2 with density exp(-f/20)."""

    def __init__(
        self,
        a: float = 1.0,
        b: float = 5.0,
    ):
        if b <= 0:
            raise ValueError("b must be positive")
        self.a = float(a)
        self.b = float(b)
        self._sigma2 = 10.0

    def sample(self, key: Array, n: int) -> Array:
        """Sample n points; returns shape (n, 2)."""
        if n <= 0:
            raise ValueError("n must be positive")
        kx, ky = jr.split(key, 2)
        sigma = jnp.sqrt(self._sigma2)
        x = self.a + sigma * jr.normal(kx, shape=(n,))
        y_std = sigma / jnp.sqrt(self.b)
        y = x**2 + y_std * jr.normal(ky, shape=(n,))
        return jnp.stack([x, y], axis=1)

    def pdf(self, x: Array) -> Array:
        """
        Evaluate density.

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

        x1 = x[:, 0]
        x2 = x[:, 1]

        exponent = -(
            (self.a - x1) ** 2 + self.b * (x2 - x1**2) ** 2
        ) / 20.0
        return jnp.exp(exponent)

