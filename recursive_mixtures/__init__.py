"""
Recursive Mixtures: A JAX-based framework for Gradient Flows on Measure Spaces.

This package implements recursive algorithms for mixture models based on
gradient flows, including Newton-Hellinger, Hellinger-Kantorovich, 
Repulsive (MMD), and Covariate-Dependent flows.
"""

from recursive_mixtures.kernels import Kernel, GaussianKernel, MaternKernel
from recursive_mixtures.measure import ParticleMeasure, Prior, GaussianPrior, MixturePrior
from recursive_mixtures.functionals import Functional, LogLikelihoodFunctional, KLFunctional, MMDFunctional
from recursive_mixtures.flows import (
    GradientFlow,
    NewtonHellingerFlow,
    HellingerKantorovichFlow,
    RepulsiveFlow,
    CovariateDependentFlow,
)
from recursive_mixtures.utils import bayesian_bootstrap, compute_sinkhorn_potentials, wasserstein_gradient

__version__ = "0.1.0"

__all__ = [
    # Kernels
    "Kernel",
    "GaussianKernel", 
    "MaternKernel",
    # Measures
    "ParticleMeasure",
    "Prior",
    "GaussianPrior",
    "MixturePrior",
    # Functionals
    "Functional",
    "LogLikelihoodFunctional",
    "KLFunctional",
    "MMDFunctional",
    # Flows
    "GradientFlow",
    "NewtonHellingerFlow",
    "HellingerKantorovichFlow",
    "RepulsiveFlow",
    "CovariateDependentFlow",
    # Utils
    "bayesian_bootstrap",
    "compute_sinkhorn_potentials",
    "wasserstein_gradient",
]

