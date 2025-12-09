# Recursive Mixtures

A JAX-based Python framework for recursive algorithms on mixture models using Gradient Flows on Measure Spaces.

## Overview

This framework implements a family of recursive algorithms for Bayesian inference on mixture models, leveraging gradient flows in the space of probability measures. The algorithms are based on the Hellinger-Kantorovich geometry, which combines the Wasserstein (optimal transport) and Hellinger (Fisher-Rao) metrics.

## Installation

```bash
pip install -r requirements.txt
```

## Algorithms

### Algorithm A: Newton-Hellinger Flow
Updates only weights while keeping atom locations fixed. Suitable for scenarios where the support of the mixture is pre-specified.

### Algorithm B: Hellinger-Kantorovich (HK) Flow
Updates both weights (Hellinger step) and atom locations (Wasserstein step). Includes optional Sinkhorn regularization for drift toward a prior.

### Algorithm C: Repulsive Flow (MMD)
Extends HK Flow with a Maximum Mean Discrepancy (MMD) based repulsive term to encourage particle diversity.

### Algorithm D: Covariate-Dependent Flow
Handles regression scenarios where atoms are regression parameters. Includes Langevin diffusion for exploration.

## Quick Start

```python
import jax.random as jr
from recursive_mixtures import (
    GaussianKernel,
    ParticleMeasure,
    GaussianPrior,
    HellingerKantorovichFlow,
)

# Initialize
key = jr.PRNGKey(42)
kernel = GaussianKernel(bandwidth=1.0)
prior = GaussianPrior(mean=0.0, std=2.0)

# Create initial particle measure
key, subkey = jr.split(key)
atoms = prior.sample(subkey, n=100)
measure = ParticleMeasure.initialize(atoms)

# Create flow
flow = HellingerKantorovichFlow(
    kernel=kernel,
    prior=prior,
    step_size=0.1,
    sinkhorn_reg=0.1,
)

# Run flow on data stream
for data_point in data_stream:
    measure = flow.step(measure, data_point)
```

## Running the Experiment

```bash
python run_experiment.py
```

This generates synthetic data from a mixture of 3 Gaussians and visualizes the HK flow convergence.

## Project Structure

```
RecursiveMixtures/
├── recursive_mixtures/
│   ├── __init__.py
│   ├── kernels.py       # Kernel functions with JAX autodiff
│   ├── measure.py       # ParticleMeasure and Prior classes
│   ├── functionals.py   # Extensible functional interface
│   ├── flows.py         # Gradient flow algorithms
│   └── utils.py         # OT wrappers and utilities
├── run_experiment.py    # Verification script
├── requirements.txt
└── README.md
```

## Key Features

- **Log-domain stability**: All weight operations use logsumexp for numerical stability
- **Vectorization**: Operations are fully vectorized using `jax.vmap`
- **JIT compilation**: Flow steps are JIT-compiled for performance
- **Extensibility**: Easy to define new functionals via abstract interface

## License

MIT License

