#!/usr/bin/env python3
"""
Verification Experiment: Hellinger-Kantorovich Flow on Gaussian Mixture.

This script demonstrates the HK flow algorithm (Algorithm B) on synthetic
data from a mixture of 3 Gaussians. It visualizes:
1. Evolution of the estimated density vs true density
2. Particle trajectories over time
3. Weight evolution heatmap
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import numpy as np

# Enable 64-bit precision for numerical stability
jax.config.update("jax_enable_x64", True)

from recursive_mixtures import (
    GaussianKernel,
    ParticleMeasure,
    GaussianPrior,
    MixturePrior,
    HellingerKantorovichFlow,
)
from recursive_mixtures.utils import (
    generate_mixture_data,
    true_mixture_density,
)


def setup_experiment():
    """Set up experiment parameters."""
    config = {
        # True mixture parameters
        'true_means': jnp.array([-2.0, 0.0, 2.0]),
        'true_stds': jnp.array([0.5, 0.8, 0.5]),
        'true_weights': jnp.array([0.3, 0.4, 0.3]),
        
        # Data generation
        'n_data': 1000,
        
        # Flow parameters
        'n_particles': 100,
        'step_size': 0.1,
        'kernel_bandwidth': 0.8,
        'sinkhorn_reg': 0.05,
        'wasserstein_weight': 0.1,
        
        # Prior parameters
        'prior_mean': 0.0,
        'prior_std': 3.0,
        
        # Recording
        'store_every': 10,
        
        # Random seed
        'seed': 42,
    }
    return config


def run_hk_flow(config):
    """
    Run the Hellinger-Kantorovich flow experiment.
    
    Returns:
        Tuple of (final_measure, history, data, config)
    """
    print("=" * 60)
    print("Hellinger-Kantorovich Flow Experiment")
    print("=" * 60)
    
    # Initialize random key
    key = jr.PRNGKey(config['seed'])
    
    # Generate synthetic data
    key, subkey = jr.split(key)
    print(f"\n1. Generating {config['n_data']} samples from mixture of 3 Gaussians...")
    print(f"   Means: {config['true_means']}")
    print(f"   Stds:  {config['true_stds']}")
    print(f"   Weights: {config['true_weights']}")
    
    data, _ = generate_mixture_data(
        subkey,
        config['n_data'],
        config['true_means'],  # Shape (K,) for 1D
        config['true_stds'],
        config['true_weights'],
    )
    data = data.squeeze()  # Shape (n_data,) for 1D
    
    # Initialize kernel
    kernel = GaussianKernel(bandwidth=config['kernel_bandwidth'])
    print(f"\n2. Kernel: Gaussian with bandwidth {config['kernel_bandwidth']}")
    
    # Initialize prior
    prior = GaussianPrior(
        mean=config['prior_mean'],
        std=config['prior_std'],
        dim=1,
    )
    print(f"   Prior: Gaussian({config['prior_mean']}, {config['prior_std']}²)")
    
    # Initialize particle measure from prior
    key, subkey = jr.split(key)
    initial_atoms = prior.sample(subkey, config['n_particles'])
    initial_measure = ParticleMeasure.initialize(initial_atoms)
    print(f"\n3. Initialized {config['n_particles']} particles from prior")
    print(f"   Initial atom range: [{initial_atoms.min():.2f}, {initial_atoms.max():.2f}]")
    
    # Create prior particle measure for Sinkhorn regularization
    key, subkey = jr.split(key)
    prior_particles = prior.to_particle_measure(subkey, config['n_particles'])
    
    # Initialize flow
    flow = HellingerKantorovichFlow(
        kernel=kernel,
        prior=prior,
        step_size=config['step_size'],
        wasserstein_weight=config['wasserstein_weight'],
        sinkhorn_reg=config['sinkhorn_reg'],
        use_sinkhorn=True,
        prior_particles=prior_particles,
    )
    print(f"\n4. Created HK Flow:")
    print(f"   Step size: {config['step_size']}")
    print(f"   Wasserstein weight: {config['wasserstein_weight']}")
    print(f"   Sinkhorn regularization: {config['sinkhorn_reg']}")
    
    # Run flow
    print(f"\n5. Running flow on {config['n_data']} data points...")
    print(f"   Storing history every {config['store_every']} steps")
    
    key, subkey = jr.split(key)
    keys = jr.split(subkey, config['n_data'])
    
    measure = initial_measure
    history = [measure]
    
    for t in range(config['n_data']):
        measure = flow.step(measure, data[t], keys[t])
        
        if (t + 1) % config['store_every'] == 0:
            history.append(measure)
            
        if (t + 1) % 200 == 0:
            ess = measure.effective_sample_size()
            print(f"   Step {t+1}/{config['n_data']}, ESS: {ess:.1f}")
    
    print(f"\n6. Flow completed! Final ESS: {measure.effective_sample_size():.1f}")
    print(f"   Recorded {len(history)} snapshots")
    
    return measure, history, data, config


def plot_results(final_measure, history, data, config):
    """
    Create visualization of the flow results.
    """
    print("\n7. Creating visualizations...")
    
    # Evaluation grid
    x_grid = jnp.linspace(-5, 5, 500)
    
    # Compute true density
    true_density = true_mixture_density(
        x_grid,
        config['true_means'],
        config['true_stds'],
        config['true_weights'],
    )
    
    # Create figure
    fig = plt.figure(figsize=(16, 12))
    
    # Color scheme - elegant dark theme
    plt.style.use('default')
    fig.patch.set_facecolor('#1a1a2e')
    
    colors = {
        'true': '#00d4ff',      # Cyan
        'estimate': '#ff6b6b',   # Coral
        'particles': '#ffd93d',  # Gold
        'trajectory': '#6bcb77', # Green
        'background': '#1a1a2e',
        'grid': '#2d2d44',
        'text': '#e8e8e8',
    }
    
    # ==========================================================================
    # Panel 1: Density Evolution
    # ==========================================================================
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.set_facecolor(colors['background'])
    
    # Plot snapshots at different times
    n_snapshots = min(6, len(history))
    snapshot_indices = np.linspace(0, len(history) - 1, n_snapshots, dtype=int)
    
    alphas = np.linspace(0.2, 1.0, n_snapshots)
    
    for idx, (snap_idx, alpha) in enumerate(zip(snapshot_indices, alphas)):
        measure = history[snap_idx]
        kernel = GaussianKernel(bandwidth=config['kernel_bandwidth'])
        
        # Kernel density estimate
        kde = measure.kernel_density(kernel, x_grid[:, None])
        
        # Normalize to integrate to 1 (approximately)
        kde = kde / (jnp.sum(kde) * (x_grid[1] - x_grid[0]))
        
        label = f't={snap_idx * config["store_every"]}' if idx in [0, n_snapshots-1] else None
        ax1.plot(x_grid, kde, color=colors['estimate'], alpha=alpha, 
                linewidth=1.5, label=label)
    
    # True density
    ax1.plot(x_grid, true_density, color=colors['true'], linewidth=2.5, 
            label='True density', linestyle='--')
    
    ax1.set_xlabel('x', fontsize=12, color=colors['text'])
    ax1.set_ylabel('Density', fontsize=12, color=colors['text'])
    ax1.set_title('Density Evolution: ρ_t → True Density', fontsize=14, 
                 color=colors['text'], fontweight='bold')
    ax1.legend(facecolor=colors['background'], edgecolor=colors['grid'],
              labelcolor=colors['text'])
    ax1.tick_params(colors=colors['text'])
    ax1.grid(True, alpha=0.3, color=colors['grid'])
    ax1.set_xlim(-5, 5)
    
    # ==========================================================================
    # Panel 2: Final Density Comparison
    # ==========================================================================
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.set_facecolor(colors['background'])
    
    kernel = GaussianKernel(bandwidth=config['kernel_bandwidth'])
    final_kde = final_measure.kernel_density(kernel, x_grid[:, None])
    final_kde = final_kde / (jnp.sum(final_kde) * (x_grid[1] - x_grid[0]))
    
    ax2.fill_between(x_grid, 0, true_density, alpha=0.3, color=colors['true'],
                    label='True density')
    ax2.plot(x_grid, true_density, color=colors['true'], linewidth=2, linestyle='--')
    
    ax2.fill_between(x_grid, 0, final_kde, alpha=0.3, color=colors['estimate'],
                    label='HK estimate')
    ax2.plot(x_grid, final_kde, color=colors['estimate'], linewidth=2)
    
    # Show particle locations
    ax2.scatter(final_measure.atoms[:, 0], 
               jnp.zeros(config['n_particles']) - 0.05,
               s=final_measure.weights * 500,
               c=colors['particles'],
               alpha=0.7,
               marker='|',
               linewidths=2,
               label='Particles')
    
    ax2.set_xlabel('x', fontsize=12, color=colors['text'])
    ax2.set_ylabel('Density', fontsize=12, color=colors['text'])
    ax2.set_title('Final Estimate vs True Density', fontsize=14,
                 color=colors['text'], fontweight='bold')
    ax2.legend(facecolor=colors['background'], edgecolor=colors['grid'],
              labelcolor=colors['text'])
    ax2.tick_params(colors=colors['text'])
    ax2.grid(True, alpha=0.3, color=colors['grid'])
    ax2.set_xlim(-5, 5)
    
    # ==========================================================================
    # Panel 3: Particle Trajectories
    # ==========================================================================
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.set_facecolor(colors['background'])
    
    # Extract trajectories
    n_history = len(history)
    n_particles = config['n_particles']
    
    trajectories = np.zeros((n_history, n_particles))
    weights_history = np.zeros((n_history, n_particles))
    
    for t, measure in enumerate(history):
        trajectories[t] = np.array(measure.atoms[:, 0])
        weights_history[t] = np.array(measure.weights)
    
    times = np.arange(n_history) * config['store_every']
    
    # Plot trajectories with weight-based opacity
    for i in range(n_particles):
        # Color based on final position
        final_pos = trajectories[-1, i]
        if final_pos < -1:
            color = '#ff6b6b'  # Red for left mode
        elif final_pos > 1:
            color = '#6bcb77'  # Green for right mode
        else:
            color = '#ffd93d'  # Yellow for center mode
        
        # Average weight for line thickness
        avg_weight = np.mean(weights_history[:, i])
        linewidth = 0.5 + avg_weight * 20
        
        ax3.plot(times, trajectories[:, i], color=color, 
                alpha=0.5, linewidth=linewidth)
    
    # Mark true means
    for mean in config['true_means']:
        ax3.axhline(y=mean, color=colors['true'], linestyle=':', 
                   alpha=0.7, linewidth=1.5)
    
    ax3.set_xlabel('Iteration', fontsize=12, color=colors['text'])
    ax3.set_ylabel('Atom Location θ', fontsize=12, color=colors['text'])
    ax3.set_title('Particle Trajectories (color = final mode)', fontsize=14,
                 color=colors['text'], fontweight='bold')
    ax3.tick_params(colors=colors['text'])
    ax3.grid(True, alpha=0.3, color=colors['grid'])
    
    # ==========================================================================
    # Panel 4: Weight Evolution Heatmap
    # ==========================================================================
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.set_facecolor(colors['background'])
    
    # Sort particles by final position for cleaner visualization
    final_order = np.argsort(trajectories[-1])
    sorted_weights = weights_history[:, final_order]
    
    # Create heatmap
    im = ax4.imshow(sorted_weights.T, aspect='auto', cmap='magma',
                   extent=[0, times[-1], 0, n_particles],
                   origin='lower')
    
    cbar = plt.colorbar(im, ax=ax4)
    cbar.set_label('Weight', fontsize=12, color=colors['text'])
    cbar.ax.yaxis.set_tick_params(color=colors['text'])
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=colors['text'])
    
    ax4.set_xlabel('Iteration', fontsize=12, color=colors['text'])
    ax4.set_ylabel('Particle (sorted by final position)', fontsize=12, 
                  color=colors['text'])
    ax4.set_title('Weight Evolution Heatmap', fontsize=14,
                 color=colors['text'], fontweight='bold')
    ax4.tick_params(colors=colors['text'])
    
    # ==========================================================================
    # Final adjustments
    # ==========================================================================
    plt.tight_layout()
    
    # Add title
    fig.suptitle('Hellinger-Kantorovich Flow: Mixture Density Estimation',
                fontsize=16, color=colors['text'], fontweight='bold', y=1.02)
    
    # Save figure
    plt.savefig('hk_flow_results.png', dpi=150, facecolor=colors['background'],
               edgecolor='none', bbox_inches='tight')
    print("   Saved: hk_flow_results.png")
    
    plt.show()
    
    return fig


def compute_diagnostics(final_measure, history, data, config):
    """Compute and print diagnostic statistics."""
    print("\n" + "=" * 60)
    print("Diagnostics")
    print("=" * 60)
    
    # Compute true moments
    true_mean = jnp.sum(config['true_weights'] * config['true_means'])
    true_var = jnp.sum(config['true_weights'] * 
                       (config['true_stds']**2 + config['true_means']**2)) - true_mean**2
    
    # Compute estimated moments
    est_mean = final_measure.mean()[0]
    est_var = final_measure.variance()[0]
    
    print(f"\nMoment Comparison:")
    print(f"  True Mean:     {true_mean:.4f}")
    print(f"  Estimated Mean: {est_mean:.4f}")
    print(f"  True Variance:  {true_var:.4f}")
    print(f"  Estimated Var:  {est_var:.4f}")
    
    # Particle statistics
    print(f"\nParticle Statistics:")
    print(f"  Final ESS: {final_measure.effective_sample_size():.1f} / {config['n_particles']}")
    print(f"  Atom range: [{final_measure.atoms.min():.2f}, {final_measure.atoms.max():.2f}]")
    print(f"  Max weight: {final_measure.weights.max():.4f}")
    print(f"  Min weight: {final_measure.weights.min():.6f}")
    
    # Mode detection
    atoms = np.array(final_measure.atoms[:, 0])
    weights = np.array(final_measure.weights)
    
    # Cluster particles into modes
    mode_centers = []
    mode_weights = []
    
    for true_mean in config['true_means']:
        mask = np.abs(atoms - float(true_mean)) < 1.0
        if np.any(mask):
            mode_centers.append(np.average(atoms[mask], weights=weights[mask]))
            mode_weights.append(np.sum(weights[mask]))
    
    print(f"\nDetected Modes:")
    for i, (center, weight) in enumerate(zip(mode_centers, mode_weights)):
        true_weight = config['true_weights'][i]
        print(f"  Mode {i+1}: center={center:.2f} (true: {config['true_means'][i]:.1f}), "
              f"weight={weight:.2f} (true: {float(true_weight):.2f})")


def main():
    """Main entry point."""
    # Setup
    config = setup_experiment()
    
    # Run experiment
    final_measure, history, data, config = run_hk_flow(config)
    
    # Compute diagnostics
    compute_diagnostics(final_measure, history, data, config)
    
    # Plot results
    fig = plot_results(final_measure, history, data, config)
    
    print("\n" + "=" * 60)
    print("Experiment Complete!")
    print("=" * 60)
    
    return final_measure, history, data


if __name__ == "__main__":
    final_measure, history, data = main()

