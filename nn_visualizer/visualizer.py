"""
Neural Network Visualizer for architecture and training visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from typing import List, Tuple

from .network import NeuralNetwork


class NeuralNetworkVisualizer:
    """
    Visualizer for neural network architecture and training.
    Creates animated visualizations of forward/backward propagation.
    """

    def __init__(self, network: NeuralNetwork, figsize: Tuple[int, int] = (16, 10)):
        self.network = network
        self.figsize = figsize

        # Colors
        self.colors = {
            'neuron_inactive': '#E0E0E0',
            'neuron_active': '#4CAF50',
            'neuron_negative': '#F44336',
            'weight_positive': '#2196F3',
            'weight_negative': '#FF9800',
            'background': '#FAFAFA'
        }

        # Create custom colormap for activations
        self.activation_cmap = LinearSegmentedColormap.from_list(
            'activation',
            ['#F44336', '#FFEB3B', '#4CAF50']
        )

    def _get_neuron_positions(self) -> List[List[Tuple[float, float]]]:
        """Calculate neuron positions for each layer."""
        positions = []
        layer_sizes = self.network.layer_sizes
        max_neurons = max(layer_sizes)

        for i, size in enumerate(layer_sizes):
            layer_positions = []
            x = i / (len(layer_sizes) - 1) if len(layer_sizes) > 1 else 0.5

            # Center neurons vertically
            start_y = 0.5 - (size - 1) / (2 * max_neurons)

            for j in range(size):
                y = start_y + j / max_neurons
                layer_positions.append((x, y))

            positions.append(layer_positions)

        return positions

    def draw_architecture(self, ax: plt.Axes, show_values: bool = False):
        """
        Draw the neural network architecture.

        Args:
            ax: Matplotlib axes
            show_values: Whether to show activation values
        """
        ax.clear()
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_facecolor(self.colors['background'])

        positions = self._get_neuron_positions()
        neuron_radius = 0.03

        # Draw connections (weights)
        for i in range(len(positions) - 1):
            for j, (x1, y1) in enumerate(positions[i]):
                for k, (x2, y2) in enumerate(positions[i + 1]):
                    weight = self.network.weights[i][j, k]

                    # Color based on weight sign
                    color = self.colors['weight_positive'] if weight >= 0 else self.colors['weight_negative']
                    alpha = min(abs(weight) / 2, 0.8)
                    linewidth = abs(weight) * 2

                    ax.plot([x1, x2], [y1, y2], color=color, alpha=alpha,
                           linewidth=max(0.5, linewidth), zorder=1)

        # Draw neurons
        for i, layer_pos in enumerate(positions):
            for j, (x, y) in enumerate(layer_pos):
                # Get activation value if available
                if show_values and len(self.network.layer_activations) > i:
                    act = self.network.layer_activations[i]
                    if act.ndim > 1:
                        value = np.mean(act[:, j]) if j < act.shape[1] else 0
                    else:
                        value = act[j] if j < len(act) else 0
                    color = self.activation_cmap(value)
                else:
                    color = self.colors['neuron_inactive']

                circle = Circle((x, y), neuron_radius, color=color,
                              ec='black', linewidth=1, zorder=2)
                ax.add_patch(circle)

                # Add value text
                if show_values and len(self.network.layer_activations) > i:
                    act = self.network.layer_activations[i]
                    if act.ndim > 1 and j < act.shape[1]:
                        value = np.mean(act[:, j])
                        ax.text(x, y, f'{value:.2f}', ha='center', va='center',
                               fontsize=7, zorder=3)

        # Layer labels
        layer_names = ['Input'] + [f'Hidden {i}' for i in range(1, len(positions) - 1)] + ['Output']
        for i, (name, layer_pos) in enumerate(zip(layer_names, positions)):
            x = layer_pos[0][0]
            ax.text(x, -0.05, name, ha='center', va='top', fontsize=10, fontweight='bold')

    def draw_decision_boundary(
        self,
        ax: plt.Axes,
        X: np.ndarray,
        y: np.ndarray,
        resolution: int = 100
    ):
        """
        Draw the decision boundary learned by the network.

        Args:
            ax: Matplotlib axes
            X: Input data (2D)
            y: Labels
            resolution: Grid resolution
        """
        ax.clear()

        # Create mesh grid
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, resolution),
            np.linspace(y_min, y_max, resolution)
        )

        # Predict on grid
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        Z = self.network.forward(grid_points)
        Z = Z.reshape(xx.shape)

        # Plot decision boundary
        ax.contourf(xx, yy, Z, levels=20, cmap='RdYlGn', alpha=0.6)
        ax.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)

        # Plot data points
        scatter = ax.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='RdYlGn',
                           edgecolors='black', s=50, zorder=5)

        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_title('Decision Boundary')

    def draw_loss_curve(self, ax: plt.Axes, losses: List[float]):
        """Draw the training loss curve."""
        ax.clear()
        ax.plot(losses, 'b-', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss')
        ax.grid(True, alpha=0.3)

        if losses:
            ax.text(0.95, 0.95, f'Current: {losses[-1]:.4f}',
                   transform=ax.transAxes, ha='right', va='top',
                   fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat'))

    def draw_gradient_flow(self, ax: plt.Axes):
        """Visualize gradient magnitudes through layers."""
        ax.clear()

        if not self.network.gradients:
            ax.text(0.5, 0.5, 'No gradients yet', ha='center', va='center')
            return

        # Calculate gradient magnitudes per layer
        grad_mags = []
        for i, grad in enumerate(self.network.gradients):
            mag = np.mean(np.abs(grad['dw']))
            grad_mags.append(mag)

        x = range(len(grad_mags))
        ax.bar(x, grad_mags, color='purple', alpha=0.7)
        ax.set_xlabel('Layer')
        ax.set_ylabel('Mean |Gradient|')
        ax.set_title('Gradient Flow')
        ax.set_xticks(x)
        ax.set_xticklabels([f'L{i+1}' for i in x])

    def create_training_visualization(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        learning_rate: float = 0.5,
        interval: int = 100
    ) -> FuncAnimation:
        """
        Create an animated visualization of training.

        Args:
            X: Training data
            y: Labels
            epochs: Number of epochs
            learning_rate: Learning rate
            interval: Animation interval in ms

        Returns:
            FuncAnimation object
        """
        # Setup figure
        fig = plt.figure(figsize=self.figsize)
        gs = gridspec.GridSpec(2, 2, figure=fig)

        ax_arch = fig.add_subplot(gs[0, 0])
        ax_boundary = fig.add_subplot(gs[0, 1])
        ax_loss = fig.add_subplot(gs[1, 0])
        ax_grad = fig.add_subplot(gs[1, 1])

        fig.suptitle('Neural Network Training Visualization', fontsize=14, fontweight='bold')

        losses = []

        def update(frame):
            # Train one epoch
            loss = self.network.backward(X, y, learning_rate)
            losses.append(loss)

            # Update visualizations
            self.draw_architecture(ax_arch, show_values=True)
            ax_arch.set_title(f'Network Architecture (Epoch {frame + 1})')

            self.draw_decision_boundary(ax_boundary, X, y)
            self.draw_loss_curve(ax_loss, losses)
            self.draw_gradient_flow(ax_grad)

            plt.tight_layout()

        anim = FuncAnimation(fig, update, frames=epochs, interval=interval, repeat=False)
        return anim

    def save_architecture_diagram(self, filename: str = 'network_architecture.png'):
        """Save a static architecture diagram."""
        fig, ax = plt.subplots(figsize=(12, 8))
        self.draw_architecture(ax, show_values=False)
        ax.set_title('Neural Network Architecture', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Architecture saved to {filename}")
