"""
Neural Network Visualizer
==========================
Interactive visualization of neural network architectures and training.
Shows forward propagation, backpropagation, and decision boundaries in real-time.

Author: Fernando Gimenez
Portfolio: LLM Engineer Position
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# NEURAL NETWORK IMPLEMENTATION
# ============================================================================

@dataclass
class LayerConfig:
    """Configuration for a neural network layer."""
    input_size: int
    output_size: int
    activation: str = "relu"


class NeuralNetwork:
    """
    A simple neural network implementation for visualization purposes.
    Includes forward/backward propagation with intermediate values stored.
    """

    def __init__(self, layer_sizes: List[int], activations: List[str] = None):
        """
        Initialize the neural network.

        Args:
            layer_sizes: List of layer sizes [input, hidden1, hidden2, ..., output]
            activations: List of activation functions for each layer
        """
        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes)

        # Default activations
        if activations is None:
            activations = ['relu'] * (self.n_layers - 2) + ['sigmoid']
        self.activations = activations

        # Initialize weights and biases
        self.weights = []
        self.biases = []

        for i in range(self.n_layers - 1):
            # Xavier initialization
            scale = np.sqrt(2.0 / layer_sizes[i])
            w = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * scale
            b = np.zeros((1, layer_sizes[i + 1]))
            self.weights.append(w)
            self.biases.append(b)

        # Storage for visualization
        self.layer_outputs = []
        self.layer_activations = []
        self.gradients = []

    def _activate(self, x: np.ndarray, activation: str) -> np.ndarray:
        """Apply activation function."""
        if activation == 'relu':
            return np.maximum(0, x)
        elif activation == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        elif activation == 'tanh':
            return np.tanh(x)
        elif activation == 'linear':
            return x
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def _activate_derivative(self, x: np.ndarray, activation: str) -> np.ndarray:
        """Derivative of activation function."""
        if activation == 'relu':
            return (x > 0).astype(float)
        elif activation == 'sigmoid':
            s = self._activate(x, 'sigmoid')
            return s * (1 - s)
        elif activation == 'tanh':
            return 1 - np.tanh(x) ** 2
        elif activation == 'linear':
            return np.ones_like(x)
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward propagation with intermediate storage.

        Args:
            X: Input data (batch_size, input_size)

        Returns:
            Output predictions
        """
        self.layer_outputs = [X]
        self.layer_activations = [X]

        current = X
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            # Linear transformation
            z = current @ w + b
            self.layer_outputs.append(z)

            # Activation
            a = self._activate(z, self.activations[i])
            self.layer_activations.append(a)

            current = a

        return current

    def backward(self, X: np.ndarray, y: np.ndarray, learning_rate: float = 0.01) -> float:
        """
        Backward propagation with gradient storage.

        Args:
            X: Input data
            y: True labels
            learning_rate: Learning rate for weight updates

        Returns:
            Loss value
        """
        m = X.shape[0]

        # Forward pass
        output = self.forward(X)

        # Compute loss (binary cross-entropy)
        epsilon = 1e-15
        output = np.clip(output, epsilon, 1 - epsilon)
        loss = -np.mean(y * np.log(output) + (1 - y) * np.log(1 - output))

        # Backward pass
        self.gradients = []

        # Output layer gradient
        delta = output - y

        for i in range(self.n_layers - 2, -1, -1):
            # Compute gradients
            dw = self.layer_activations[i].T @ delta / m
            db = np.mean(delta, axis=0, keepdims=True)

            self.gradients.insert(0, {'dw': dw, 'db': db, 'delta': delta.copy()})

            if i > 0:
                # Backpropagate error
                delta = (delta @ self.weights[i].T) * self._activate_derivative(
                    self.layer_outputs[i], self.activations[i - 1]
                )

        # Update weights
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * self.gradients[i]['dw']
            self.biases[i] -= learning_rate * self.gradients[i]['db']

        return loss

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return (self.forward(X) > 0.5).astype(int)


# ============================================================================
# VISUALIZATION
# ============================================================================

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


# ============================================================================
# DATA GENERATION
# ============================================================================

def generate_spiral_data(n_samples: int = 200, noise: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """Generate spiral classification data."""
    n = n_samples // 2

    # Class 0: Spiral
    theta0 = np.linspace(0, 3 * np.pi, n)
    r0 = theta0 / (3 * np.pi)
    x0 = r0 * np.cos(theta0) + noise * np.random.randn(n)
    y0 = r0 * np.sin(theta0) + noise * np.random.randn(n)

    # Class 1: Opposite spiral
    theta1 = np.linspace(0, 3 * np.pi, n) + np.pi
    r1 = theta1 / (3 * np.pi)
    x1 = r1 * np.cos(theta1) + noise * np.random.randn(n)
    y1 = r1 * np.sin(theta1) + noise * np.random.randn(n)

    X = np.vstack([np.column_stack([x0, y0]), np.column_stack([x1, y1])])
    y = np.array([0] * n + [1] * n).reshape(-1, 1)

    return X, y


def generate_xor_data(n_samples: int = 200, noise: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """Generate XOR classification data."""
    n = n_samples // 4

    # Four clusters
    centers = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    labels = [0, 1, 1, 0]

    X_list = []
    y_list = []

    for center, label in zip(centers, labels):
        X_list.append(np.random.randn(n, 2) * noise + np.array(center))
        y_list.extend([label] * n)

    X = np.vstack(X_list)
    y = np.array(y_list).reshape(-1, 1)

    return X, y


def generate_circles_data(n_samples: int = 200, noise: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """Generate concentric circles data."""
    n = n_samples // 2

    # Inner circle
    theta_inner = np.random.uniform(0, 2 * np.pi, n)
    r_inner = 0.3 + noise * np.random.randn(n)
    x_inner = r_inner * np.cos(theta_inner)
    y_inner = r_inner * np.sin(theta_inner)

    # Outer circle
    theta_outer = np.random.uniform(0, 2 * np.pi, n)
    r_outer = 0.8 + noise * np.random.randn(n)
    x_outer = r_outer * np.cos(theta_outer)
    y_outer = r_outer * np.sin(theta_outer)

    X = np.vstack([
        np.column_stack([x_inner, y_inner]),
        np.column_stack([x_outer, y_outer])
    ])
    y = np.array([0] * n + [1] * n).reshape(-1, 1)

    return X, y


# ============================================================================
# DEMO
# ============================================================================

def main():
    """Demo of the Neural Network Visualizer."""
    print("=" * 60)
    print("NEURAL NETWORK VISUALIZER")
    print("=" * 60)

    # Generate data
    print("\nGenerating spiral dataset...")
    X, y = generate_spiral_data(n_samples=300, noise=0.15)

    # Normalize data
    X = (X - X.mean(axis=0)) / X.std(axis=0)

    print(f"Data shape: X={X.shape}, y={y.shape}")

    # Create network
    print("\nCreating neural network [2 -> 8 -> 8 -> 4 -> 1]...")
    network = NeuralNetwork(
        layer_sizes=[2, 8, 8, 4, 1],
        activations=['relu', 'relu', 'relu', 'sigmoid']
    )

    # Create visualizer
    visualizer = NeuralNetworkVisualizer(network)

    # Save initial architecture
    visualizer.save_architecture_diagram('nn_architecture_initial.png')

    # Train and visualize
    print("\nTraining network...")

    # Create static visualization of training progress
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    epochs_to_show = [0, 20, 50, 100, 200, 500]
    losses = []

    for idx, target_epoch in enumerate(epochs_to_show):
        # Train to target epoch
        while len(losses) < target_epoch:
            loss = network.backward(X, y, learning_rate=0.5)
            losses.append(loss)

        # Run forward pass for visualization
        network.forward(X)

        # Plot decision boundary
        ax = axes[idx // 3, idx % 3]
        visualizer.draw_decision_boundary(ax, X, y)
        ax.set_title(f'Epoch {target_epoch}, Loss: {losses[-1] if losses else 0:.4f}')

    plt.suptitle('Neural Network Learning Progress', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('nn_training_progress.png', dpi=150, bbox_inches='tight')
    print("Training progress saved to 'nn_training_progress.png'")

    # Final architecture with activations
    fig, ax = plt.subplots(figsize=(12, 8))
    visualizer.draw_architecture(ax, show_values=True)
    ax.set_title('Final Network Architecture with Activations', fontsize=14, fontweight='bold')
    plt.savefig('nn_architecture_final.png', dpi=150, bbox_inches='tight')
    print("Final architecture saved to 'nn_architecture_final.png'")

    # Loss curve
    fig, ax = plt.subplots(figsize=(10, 6))
    visualizer.draw_loss_curve(ax, losses)
    plt.savefig('nn_loss_curve.png', dpi=150, bbox_inches='tight')
    print("Loss curve saved to 'nn_loss_curve.png'")

    # Calculate final accuracy
    predictions = network.predict(X)
    accuracy = np.mean(predictions == y)
    print(f"\nFinal accuracy: {accuracy:.2%}")

    print("\n" + "=" * 60)
    print("Visualization complete!")
    print("Generated files:")
    print("  - nn_architecture_initial.png")
    print("  - nn_architecture_final.png")
    print("  - nn_training_progress.png")
    print("  - nn_loss_curve.png")
    print("=" * 60)

    plt.show()


if __name__ == "__main__":
    main()
