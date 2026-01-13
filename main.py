#!/usr/bin/env python3
"""
Neural Network Visualizer - Demo Entry Point
=============================================
Run this script to see the neural network visualization in action.

Usage:
    python main.py
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from nn_visualizer import NeuralNetwork, NeuralNetworkVisualizer, generate_spiral_data


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
