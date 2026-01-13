"""
Neural Network implementation for visualization purposes.
"""

import numpy as np
from typing import List


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
