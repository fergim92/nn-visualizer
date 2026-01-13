# Neural Network Visualizer

Interactive visualization of neural network architectures and training dynamics. Watch forward propagation, backpropagation, and decision boundaries evolve in real-time.

## Overview

This tool provides educational visualizations of how neural networks learn, making it easier to understand:
- Network architecture and weight connections
- Activation patterns during forward propagation
- Gradient flow during backpropagation
- Decision boundary evolution during training

## Features

### Visualizations

| Visualization | Description |
|---------------|-------------|
| **Architecture Diagram** | Network structure with neurons and weighted connections |
| **Decision Boundary** | Classification regions learned by the network |
| **Loss Curve** | Training loss over epochs |
| **Gradient Flow** | Gradient magnitudes per layer (detects vanishing/exploding gradients) |
| **Training Animation** | Real-time animated training visualization |

### Neural Network Implementation

- Fully connected layers with configurable sizes
- Activation functions: ReLU, Sigmoid, Tanh, Linear
- Xavier weight initialization
- Binary cross-entropy loss
- Gradient clipping and backpropagation

### Dataset Generators

- **Spiral**: Two interleaved spirals (non-linearly separable)
- **XOR**: Classic XOR problem with four clusters
- **Circles**: Concentric circles classification

## Installation

```bash
pip install -r requirements.txt
```

### Requirements
- NumPy >= 1.24.0
- Matplotlib >= 3.7.0

## Usage

### Quick Demo

```bash
python nn_visualizer.py
```

This generates:
- `nn_architecture_initial.png` - Initial network structure
- `nn_architecture_final.png` - Final network with activation values
- `nn_training_progress.png` - Decision boundary at different epochs
- `nn_loss_curve.png` - Training loss over time

### Custom Network

```python
from nn_visualizer import NeuralNetwork, NeuralNetworkVisualizer, generate_spiral_data

# Generate data
X, y = generate_spiral_data(n_samples=300, noise=0.15)

# Create network
network = NeuralNetwork(
    layer_sizes=[2, 16, 16, 8, 1],
    activations=['relu', 'relu', 'relu', 'sigmoid']
)

# Create visualizer
visualizer = NeuralNetworkVisualizer(network)

# Train and visualize
for epoch in range(500):
    loss = network.backward(X, y, learning_rate=0.5)
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Save final visualization
visualizer.save_architecture_diagram('my_network.png')
```

### Animated Training

```python
# Create animated visualization
anim = visualizer.create_training_visualization(
    X, y,
    epochs=200,
    learning_rate=0.5,
    interval=100  # ms between frames
)

# Save as video (requires ffmpeg)
anim.save('training.mp4', writer='ffmpeg', fps=10)

# Or display in notebook
from IPython.display import HTML
HTML(anim.to_jshtml())
```

### Decision Boundary Plot

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 8))
visualizer.draw_decision_boundary(ax, X, y, resolution=150)
plt.savefig('decision_boundary.png', dpi=150)
```

## Architecture Visualization

The architecture diagram shows:
- **Neurons**: Colored by activation value (red=0, yellow=0.5, green=1)
- **Connections**: Width proportional to weight magnitude
- **Connection Color**: Blue for positive weights, orange for negative

```
Input Layer    Hidden Layers    Output Layer
    O ─────────── O ─────────── O
    O ─────────── O ─────────── │
                  O             │
                                ▼
                            Prediction
```

## Network Configuration

### Layer Sizes

```python
# Simple network for XOR
network = NeuralNetwork([2, 4, 1])

# Deep network for complex patterns
network = NeuralNetwork([2, 32, 32, 16, 8, 1])
```

### Activation Functions

```python
network = NeuralNetwork(
    layer_sizes=[2, 8, 8, 1],
    activations=['relu', 'tanh', 'sigmoid']  # One per layer transition
)
```

| Activation | Use Case |
|------------|----------|
| `relu` | Hidden layers (default) |
| `sigmoid` | Binary output layer |
| `tanh` | Hidden layers, centered output |
| `linear` | Regression output |

## Understanding the Visualizations

### Decision Boundary
- **Green region**: Predicted class 1
- **Red region**: Predicted class 0
- **Black line**: Decision boundary (0.5 threshold)
- **Points**: Training data colored by true label

### Gradient Flow
- Monitors gradient magnitudes per layer
- **Vanishing gradients**: Very small bars in early layers
- **Exploding gradients**: Very large bars
- **Healthy training**: Similar magnitude across layers

### Loss Curve
- Should decrease over training
- Plateau indicates convergence
- Oscillation may indicate high learning rate

## Example Output

After running the demo, you'll see the network learn to classify spiral data:

| Epoch 0 | Epoch 100 | Epoch 500 |
|---------|-----------|-----------|
| Random boundary | Emerging pattern | Clean separation |

## Extending the Project

### Add New Activation
```python
def _activate(self, x, activation):
    if activation == 'swish':
        return x * self._activate(x, 'sigmoid')
    # ... existing activations
```

### Add Regularization
```python
# L2 regularization in backward pass
l2_penalty = 0.01 * sum(np.sum(w**2) for w in self.weights)
loss += l2_penalty
```

### Multi-class Classification
Replace sigmoid with softmax and use cross-entropy loss.

## License

MIT License
