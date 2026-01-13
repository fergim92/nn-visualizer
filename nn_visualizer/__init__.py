"""
Neural Network Visualizer
==========================
Interactive visualization of neural network architectures and training.
Shows forward propagation, backpropagation, and decision boundaries in real-time.
"""

from .models import LayerConfig
from .network import NeuralNetwork
from .visualizer import NeuralNetworkVisualizer
from .data_generators import generate_spiral_data, generate_xor_data, generate_circles_data

__version__ = "1.0.0"
__author__ = "Fernando Gimenez"

__all__ = [
    "LayerConfig",
    "NeuralNetwork",
    "NeuralNetworkVisualizer",
    "generate_spiral_data",
    "generate_xor_data",
    "generate_circles_data",
]
