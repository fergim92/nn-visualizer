"""
Data models for the neural network visualizer.
"""

from dataclasses import dataclass


@dataclass
class LayerConfig:
    """Configuration for a neural network layer."""
    input_size: int
    output_size: int
    activation: str = "relu"
