"""
Data generators for neural network visualization demos.
"""

import numpy as np
from typing import Tuple


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
