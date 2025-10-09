"""Geometry helpers for FC layer drawing."""
from __future__ import annotations

from typing import Iterable, List

import numpy as np

__all__ = [
    "compute_y_positions",
    "compute_node_radius",
    "compute_label_font_size",
    "compute_weight_label_size",
    "pick_x_positions",
]


PADDING_Y = 60


def compute_y_positions(canvas_h: int, count: int, padding_y: int = PADDING_Y) -> np.ndarray:
    """Evenly space ``count`` positions along the canvas height."""
    if count <= 0:
        return np.array([], dtype=float)
    if count == 1:
        return np.array([canvas_h / 2], dtype=float)
    return np.linspace(padding_y, canvas_h - padding_y, count, dtype=float)


def compute_node_radius(spacing: float) -> float:
    """Derive a node radius from the vertical spacing."""
    return max(3.0, min(10.0, spacing * 0.45))


def compute_label_font_size(radius: float) -> int:
    """Pick a font size that fits inside a node of the given radius."""
    return max(6, int(radius * 0.8))


def compute_weight_label_size(canvas_h: int, spacing: float, norm_weight: float | None = None) -> int:
    """Determine the font size for highlighted weight labels."""
    base = max(8, min(14, int(canvas_h * 0.02 + spacing * 0.05)))
    if norm_weight is not None:
        base = max(8, min(16, base + int(abs(norm_weight) * 3)))
    return base


def pick_x_positions(canvas_w: int, num_layers: int) -> List[float]:
    """Return evenly spaced x-coordinates for a given number of layers."""
    if num_layers <= 0:
        return []
    if num_layers == 1:
        return [canvas_w / 2]
    
    # Use a margin on the left and right
    margin = canvas_w * 0.1
    return np.linspace(margin, canvas_w - margin, num_layers, dtype=float).tolist()