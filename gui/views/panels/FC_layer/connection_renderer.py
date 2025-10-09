"""Connection rendering utilities for FC explorer."""
from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

from .layout_utils import compute_weight_label_size

__all__ = ["ConnectionRenderer"]

Coords = Sequence[Tuple[float, float]]


def _to_numpy(matrix) -> Optional[np.ndarray]:
    if matrix is None:
        return None
    tensor = matrix
    if hasattr(tensor, "detach"):
        tensor = tensor.detach()
    if hasattr(tensor, "cpu"):
        tensor = tensor.cpu()
    return np.asarray(tensor, dtype=float)


def _estimate_spacing(coords: Coords) -> float:
    if len(coords) < 2:
        return 0.0
    diffs = [abs(coords[i + 1][1] - coords[i][1]) for i in range(len(coords) - 1)]
    return float(np.mean(diffs)) if diffs else 0.0


class ConnectionRenderer:
    """Draws weight connections between consecutive FC layers."""

    def __init__(self, canvas, edge_threshold: float):
        self.canvas = canvas
        self.edge_threshold = edge_threshold

    def draw_pair(
        self,
        coords1: Coords,
        coords2: Coords,
        weights,
        idx1: Sequence[int],
        idx2: Sequence[int],
        name1: str,
        name2: str,
        selected_neuron: Tuple[str, int] | None,
        activations1: Iterable[float],
    ) -> None:
        if not coords1 or not coords2:
            return

        weight_matrix = _to_numpy(weights)
        if weight_matrix is None:
            return

        weight_matrix = weight_matrix.T
        weight_matrix = weight_matrix[np.ix_(idx1, idx2)]

        if weight_matrix.shape != (len(coords1), len(coords2)):
            return

        abs_max = np.max(np.abs(weight_matrix))
        if abs_max < 1e-9:
            return

        sel_layer, sel_idx = selected_neuron if selected_neuron else (None, None)
        if sel_layer and sel_layer not in (name1, name2):
            return

        canvas_h = self.canvas.winfo_height()
        spacing = _estimate_spacing(coords2)

        activations_np = np.asarray(activations1)

        for i, (x1, y1) in enumerate(coords1):
            if sel_layer == name1 and sel_idx != i:
                continue
            for j, (x2, y2) in enumerate(coords2):
                if sel_layer == name2 and sel_idx != j:
                    continue

                weight = weight_matrix[i, j]
                norm = weight / abs_max
                if abs(norm) <= self.edge_threshold:
                    continue

                highlighted = (sel_layer == name1 and sel_idx == i) or (
                    sel_layer == name2 and sel_idx == j
                )

                color = ("#ff6b6b" if weight > 0 else "#ff9999") if highlighted else (
                    "black" if weight > 0 else "lightgrey"
                )
                thickness = (1.0 + abs(norm) * 4.0) if highlighted else (
                    0.5 + abs(norm) * 3.0
                )

                self.canvas.create_line(x1, y1, x2, y2, fill=color, width=thickness, tags="network_item")

                if highlighted:
                    activation_value = activations_np[i]
                    activation_flow = activation_value * weight

                    font_size = compute_weight_label_size(canvas_h, spacing, norm)
                    mid_x = x1 + 0.6 * (x2 - x1)
                    mid_y = y1 + 0.6 * (y2 - y1)
                    self.canvas.create_text(
                        mid_x,
                        mid_y,
                        text=f"{activation_flow:.2f}",
                        font=("", font_size, "bold"),
                        fill="#00008B",
                        tags="network_item",
                    )

