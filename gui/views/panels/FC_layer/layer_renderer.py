"""Layer renderer for FC visualisation."""
from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .layout_utils import (
    PADDING_Y,
    compute_label_font_size,
    compute_node_radius,
    compute_y_positions,
)

__all__ = ["LayerRenderer"]

NodeCoords = List[Tuple[float, float]]


class LayerRenderer:
    """Draws neuron columns for the FC explorer."""

    def __init__(self, canvas, neuron_tags: Dict[int, Tuple[str, int]], icon_factory=None):
        self.canvas = canvas
        self.neuron_tags = neuron_tags
        self.icon_factory = icon_factory

    def draw_layer(
        self,
        vector: Sequence[float],
        x_center: float,
        title: str,
        is_subsampled: bool,
        indices: Sequence[int],
        selected_neuron: Tuple[str, int] | None,
        is_output_layer: bool = False,
    ) -> NodeCoords:
        canvas_h = self.canvas.winfo_height()
        self.canvas.create_text(
            x_center,
            canvas_h - 15,
            text=f"{title} ({len(vector)})",
            font=("", 9, "bold"),
            anchor="s",
            tags="network_item",
        )
        if is_subsampled:
            self.canvas.create_text(
                x_center,
                canvas_h - 30,
                text="(subsampled)",
                font=("", 7, "italic"),
                anchor="s",
                fill="gray",
                tags="network_item",
            )

        if len(vector) == 0:
            return []

        values = np.asarray(vector)
        y_positions = compute_y_positions(canvas_h, len(values), PADDING_Y)
        if len(y_positions) > 1:
            spacing = y_positions[1] - y_positions[0]
        else:
            spacing = canvas_h
        radius = compute_node_radius(spacing)

        coords: NodeCoords = []
        selected = selected_neuron

        for i, y in enumerate(y_positions):
            idx = int(indices[i]) if len(indices) > i else i
            is_selected = selected == (title, i)

            if is_output_layer and self.icon_factory is not None:
                icon = self.icon_factory.get_by_index(idx)
                if icon is not None:
                    if is_selected:
                        highlight_radius = max(icon.width(), icon.height()) / 2 + 6
                        ring_id = self.canvas.create_oval(
                            x_center - highlight_radius,
                            y - highlight_radius,
                            x_center + highlight_radius,
                            y + highlight_radius,
                            outline="#0d47a1",
                            width=3,
                            tags="network_item",
                        )
                        self.neuron_tags[ring_id] = (title, i)
                        self.canvas.tag_lower(ring_id)
                    image_id = self.canvas.create_image(x_center, y, image=icon, tags="network_item",)
                    self.neuron_tags[image_id] = (title, i)
                    coords.append((x_center, y))
                    continue

            fill = "#ff6b6b" if is_selected else "#cce7ff"
            outline = "red" if is_selected else "gray30"
            width = 2 if is_selected else 1

            item_id = self.canvas.create_oval(
                x_center - radius,
                y - radius,
                x_center + radius,
                y + radius,
                fill=fill,
                outline=outline,
                width=width,
                tags="network_item",
            )
            self.neuron_tags[item_id] = (title, i)
            coords.append((x_center, y))

            if radius > 6:
                font_size = compute_label_font_size(radius)
                self.canvas.create_text(
                    x_center,
                    y,
                    text=str(idx),
                    font=("", font_size, "bold"),
                    fill="gray10",
                    state="disabled",
                    tags="network_item",
                )

        return coords
