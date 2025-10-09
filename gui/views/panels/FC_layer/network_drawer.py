"""Orchestrator for FC network visualisation."""
from __future__ import annotations

from typing import Dict, List

from .connection_renderer import ConnectionRenderer
from .constants import MAX_NEURONS_TO_DRAW
from .data_prep import prepare_single_layer_data
from .layer_renderer import LayerRenderer
from .layout_utils import pick_x_positions
from .output_icons import OutputIconFactory


class NetworkDrawer:
    """Coordinates drawing of neurons and connections."""

    def __init__(self, panel):
        self.panel = panel
        self.canvas = panel.canvas
        self.icon_factory = OutputIconFactory()
        self.layer_renderer = LayerRenderer(self.canvas, panel.neuron_tags, self.icon_factory)

    def draw_network(self, layers_data: List[Dict], threshold: float):
        self.canvas.delete("network_item")
        self.panel.neuron_tags.clear()
        self.panel.update_idletasks()

        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()

        if not layers_data or all(d.get("activation") is None for d in layers_data):
            self.canvas.create_text(
                canvas_w / 2, canvas_h / 2,
                text="FC activation data not available.",
                anchor="center", font=("", 10), tags="network_item"
            )
            return

        num_layers = len(layers_data)
        x_positions = pick_x_positions(canvas_w, num_layers)
        selected = self.panel.selected_neuron

        # --- Process and store all layer data first ---
        processed_layers = []
        for i, layer_info in enumerate(layers_data):
            vector, subsampled, indices = prepare_single_layer_data(
                layer_info["activation"], MAX_NEURONS_TO_DRAW
            )
            processed_layers.append({
                "name": layer_info["name"],
                "vector": vector,
                "subsampled": subsampled,
                "indices": indices,
                "x_pos": x_positions[i],
                "coords": [] # To be filled next
            })

        # --- First Pass: Draw all layers and store their node coordinates ---
        for i in range(num_layers):
            layer = processed_layers[i]
            # Check if it's the last layer to determine if it's an output layer
            is_output = (i == num_layers - 1)
            
            coords = self.layer_renderer.draw_layer(
                layer["vector"], layer["x_pos"], layer["name"],
                layer["subsampled"], layer["indices"], selected,
                is_output_layer=is_output
            )
            layer["coords"] = coords

        # --- Second Pass: Draw all connections between layers ---
        connection_renderer = ConnectionRenderer(self.canvas, threshold)
        for i in range(num_layers - 1):
            layer1 = processed_layers[i]
            layer2 = processed_layers[i+1]
            
            # The weights connecting layer1 and layer2 are stored in layer2's data
            weight_tensor = layers_data[i+1]["weight"]

            connection_renderer.draw_pair(
                layer1["coords"], layer2["coords"],
                weight_tensor,
                layer1["indices"], layer2["indices"],
                layer1["name"], layer2["name"],
                selected,
                layer1["vector"]
            )

        # --- Third Pass: Redraw layers so nodes are on top of connections ---
        for i in range(num_layers):
            layer = processed_layers[i]
            is_output = (i == num_layers - 1)
            self.layer_renderer.draw_layer(
                layer["vector"], layer["x_pos"], layer["name"],
                layer["subsampled"], layer["indices"], selected,
                is_output_layer=is_output
            )

        self.canvas.create_text(
            canvas_w / 2, 10,
            text="Right-click a neuron to highlight connections | Double-click to reset view",
            font=("", 8), fill="gray50", anchor="n", tags="network_item"
        )