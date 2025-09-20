"""Compatibility layer exposing utility helpers for legacy imports."""
from __future__ import annotations

from core.utils import (
    find_datasets,
    find_models_in_dataset,
    load_class_map,
    get_model_layers,
    format_weight,
)
from gui.widgets.feature_map_canvas import visualize_feature_maps
from gui.widgets.vector_canvas import visualize_vector
from gui.widgets.grid_overlay import draw_pixel_grid, draw_detail_pixel_grid

__all__ = [
    "find_datasets",
    "find_models_in_dataset",
    "load_class_map",
    "get_model_layers",
    "format_weight",
    "visualize_feature_maps",
    "visualize_vector",
    "draw_pixel_grid",
    "draw_detail_pixel_grid",
]
