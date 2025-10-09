"""Compatibility module re-exporting GUI controllers."""
from __future__ import annotations

from gui.controllers.dataset_controller import load_model_and_data, on_dataset_selected
from gui.controllers.main_controller import (
    populate_layer_selector,
    update_all_visuals,
    update_input_panel,
    update_activation_panels,
)

from gui.controllers.cnn_controller import (
    on_grid_click,
    update_activation_detail,
    update_kernel_panel,
    on_detail_canvas_resize,
)

from gui.controllers.navigation_controller import (
    enable_controls,
    on_canvas_configure,
    on_mousewheel,
)
from gui.controllers.event_wiring import setup_callbacks
from gui.widgets.channel_viewer import create_channel_viewer

__all__ = [
    "on_dataset_selected",
    "load_model_and_data",
    "populate_layer_selector",
    "update_all_visuals",
    "update_input_panel",
    "update_activation_panels",
    "on_grid_click",
    "update_activation_detail",
    "update_kernel_panel",
    "on_detail_canvas_resize",
    "on_canvas_configure",
    "on_mousewheel",
    "enable_controls",
    "setup_callbacks",
    "create_channel_viewer",
]
