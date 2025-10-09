"""Centralized wiring for Tkinter event handlers."""
from __future__ import annotations

from gui.controllers import main_controller, cnn_controller, navigation_controller
from gui.controllers.dataset_controller import load_model_and_data, on_dataset_selected


def setup_callbacks(app) -> None:
    app.dataset_selector.bind("<<ComboboxSelected>>", lambda event: on_dataset_selected(app, event))
    app.model_selector.bind("<<ComboboxSelected>>", lambda event: load_model_and_data(app))

    app.cnn_radio.config(command=lambda: main_controller.populate_layer_selector(app))
    app.fc_radio.config(command=lambda: main_controller.populate_layer_selector(app))
    app.layer_selector.bind("<<ComboboxSelected>>", lambda event: main_controller.update_all_visuals(app))

    app.input_panel.prev_button.config(command=lambda: navigation_controller.show_previous_image(app))
    app.input_panel.next_button.config(command=lambda: navigation_controller.show_next_image(app))
    app.index_entry.bind("<Return>", lambda event: main_controller.update_all_visuals(app))

    app.grid_checkbutton.config(command=lambda: main_controller.update_input_panel(app))

    canvas = app.activation_panel.canvas
    canvas.bind("<Button-1>", lambda event: cnn_controller.on_grid_click(app, event))
    canvas.bind("<Configure>", lambda event: navigation_controller.on_canvas_configure(app, event))
    canvas.bind("<MouseWheel>", lambda event: navigation_controller.on_mousewheel(app, event))
    canvas.bind("<Button-4>", lambda event: navigation_controller.on_mousewheel(app, event))
    canvas.bind("<Button-5>", lambda event: navigation_controller.on_mousewheel(app, event))

    app.detail_panel.canvas.bind("<Configure>", lambda event: cnn_controller.on_detail_canvas_resize(app, event))
    app.detail_grid_checkbutton.config(command=lambda: cnn_controller.on_detail_canvas_resize(app, None))

    navigation_controller.enable_controls(app, False)


__all__ = ["setup_callbacks"]
