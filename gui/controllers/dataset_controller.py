"""Callbacks related to dataset and model selection."""
from __future__ import annotations

import os
from tkinter import messagebox

from core.utils import find_models_in_dataset, load_class_map
from gui.services import dataset_service, model_loader
from gui.controllers import activation_controller
from gui.controllers.navigation_controller import enable_controls


def on_dataset_selected(app, event=None) -> None:
    dataset_path = app.dataset_selector.get()
    if not dataset_path:
        return

    state = app.state
    state.reset_dataset()
    state.reset_model()

    app.model_selector.set("")
    app.layer_selector.set("")

    try:
        models_found = find_models_in_dataset(dataset_path)
        app.model_selector["values"] = models_found
        if models_found:
            app.model_selector.set(models_found[0])
            load_model_and_data(app)
        else:
            app.status_var.set(f"Dataset selected, but no model (.pth) found in '{dataset_path}'.")
            enable_controls(app, False)
    except Exception as exc:  # pylint: disable=broad-except
        messagebox.showerror("Error", f"Could not process dataset directory: {exc}")


def load_model_and_data(app) -> None:
    dataset_path = app.dataset_selector.get()
    model_path = app.model_selector.get()
    if not dataset_path or not model_path:
        enable_controls(app, False)
        return

    state = app.state

    try:
        state.reset_dataset()
        images_zip, labels_zip, filenames = dataset_service.open_archives(dataset_path)
        state.images_zip = images_zip
        state.labels_zip = labels_zip
        state.file_names = filenames
        state.class_map = load_class_map(dataset_path)

        if not state.file_names:
            raise RuntimeError("No PNG images found inside images.zip")

        img_size = dataset_service.peek_image_size(state.images_zip, state.file_names[0])
        num_classes = len(state.class_map)

        state.model = model_loader.load_model(model_path, img_size=img_size, num_classes=num_classes)
        state.model_layers = model_loader.collect_layer_metadata(state.model)
        state.selected_layer_name = None
        state.selected_filter_index = -1
        state.selected_kernel_channel = 0
        state.last_detail_maps = None
        state.last_detail_index = -1

        activation_controller.populate_layer_selector(app)
        app.status_var.set(f"Model '{os.path.basename(model_path)}' loaded. Ready to explore.")
        enable_controls(app, True)
        activation_controller.update_all_visuals(app)
    except Exception as exc:  # pylint: disable=broad-except
        messagebox.showerror("Loading Error", f"An error occurred: {exc}")
        enable_controls(app, False)


__all__ = ["on_dataset_selected", "load_model_and_data"]
