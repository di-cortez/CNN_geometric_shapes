"""High-level controllers for main application state and input panel."""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image, ImageTk
from torchvision import transforms

from gui.services import dataset_service
from gui.widgets.grid_overlay import draw_pixel_grid
from . import cnn_controller, fc_controller


def populate_layer_selector(app) -> None:
    """Populates the layer dropdown based on the selected view (CNN/FC)."""
    layer_type = app.layer_type_var.get()
    app.switch_view(layer_type)
    layers = app.state.model_layers.get(layer_type, []) if app.state.model_layers else []
    app.layer_selector["values"] = layers
    app.state.selected_layer_name = None
    app.state.selected_filter_index = -1
    if layers:
        app.layer_selector.set(layers[0])
        is_fc = layer_type == "FC"
        app.layer_selector.config(state="disabled" if is_fc else "readonly")
    else:
        app.layer_selector.set("")
        app.layer_selector.config(state="readonly")
    update_all_visuals(app)


def update_all_visuals(app) -> None:
    """Updates all visual components of the application."""
    state = app.state
    if not (state.model and state.images_zip and state.file_names):
        return
    update_input_panel(app)
    update_activation_panels(app)


def update_input_panel(app) -> None:
    """Updates the input image panel with the current sample."""
    state = app.state
    try:
        idx = int(app.index_var.get())
        if not (0 <= idx < len(state.file_names)):
            return

        filename = state.file_names[idx]
        image = dataset_service.read_image(state.images_zip, filename)

        app.image_canvas.update_idletasks()
        canvas_width = app.image_canvas.winfo_width() or 280
        canvas_height = app.image_canvas.winfo_height() or 280
        new_size = min(canvas_width, canvas_height)
        resized = image.resize((new_size, new_size), Image.NEAREST)

        tk_image = ImageTk.PhotoImage(resized)
        app.image_canvas.delete("all")
        app.image_canvas.create_image(0, 0, image=tk_image, anchor="nw")
        app.image_canvas.image = tk_image

        if app.show_grid_var.get():
            draw_pixel_grid(app.image_canvas, image.size, (new_size, new_size), (0, 0))

        # Run model prediction
        input_tensor = transforms.ToTensor()(image).unsqueeze(0)
        with torch.no_grad():
            outputs = state.model(input_tensor)
        _, pred_id = torch.max(outputs, 1)

        # Update labels
        label_id = int(state.labels_zip.read(filename.replace(".png", ".txt")).decode())
        app.label_var.set(f"True Label: {label_id} ({state.class_map.get(label_id, '?')})")
        app.pred_var.set(f"Prediction: {pred_id.item()} ({state.class_map.get(pred_id.item(), '?')})")
        
        # --- THIS IS THE MISSING CODE BLOCK ---
        # Calculate the most frequent colors to identify shape and background
        arr = np.array(image)
        flat = arr.reshape(-1, 3)
        colors, counts = np.unique(flat, axis=0, return_counts=True)
        order = np.argsort(counts)[::-1]
        
        bg_rgb = tuple(int(x) for x in colors[order[0]]) if len(order) > 0 else (0, 0, 0)
        shape_rgb = tuple(int(x) for x in colors[order[1]]) if len(order) > 1 else bg_rgb

        app.shape_color_var.set(f"Shape Color: {shape_rgb}")
        app.bg_color_var.set(f"Background Color: {bg_rgb}")
        # --- END OF MISSING CODE BLOCK ---

        app.status_var.set(f"Showing image {idx}/{len(state.file_names) - 1}")

    except Exception as exc:
        app.status_var.set(f"Error: {exc}")
        # --- ALSO RESTORE THESE LINES IN THE ERROR CASE ---
        app.shape_color_var.set("Shape Color: (---)")
        app.bg_color_var.set("Background Color: (---)")

def update_activation_panels(app) -> None:
    """Routes the update request to the appropriate controller (CNN or FC)."""
    layer_type = app.layer_type_var.get()

    # Clear panels to prevent leftover visuals from other views
    cnn_controller.clear_panels(app)
    fc_controller.clear_panels(app)

    if not app.state.model:
        return

    if layer_type == "FC":
        fc_controller.update_fc_view(app)
    else: # CNN
        cnn_controller.update_cnn_view(app)