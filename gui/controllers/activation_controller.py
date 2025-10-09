# """Visualization and activation-related controllers."""
# from __future__ import annotations

# import numpy as np
# import torch
# from PIL import Image, ImageTk
# from torchvision import transforms
# from tkinter import ttk

# from gui.services import dataset_service
# from gui.widgets.feature_map_canvas import visualize_feature_maps
# from gui.widgets.vector_canvas import visualize_vector
# from gui.widgets.grid_overlay import draw_pixel_grid, draw_detail_pixel_grid
# from gui.widgets.channel_viewer import create_channel_viewer
# from gui.views.panels.fc_view_panel import FCViewPanel


# def populate_layer_selector(app) -> None:
#     layer_type = app.layer_type_var.get()

#     app.switch_view(layer_type)

#     layers = app.state.model_layers.get(layer_type, []) if app.state.model_layers else []
#     app.layer_selector["values"] = layers

#     app.state.selected_layer_name = None
#     app.state.selected_filter_index = -1
#     app.state.selected_kernel_channel = 0
#     app.state.last_detail_maps = None
#     app.state.last_detail_index = -1

#     if layers:
#         app.layer_selector.set(layers[0])
#         app.layer_selector.config(state="disabled" if layer_type == "FC" else "normal")
#     else:
#         app.layer_selector.set("")
#         app.layer_selector.config(state="normal")
#     update_all_visuals(app)


# def update_all_visuals(app) -> None:
#     state = app.state
#     if not (state.model and state.images_zip and state.file_names):
#         return
#     update_input_panel(app)
#     update_activation_panels(app)


# def update_input_panel(app) -> None:
#     state = app.state
#     try:
#         idx = int(app.index_var.get())
#         if not (0 <= idx < len(state.file_names)):
#             app.status_var.set("Invalid index.")
#             return

#         filename = state.file_names[idx]
#         image = dataset_service.read_image(state.images_zip, filename)
#         original_size = image.size

#         app.image_canvas.update_idletasks()
#         canvas_width = app.image_canvas.winfo_width() or 280
#         canvas_height = app.image_canvas.winfo_height() or 280
#         if canvas_width < 20 or canvas_height < 20:
#             canvas_width = canvas_height = 280
#         new_size = min(canvas_width, canvas_height)
#         canvas_size = (new_size, new_size)
#         resized = image.resize(canvas_size, Image.NEAREST)

#         tk_image = ImageTk.PhotoImage(resized)
#         app.image_canvas.delete("all")
#         x_offset = (canvas_width - new_size) // 2
#         y_offset = (canvas_height - new_size) // 2
#         app.image_canvas.create_image(x_offset, y_offset, image=tk_image, anchor="nw")
#         app.image_canvas.image = tk_image

#         if app.show_grid_var.get():
#             draw_pixel_grid(app.image_canvas, original_size, canvas_size, (x_offset, y_offset))

#         input_tensor = transforms.ToTensor()(image).unsqueeze(0)
#         with torch.no_grad():
#             outputs = state.model(input_tensor)
#         _, pred_id = torch.max(outputs, 1)
#         pred_id = pred_id.item()

#         label_text_path = filename.replace(".png", ".txt")
#         label_id = int(state.labels_zip.read(label_text_path).decode("utf-8").strip())

#         app.label_var.set(f"True Label: {label_id} ({state.class_map.get(label_id, '?')})")
#         app.pred_var.set(f"Prediction: {pred_id} ({state.class_map.get(pred_id, '?')})")

#         arr = np.array(image)
#         flat = arr.reshape(-1, 3)
#         colors, counts = np.unique(flat, axis=0, return_counts=True)
#         order = np.argsort(counts)[::-1]
#         bg_rgb = tuple(int(x) for x in colors[order[0]]) if len(order) else (0, 0, 0)
#         shape_rgb = bg_rgb
#         for candidate in order[1:]:
#             shape_rgb = tuple(int(x) for x in colors[candidate])
#             break

#         app.shape_color_var.set(f"Shape Color: ({shape_rgb[0]}, {shape_rgb[1]}, {shape_rgb[2]})")
#         app.bg_color_var.set(f"Background Color: ({bg_rgb[0]}, {bg_rgb[1]}, {bg_rgb[2]})")
#         app.status_var.set(f"Showing image {idx}/{len(state.file_names) - 1}")
#     except Exception as exc:  # pylint: disable=broad-except
#         app.status_var.set(f"Error displaying image: {exc}")
#         app.shape_color_var.set("Shape Color: (---)")
#         app.bg_color_var.set("Background Color: (---)")



# def update_activation_panels(app) -> None:
#     state = app.state
#     layer_type = app.layer_type_var.get()

#     # Clear all panels initially to prevent leftover visuals
#     app.fc_panel.update_view([])
#     update_activation_detail(app, None, -1)
#     update_kernel_panel(app, None, -1)
#     app.activation_grid_canvas.delete("all")

#     if not state.model:
#         return

#     if layer_type == "FC":
#         fc_layer_info = state.model_layers.get("FC", [])
#         if not fc_layer_info:
#             app.fc_panel.update_view([])
#             return

#         layers_data = []
#         activations = state.model.activations

#         # --- Step 1: Find the true input to the first FC layer ---
#         first_fc_name = fc_layer_info[0].split(" ")[0]
#         all_layer_names = [name for name, _ in state.model.named_modules()]
#         input_tensor = None
#         try:
#             first_fc_idx = all_layer_names.index(first_fc_name)
#             if first_fc_idx > 0:
#                 input_source_name = all_layer_names[first_fc_idx - 1]
#                 input_tensor = activations.get(input_source_name)
#         except (ValueError, IndexError):
#             pass # input_tensor remains None

#         # Add the "Input" layer as the first element
#         layers_data.append({
#             "name": "Input",
#             "activation": input_tensor.view(input_tensor.size(0), -1) if input_tensor is not None else None,
#             "weight": None, # Input layer has no incoming weights
#             "bias": None,
#         })

#         # --- Step 2: Loop through all detected FC layers ---
#         for info_str in fc_layer_info:
#             layer_name = info_str.split(" ")[0]
#             layer_module = getattr(state.model, layer_name, None)
            
#             if layer_module and hasattr(layer_module, 'weight'):
#                 layer_activation = activations.get(layer_name)
#                 layer_weight = layer_module.weight.data if hasattr(layer_module, 'weight') else None
#                 layer_bias = layer_module.bias.data if hasattr(layer_module, 'bias') and layer_module.bias is not None else None

#                 layers_data.append({
#                     "name": layer_name,
#                     "activation": layer_activation,
#                     "weight": layer_weight,
#                     "bias": layer_bias,
#                 })
        
#         app.fc_panel.update_view(layers_data)
#         return # End of FC logic

#     # --- CNN Visualization Logic ---
#     layer_label = app.layer_selector.get()
#     if not layer_label:
#         return # Nothing selected, nothing to do

#     layer_name = layer_label.split(" ")[0]
#     activation_tensor = state.model.activations.get(layer_name)

#     if activation_tensor is None:
#         return # No activation data for this layer

#     if state.selected_layer_name != layer_name:
#         state.selected_layer_name = layer_name
#         state.selected_filter_index = -1 # Reset filter selection when layer changes

#     state.tk_grid_images.clear()

#     # We are in the 'CNN' view, so we always treat the activations as feature maps.
#     # This correctly handles conv, pool, ReLU, and other CNN-related layers.
#     feature_maps = activation_tensor.squeeze(0)
#     visualize_feature_maps(app.activation_grid_canvas, feature_maps, state.tk_grid_images)

#     # Update detail and kernel panels if a valid filter is selected for a conv layer
#     if "conv" in layer_name and 0 <= state.selected_filter_index < feature_maps.shape[0]:
#         update_activation_detail(app, feature_maps, state.selected_filter_index)
#         update_kernel_panel(app, layer_name, state.selected_filter_index)
#     else:
#         # Otherwise, ensure they are cleared
#         update_activation_detail(app, None, -1)
#         update_kernel_panel(app, None, -1)


# def on_grid_click(app, event) -> None:
#     canvas = event.widget
#     if not hasattr(canvas, "viz_info"):
#         return
#     info = canvas.viz_info
#     x_coord, y_coord = event.x, event.y
#     col = x_coord // (info["img_w"] + info["pad"])
#     row = y_coord // (info["img_h"] + info["pad"])
#     filter_index = row * info["cols"] + col

#     maps = info.get("maps")
#     if maps is None or not (0 <= filter_index < maps.shape[0]):
#         return

#     layer_label = app.layer_selector.get()
#     if not layer_label:
#         return
#     layer_name = layer_label.split(" ")[0]

#     app.state.selected_layer_name = layer_name
#     app.state.selected_filter_index = filter_index
#     app.state.selected_kernel_channel = 0

#     update_activation_detail(app, maps, filter_index)
#     update_kernel_panel(app, layer_name, filter_index)


# def update_activation_detail(app, feature_maps, filter_index: int) -> None:
#     state = app.state
#     state.last_detail_maps = feature_maps
#     state.last_detail_index = filter_index

#     app.detail_canvas.delete("all")
#     if feature_maps is None or filter_index < 0:
#         app.detail_frame.config(text="Activation Detail")
#         app.stats_var.set("Stats: (select a filter)")
#         return

#     app.detail_frame.config(text=f"Activation Detail (Filter {filter_index + 1} of {feature_maps.shape[0]})")
#     selected_map = feature_maps[filter_index]
#     min_val = selected_map.min().item()
#     max_val = selected_map.max().item()
#     mean_val = selected_map.mean().item()
#     app.stats_var.set(f"Stats: Min={min_val:.4f} | Max={max_val:.4f} | Mean={mean_val:.4f}")

#     normalized = (selected_map - min_val) / (max_val - min_val + 1e-8)
#     detail_data = (normalized * 255).byte().numpy()

#     app.detail_canvas.update_idletasks()
#     width = app.detail_canvas.winfo_width()
#     height = app.detail_canvas.winfo_height()
#     size = max(16, min(width, height) - 10)
#     if size < 16:
#         return

#     pil_img = Image.fromarray(detail_data).resize((size, size), Image.NEAREST)
#     state.tk_detail_image = ImageTk.PhotoImage(pil_img)
#     app.detail_canvas.create_image(5, 5, image=state.tk_detail_image, anchor="nw")

#     if app.show_detail_grid_var.get():
#         draw_detail_pixel_grid(app.detail_canvas, selected_map.shape, (size, size), (5, 5))


# def update_kernel_panel(app, layer_name, filter_index: int) -> None:
#     state = app.state
#     for widget in app.kernel_panel_frame.winfo_children():
#         widget.destroy()

#     if not layer_name or "conv" not in layer_name or filter_index < 0:
#         ttk.Label(app.kernel_panel_frame, text="Select a CNN filter to see its kernel.").pack(pady=20)
#         return

#     layer_attr = layer_name.split(" ")[0]
#     layer = getattr(state.model, layer_attr, None)
#     if layer is None:
#         return

#     weights = layer.weight.data[filter_index]
#     bias = layer.bias.data[filter_index]
#     in_channels = weights.shape[0]

#     init_channel = state.selected_kernel_channel
#     if not (0 <= init_channel < in_channels):
#         init_channel = 0
#     state.selected_kernel_channel = init_channel

#     ttk.Label(app.kernel_panel_frame, text=f"Input Channels: {in_channels}", font=("", 10)).pack(pady=(5, 0))
#     ttk.Label(app.kernel_panel_frame, text=f"Bias: b = {bias.item():>7.4f}", font=("", 10, "bold")).pack(pady=(0, 5))

#     def on_channel_change(index: int) -> None:
#         state.selected_kernel_channel = index

#     create_channel_viewer(app.kernel_panel_frame, "Viewer A", weights, in_channels, on_channel_change, initial_index=init_channel)
#     ttk.Separator(app.kernel_panel_frame, orient="horizontal").pack(fill="x", pady=10, padx=5)
#     create_channel_viewer(app.kernel_panel_frame, "Viewer B", weights, in_channels, on_channel_change, initial_index=init_channel)


# def on_detail_canvas_resize(app, _event=None) -> None:
#     update_activation_detail(app, app.state.last_detail_maps, app.state.last_detail_index)


# __all__ = [
#     "populate_layer_selector",
#     "update_all_visuals",
#     "update_input_panel",
#     "update_activation_panels",
#     "on_grid_click",
#     "update_activation_detail",
#     "update_kernel_panel",
#     "on_detail_canvas_resize",
# ]
