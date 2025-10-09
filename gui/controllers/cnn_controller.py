"""Controllers for the CNN visualization view."""
from __future__ import annotations

from PIL import Image, ImageTk
from tkinter import ttk

from gui.widgets.feature_map_canvas import visualize_feature_maps
from gui.widgets.grid_overlay import draw_detail_pixel_grid
from gui.widgets.channel_viewer import create_channel_viewer


def clear_panels(app) -> None:
    """Clears all visual elements related to the CNN view."""
    update_activation_detail(app, None, -1)
    update_kernel_panel(app, None, -1)
    app.activation_grid_canvas.delete("all")

def update_cnn_view(app) -> None:
    """Updates all panels in the CNN view."""
    state = app.state
    layer_label = app.layer_selector.get()
    if not layer_label:
        return

    layer_name = layer_label.split(" ")[0]
    activation_tensor = state.model.activations.get(layer_name)
    if activation_tensor is None:
        return

    if state.selected_layer_name != layer_name:
        state.selected_layer_name = layer_name
        state.selected_filter_index = -1

    state.tk_grid_images.clear()
    feature_maps = activation_tensor.squeeze(0)
    visualize_feature_maps(app.activation_grid_canvas, feature_maps, state.tk_grid_images)

    if "conv" in layer_name and 0 <= state.selected_filter_index < feature_maps.shape[0]:
        update_activation_detail(app, feature_maps, state.selected_filter_index)
        update_kernel_panel(app, layer_name, state.selected_filter_index)
    else:
        update_activation_detail(app, None, -1)
        update_kernel_panel(app, None, -1)


def on_grid_click(app, event) -> None:
    # (This function is correct and unchanged)
    canvas = event.widget
    if not hasattr(canvas, "viz_info"): return
    info, x, y = canvas.viz_info, event.x, event.y
    
    col = x // (info["img_w"] + info["pad"])
    row = y // (info["img_h"] + info["pad"])
    filter_index = row * info["cols"] + col

    maps = info.get("maps")
    if maps is None or not (0 <= filter_index < maps.shape[0]): return

    app.state.selected_filter_index = filter_index
    update_cnn_view(app)


def update_activation_detail(app, feature_maps, filter_index: int) -> None:
    # (This function is correct and unchanged)
    state = app.state
    state.last_detail_maps = feature_maps
    state.last_detail_index = filter_index

    app.detail_canvas.delete("all")
    if feature_maps is None or filter_index < 0:
        app.detail_frame.config(text="Activation Detail")
        app.stats_var.set("Stats: (select a filter)")
        return

    app.detail_frame.config(text=f"Activation Detail (Filter {filter_index + 1})")
    selected_map = feature_maps[filter_index]
    min_val, max_val = selected_map.min().item(), selected_map.max().item()
    app.stats_var.set(f"Stats: Min={min_val:.4f} | Max={max_val:.4f}")

    normalized = (selected_map - min_val) / (max_val - min_val + 1e-8)
    detail_data = (normalized * 255).byte().numpy()

    app.detail_canvas.update_idletasks()
    size = max(16, min(app.detail_canvas.winfo_width(), app.detail_canvas.winfo_height()) - 10)
    
    pil_img = Image.fromarray(detail_data).resize((size, size), Image.NEAREST)
    state.tk_detail_image = ImageTk.PhotoImage(pil_img)
    app.detail_canvas.create_image(5, 5, image=state.tk_detail_image, anchor="nw")

    if app.show_detail_grid_var.get():
        draw_detail_pixel_grid(app.detail_canvas, selected_map.shape, (size, size), (5, 5))


# --- THIS IS THE CORRECTED FUNCTION ---
def update_kernel_panel(app, layer_name: str | None, filter_index: int) -> None:
    """Updates the panel showing kernel weights for a selected filter."""
    state = app.state
    for widget in app.kernel_panel_frame.winfo_children():
        widget.destroy()

    if not layer_name or "conv" not in layer_name or filter_index < 0:
        ttk.Label(app.kernel_panel_frame, text="Select a CNN filter to see its kernel.").pack(pady=20)
        return

    layer = getattr(state.model, layer_name.split(" ")[0], None)
    if not layer or not hasattr(layer, 'weight') or not hasattr(layer, 'bias'):
        return

    weights = layer.weight.data[filter_index]
    bias = layer.bias.data[filter_index]
    in_channels = weights.shape[0]

    # Restore the logic to handle channel changes
    init_channel = state.selected_kernel_channel
    if not (0 <= init_channel < in_channels):
        init_channel = 0
    state.selected_kernel_channel = init_channel

    ttk.Label(app.kernel_panel_frame, text=f"Input Channels: {in_channels}", font=("", 10)).pack(pady=(5, 0))
    ttk.Label(app.kernel_panel_frame, text=f"Bias: b = {bias.item():>7.4f}", font=("", 10, "bold")).pack(pady=(0, 5))

    # Define the callback function that create_channel_viewer needs
    def on_channel_change(index: int) -> None:
        state.selected_kernel_channel = index

    # Restore the original dual-viewer functionality
    create_channel_viewer(app.kernel_panel_frame, "Viewer A", weights, in_channels, on_channel_change, initial_index=init_channel)
    ttk.Separator(app.kernel_panel_frame, orient="horizontal").pack(fill="x", pady=10, padx=5)
    create_channel_viewer(app.kernel_panel_frame, "Viewer B", weights, in_channels, on_channel_change, initial_index=init_channel)


def on_detail_canvas_resize(app, _event=None) -> None:
    """Callback to redraw the detail canvas on resize."""
    update_activation_detail(app, app.state.last_detail_maps, app.state.last_detail_index)