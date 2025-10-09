"""Rendering helpers for convolutional activation grids."""
from __future__ import annotations

from typing import List

import numpy as np
from PIL import Image, ImageTk
from tkinter import Canvas



def visualize_feature_maps(canvas: Canvas, feature_maps, tk_images: List[ImageTk.PhotoImage]) -> None:
    """Render feature maps with overlayed indices onto the provided canvas."""
    canvas.delete("all")
    tk_images.clear()

    if feature_maps is None or feature_maps.shape[0] == 0:
        canvas.config(scrollregion=canvas.bbox("all"))
        return

    num_maps = feature_maps.shape[0]
    
    # --- Start of New/Modified Code ---

    # 1. Get the current canvas width to make the layout responsive.
    canvas.update_idletasks()
    canvas_width = canvas.winfo_width()
    
    # Define padding for the grid
    padding = 10
    
    # Handle cases where the canvas might be too small
    if canvas_width < 50:
        canvas_width = 50

    # 2. Dynamically calculate the number of columns for a squarish layout.
    columns = int(np.ceil(np.sqrt(num_maps))) if num_maps > 0 else 1
    
    # 3. Dynamically calculate image size to fill the available width.
    # The formula accounts for the total space taken by images and padding.
    image_size = (canvas_width - (columns + 1) * padding) / columns
    image_size = int(max(16, image_size)) # Enforce a minimum size for visibility.

    # --- End of New/Modified Code ---

    canvas.viz_info = {
        "maps": feature_maps,
        "cols": columns,
        "img_w": image_size,
        "img_h": image_size,
        "pad": padding,
    }

    for idx, fmap in enumerate(feature_maps):
        row, col = divmod(idx, columns)
        x0 = col * (image_size + padding) + padding
        y0 = row * (image_size + padding) + padding

        normalized = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-8)
        data = (normalized * 255).byte().cpu().numpy() if hasattr(normalized, "cpu") else (normalized * 255).astype(np.uint8)
        pil_img = Image.fromarray(data).resize((image_size, image_size), Image.NEAREST)
        tk_img = ImageTk.PhotoImage(pil_img)
        tk_images.append(tk_img)

        canvas.create_image(x0, y0, image=tk_img, anchor="nw")
        label = f"{idx + 1}/{num_maps}"
        # Adding a text shadow for better readability
        canvas.create_text(x0 + 1, y0 + 1, text=label, font=("Arial", 8, "bold"), fill="black", anchor="nw")
        canvas.create_text(x0, y0, text=label, font=("Arial", 8, "bold"), fill="white", anchor="nw")

    # 4. Update the scroll region to match the content size.
    # This makes the scrollbars functional for the dynamic grid.
    canvas.config(scrollregion=canvas.bbox("all"))