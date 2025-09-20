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

    num_maps = feature_maps.shape[0] if feature_maps is not None else 0
    if num_maps == 0:
        return

    image_size = 64
    padding = 5
    columns = 4

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
        canvas.create_text(x0 + 1, y0 + 1, text=label, font=("Arial", 8, "bold"), fill="black", anchor="nw")
        canvas.create_text(x0, y0, text=label, font=("Arial", 8, "bold"), fill="white", anchor="nw")
