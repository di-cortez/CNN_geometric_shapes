"""Visualization helpers for fully-connected activation vectors."""
from __future__ import annotations

import numpy as np
from tkinter import Canvas
from typing import Iterable


def visualize_vector(canvas: Canvas, vector, original_shape: Iterable[int]) -> None:
    """Draw a heatmap representation of ``vector`` on ``canvas``."""
    canvas.delete("all")
    if vector is None:
        return

    canvas.update_idletasks()
    width, height = canvas.winfo_width(), canvas.winfo_height()
    if width < 10 or height < 10:
        return

    canvas.create_text(5, 5, text=f"Shape: {tuple(original_shape)}", anchor="nw", fill="white", font=("Arial", 9))

    flattened = np.asarray(vector).flatten()
    if flattened.size == 0:
        return

    v_min, v_max = flattened.min(), flattened.max()
    if v_max == v_min:
        norm = np.zeros_like(flattened)
    else:
        norm = (flattened - v_min) / (v_max - v_min)

    rows = 32
    cols = int(np.ceil(flattened.size / rows)) or 1
    cell_w = width / cols
    cell_h = (height - 25) / rows

    for idx, value in enumerate(norm):
        row, col = divmod(idx, cols)
        red = int(value * 255)
        blue = int((1 - value) * 255)
        color = f'#{red:02x}00{blue:02x}'
        x0 = col * cell_w
        y0 = row * cell_h + 25
        x1 = x0 + cell_w
        y1 = y0 + cell_h
        canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline="")
