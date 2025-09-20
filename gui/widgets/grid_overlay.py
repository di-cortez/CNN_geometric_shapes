"""Canvas overlay helpers for pixel-aligned grids."""
from __future__ import annotations

from tkinter import Canvas
from typing import Tuple


def draw_pixel_grid(canvas: Canvas, original_size: Tuple[int, int], canvas_size: Tuple[int, int], offset: Tuple[int, int]) -> None:
    """Draw an overlay grid aligned with the original image pixels."""
    canvas.delete("pixel_grid")
    offset_x, offset_y = offset
    width, height = canvas_size
    step_x = width / max(original_size[0], 1)
    step_y = height / max(original_size[1], 1)

    for idx in range(1, original_size[0]):
        x = offset_x + idx * step_x
        canvas.create_line(x, offset_y, x, offset_y + height, fill="gray50", tags="pixel_grid", width=0.5)

    for idx in range(1, original_size[1]):
        y = offset_y + idx * step_y
        canvas.create_line(offset_x, y, offset_x + width, y, fill="gray50", tags="pixel_grid", width=0.5)


def draw_detail_pixel_grid(canvas: Canvas, original_shape: Tuple[int, int], resized_size: Tuple[int, int], offset: Tuple[int, int]) -> None:
    """Overlay a grid on the detailed activation preview."""
    canvas.delete("detail_grid")
    offset_x, offset_y = offset
    res_w, res_h = resized_size
    orig_h, orig_w = original_shape
    if orig_w == 0 or orig_h == 0:
        return

    step_x = res_w / orig_w
    step_y = res_h / orig_h

    for idx in range(1, orig_w):
        x = offset_x + idx * step_x
        canvas.create_line(x, offset_y, x, offset_y + res_h, fill="gray50", tags="detail_grid", width=0.5)

    for idx in range(1, orig_h):
        y = offset_y + idx * step_y
        canvas.create_line(offset_x, y, offset_x + res_w, y, fill="gray50", tags="detail_grid", width=0.5)
