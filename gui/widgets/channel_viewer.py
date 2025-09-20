"""Reusable widget factory for inspecting individual convolution channels."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable

from core.utils.formatting import format_weight


def create_channel_viewer(parent, title: str, weights, in_channels: int, on_channel_change: Callable[[int], None], initial_index: int = 0):
    """Create a viewer for a kernel slice and wire navigation callbacks."""
    frame = ttk.Frame(parent)
    frame.pack(pady=5, padx=5, fill="x")

    header = ttk.Frame(frame)
    header.pack()
    ttk.Label(header, text=title, font=("", 10, "bold")).pack(side="left")
    sum_var = tk.StringVar(value="Sum: 0.0000")
    ttk.Label(header, textvariable=sum_var, font=("", 10)).pack(side="left", padx=(8, 0))

    nav_frame = ttk.Frame(frame)
    nav_frame.pack()

    matrix_frame = ttk.Frame(frame, relief="sunken", borderwidth=2)
    matrix_frame.pack(padx=10, pady=5)

    kernel_size = weights.shape[1]
    index = max(0, min(initial_index, in_channels - 1)) if in_channels else 0
    channel_var = tk.IntVar(value=index)

    labels = [[ttk.Label(matrix_frame, width=10, anchor="center") for _ in range(kernel_size)] for _ in range(kernel_size)]
    for row in range(kernel_size):
        for col in range(kernel_size):
            labels[row][col].grid(row=row, column=col, padx=1, pady=1)

    channel_label_var = tk.StringVar()

    def update_matrix() -> None:
        current = channel_var.get()
        channel_label_var.set(f"{current + 1}/{in_channels}")
        kernel_slice = weights[current]
        kernel_sum = float(kernel_slice.sum().item())
        sum_var.set(f"Sum: {kernel_sum:.4f}")
        for r in range(kernel_size):
            for c in range(kernel_size):
                labels[r][c].config(text=format_weight(kernel_slice[r, c].item()))
        on_channel_change(current)

    def navigate(direction: int) -> None:
        current = channel_var.get()
        new_index = (current + direction) % in_channels if in_channels else 0
        channel_var.set(new_index)
        update_matrix()

    ttk.Button(nav_frame, text="< Prev", command=lambda: navigate(-1)).pack(side="left")
    ttk.Label(nav_frame, textvariable=channel_label_var, width=6, anchor="center").pack(side="left")
    ttk.Button(nav_frame, text="Next >", command=lambda: navigate(1)).pack(side="left")

    update_matrix()
    return frame
