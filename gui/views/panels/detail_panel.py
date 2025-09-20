"""Panel displaying a single activation map with detail overlays."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk


class DetailPanel(ttk.LabelFrame):
    def __init__(self, parent, *, show_grid_var: tk.BooleanVar, stats_var: tk.StringVar) -> None:
        super().__init__(parent, text="Activation Detail", padding=5)

        self.canvas = tk.Canvas(self, bg="gray10")
        self.canvas.pack(fill="both", expand=True, pady=(0, 5))

        controls = ttk.Frame(self)
        controls.pack(fill="x", anchor="w")
        self.grid_checkbutton = ttk.Checkbutton(controls, text="Grid", variable=show_grid_var)
        self.grid_checkbutton.pack(side="left")
        ttk.Label(controls, textvariable=stats_var).pack(side="left", padx=10)


__all__ = ["DetailPanel"]
