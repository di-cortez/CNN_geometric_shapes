"""Input panel containing image preview and metadata labels."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk


class InputPanel(ttk.LabelFrame):
    def __init__(
        self,
        parent,
        *,
        index_var: tk.StringVar,
        show_grid_var: tk.BooleanVar,
        label_var: tk.StringVar,
        pred_var: tk.StringVar,
        shape_color_var: tk.StringVar,
        bg_color_var: tk.StringVar,
    ) -> None:
        super().__init__(parent, text="Input Image", padding=10)

        self.control_frame = ttk.Frame(self)
        self.control_frame.pack(pady=(0, 10))

        self.prev_button = ttk.Button(self.control_frame, text="< Prev")
        self.prev_button.pack(side="left")

        self.index_entry = ttk.Entry(self.control_frame, textvariable=index_var, width=8)
        self.index_entry.pack(side="left", padx=5)

        self.next_button = ttk.Button(self.control_frame, text="Next >")
        self.next_button.pack(side="left")

        self.image_canvas = tk.Canvas(self, borderwidth=2, relief="solid", bg="black")
        self.image_canvas.pack(pady=5, fill="both", expand=True)

        options_frame = ttk.Frame(self)
        options_frame.pack(anchor="w", pady=(5, 0))
        self.grid_checkbutton = ttk.Checkbutton(options_frame, text="Grid", variable=show_grid_var)
        self.grid_checkbutton.pack(side="left")

        ttk.Label(self, textvariable=label_var).pack(anchor="w", pady=(5, 0))
        ttk.Label(self, textvariable=pred_var).pack(anchor="w")

        color_frame = ttk.Frame(self)
        color_frame.pack(anchor="w", pady=(5, 0))
        ttk.Label(color_frame, textvariable=shape_color_var).pack(side="left")
        ttk.Label(color_frame, textvariable=bg_color_var).pack(side="left", padx=(12, 0))


__all__ = ["InputPanel"]
