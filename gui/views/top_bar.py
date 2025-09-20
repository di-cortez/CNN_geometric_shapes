"""Top bar containing dataset and layer selection widgets."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Sequence


class TopBar(ttk.Frame):
    def __init__(
        self,
        parent,
        *,
        dataset_options: Sequence[str],
        layer_type_var: tk.StringVar,
    ) -> None:
        super().__init__(parent, padding=(0, 0, 0, 10))

        self.columnconfigure(1, weight=1)

        left_frame = ttk.Frame(self)
        left_frame.grid(row=0, column=0, sticky="w")

        ttk.Label(left_frame, text="Select Dataset:").pack(anchor="w")
        self.dataset_selector = ttk.Combobox(left_frame, values=list(dataset_options), width=50, state="readonly")
        self.dataset_selector.pack(fill="x")

        ttk.Label(left_frame, text="Select Model:").pack(anchor="w", pady=(5, 0))
        self.model_selector = ttk.Combobox(left_frame, width=50, state="readonly")
        self.model_selector.pack(fill="x")

        right_frame = ttk.Frame(self)
        right_frame.grid(row=0, column=2, sticky="e")

        self.cnn_radio = ttk.Radiobutton(right_frame, text="CNN", variable=layer_type_var, value="CNN")
        self.cnn_radio.pack(side="left")
        self.fc_radio = ttk.Radiobutton(right_frame, text="FC", variable=layer_type_var, value="FC")
        self.fc_radio.pack(side="left", padx=(5, 15))

        ttk.Label(right_frame, text="Select Layer:").pack(anchor="w")
        self.layer_selector = ttk.Combobox(right_frame, width=35, state="readonly")
        self.layer_selector.pack(fill="x")


__all__ = ["TopBar"]
