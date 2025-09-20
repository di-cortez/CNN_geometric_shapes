"""Panel hosting the activation grid canvas with scrollbars."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk


class ActivationPanel(ttk.LabelFrame):
    def __init__(self, parent) -> None:
        super().__init__(parent, text="Activation Grid", padding=5)

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(self, bg="gray20")
        self.canvas.grid(row=0, column=0, sticky="nsew")

        v_scroll = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        v_scroll.grid(row=0, column=1, sticky="ns")
        h_scroll = ttk.Scrollbar(self, orient="horizontal", command=self.canvas.xview)
        h_scroll.grid(row=1, column=0, sticky="ew")

        self.canvas.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)


__all__ = ["ActivationPanel"]
