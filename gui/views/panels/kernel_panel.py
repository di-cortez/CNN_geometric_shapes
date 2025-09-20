"""Panel reserving space for convolution kernel viewers."""
from __future__ import annotations

from tkinter import ttk


class KernelPanel(ttk.LabelFrame):
    def __init__(self, parent) -> None:
        super().__init__(parent, text="Kernel Panel", padding=5)
        ttk.Label(self, text="Select a CNN filter to see its kernel.").pack(pady=20)


__all__ = ["KernelPanel"]
