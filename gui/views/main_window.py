"""Main Tkinter application frame assembling all GUI components."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from core.utils import find_datasets
from gui.controllers.event_wiring import setup_callbacks
from gui.controllers.dataset_controller import on_dataset_selected
from gui.state import UIState
from gui.views.top_bar import TopBar
from gui.views.panels.input_panel import InputPanel
from gui.views.panels.activation_panel import ActivationPanel
from gui.views.panels.kernel_panel import KernelPanel
from gui.views.panels.detail_panel import DetailPanel
from gui.views.panels.fc_view_panel import FCViewPanel


class Application(ttk.Frame):
    """Compose the main window widgets and expose them to controllers."""

    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.state = UIState()

        parent.title("Neural Network Analyzer")
        parent.geometry("1300x800")

        self._init_variables()
        self._create_widgets()
        setup_callbacks(self)

        self._auto_load_first_dataset()

    def _init_variables(self) -> None:
        self.index_var = tk.StringVar(value="0")
        self.show_grid_var = tk.BooleanVar(value=False)
        self.show_detail_grid_var = tk.BooleanVar(value=False)
        self.layer_type_var = tk.StringVar(value="CNN")
        self.label_var = tk.StringVar(value="True Label:")
        self.pred_var = tk.StringVar(value="Prediction:")
        self.shape_color_var = tk.StringVar(value="Shape Color: (---)")
        self.bg_color_var = tk.StringVar(value="Background Color: (---)")
        self.stats_var = tk.StringVar(value="Stats: (select a filter)")
        self.status_var = tk.StringVar(value="Ready. Please select a dataset to begin.")

    def _create_widgets(self) -> None:
        datasets = find_datasets()

        self.top_bar = TopBar(self, dataset_options=datasets, layer_type_var=self.layer_type_var)
        self.top_bar.pack(fill="x")

        self.dataset_selector = self.top_bar.dataset_selector
        self.model_selector = self.top_bar.model_selector
        self.layer_selector = self.top_bar.layer_selector
        self.cnn_radio = self.top_bar.cnn_radio
        self.fc_radio = self.top_bar.fc_radio

        main_frame = ttk.Frame(self)
        main_frame.pack(fill="both", expand=True)
        main_frame.columnconfigure(0, weight=35)
        main_frame.columnconfigure(1, weight=40)
        main_frame.columnconfigure(2, weight=25)
        main_frame.rowconfigure(0, weight=1)

        self.input_panel = InputPanel(
            main_frame,
            index_var=self.index_var,
            show_grid_var=self.show_grid_var,
            label_var=self.label_var,
            pred_var=self.pred_var,
            shape_color_var=self.shape_color_var,
            bg_color_var=self.bg_color_var,
        )
        self.input_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 5))

        self.center_container = ttk.Frame(main_frame)
        self.center_container.grid(row=0, column=1, sticky="nsew", padx=5)
        self.center_container.rowconfigure(0, weight=50)
        self.center_container.rowconfigure(1, weight=50)
        self.center_container.columnconfigure(0, weight=1)

        self.activation_panel = ActivationPanel(self.center_container)
        self.activation_panel.grid(row=0, column=0, sticky="nsew", pady=(0, 5))

        self.kernel_panel = KernelPanel(self.center_container)
        self.kernel_panel.grid(row=1, column=0, sticky="nsew")

        self.right_container = ttk.Frame(main_frame)
        self.right_container.grid(row=0, column=2, sticky="nsew", padx=(5, 0))
        self.right_container.rowconfigure(0, weight=1)
        self.right_container.columnconfigure(0, weight=1)

        self.detail_panel = DetailPanel(
            self.right_container,
            show_grid_var=self.show_detail_grid_var,
            stats_var=self.stats_var,
        )
        self.detail_panel.grid(row=0, column=0, sticky="nsew")


        self.fc_panel = FCViewPanel(main_frame)

        # Attribute aliases expected by controller functions
        self.image_canvas = self.input_panel.image_canvas
        self.grid_checkbutton = self.input_panel.grid_checkbutton
        self.activation_grid_canvas = self.activation_panel.canvas
        self.kernel_panel_frame = self.kernel_panel
        self.detail_frame = self.detail_panel
        self.detail_canvas = self.detail_panel.canvas
        self.detail_grid_checkbutton = self.detail_panel.grid_checkbutton
        self.control_frame = self.input_panel.control_frame
        self.prev_button = self.input_panel.prev_button
        self.next_button = self.input_panel.next_button
        self.index_entry = self.input_panel.index_entry

    def _auto_load_first_dataset(self) -> None:
        """Checks for datasets and loads the first one found."""
        available_datasets = self.dataset_selector["values"]
        if available_datasets:
            self.dataset_selector.set(available_datasets[0])
            self.after(100, lambda: on_dataset_selected(self))

    def switch_view(self, view_type: str):
        """Switches between 'CNN' and 'FC' visualization views."""
        if view_type == "CNN":
            self.fc_panel.grid_remove()
            self.center_container.grid()
            self.right_container.grid()
        elif view_type == "FC":
            self.center_container.grid_remove()
            self.right_container.grid_remove()
            self.fc_panel.grid(row=0, column=1, columnspan=2, sticky="nsew", padx=5)


__all__ = ["Application"]
