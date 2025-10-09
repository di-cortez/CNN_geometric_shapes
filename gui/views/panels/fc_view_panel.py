# fc_view_panel.py
"""
The main Tkinter widget for the fully connected layer visualization.
"""
import tkinter as tk
from tkinter import ttk
from gui.views.panels.FC_layer.network_drawer import NetworkDrawer
from gui.views.panels.FC_layer.canvas_event_handler import CanvasEventHandler

class FCViewPanel(ttk.Frame):
    """A panel for visualizing a fully connected neural network."""

    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)

        # --- Variable to hold the threshold value ---
        self.edge_threshold_var = tk.DoubleVar(value = 0.001)
        self.neuron_selectors = {} # Maps layer_name -> combobox widget

        self._create_top_bar()

        # Core components
        self.canvas = tk.Canvas(self, bg="white", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, side="bottom")
        self.zoom_level = 1.0

        # State management
        self.last_data = None
        self.selected_neuron = None  # Tuple: (layer_name, neuron_index)
        self.neuron_tags = {}        # Maps canvas item ID to neuron info

        # Delegate drawing and event handling to specialized classes
        self.drawer = NetworkDrawer(self)
        self.event_handler = CanvasEventHandler(self)

    def _create_top_bar(self):
        """Creates the top bar with visualization controls."""
        top_bar = ttk.Frame(self)
        top_bar.pack(fill="x", side="top", padx=5, pady=5)

        # --- Threshold control remains on the left ---
        ttk.Label(top_bar, text="Edge Threshold:").pack(side="left", padx=(0, 5))
        self.threshold_entry = ttk.Entry(
            top_bar, width=8, textvariable=self.edge_threshold_var
        )
        self.threshold_entry.pack(side="left")
        self.threshold_entry.bind("<Return>", self._trigger_redraw)

        # --- NEW: A container frame for the dynamic selectors ---
        self.selectors_frame = ttk.Frame(top_bar)
        self.selectors_frame.pack(side="left", padx=(20, 0))

    def _trigger_redraw(self, event=None):
        """Helper function to redraw the view using the last known data."""
        if self.last_data:
            self.focus_set()
            self.update_view(self.last_data)

    #     self.after(10, redraw_and_restore)
    def _update_neuron_selectors(self, layers_data: list):
        """Dynamically builds and updates neuron selector comboboxes."""
        # Clear any old widgets from the frame
        for widget in self.selectors_frame.winfo_children():
            widget.destroy()
        self.neuron_selectors.clear()
        
        # Create a selector for each layer except the last one
        for i, layer_info in enumerate(layers_data):
            if i == len(layers_data) - 1: # No selector for the final output layer
                continue

            layer_name = layer_info["name"]
            
            # Create the label and combobox for the current layer
            container = ttk.Frame(self.selectors_frame)
            container.pack(side="left", padx=(0, 15))
            
            ttk.Label(container, text=f"{layer_name} Neuron:").pack(side="left", padx=(0, 5))
            selector = ttk.Combobox(container, state="readonly", width=15)
            selector.pack(side="left")
            selector.bind("<<ComboboxSelected>>", self._on_neuron_select_from_combobox)
            
            # Store the widget for later access
            self.neuron_selectors[layer_name] = selector
            
            # Populate the combobox with neuron indices
            activation_tensor = layer_info.get("activation")
            if activation_tensor is not None:
                num_neurons = activation_tensor.shape[-1]
                values = ["Deselect"] + [f"Neuron {i}" for i in range(num_neurons)]
                selector["values"] = values
            else:
                selector["values"] = ["Deselect"]
            
            # Sync the combobox with the current selection state
            if self.selected_neuron and self.selected_neuron[0] == layer_name:
                selector.set(f"Neuron {self.selected_neuron[1]}")
            else:
                selector.set("Deselect")

    # --- NEW: A single handler for all dynamic comboboxes ---
    def _on_neuron_select_from_combobox(self, event):
        """Handles neuron selection from any of the dynamic comboboxes."""
        clicked_widget = event.widget
        value = clicked_widget.get()
        
        # Find which layer this combobox belongs to
        selected_layer_name = None
        for name, selector in self.neuron_selectors.items():
            if selector == clicked_widget:
                selected_layer_name = name
                break
        
        if not selected_layer_name:
            return

        if value == "Deselect":
            self.selected_neuron = None
        else:
            index = int(value.split(" ")[1])
            self.selected_neuron = (selected_layer_name, index)
            
        if self.last_data:
            self.update_view(self.last_data)


    def update_view(self, layers_data: list):
        """Updates the visualization with a list of layer data."""
        xview_state = self.canvas.xview()
        yview_state = self.canvas.yview()

        self.last_data = layers_data
        
        # --- MODIFIED: Call the new selector update method ---
        self._update_neuron_selectors(layers_data)

        try:
            threshold = self.edge_threshold_var.get()
        except (tk.TclError, ValueError):
            threshold = 0.1
            self.edge_threshold_var.set(threshold)

        def redraw_and_restore():
            self.drawer.draw_network(layers_data, threshold)
            self.canvas.scale("all", 0, 0, self.zoom_level, self.zoom_level)
            self.canvas.xview_moveto(xview_state[0])
            self.canvas.yview_moveto(yview_state[0])

        self.after(10, redraw_and_restore)