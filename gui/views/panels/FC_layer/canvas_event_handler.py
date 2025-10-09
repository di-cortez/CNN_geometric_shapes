# canvas_event_handler.py
"""
Handles user interaction events for the canvas (zoom, pan, click).
"""

class CanvasEventHandler:
    """Manages canvas events like zoom, pan, and neuron selection."""

    def __init__(self, panel):
        """
        Initializes the event handler and binds events to the canvas.
        Args:
            panel: The parent FCViewPanel instance.
        """
        self.panel = panel
        self.canvas = panel.canvas
        self._bind_events()

    def _bind_events(self):
        """Binds all necessary mouse events to their handler methods."""
        # Zooming
        self.canvas.bind("<Control-Button-4>", self._zoom)      # Linux zoom in
        self.canvas.bind("<Control-Button-5>", self._zoom)      # Linux zoom out
        self.canvas.bind("<Control-MouseWheel>", self._zoom)    # Windows/macOS zoom
        # Panning
        self.canvas.bind("<ButtonPress-1>", self._pan_start)
        self.canvas.bind("<B1-Motion>", self._pan_move)
        self.canvas.bind("<ButtonRelease-1>", self._pan_end)
        # Selection and Reset
        self.canvas.bind("<Double-Button-1>", self._reset_view)
        self.canvas.bind("<Button-3>", self._on_neuron_click)  # Right-click for selection

    def _zoom(self, event):
        """Zooms the canvas view in or out."""
        factor = 1.1 if (event.num == 4 or event.delta > 0) else 0.9
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        self.canvas.scale("all", x, y, factor, factor)

    def _pan_start(self, event):
        """Records the starting point for a pan operation."""
        # Use find_overlapping for a more precise hit-test.
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        
        # Find items directly under the cursor
        overlapping_items = self.canvas.find_overlapping(canvas_x, canvas_y, canvas_x, canvas_y)
        
        # Check if any of the overlapping items are neurons
        is_on_neuron = any(item_id in self.panel.neuron_tags for item_id in overlapping_items)

        if is_on_neuron:
            return # Correctly ignore clicks on neurons and do not pan
            
        self.canvas.config(cursor="fleur")
        self.canvas.scan_mark(event.x, event.y)

    def _pan_move(self, event):
        """Drags the canvas view."""
        self.canvas.scan_dragto(event.x, event.y, gain=1)
        
    def _pan_end(self, event):
        """Resets the cursor after panning."""
        self.canvas.config(cursor="")

    def _reset_view(self, event=None):
        """Resets the view by redrawing the network."""
        if self.panel.last_data:
            self.panel.selected_neuron = None
            self.panel.update_view(self.panel.last_data)

    def _on_neuron_click(self, event):
        """Handles neuron selection and triggers a redraw."""
        item = self.canvas.find_closest(self.canvas.canvasx(event.x), self.canvas.canvasy(event.y))
        if not item: return

        item_id = item[0]
        if item_id in self.panel.neuron_tags:
            clicked_neuron = self.panel.neuron_tags[item_id]
            
            # Toggle selection: if clicking the same neuron, deselect it
            if self.panel.selected_neuron == clicked_neuron:
                self.panel.selected_neuron = None
            else:
                self.panel.selected_neuron = clicked_neuron
            
            # Redraw to reflect the new selection
            if self.panel.last_data:
                self.panel.update_view(self.panel.last_data)