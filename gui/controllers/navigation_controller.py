"""Navigation-related callbacks for dataset browsing."""
from __future__ import annotations

from gui.controllers import activation_controller


def show_previous_image(app) -> None:
    try:
        current = int(app.index_var.get())
    except ValueError:
        current = 0
    app.index_var.set(str(max(0, current - 1)))
    activation_controller.update_all_visuals(app)


def show_next_image(app) -> None:
    try:
        current = int(app.index_var.get())
    except ValueError:
        current = 0
    app.index_var.set(str(current + 1))
    activation_controller.update_all_visuals(app)


def enable_controls(app, is_enabled: bool) -> None:
    state = "normal" if is_enabled else "disabled"
    for child in app.input_panel.control_frame.winfo_children():
        child.configure(state=state)


def on_canvas_configure(app, _event=None) -> None:
    def _update_scrollregion() -> None:
        bbox = app.activation_panel.canvas.bbox("all")
        app.activation_panel.canvas.configure(scrollregion=bbox)
    app.after_idle(_update_scrollregion)


def on_mousewheel(app, event) -> None:
    direction = 1 if getattr(event, "num", None) == 5 or getattr(event, "delta", 0) < 0 else -1
    app.activation_panel.canvas.yview_scroll(direction, "units")


__all__ = [
    "show_previous_image",
    "show_next_image",
    "enable_controls",
    "on_canvas_configure",
    "on_mousewheel",
]
