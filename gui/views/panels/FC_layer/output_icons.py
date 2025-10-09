"""Helper utilities to render shape icons for the FC output layer."""
from __future__ import annotations

import contextlib
import random
from typing import Dict

import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageTk

from shapes import SHAPE_FUNCTIONS, SHAPE_IDS

_DRAW_SIZE = 72
_DEFAULT_ICON_SIZE = 34
_ICON_FILL_HEX = "#123b8c"
_BACKGROUND_RGBA = (0, 0, 0, 0)
_ICON_FILL_RGB = ImageColor.getrgb(_ICON_FILL_HEX)



@contextlib.contextmanager
def _stable_random(seed: int):
    """Temporarily seed RNGs so generated icons remain deterministic."""
    py_state = random.getstate()
    np_state = np.random.get_state()
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)


class OutputIconFactory:
    """Pre-renders Tkinter PhotoImages for each output class."""

    def __init__(self, icon_size: int = _DEFAULT_ICON_SIZE):
        self.icon_size = icon_size
        self._icons: Dict[int, ImageTk.PhotoImage] = {}
        self._icons_by_name: Dict[str, ImageTk.PhotoImage] = {}
        self._build_icons()

    def get_by_index(self, class_index: int) -> ImageTk.PhotoImage | None:
        """Return the PhotoImage that represents the given class index."""
        return self._icons.get(class_index)

    def get_by_name(self, class_name: str) -> ImageTk.PhotoImage | None:
        """Return the PhotoImage for a class label."""
        return self._icons_by_name.get(class_name)

    def _build_icons(self) -> None:
        for shape_name, draw_func in SHAPE_FUNCTIONS.items():
            icon = self._render_icon(draw_func, shape_name)
            if icon is None:
                continue
            photo = ImageTk.PhotoImage(icon)
            self._icons_by_name[shape_name] = photo
            class_index = SHAPE_IDS.get(shape_name)
            if class_index is not None:
                self._icons[class_index] = photo

    def _render_icon(self, draw_func, expected_name: str):
        """Render a single icon using the provided draw function."""
        canvas = Image.new("RGBA", (_DRAW_SIZE, _DRAW_SIZE), _BACKGROUND_RGBA)
        draw = ImageDraw.Draw(canvas)

        rendered_name = None
        for attempt in range(5):
            seed = hash((expected_name, attempt)) & 0xFFFFFFFF
            with _stable_random(seed):
                rendered_name = draw_func(
                    draw,
                    _DRAW_SIZE,
                    _ICON_FILL_RGB,
                    background_color=_BACKGROUND_RGBA,
                )
            if rendered_name == expected_name:
                break
        if rendered_name != expected_name:
            return None

        return canvas.resize((self.icon_size, self.icon_size), Image.LANCZOS)


__all__ = ["OutputIconFactory"]
