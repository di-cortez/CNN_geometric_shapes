"""Mutable state container shared across GUI controllers."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch


@dataclass
class UIState:
    model: Optional[torch.nn.Module] = None
    images_zip: Optional[object] = None
    labels_zip: Optional[object] = None
    file_names: List[str] = field(default_factory=list)
    class_map: Dict[int, str] = field(default_factory=dict)
    model_layers: Dict[str, List[str]] = field(default_factory=dict)
    last_detail_maps: Optional[object] = None
    last_detail_index: int = -1
    tk_grid_images: List[object] = field(default_factory=list)
    tk_detail_image: Optional[object] = None
    selected_layer_name: Optional[str] = None
    selected_filter_index: int = -1
    selected_kernel_channel: int = 0

    def reset_dataset(self) -> None:
        self.close_archives()
        self.file_names.clear()
        self.class_map.clear()
        self.last_detail_maps = None
        self.last_detail_index = -1
        self.selected_layer_name = None
        self.selected_filter_index = -1
        self.selected_kernel_channel = 0
        self.tk_grid_images.clear()
        self.tk_detail_image = None

    def reset_model(self) -> None:
        self.model = None
        self.model_layers.clear()
        self.selected_layer_name = None
        self.selected_filter_index = -1
        self.selected_kernel_channel = 0
        self.last_detail_maps = None
        self.last_detail_index = -1
        self.tk_detail_image = None

    def close_archives(self) -> None:
        if hasattr(self.images_zip, "close"):
            self.images_zip.close()
        if hasattr(self.labels_zip, "close"):
            self.labels_zip.close()
        self.images_zip = None
        self.labels_zip = None


__all__ = ["UIState"]
