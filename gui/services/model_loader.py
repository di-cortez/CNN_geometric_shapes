"""Helpers for instantiating and preparing models for GUI exploration."""
from __future__ import annotations

import torch
from torch import nn

from model import SimpleCNN
from core.utils.model import get_model_layers


def load_model(model_path: str, img_size: int, num_classes: int) -> nn.Module:
    """Instantiate ``SimpleCNN`` and load weights from disk."""
    model = SimpleCNN(dropout=0.4, img_size=img_size, num_classes=num_classes)
    state_dict = torch.load(model_path, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()
    return model


def collect_layer_metadata(model: nn.Module):
    """Return grouped layer names for the configured model."""
    return get_model_layers(model)
