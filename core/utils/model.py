"""Utilities for introspecting PyTorch models used by the GUI."""
from __future__ import annotations

from typing import Dict, List

import torch.nn as nn


def get_model_layers(model: nn.Module) -> Dict[str, List[str]]:
    """Group model layers into CNN and FC buckets with readable labels."""
    layers: Dict[str, List[str]] = {"CNN": [], "FC": []}
    for name, module in model.named_modules():
        if not name:
            continue
        module_name = type(module).__name__
        if isinstance(module, (nn.Conv2d, nn.MaxPool2d, nn.ReLU, nn.BatchNorm2d)):
            layers["CNN"].append(f"{name} ({module_name})")
        elif isinstance(module, (nn.Linear, nn.Dropout)):
            layers["FC"].append(f"{name} ({module_name})")
    return layers
