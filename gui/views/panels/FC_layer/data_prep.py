"""Data preparation helpers for FC layer visualisation."""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

__all__ = ["prepare_single_layer_data"]


def _tensor_to_numpy(obj) -> np.ndarray:
    """Convert an activation tensor/array to a 1D numpy array."""
    if obj is None:
        return None  # type: ignore[return-value]

    array = obj
    if hasattr(array, "detach"):
        array = array.detach()
    if hasattr(array, "cpu"):
        array = array.cpu()
    array = np.asarray(array)
    array = np.squeeze(array)
    if array.ndim > 1:
        array = array.reshape(-1)
    return array.astype(np.float32, copy=False)


def prepare_single_layer_data(
    activation_tensor: object,
    max_neurons: int,
) -> Tuple[np.ndarray, bool, np.ndarray]:
    """Subsample and normalise a single activation vector for drawing."""
    vector = _tensor_to_numpy(activation_tensor)

    if vector is None or vector.size == 0:
        return np.array([], dtype=np.float32), False, np.array([], dtype=int)

    if vector.size > max_neurons:
        indices = np.linspace(0, vector.size - 1, max_neurons, dtype=int)
        display_vector = vector[indices]
        is_subsampled = True
    else:
        indices = np.arange(vector.size, dtype=int)
        display_vector = vector
        is_subsampled = False

    return display_vector, is_subsampled, indices