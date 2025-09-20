"""Dataset discovery and metadata utilities."""
from __future__ import annotations

import glob
import os
from typing import Dict, List


def find_datasets(base_path: str = ".") -> List[str]:
    """Return dataset directories under ``base_path`` (non-hidden, non-files)."""
    entries = []
    for name in os.listdir(base_path):
        full_path = os.path.join(base_path, name)
        if os.path.isdir(full_path) and not name.startswith('.'):
            entries.append(name)
    return sorted(entries)


def find_models_in_dataset(dataset_path: str) -> List[str]:
    """Return absolute paths to ``.pth`` files inside ``dataset_path``."""
    pattern = os.path.join(dataset_path, "*.pth")
    return sorted(glob.glob(pattern))


def load_class_map(dataset_path: str, filename: str = "shape_ids.txt") -> Dict[int, str]:
    """Load ``shape_ids`` mappings from the provided dataset directory."""
    mapping: Dict[int, str] = {}
    file_path = os.path.join(dataset_path, filename)
    if not os.path.exists(file_path):
        return mapping

    with open(file_path, "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split(":")
            if len(parts) == 2:
                try:
                    mapping[int(parts[0])] = parts[1].strip()
                except ValueError:
                    continue
    return mapping
