"""Dataset discovery and metadata utilities."""
from __future__ import annotations

import glob
import os
from typing import Dict, List, Set


def find_datasets(base_path: str = ".") -> List[str]:
    """Return dataset directories under ``base_path`` (non-hidden, non-files)."""
    excluded_dirs: Set[str] = {'gui', 'core', '__pycache__', '.git', '.venv'}
    dataset_dirs = []
    try:
        for entry in os.scandir(base_path):
            if (entry.is_dir() and 
                entry.name not in excluded_dirs):
                dataset_dirs.append(entry.name)
                
    except FileNotFoundError:
        return []
        
    return sorted(dataset_dirs)


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
