"""Dataset IO helpers for the GUI controllers."""
from __future__ import annotations

import io
import os
import zipfile
from typing import List, Tuple

from PIL import Image


def open_archives(dataset_path: str) -> Tuple[zipfile.ZipFile, zipfile.ZipFile, List[str]]:
    """Open dataset archives and return sorted image filenames."""
    images_zip_path = os.path.join(dataset_path, "images.zip")
    labels_zip_path = os.path.join(dataset_path, "labels.zip")
    images_zip = zipfile.ZipFile(images_zip_path, "r")
    labels_zip = zipfile.ZipFile(labels_zip_path, "r")
    filenames = sorted(name for name in images_zip.namelist() if name.endswith(".png"))
    return images_zip, labels_zip, filenames


def read_image(images_zip: zipfile.ZipFile, filename: str) -> Image.Image:
    """Load an image from the dataset archive as a PIL RGB image."""
    with images_zip.open(filename) as file_handle:
        return Image.open(io.BytesIO(file_handle.read())).convert("RGB")

def peek_image_size(images_zip: zipfile.ZipFile, filename: str) -> int:
    """Return the width of the image assumed to be square."""
    with images_zip.open(filename) as file_handle:
        with Image.open(io.BytesIO(file_handle.read())) as img:
            return img.size[0]

