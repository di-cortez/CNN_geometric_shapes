# Arquivo: dataset.py
import torch
import zipfile
import io
from PIL import Image
from torch.utils.data import Dataset

class ShapeDataset(Dataset):
    def __init__(self, img_zip_path, label_zip_path, transform=None):
        self.img_path = zipfile.ZipFile(img_zip_path, 'r')
        self.labels_path = zipfile.ZipFile(label_zip_path, 'r')
        self.transform = transform
        self.file_names = sorted(
            [name for name in self.img_path.namelist() if name.endswith(".png")]
        )

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        img_name = self.file_names[idx]
        img_bytes = self.img_path.read(img_name)
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label_name = img_name.replace(".png", ".txt")
        label_content = self.labels_path.read(label_name).decode("utf-8")
        label = int(label_content.strip())

        # A nova função de perda (CrossEntropyLoss) espera rótulos do tipo Long (inteiro)
        return image, torch.tensor(label, dtype=torch.long)