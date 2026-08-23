import torch
import pandas as pd

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# Augmentation for training
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# No augmentation for validation
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


class CarDamageDataset(Dataset):

    def __init__(self, csv_file, image_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]

        image = Image.open(
            f"{self.image_dir}/{row['file_name']}"
        ).convert("RGB")

        labels = torch.tensor(
            row[
                [
                    "dent",
                    "scratch",
                    "crack",
                    "glass_shatter",
                    "lamp_broken",
                    "tire_flat"
                ]
            ].values.astype("float32")
        )

        if self.transform:
            image = self.transform(image)

        return image, labels