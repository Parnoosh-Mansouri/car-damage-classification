import pandas as pd
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


class CarDamageDataset(Dataset):

    # Initialize the dataset
    def __init__(self, csv_file, image_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_dir = Path(image_dir)
        self.transform = transform

    # Return the number of samples
    def __len__(self):
        return len(self.data)

    # Return one image and its labels
    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Build the image path
        image_path = self.image_dir / row["file_name"]

        # Open the image and convert it to RGB
        image = Image.open(image_path).convert("RGB")

        # Get the six damage labels
        labels = row[
            [
                "dent",
                "scratch",
                "crack",
                "glass_shatter",
                "lamp_broken",
                "tire_flat"
            ]
        ].values

        # Convert labels to a PyTorch tensor
        labels = torch.tensor(
            labels.astype(float),
            dtype=torch.float32
        )

        # Apply image transformations
        if self.transform:
            image = self.transform(image)

        return image, labels


if __name__ == "__main__":

    # Create the dataset
    dataset = CarDamageDataset(
        "data/train.csv",
        "dataset/CarDD_COCO/Train2017",
        transform=transform
    )

    print("Number of samples:", len(dataset))

    # Create a DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True
    )

    # Get the first batch
    images, labels = next(iter(dataloader))

    print("Images shape:", images.shape)
    print("Labels shape:", labels.shape)