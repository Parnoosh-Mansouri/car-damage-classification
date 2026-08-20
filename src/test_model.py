import torch
from torch.utils.data import DataLoader

from dataset import CarDamageDataset, transform
from model import CarDamageModel


if __name__ == "__main__":

    # Select GPU if available
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Create the dataset
    dataset = CarDamageDataset(
        "data/train.csv",
        "dataset/CarDD_COCO/Train2017",
        transform=transform
    )

    # Create the DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True
    )

    # Create the model
    model = CarDamageModel()

    # Move the model to the GPU
    model = model.to(device)

    # Get one batch
    images, labels = next(iter(dataloader))

    # Move the images to the GPU
    images = images.to(device)

    # Run the images through the model
    outputs = model(images)

    print("Device:", device)
    print("Images shape:", images.shape)
    print("Labels shape:", labels.shape)
    print("Outputs shape:", outputs.shape)