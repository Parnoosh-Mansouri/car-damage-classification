import torch
from torch.utils.data import DataLoader
from dataset import CarDamageDataset, transform
from model import CarDamageModel


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# Load one batch
dataset = CarDamageDataset(
    "data/train.csv",
    "dataset/CarDD_COCO/train2017",
    transform
)

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)

images, labels = next(iter(loader))

images = images.to(device)
labels = labels.to(device)


# Test model forward pass
model = CarDamageModel().to(device)
model.eval()

with torch.no_grad():
    outputs = model(images)


print("Device:", device)
print("Images shape:", images.shape)
print("Labels shape:", labels.shape)
print("Outputs shape:", outputs.shape)