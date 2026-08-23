import torch
import numpy as np

from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from dataset_augmented import (
    CarDamageDataset,
    val_transform
)

from model import CarDamageModel


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CLASS_NAMES = [
    "dent",
    "scratch",
    "crack",
    "glass_shatter",
    "lamp_broken",
    "tire_flat"
]


# Load validation data
dataset = CarDamageDataset(
    "data/val.csv",
    "dataset/CarDD_COCO/val2017",
    val_transform
)

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=False
)


# Load augmented model
model = CarDamageModel().to(device)

model.load_state_dict(
    torch.load(
        "best_model_augmented.pth",
        map_location=device
    )
)

model.eval()


all_probs = []
all_labels = []


with torch.no_grad():

    for images, labels in loader:
        images = images.to(device)

        outputs = model(images)

        probs = torch.sigmoid(outputs)

        all_probs.append(
            probs.cpu().numpy()
        )

        all_labels.append(
            labels.numpy()
        )


all_probs = np.concatenate(all_probs)
all_labels = np.concatenate(all_labels)


# Find the best threshold for each class
best_thresholds = []

print("\nBest threshold for each class:\n")

for i, class_name in enumerate(CLASS_NAMES):

    best_f1 = 0
    best_threshold = 0.5

    for threshold in np.arange(
        0.1, 0.95, 0.05
    ):

        predictions = (
            all_probs[:, i] >= threshold
        ).astype(int)

        f1 = f1_score(
            all_labels[:, i],
            predictions,
            zero_division=0
        )

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    best_thresholds.append(
        best_threshold
    )

    print(
        f"{class_name:15s} "
        f"Threshold: {best_threshold:.2f} "
        f"F1: {best_f1:.4f}"
    )


print("Thresholds:")
print(best_thresholds)