import torch
import numpy as np

from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

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

THRESHOLDS = [
    0.35,
    0.45,
    0.20,
    0.50,
    0.30,
    0.35
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


# Load fine-tuned model
model = CarDamageModel().to(device)

model.load_state_dict(
    torch.load(
        "best_model_finetuned.pth",
        map_location=device
    )
)

model.eval()

predictions = []
labels_list = []


with torch.no_grad():

    for images, labels in loader:

        images = images.to(device)

        outputs = model(images)

        probabilities = torch.sigmoid(
            outputs
        ).cpu().numpy()

        thresholds = np.array(
            THRESHOLDS
        )

        batch_predictions = (
            probabilities >= thresholds
        ).astype(int)

        predictions.append(
            batch_predictions
        )

        labels_list.append(
            labels.numpy()
        )


predictions = np.concatenate(
    predictions
)

labels_list = np.concatenate(
    labels_list
)


print("\nClassification Report:\n")

print(
    classification_report(
        labels_list,
        predictions,
        target_names=CLASS_NAMES,
        zero_division=0
    )
)