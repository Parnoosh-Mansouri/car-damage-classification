import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score

from dataset_augmented import (
    CarDamageDataset,
    train_transform,
    val_transform
)

from model import CarDamageModel


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BATCH_SIZE = 32
EPOCHS = 5
LR = 0.0001


# Prepare datasets
train_dataset = CarDamageDataset(
    "data/train.csv",
    "dataset/CarDD_COCO/train2017",
    train_transform
)

val_dataset = CarDamageDataset(
    "data/val.csv",
    "dataset/CarDD_COCO/val2017",
    val_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE
)


# Load pretrained model
model = CarDamageModel().to(device)

for param in model.model.parameters():
    param.requires_grad = False

for param in model.model.layer4.parameters():
    param.requires_grad = True

for param in model.model.fc.parameters():
    param.requires_grad = True


criterion = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(
    filter(
        lambda p: p.requires_grad,
        model.parameters()
    ),
    lr=LR
)

best_loss = float("inf")


# Fine-tuning loop
for epoch in range(EPOCHS):

    model.train()
    train_loss = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    model.eval()
    val_loss = 0

    predictions = []
    true_labels = []

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            val_loss += criterion(
                outputs, labels
            ).item()

            predictions.append(
                (torch.sigmoid(outputs) >= 0.5).cpu()
            )

            true_labels.append(
                labels.cpu()
            )

    val_loss /= len(val_loader)

    predictions = torch.cat(
        predictions
    ).numpy()

    true_labels = torch.cat(
        true_labels
    ).numpy()

    precision = precision_score(
        true_labels,
        predictions,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        true_labels,
        predictions,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        true_labels,
        predictions,
        average="macro",
        zero_division=0
    )

    print(
        f"\nEpoch [{epoch + 1}/{EPOCHS}]"
    )

    print(
        f"Train Loss: {train_loss:.4f}"
    )

    print(
        f"Val Loss: {val_loss:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall: {recall:.4f}"
    )

    print(
        f"F1: {f1:.4f}"
    )

    if val_loss < best_loss:

        best_loss = val_loss

        torch.save(
            model.state_dict(),
            "best_model_finetuned.pth"
        )

        print(
            "Best fine-tuned model saved!"
        )


print("\nFine-tuning completed!")