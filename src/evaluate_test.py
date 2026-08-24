import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

from dataset_augmented import CarDamageDataset, val_transform
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

# Thresholds selected using the validation set
THRESHOLDS = [
    0.35,
    0.45,
    0.20,
    0.50,
    0.30,
    0.35
]


# Load test data
test_dataset = CarDamageDataset(
    "data/test.csv",
    "dataset/CarDD_COCO/test2017",
    transform=val_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)


# Load the final fine-tuned model
model = CarDamageModel().to(device)

model.load_state_dict(
    torch.load(
        "best_model_finetuned.pth",
        map_location=device
    )
)

model.eval()


all_predictions = []
all_labels = []


# Evaluate only on the unseen test set
with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        probabilities = torch.sigmoid(outputs)

        thresholds = torch.tensor(
            THRESHOLDS,
            device=device
        )

        predictions = (
            probabilities >= thresholds
        ).float()

        all_predictions.append(
            predictions.cpu()
        )

        all_labels.append(labels)


all_predictions = torch.cat(
    all_predictions
).numpy()

all_labels = torch.cat(
    all_labels
).numpy()


# Final test-set evaluation
print("\nFinal Test Classification Report:\n")

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=CLASS_NAMES,
        zero_division=0
    )
)