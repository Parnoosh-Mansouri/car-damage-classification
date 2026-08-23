import sys
import torch
from PIL import Image

from model import CarDamageModel
from dataset_augmented import val_transform


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


# Load the final model
model = CarDamageModel().to(device)

model.load_state_dict(
    torch.load(
        "best_model_finetuned.pth",
        map_location=device
    )
)

model.eval()


# Load and preprocess image
image_path = sys.argv[1]

image = Image.open(
    image_path
).convert("RGB")

image = val_transform(image)
image = image.unsqueeze(0).to(device)


# Make prediction
with torch.no_grad():

    outputs = model(image)

    probabilities = torch.sigmoid(
        outputs
    )[0]


# Display results
print("\nPrediction:\n")

found = False

for name, probability, threshold in zip(
    CLASS_NAMES,
    probabilities,
    THRESHOLDS
):

    probability = probability.item()

    if probability >= threshold:

        print(
            f"{name:15s} "
            f"{probability:.2%}"
        )

        found = True


if not found:
    print("No damage detected.")