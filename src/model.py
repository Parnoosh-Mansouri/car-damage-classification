import torch
import torch.nn as nn
from torchvision import models


class CarDamageModel(nn.Module):

    # Initialize the model
    def __init__(self, num_classes=6):
        super().__init__()

        # Load a pretrained ResNet18
        self.model = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )

        # Get the number of input features of the classifier
        num_features = self.model.fc.in_features

        # Replace the original classifier with our 6-class classifier
        self.model.fc = nn.Linear(num_features, num_classes)

    # Define the forward pass
    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":

    # Select GPU if available
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Create the model
    model = CarDamageModel()

    # Move the model to the selected device
    model = model.to(device)

    print("Device:", device)
    print("Model created successfully!")