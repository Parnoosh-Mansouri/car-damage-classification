import torch
import torch.nn as nn
from torchvision import models


class CarDamageModel(nn.Module):

    def __init__(self):
        super().__init__()

        # Pretrained ResNet18
        self.model = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )

        self.model.fc = nn.Linear(
            self.model.fc.in_features,
            6
        )

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = CarDamageModel().to(device)

    x = torch.randn(
        2, 3, 224, 224
    ).to(device)

    print("Device:", device)
    print("Output shape:", model(x).shape)