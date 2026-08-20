import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


class CarDamageDataset(Dataset):

    def __init__(self, csv_file, image_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]

        image = Image.open(
            f"{self.image_dir}/{row['file_name']}"
        ).convert("RGB")

        labels = torch.tensor(
            row[
                [
                    "dent",
                    "scratch",
                    "crack",
                    "glass_shatter",
                    "lamp_broken",
                    "tire_flat"
                ]
            ].values.astype("float32")
        )

        if self.transform:
            image = self.transform(image)

        return image, labels


if __name__ == "__main__":
    dataset = CarDamageDataset(
        "data/train.csv",
        "dataset/CarDD_COCO/train2017",
        transform
    )

    image, label = dataset[0]

    print("Number of samples:", len(dataset))
    print("Image shape:", image.shape)
    print("Label:", label)