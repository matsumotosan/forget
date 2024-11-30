import os
from typing import Any, Union, Optional, Callable
from PIL import Image
import pandas as pd
from pathlib import Path
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import download_and_extract_archive


class MUFAC(VisionDataset):
    base_dir = "mufac"
    url = "https://postechackr-my.sharepoint.com/:u:/g/personal/dongbinna_postech_ac_kr/EbMhBPnmIb5MutZvGicPKggBWKm5hLs0iwKfGW7_TwQIKg?download=1"
    filename = "mufac.zip"
    alpha2idx = {
        "a": 0,
        "b": 1,
        "c": 2,
        "d": 3,
        "e": 4,
        "f": 5,
        "g": 6,
        "h": 7,
    }

    idx2age = {
        0: "0-6 years old",
        1: "7-12 years old",
        2: "13-19 years old",
        3: "20-30 years old",
        4: "31-45 years old",
        5: "46-55 years old",
        6: "56-66 years old",
        7: "67-80 years old",
    }

    meta_paths = {
        "train": "custom_train_dataset.csv",
        "val": "custom_val_dataset.csv",
        "test": "custom_test_dataset.csv",
    }

    img_dir = {
        "train": "train_images",
        "val": "val_images",
        "test": "test_images",
    }

    def __init__(
        self,
        root: str,
        stage: str = "train",
        transform: Optional[Callable] = None,
    ) -> None:

        self.root = root
        self.stage = stage
        self.transform = transform

        self.data, self.targets = self.parse_meta_data()

    def parse_meta_data(self):
        meta = pd.read_csv(os.path.join(self.root, self.base_dir, self.meta_paths[self.stage]))
        data = meta["image_path"].tolist()
        targets = [self.alpha2idx[c] for c in meta["age_class"].tolist()]
        return data, targets

    def __getitem__(self, idx):
        image_path, target = self.data[idx], self.targets[idx]
        img = Image.open(os.path.join(self.root, self.base_dir, self.img_dir[self.stage], image_path))

        if self.transform:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.targets)
