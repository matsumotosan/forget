from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from rich import print
from torch.nn.modules import KLDivLoss
from mufac_dataset import MUFAC
from unlearn import unlearn
from unlearning_datamodule import CIFAR10UnlearningDataModule, mufac_transform
from utils import evaluate, setup_log_dir, save_json
from models import load_resnet18

DATASET = "mufac"

DATA_DIR = "./data"
MODEL_DIR = "./models"
LOG_DIR = "./logs"

# forget classes (one or more of the following)
# "0-6 years old"
# "7-12 years old"
# "13-19 years old"
# "20-30 years old"
# "31-45 years old"
# "46-55 years old"
# "56-66 years old"
# "67-80 years old"
FORGET_CLASS = ("0-6 years old",)

BATCH_SIZE = 64
TRAIN_EPOCHS = 3
UNLEARN_EPOCHS = 20
LEARNING_RATE = 1e-3
UNLEARNING_RATE = 1e-3
RETAIN_RATE = 1e-3

FROM_SCRATCH = False
FORGET = True
RETAIN = False


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    experiment_dir = setup_log_dir(LOG_DIR, DATASET)

    # Save experiment settings
    params = {
        "dataset": DATASET,
        "forget_class": FORGET_CLASS,
        "batch_size": BATCH_SIZE,
        "train_epochs": TRAIN_EPOCHS,
        "unlearn_epochs": UNLEARN_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "unlearning_rate": UNLEARNING_RATE,
        "retain_rate": RETAIN_RATE,
        "forget_step": FORGET,
        "retain_step": RETAIN,
    }
    save_json(f"{experiment_dir}/params.json", params)

    ds = MUFAC(
        root=DATA_DIR,
        transform=mufac_transform,
    )

    img, label = ds[0]
    print(img.shape)
    print(label)


if __name__ == "__main__":
    main()
