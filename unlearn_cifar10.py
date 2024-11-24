import os
import requests
from pathlib import Path
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from rich import print
from torch.nn.modules import KLDivLoss
from torchvision.datasets.cifar import CIFAR10
from unlearn import unlearn
from unlearning_datamodule import CIFAR10UnlearningDataModule, cifar10_transform
from utils import evaluate, train
from torchvision.models import resnet18
from torch.utils.data import DataLoader

DATA_DIR = "./data"
MODEL_DIR = "./models"
FIG_DIR = "./figures"

BATCH_SIZE = 64
TRAIN_EPOCHS = 3
UNLEARN_EPOCHS = 3
LEARNING_RATE = 1e-3
UNLEARNING_RATE = 1e-3
RETAIN_RATE = 1e-3

sns.set_theme()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_resnet18(weights_path="models/weights_resnet18_cifar10.pth"):
    """Load ResNet18 model for CIFAR10."""
    if not os.path.exists(weights_path):
        response = requests.get(
            "https://storage.googleapis.com/unlearning-challenge/weights_resnet18_cifar10.pth"
        )
        open(weights_path, "wb").write(response.content)

    weights = torch.load(weights_path, map_location=DEVICE)
    model = resnet18(num_classes=10)
    model.load_state_dict(weights)
    return model


def main():
    # Choose from ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    forget_class = ("airplane",)
    ds = CIFAR10(root=DATA_DIR)

    # Initialize unlearning datamodule
    dm = CIFAR10UnlearningDataModule(
        data_dir=DATA_DIR,
        forget_class=[ds.class_to_idx[c] for c in forget_class],
    )

    dm.setup()

    # Get dataloader for each dataset split
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()
    forget_loader = dm.forget_dataloader()
    retain_loader = dm.retain_dataloader()

    # Set model paths
    trained_model_path = f"{MODEL_DIR}/weights_resnet18_cifar10.pt"
    retrained_model_path = Path(f"{MODEL_DIR}/retrained_cifar10.pt")
    criterion = nn.CrossEntropyLoss()

    # Train on original training dataset
    print("=== Standard training ===")
    trained_model = load_resnet18(weights_path=trained_model_path)
    acc_trained = evaluate(trained_model, test_loader, 10)
    print(f"Trained accuracy: {acc_trained}")

    # # Unlearning with KL divergence loss (with retain step)
    # print("\n=== Finetune with KLDiv loss (with retain step) ===")
    # unlearned_model = deepcopy(trained_model)
    #
    # unlearned_initial_acc = evaluate(unlearned_model, test_loader, 10)
    # print(f"Trained accuracy (starting point for unlearning): {unlearned_initial_acc}")
    #
    # forget_optimizer = optim.Adam(unlearned_model.parameters(), lr=UNLEARNING_RATE)
    # retain_optimizer = optim.Adam(unlearned_model.parameters(), lr=RETAIN_RATE)
    # forget_criterion = KLDivLoss(reduction="batchmean")
    #
    # unlearn(
    #     model=unlearned_model,
    #     retain_dataloader=retain_loader,
    #     forget_dataloader=forget_loader,
    #     val_dataloader=val_loader,
    #     retain_optimizer=retain_optimizer,
    #     forget_optimizer=forget_optimizer,
    #     epochs=UNLEARN_EPOCHS,
    #     forget_criterion=forget_criterion,
    #     forget_step=False,
    #     retain_step=True,
    # )
    #
    # acc_unlearned = evaluate(unlearned_model, test_loader, 10)
    # print(f"Unlearned accuracy (with retain step): {acc_unlearned}")
    # print(
    #     f"Change in accuracy (acc_unlearned - acc_trained): {acc_unlearned - acc_trained}"
    # )
    #
    # # Plot per-class accuracy
    # acc_df = pd.DataFrame(
    #     {
    #         "class": np.arange(10),
    #         "acc_trained": acc_trained.cpu().detach().numpy(),
    #         # "acc_retrained": acc_retrained.cpu().detach().numpy(),
    #         "acc_unlearned": acc_unlearned.cpu().detach().numpy(),
    #     }
    # )
    #
    # acc_df = acc_df.melt(
    #     id_vars="class", value_vars=["acc_trained", "acc_retrained", "acc_unlearned"]
    # )
    #
    # f, ax = plt.subplots(figsize=(8, 5))
    # sns.barplot(data=acc_df, x="class", y="value", hue="variable")
    # ax.set_title("Classwise accuracy on cifar10")
    # f.tight_layout()
    # plt.savefig(f"{FIG_DIR}/unlearn_cifar10_class_acc.png", dpi=300)
    # plt.show()


if __name__ == "__main__":
    main()
