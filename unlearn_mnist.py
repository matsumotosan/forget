from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from rich import print
from torch.nn.modules import KLDivLoss
from unlearn import unlearn
from unlearning_datamodule import MNISTUnlearningDataModule
from utils import evaluate, train

DATA_DIR = "./data"
MODEL_DIR = "./models"
FIG_DIR = "./figures"

BATCH_SIZE = 64
TRAIN_EPOCHS = 3
UNLEARN_EPOCHS = 3
LEARNING_RATE = 1e-3
UNLEARNING_RATE = 1e-3
RETAIN_RATE = 1e-3

FROM_SCRATCH = True

sns.set_theme()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def main():
    forget_class = (5, 7)

    # Initialize unlearning datamodule
    dm = MNISTUnlearningDataModule(
        data_dir=DATA_DIR,
        forget_class=torch.tensor(forget_class),
    )

    dm.setup()

    # Get dataloader for each dataset split
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()
    forget_loader = dm.forget_dataloader()
    retain_loader = dm.retain_dataloader()

    # Set model paths
    trained_model_path = Path(f"{MODEL_DIR}/trained_mnist.pt")
    retrained_model_path = Path(f"{MODEL_DIR}/retrained_mnist.pt")
    criterion = nn.CrossEntropyLoss()

    # Train on original training dataset
    print("=== Standard training ===")
    trained_model = SimpleNet().to(device)
    if trained_model_path.exists() and not FROM_SCRATCH:
        print(f"Loading pretrained model from {trained_model_path}.")
        trained_model.load_state_dict(torch.load(trained_model_path, weights_only=True))
    else:
        optimizer = optim.Adam(trained_model.parameters(), lr=LEARNING_RATE)
        train(trained_model, train_loader, criterion, optimizer, TRAIN_EPOCHS, device)

        print(f"Saving trained model to {trained_model_path}.")
        torch.save(trained_model.state_dict(), trained_model_path)

    acc_trained = evaluate(trained_model, test_loader, 10, device)
    print(f"Trained accuracy: {acc_trained}")

    # Retrain on retain dataset (gold standard)
    print("\n=== Retrain on retain dataset (gold standard) ===")
    retrained_model = SimpleNet().to(device)
    if retrained_model_path.exists() and not FROM_SCRATCH:
        print(f"Loading retrained model from {retrained_model_path}.")
        retrained_model.load_state_dict(
            torch.load(retrained_model_path, weights_only=True)
        )
    else:
        optimizer = optim.Adam(retrained_model.parameters(), lr=LEARNING_RATE)
        train(
            retrained_model, retain_loader, criterion, optimizer, TRAIN_EPOCHS, device
        )

        print(f"Saving retrained model to {retrained_model_path}.")
        torch.save(retrained_model.state_dict(), retrained_model_path)

    acc_retrained = evaluate(retrained_model, test_loader, 10, device)
    print(f"Retrained accuracy: {acc_retrained}")

    # Unlearning with KL divergence loss (with retain step)
    print("\n=== Finetune with KLDiv loss (with retain step) ===")
    unlearned_model = trained_model

    unlearned_initial_acc = evaluate(unlearned_model, test_loader, 10, device)
    print(f"Trained accuracy (starting point for unlearning): {unlearned_initial_acc}")

    forget_optimizer = optim.Adam(unlearned_model.parameters(), lr=UNLEARNING_RATE)
    retain_optimizer = optim.Adam(unlearned_model.parameters(), lr=RETAIN_RATE)
    forget_criterion = KLDivLoss(reduction="batchmean")

    unlearn(
        model=unlearned_model,
        retain_dataloader=retain_loader,
        forget_dataloader=forget_loader,
        val_dataloader=val_loader,
        retain_optimizer=retain_optimizer,
        forget_optimizer=forget_optimizer,
        epochs=UNLEARN_EPOCHS,
        device=device,
        forget_criterion=forget_criterion,
        forget_step=False,
        retain_step=True,
    )

    acc_unlearned = evaluate(unlearned_model, test_loader, 10, device)
    print(f"Unlearned accuracy (with retain step): {acc_unlearned}")
    print(
        f"Change in accuracy (acc_unlearned - acc_trained): {acc_unlearned - acc_trained}"
    )

    # Plot per-class accuracy
    acc_df = pd.DataFrame(
        {
            "class": np.arange(10),
            "acc_trained": acc_trained.cpu().detach().numpy(),
            "acc_retrained": acc_retrained.cpu().detach().numpy(),
            "acc_unlearned": acc_unlearned.cpu().detach().numpy(),
        }
    )

    acc_df = acc_df.melt(
        id_vars="class", value_vars=["acc_trained", "acc_retrained", "acc_unlearned"]
    )

    f, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=acc_df, x="class", y="value", hue="variable")
    ax.set_title("Classwise accuracy on MNIST")
    f.tight_layout()
    plt.savefig(f"{FIG_DIR}/unlearn_mnist_class_acc.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
