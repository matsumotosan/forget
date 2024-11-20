from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from rich import print
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader, Subset, random_split
from torchmetrics.classification import MulticlassAccuracy
from torchvision import datasets, transforms
from unlearn import unlearn
from torch import Tensor

DATA_DIR = "./data"
MODEL_DIR = "./models"
FIG_DIR = "./figures"

BATCH_SIZE = 64
TRAIN_EPOCHS = 3
UNLEARN_EPOCHS = 5
LEARNING_RATE = 1e-3
UNLEARNING_RATE = 1e-3
RETAIN_RATE = 1e-3

sns.set_theme()


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


def train(model, dataloader, criterion, optimizer, epochs, verbose: bool = True):
    model.train()
    accuracy = MulticlassAccuracy(num_classes=10, average=None)

    for epoch in range(epochs):
        accuracy.reset()
        running_loss = 0

        for images, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            accuracy.update(outputs, labels)

        if verbose:
            print(
                f"Epoch {epoch+1} - "
                f"loss: {running_loss/len(dataloader)}, "
                f"acc: {accuracy.compute()}"
            )


def evaluate(model, dataloader):
    accuracy = MulticlassAccuracy(num_classes=10, average=None)

    model.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images)
            accuracy.update(outputs, labels)

    return accuracy.compute()


class JSDLoss(_Loss):
    """Jensen-Shannon divergence loss."""

    def __init__(
        self,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
        log_target: bool = False,
    ) -> None:
        super().__init__(size_average, reduce, reduction)
        self.log_target = log_target

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        M = 0.5 * (input + target)
        return 0.5 * F.kl_div(
            input, M, reduction=self.reduction, log_target=self.log_target
        ) + 0.5 * F.kl_div(
            target, M, reduction=self.reduction, log_target=self.log_target
        )


def main():
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_dataset = datasets.MNIST(
        root=DATA_DIR, train=True, download=True, transform=transform
    )

    test_dataset = datasets.MNIST(
        root=DATA_DIR, train=False, download=True, transform=transform
    )

    train_dataset, val_dataset = random_split(train_dataset, [0.9, 0.1])

    # Split training dataset into retain and forget subsets
    forget_class = 5
    forget_idx = np.where(
        train_dataset.dataset.targets[train_dataset.indices] == forget_class
    )[0]
    retain_idx = np.where(
        train_dataset.dataset.targets[train_dataset.indices] != forget_class
    )[0]

    forget_dataset = Subset(train_dataset, forget_idx)
    retain_dataset = Subset(train_dataset, retain_idx)

    # Create dataloaders for each dataset split
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    forget_loader = DataLoader(forget_dataset, batch_size=BATCH_SIZE, shuffle=True)
    retain_loader = DataLoader(retain_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Set model paths
    trained_model_path = Path(f"{MODEL_DIR}/trained_mnist.pt")
    retrained_model_path = Path(f"{MODEL_DIR}/retrained_mnist.pt")
    criterion = nn.CrossEntropyLoss()

    # Train on original training dataset
    print("=== Standard training ===")
    trained_model = SimpleNet()
    if trained_model_path.exists():
        print(f"Loading pretrained model from {trained_model_path}.")
        trained_model.load_state_dict(torch.load(trained_model_path, weights_only=True))
    else:
        optimizer = optim.Adam(trained_model.parameters(), lr=LEARNING_RATE)
        train(trained_model, train_loader, criterion, optimizer, TRAIN_EPOCHS)

        print(f"Saving trained model to {trained_model_path}.")
        torch.save(trained_model.state_dict(), trained_model_path)

    acc_trained = evaluate(trained_model, test_loader)
    print(f"Trained accuracy: {acc_trained}")

    # Retrain on retain dataset (gold standard)
    print("\n=== Retrain on retain dataset (gold standard) ===")
    retrained_model = SimpleNet()
    if retrained_model_path.exists():
        print(f"Loading retrained model from {retrained_model_path}.")
        retrained_model.load_state_dict(
            torch.load(retrained_model_path, weights_only=True)
        )
    else:
        optimizer = optim.Adam(retrained_model.parameters(), lr=LEARNING_RATE)
        train(retrained_model, retain_loader, criterion, optimizer, TRAIN_EPOCHS)

        print(f"Saving retrained model to {retrained_model_path}.")
        torch.save(retrained_model.state_dict(), retrained_model_path)

    acc_retrained = evaluate(retrained_model, test_loader)
    print(f"Retrained accuracy: {acc_retrained}")

    # Unlearning with KL divergence loss (with retain step)
    print("\n=== Finetune with KLDiv loss (with retain step) ===")
    unlearned_model = deepcopy(trained_model)

    unlearned_initial_acc = evaluate(unlearned_model, test_loader)
    print(f"Trained accuracy (starting point for unlearning): {unlearned_initial_acc}")

    forget_optimizer = optim.Adam(unlearned_model.parameters(), lr=UNLEARNING_RATE)
    retain_optimizer = optim.Adam(unlearned_model.parameters(), lr=RETAIN_RATE)

    unlearn(
        unlearned_model,
        retain_loader,
        forget_loader,
        val_loader,
        retain_optimizer,
        forget_optimizer,
        UNLEARN_EPOCHS,
        retain_step=True,
    )

    acc_unlearned = evaluate(unlearned_model, test_loader)
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
