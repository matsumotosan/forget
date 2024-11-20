import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
from torchmetrics.classification import MulticlassAccuracy
from torchvision import datasets, transforms
from rich import print
from pathlib import Path
from copy import deepcopy


DATA_DIR = "./data"
MODEL_DIR = "./models"
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 1e-3


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


def train(model, dataloader, criterion, optimizer, epochs):
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


def unlearn(
    model,
    retain_dataloader,
    forget_dataloader,
    val_dataloader,
    retain_optimizer,
    forget_optimizer,
    epochs,
    forget_criterion=nn.KLDivLoss(reduction="batchmean"),
    retain_loop: bool = True,
    retain_criterion: nn.Module = nn.CrossEntropyLoss(),
):
    retain_acc = MulticlassAccuracy(num_classes=10, average=None)
    val_acc = MulticlassAccuracy(num_classes=10, average=None)

    for epoch in range(epochs):
        model.train()

        retain_acc.reset()
        val_acc.reset()

        running_forget_loss = 0
        running_retain_loss = 0
        running_val_loss = 0

        # Forget loop
        for images, _ in forget_dataloader:
            forget_optimizer.zero_grad()
            outputs = model(images).softmax(dim=1)

            # Minimize KL divergence from uniform logits
            uniform_logits = torch.ones_like(outputs) / len(outputs)
            forget_loss = forget_criterion(outputs, uniform_logits)
            forget_loss.backward()
            running_forget_loss += forget_loss.item()
            forget_optimizer.step()

        # Retain loop
        if retain_loop:
            for images, labels in retain_dataloader:
                retain_optimizer.zero_grad()
                outputs = model(images)
                retain_loss = retain_criterion(outputs, labels)
                retain_loss.backward()
                retain_optimizer.step()
                running_retain_loss += retain_loss.item()
                retain_acc.update(outputs, labels)

        # Validation on all classes
        model.eval()
        with torch.no_grad():
            for images, labels in val_dataloader:
                outputs = model(images)
                val_loss = retain_criterion(outputs, labels)
                running_val_loss += val_loss.item()
                val_acc.update(outputs, labels)

        print(
            f"Epoch {epoch+1} - "
            f"forget_loss: {running_forget_loss/len(retain_dataloader)}, "
            f"val_loss: {running_val_loss/len(val_dataloader)}, "
            f"val_acc: {val_acc.compute()}"
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

    # Train with full dataset
    print("=== Standard training ===")
    trained_model = SimpleNet()
    if trained_model_path.exists():
        print(f"Loading pretrained model from {trained_model_path}.")
        trained_model.load_state_dict(torch.load(trained_model_path, weights_only=True))
    else:
        optimizer = optim.Adam(trained_model.parameters(), lr=LEARNING_RATE)
        train(trained_model, train_loader, criterion, optimizer, EPOCHS)

        print(f"Saving trained model to {trained_model_path}.")
        torch.save(trained_model.state_dict(), trained_model_path)

    trained_acc = evaluate(trained_model, test_loader)
    print(f"Trained accuracy: {trained_acc}")

    # Retrain with retain dataset (gold standard)
    print("\n=== Retrain on retain dataset (gold standard) ===")
    retrained_model = SimpleNet()
    if retrained_model_path.exists():
        print(f"Loading retrained model from {retrained_model_path}.")
        retrained_model.load_state_dict(
            torch.load(retrained_model_path, weights_only=True)
        )
    else:
        optimizer = optim.Adam(retrained_model.parameters(), lr=LEARNING_RATE)
        train(retrained_model, retain_loader, criterion, optimizer, EPOCHS)

        print(f"Saving retrained model to {retrained_model_path}.")
        torch.save(retrained_model.state_dict(), retrained_model_path)

    retrained_acc = evaluate(retrained_model, test_loader)
    print(f"Retrained accuracy: {retrained_acc}")

    # Unlearning with KL divergence loss (with retain step)
    print("\n=== Finetune with KLDiv loss (with retain step) ===")
    unlearned_model = deepcopy(trained_model)

    unlearned_initial_acc = evaluate(unlearned_model, test_loader)
    print(f"Trained accuracy: {unlearned_initial_acc}")

    forget_optimizer = optim.Adam(unlearned_model.parameters(), lr=LEARNING_RATE)
    retain_optimizer = optim.Adam(unlearned_model.parameters(), lr=LEARNING_RATE)

    unlearn(
        unlearned_model,
        retain_loader,
        forget_loader,
        val_loader,
        retain_optimizer,
        forget_optimizer,
        EPOCHS,
        retain_loop=True,
    )

    unlearned_acc = evaluate(unlearned_model, test_loader)
    print(f"Unlearned accuracy (with retain step): {unlearned_acc}")


if __name__ == "__main__":
    main()
