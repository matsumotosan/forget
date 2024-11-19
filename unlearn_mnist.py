import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchmetrics.classification import MulticlassAccuracy
from torchvision import datasets, transforms
from rich import print


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


BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 1e-3

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

# Split training dataset into retain and forget subsets
train_dataset = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)

forget_class = 5
forget_idx = np.where(train_dataset.targets == forget_class)[0]
retain_idx = np.where(train_dataset.targets != forget_class)[0]

forget_dataset = Subset(train_dataset, forget_idx)
retain_dataset = Subset(train_dataset, retain_idx)

test_dataset = datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
forget_loader = DataLoader(forget_dataset, batch_size=BATCH_SIZE, shuffle=True)
retain_loader = DataLoader(retain_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


def train(model, dataloader, criterion, optimizer, epochs):
    model.train()
    accuracy = MulticlassAccuracy(num_classes=len(train_dataset.classes), average=None)

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
    model.eval()
    accuracy = MulticlassAccuracy(num_classes=len(train_dataset.classes), average=None)

    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images)
            accuracy.update(outputs, labels)

    return accuracy.compute()


def unlearn(
    model,
    retain_dataloader,
    forget_dataloader,
    epochs,
    retain_loop: bool = True,
):
    model.train()
    accuracy = MulticlassAccuracy(num_classes=len(train_dataset.classes), average=None)

    for epoch in range(epochs):
        accuracy.reset()
        running_loss = 0

        # Forget loop
        for images, labels in forget_dataloader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            accuracy.update(outputs, labels)

        # Retain loop
        if retain_loop:
            for images, labels in retain_dataloader:
                pass

        print(
            f"Epoch {epoch+1} - "
            f"loss: {running_loss/len(retain_dataloader)}, "
            f"acc: {accuracy.compute()}"
        )


# Train with full dataset
model = SimpleNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

train(model, train_loader, criterion, optimizer, EPOCHS)
initial_accuracy = evaluate(model, test_loader)
print(f"Accuracy: {initial_accuracy}")

# Train with retain dataset (gold standard)
model = SimpleNet()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
train(model, retain_loader, criterion, optimizer, EPOCHS)
unlearn_exact_accuracy = evaluate(model, test_loader)
print(f"Unlearned accuracy (exact): {unlearn_exact_accuracy}%")

# Unlearning with KL divergence loss
# model = SimpleNet()
# optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
# unlearn(model, retain_loader, forget_loader, criterion, optimizer, EPOCHS)
# unlearn_inexact_accuracy = evaluate(model, test_loader)
# print(f"Unlearned accuracy (inexact): {unlearn_inexact_accuracy}%")
