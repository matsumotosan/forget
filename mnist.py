import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchmetrics.classification import MulticlassAccuracy
from torchvision import datasets, transforms


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
train_dataset = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# def calc_accuracy():
#     model.eval()
#     correct = {i: 0 for i in range(10)}
#     total = {i: 0 for i in range(10)}
#
#     with torch.no_grad():
#         for images, labels in loader:
#             outputs = model(images)
#             _, predicted = torch.max(outputs.data, 1)
#             for label, prediction in zip(labels, predicted):
#                 if label == prediction:
#                     correct[label.item()] += 1
#                 total[label.item()] += 1
#
#     accuracy_per_class = {
#         label: (correct[label] / total[label] * 100) if total[label] > 0 else 0
#         for label in range(10)
#     }
#     return total_acc, accuracy_per_class


def train(model, dataloader, criterion, optimizer, epochs):
    model.train()
    # accuracy = Accuracy(task="multiclass", num_classes=len(train_dataset.classes))
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
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy


# train with full dataset
model = SimpleNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train(model, train_loader, criterion, optimizer, EPOCHS)
initial_accuracy = evaluate(model, test_loader)
print(f"Accuracy: {initial_accuracy}%")

# train with subset (remove '5')
indices = np.where(train_dataset.targets != 5)[0]
filtered_train_dataset = Subset(train_dataset, indices)
filtered_train_loader = DataLoader(filtered_train_dataset, batch_size=64, shuffle=True)

unlearned_model = SimpleNet()
optimizer = optim.Adam(unlearned_model.parameters(), lr=LEARNING_RATE)
train(unlearned_model, filtered_train_loader, criterion, optimizer, EPOCHS)
unlearned_accuracy = evaluate(unlearned_model, test_loader)
print(f"Unlearning accuracy (exact): {unlearned_accuracy}%")
