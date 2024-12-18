import numpy as np
import torch.nn as nn
import json
import os
import torch
from datetime import datetime
from torchmetrics.classification import MulticlassAccuracy
from tqdm import tqdm


cifar10_idx2class = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck",
}

mufac_idx2class = {
    0: "0-6 years old",
    1: "7-12 years old",
    2: "13-19 years old",
    3: "20-30 years old",
    4: "31-45 years old",
    5: "46-55 years old",
    6: "56-66 years old",
    7: "67-80 years old"
}

cifar10_class2idx = {v: k for k, v in cifar10_idx2class.items()}
mufac_class2idx = {v: k for k, v in mufac_idx2class.items()}

def train(
    model, dataloader, criterion, optimizer, epochs, device, verbose: bool = True
):
    model.train()
    accuracy = MulticlassAccuracy(num_classes=10, average=None).to(device)

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}")
        accuracy.reset()
        running_loss = 0

        for images, labels in tqdm(dataloader, desc="train"):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            accuracy.update(outputs, labels)

        if verbose:
            print(
                f"loss: {running_loss/len(dataloader)}, " f"acc: {accuracy.compute()}"
            )


def evaluate(model, dataloader, n_classes, device, forget: bool = False):
    accuracy = MulticlassAccuracy(n_classes, average=None).to(device)
    criterion = nn.CrossEntropyLoss()
    loss = 0

    model.eval()
    for images, labels in dataloader:
        images = images.to(device)
        outputs = model(images)

        if forget:
            labels = torch.ones_like(labels)
        labels = labels.to(device)

        loss += criterion(outputs, labels).item()
        accuracy.update(outputs, labels)
    
    loss /= len(dataloader)
    return accuracy.compute().tolist(), loss


def per_sample_loss(model, dataloader, device):
    criterion = nn.CrossEntropyLoss(reduction="none")
    losses = np.array([])

    model.eval()
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels).detach().cpu().numpy()
        losses = np.concatenate((losses, loss))
    
    return losses


def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f)


def read_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data


def setup_log_dir(log_dir, dataset):
    now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    experiment_dir = f"{log_dir}/{dataset}/{now}"
    os.makedirs(experiment_dir)
    return experiment_dir
