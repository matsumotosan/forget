import torch
from torchmetrics.classification import MulticlassAccuracy


def train(model, dataloader, criterion, optimizer, epochs, device, verbose: bool = True):
    model.train()
    accuracy = MulticlassAccuracy(num_classes=10, average=None).to(device)

    for epoch in range(epochs):
        accuracy.reset()
        running_loss = 0

        for images, labels in dataloader:
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
                f"Epoch {epoch+1} - "
                f"loss: {running_loss/len(dataloader)}, "
                f"acc: {accuracy.compute()}"
            )


def evaluate(model, dataloader, n_classes, device):
    accuracy = MulticlassAccuracy(n_classes, average=None).to(device)

    model.eval()
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        accuracy.update(outputs, labels)

    return accuracy.compute()
