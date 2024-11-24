import torch
import torch.nn as nn
from rich import print
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy
import torch.nn.functional as F


def unlearn(
    model: nn.Module,
    retain_dataloader: DataLoader,
    forget_dataloader: DataLoader,
    val_dataloader: DataLoader,
    retain_optimizer: Optimizer,
    forget_optimizer: Optimizer,
    epochs: int,
    forget_step: bool = True,
    retain_step: bool = True,
    forget_criterion: nn.Module = nn.KLDivLoss(reduction="batchmean"),
    retain_criterion: nn.Module = nn.CrossEntropyLoss(),
    verbose: bool = True,
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

        # Forget step
        if forget_step:
            for images, _ in forget_dataloader:
                forget_optimizer.zero_grad()
                outputs = F.log_softmax(model(images), dim=1)

                # Minimize KL divergence from uniform logits
                uniform_logits = F.softmax(torch.ones_like(outputs), dim=1)
                forget_loss = forget_criterion(outputs, uniform_logits)
                forget_loss.backward()
                running_forget_loss += forget_loss.item()
                forget_optimizer.step()

        # Retain step
        if retain_step:
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

        if verbose:
            print(
                f"Epoch {epoch+1} - "
                f"forget_loss: {running_forget_loss/len(retain_dataloader)}, "
                f"val_loss: {running_val_loss/len(val_dataloader)}, "
                f"val_acc: {val_acc.compute()}"
            )
