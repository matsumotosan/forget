import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from rich import print
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy
from tqdm import tqdm
from utils import save_json, evaluate


def print_metrics(metrics, epoch=-1) -> None:
    print(
        f"forget_loss: {metrics['forget_loss'][epoch]}\n"
        f"retain_loss: {metrics['retain_loss'][epoch]}\n"
        f"val_loss: {metrics['val_loss'][epoch]}\n"
        f"forget_acc: {metrics['forget_acc'][epoch]}\n"
        f"retain_acc: {metrics['retain_acc'][epoch]}\n"
        f"val_acc: {metrics['val_acc'][epoch]}\n"
    )


def unlearn(
    model: nn.Module,
    num_classes: int,
    retain_dataloader: DataLoader,
    forget_dataloader: DataLoader,
    val_dataloader: DataLoader,
    retain_optimizer: Optimizer,
    forget_optimizer: Optimizer,
    unlearn_epochs: int,
    device,
    log_dir: str,
    forget_step: bool = True,
    retain_step: bool = True,
    forget_criterion: nn.Module = nn.KLDivLoss(reduction="batchmean"),
    retain_criterion: nn.Module = nn.CrossEntropyLoss(),
    verbose: bool = True,
):
    # Save initial trained model
    ckpt_dir = f"{log_dir}/ckpt"
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    torch.save(model, f"{ckpt_dir}/epoch-0.pt")

    metrics = {
        "forget_loss": [],
        "forget_acc": [],
        "retain_loss": [],
        "retain_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    # Save initial metrics
    forget_acc_initial, forget_loss_initial = evaluate(model, forget_dataloader, num_classes, device, forget=True)
    retain_acc_initial, retain_loss_initial = evaluate(model, retain_dataloader, num_classes, device)
    val_acc_initial, val_loss_initial = evaluate(model, val_dataloader, num_classes, device)

    metrics["forget_loss"].append(forget_loss_initial)
    metrics["retain_loss"].append(retain_loss_initial)
    metrics["val_loss"].append(val_loss_initial)

    metrics["forget_acc"].append(forget_acc_initial)
    metrics["retain_acc"].append(retain_acc_initial)
    metrics["val_acc"].append(val_acc_initial)

    if verbose:
        print("Metrics before unlearning - ")
        print_metrics(metrics)

    forget_acc = MulticlassAccuracy(num_classes=num_classes, average=None).to(device)
    retain_acc = MulticlassAccuracy(num_classes=num_classes, average=None).to(device)
    val_acc = MulticlassAccuracy(num_classes=num_classes, average=None).to(device)

    for epoch in range(unlearn_epochs):
        print(f"Epoch {epoch+1}")
        model.train()

        retain_acc.reset()
        val_acc.reset()

        running_forget_loss = 0
        running_retain_loss = 0
        running_val_loss = 0

        # Forget step
        if forget_step:
            for images, labels in tqdm(forget_dataloader, desc="forget"):
                images = images.to(device)
                labels = labels.to(device)

                forget_optimizer.zero_grad()
                outputs = F.log_softmax(model(images), dim=1)

                # Minimize KL divergence from uniform logits
                uniform_logits = F.softmax(torch.ones_like(outputs), dim=1)
                forget_loss = forget_criterion(outputs, uniform_logits)
                forget_loss.backward()
                running_forget_loss += forget_loss.item()
                forget_optimizer.step()
                forget_acc.update(outputs, labels)

        # Retain step
        if retain_step:
            for images, labels in tqdm(retain_dataloader, desc="retain"):
                images = images.to(device)
                labels = labels.to(device)

                retain_optimizer.zero_grad()
                outputs = model(images)
                retain_loss = retain_criterion(outputs, labels)
                retain_loss.backward()
                retain_optimizer.step()
                running_retain_loss += retain_loss.item()
                retain_acc.update(outputs, labels)

        # Validation on all classes
        model.eval()
        for images, labels in tqdm(val_dataloader, desc="val"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            val_loss = retain_criterion(outputs, labels)
            running_val_loss += val_loss.item()
            val_acc.update(outputs, labels)

        # Update metrics
        metrics["forget_loss"].append(running_forget_loss / len(forget_dataloader))
        metrics["retain_loss"].append(running_retain_loss / len(retain_dataloader))
        metrics["val_loss"].append(running_val_loss / len(val_dataloader))

        metrics["forget_acc"].append(forget_acc.compute().tolist())
        metrics["retain_acc"].append(retain_acc.compute().tolist())
        metrics["val_acc"].append(val_acc.compute().tolist())

        if verbose:
            print_metrics(metrics)

        # Save model
        torch.save(model, f"{ckpt_dir}/epoch-{epoch+1}.pt")

    # Save metrics
    print(f"Saving metrics to {log_dir}.")
    save_json(f"{log_dir}/metrics.json", metrics)