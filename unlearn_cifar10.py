from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from rich import print
from torch.nn.modules import KLDivLoss
from torchvision.datasets.cifar import CIFAR10
from unlearn import unlearn
from unlearning_datamodule import CIFAR10UnlearningDataModule
from utils import evaluate, train, setup_log_dir
from models import load_resnet18

DATASET = "cifar10"

DATA_DIR = "./data"
MODEL_DIR = "./models"
LOG_DIR = "./logs"

BATCH_SIZE = 16
TRAIN_EPOCHS = 3
UNLEARN_EPOCHS = 3
LEARNING_RATE = 1e-3
UNLEARNING_RATE = 1e-3
RETAIN_RATE = 1e-3

FROM_SCRATCH = False
FORGET = True
RETAIN = True


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    experiment_dir = setup_log_dir(LOG_DIR, DATASET)

    # Choose from ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    forget_class = ("airplane", "ship")
    ds = CIFAR10(root=DATA_DIR, download=True)

    # Initialize unlearning datamodule
    dm = CIFAR10UnlearningDataModule(
        data_dir=DATA_DIR,
        forget_class=[ds.class_to_idx[c] for c in forget_class],
        batch_size=BATCH_SIZE,
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
    model = load_resnet18(trained_model_path, device).to(device)
    acc_trained = evaluate(model, test_loader, 10, device)
    print(f"Trained accuracy: {acc_trained}")

    # Unlearning with KL divergence loss (with retain step)
    print("\n=== Finetune with KLDiv loss (with retain step) ===")
    unlearned_model = model

    unlearned_initial_acc = evaluate(unlearned_model, test_loader, 10, device)
    print(f"Trained accuracy (starting point for unlearning): {unlearned_initial_acc}")

    forget_optimizer = optim.Adam(unlearned_model.parameters(), lr=UNLEARNING_RATE)
    retain_optimizer = optim.Adam(unlearned_model.parameters(), lr=RETAIN_RATE)
    forget_criterion = KLDivLoss(reduction="batchmean")

    if not FORGET or not RETAIN:
        raise ValueError("At least one of FORGET or RETAIN must be True.")

    unlearn(
        model=unlearned_model,
        retain_dataloader=retain_loader,
        forget_dataloader=forget_loader,
        val_dataloader=val_loader,
        retain_optimizer=retain_optimizer,
        forget_optimizer=forget_optimizer,
        unlearn_epochs=UNLEARN_EPOCHS,
        device=device,
        log_dir=experiment_dir,
        forget_criterion=forget_criterion,
        forget_step=True,
        retain_step=True,
    )

    acc_unlearned = evaluate(unlearned_model, test_loader, 10, device)
    base_msg = "Unlearned accuracy"
    if FORGET and not RETAIN:
        setting = "forget"
    if RETAIN and not FORGET:
        setting = "retain"
    else:
        setting = "forget and retain"

    print(f"{base_msg} ({setting}): {acc_unlearned}")
    print(
        f"Change in accuracy (acc_unlearned - acc_trained): {acc_unlearned - acc_trained}"
    )


if __name__ == "__main__":
    main()
