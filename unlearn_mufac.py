from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from rich import print
from torch.nn.modules import KLDivLoss
from unlearn import unlearn
from unlearning_datamodule import MUFACUnlearningDataModule
from utils import evaluate, setup_log_dir, save_json, mufac_class2idx
from torchvision.models import resnet18

DATASET = "mufac"

DATA_DIR = "./data"
MODEL_DIR = "./models"
LOG_DIR = "./logs"

# forget classes (one or more of the following)
# "0-6 years old"
# "7-12 years old"
# "13-19 years old"
# "20-30 years old"
# "31-45 years old"
# "46-55 years old"
# "56-66 years old"
# "67-80 years old"
FORGET_CLASS = ("0-6 years old",)

BATCH_SIZE = 64
TRAIN_EPOCHS = 3
UNLEARN_EPOCHS = 20
LEARNING_RATE = 1e-3
UNLEARNING_RATE = 1e-3
RETAIN_RATE = 1e-3

FROM_SCRATCH = False
FORGET = True
RETAIN = True


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    experiment_dir = setup_log_dir(LOG_DIR, DATASET)

    # Save experiment settings
    params = {
        "dataset": DATASET,
        "forget_class": FORGET_CLASS,
        "batch_size": BATCH_SIZE,
        "train_epochs": TRAIN_EPOCHS,
        "unlearn_epochs": UNLEARN_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "unlearning_rate": UNLEARNING_RATE,
        "retain_rate": RETAIN_RATE,
        "forget_step": FORGET,
        "retain_step": RETAIN,
    }
    save_json(f"{experiment_dir}/params.json", params)

    dm = MUFACUnlearningDataModule(
        data_dir=DATA_DIR,
        forget_class=[mufac_class2idx[c] for c in FORGET_CLASS],
    )

    dm.setup()

    # Get dataloader for each dataset split
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()
    forget_loader = dm.forget_dataloader()
    retain_loader = dm.retain_dataloader()

    # Set model paths
    trained_model_path = f"{MODEL_DIR}/pretrained_mufac.pt"
    retrained_model_path = Path(f"{MODEL_DIR}/retrained_mufac.pt")
    criterion = nn.CrossEntropyLoss()

    # Train on original training dataset
    print("=== Standard training ===")
    model = resnet18(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 8)
    model.load_state_dict(torch.load(trained_model_path))
    model.to(device)

    acc_trained, loss_trained = evaluate(model, test_loader, 8, device)
    print(f"Trained accuracy: {acc_trained}")

    # Unlearning with KL divergence loss (with retain step)
    print("\n=== Finetune with KLDiv loss ===")
    forget_optimizer = optim.SGD(model.parameters(), lr=UNLEARNING_RATE)
    retain_optimizer = optim.SGD(model.parameters(), lr=RETAIN_RATE)
    forget_criterion = KLDivLoss(reduction="batchmean")

    if not FORGET and not RETAIN:
        raise ValueError("At least one of FORGET or RETAIN must be True.")

    unlearn(
        model=model,
        num_classes=8,
        retain_dataloader=retain_loader,
        forget_dataloader=forget_loader,
        val_dataloader=val_loader,
        retain_optimizer=retain_optimizer,
        forget_optimizer=forget_optimizer,
        unlearn_epochs=UNLEARN_EPOCHS,
        device=device,
        log_dir=experiment_dir,
        forget_criterion=forget_criterion,
        forget_step=FORGET,
        retain_step=RETAIN,
    )

    acc_unlearned, loss_unlearned = evaluate(model, test_loader, 8, device)

    # Print final metrics
    base_msg = "Unlearned accuracy"
    if FORGET and not RETAIN:
        setting = "forget"
    elif RETAIN and not FORGET:
        setting = "retain"
    else:
        setting = "forget and retain"

    print(f"{base_msg} ({setting}): {acc_unlearned}")
    acc_delta = [x - y for x, y in zip(acc_unlearned, acc_trained)]
    print(
        f"Change in accuracy (acc_unlearned - acc_trained): {acc_delta}"
    )


if __name__ == "__main__":
    main()
