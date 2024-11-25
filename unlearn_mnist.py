from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from rich import print
from torch.nn.modules import KLDivLoss
from unlearn import unlearn
from unlearning_datamodule import MNISTUnlearningDataModule
from utils import evaluate, train, setup_log_dir
from models import get_model

DATASET = "mnist"

DATA_DIR = "./data"
MODEL_DIR = "./models"
FIG_DIR = "./figures"
LOG_DIR = "./logs"

BATCH_SIZE = 64
TRAIN_EPOCHS = 3
UNLEARN_EPOCHS = 3
LEARNING_RATE = 1e-3
UNLEARNING_RATE = 1e-3
RETAIN_RATE = 1e-3

FROM_SCRATCH = True
FORGET = True
RETAIN = True


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    experiment_dir = setup_log_dir(LOG_DIR, DATASET)
    forget_class = (5, 7)

    model = get_model(MODEL_DIR, DATASET, device)

    # Initialize unlearning datamodule
    dm = MNISTUnlearningDataModule(
        data_dir=DATA_DIR,
        forget_class=torch.tensor(forget_class),
    )

    dm.setup()

    # Get dataloader for each dataset split
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()
    forget_loader = dm.forget_dataloader()
    retain_loader = dm.retain_dataloader()

    # Set model paths
    trained_model_path = Path(f"{MODEL_DIR}/trained_mnist.pt")
    retrained_model_path = Path(f"{MODEL_DIR}/retrained_mnist.pt")
    criterion = nn.CrossEntropyLoss()

    # Train on original training dataset
    print("=== Standard training ===")
    if trained_model_path.exists() and not FROM_SCRATCH:
        print(f"Loading pretrained model from {trained_model_path}.")
        model.load_state_dict(torch.load(trained_model_path, weights_only=True))
    else:
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        train(model, train_loader, criterion, optimizer, TRAIN_EPOCHS, device)

        print(f"Saving trained model to {trained_model_path}.")
        torch.save(model.state_dict(), trained_model_path)

    acc_trained = evaluate(model, test_loader, 10, device)
    print(f"Trained accuracy: {acc_trained}")

    # Retrain on retain dataset (gold standard)
    print("\n=== Retrain on retain dataset (gold standard) ===")
    retrained_model = model = get_model(MODEL_DIR, DATASET, device)
    if retrained_model_path.exists() and not FROM_SCRATCH:
        print(f"Loading retrained model from {retrained_model_path}.")
        retrained_model.load_state_dict(
            torch.load(retrained_model_path, weights_only=True)
        )
    else:
        optimizer = optim.Adam(retrained_model.parameters(), lr=LEARNING_RATE)
        train(
            retrained_model, retain_loader, criterion, optimizer, TRAIN_EPOCHS, device
        )

        print(f"Saving retrained model to {retrained_model_path}.")
        torch.save(retrained_model.state_dict(), retrained_model_path)

    acc_retrained = evaluate(retrained_model, test_loader, 10, device)
    print(f"Retrained accuracy: {acc_retrained}")

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
        forget_step=FORGET,
        retain_step=RETAIN,
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
