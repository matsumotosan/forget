import numpy as np
import torch
from sklearn import linear_model, model_selection
import matplotlib.pyplot as plt

from models import load_resnet18
from unlearning_datamodule import CIFAR10UnlearningDataModule
from utils import evaluate, cifar_idx2class, read_json

DATASET = "cifar10"

DATA_DIR = "./data"
MODEL_DIR = "./models"
LOG_DIR = "./logs"

N_SPLITS = 10
RANDOM_STATE = 42
BATCH_SIZE = 64
FORGET_CLASS = ("airplane", "ship")

EXPERIMENT_DIR = ""


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    params = read_json(f"{EXPERIMENT_DIR}/params.json")

    # Initialize unlearning datamodule
    dm = CIFAR10UnlearningDataModule(
        data_dir=DATA_DIR,
        forget_class=[cifar_idx2class[c] for c in params["forget_class"]],
        batch_size=BATCH_SIZE,
    )

    dm.setup()

    # Get dataloader for each dataset split
    test_loader = dm.test_dataloader()
    forget_loader = dm.forget_dataloader()

    # Get original trained and unlearned models
    trained_model_path = f"{MODEL_DIR}/weights_resnet18_cifar10.pt"
    unlearned_model_path = f"{MODEL_DIR}/weights_resnet18_cifar10.pt"

    trained_model = load_resnet18(trained_model_path, device).to(device)
    unlearned_model = load_resnet18(unlearned_model_path, device).to(device)

    # Compute losses
    forget_loss_trained, _ = evaluate(trained_model, forget_loader, 10, device)
    forget_loss_unlearned, _ = evaluate(unlearned_model, forget_loader, 10, device)
    test_loss_trained, _ = evaluate(trained_model, test_loader, 10, device)
    test_loss_unlearned, _ = evaluate(unlearned_model, test_loader, 10, device)

    plt.hist(forget_loss_trained, density=True, alpha=0.5, bins=50, label="Test set")
    plt.hist(forget_loss_unlearned, density=True, alpha=0.5, bins=50, label="Train set")
    
    # # Define adversarial model
    # attacker = linear_model.LogisticRegression()
    # cv = model_selection.StratifiedShuffleSplit(
    #     n_splits=N_SPLITS, random_state=RANDOM_STATE
    # )
    #
    # score = model_selection.cross_val_score(
    #     attacker, sample_loss, members, cv=cv, scoring="accuracy"
    # )
    #
    # np.random.shuffle(forget_losses)
    # forget_losses = forget_losses[: len(test_losses)]
    #
    # samples_mia = np.concatenate((test_losses, forget_losses)).reshape((-1, 1))
    # labels_mia = [0] * len(test_losses) + [1] * len(forget_losses)


if __name__ == "__main__":
    main()
