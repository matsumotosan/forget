import os
from argparse import ArgumentParser
import numpy as np
import torch
from sklearn import linear_model, model_selection
import matplotlib.pyplot as plt
import seaborn as sns

from models import load_resnet18
from unlearning_datamodule import CIFAR10UnlearningDataModule
from utils import per_sample_loss, cifar10_class2idx, read_json

DATA_DIR = "./data"
MODEL_DIR = "./models"
LOG_DIR = "./logs"

N_SPLITS = 10
SEED = 42
BATCH_SIZE = 64


def main(args):
    os.makedirs(args.fig_dir, exist_ok=True)
    sns.set_theme()
    sns.set_style("ticks")
    sns.set_context("talk")

    np.random.seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    params = read_json(f"{args.exp_dir}/params.json")

    # Initialize unlearning datamodule
    dm = CIFAR10UnlearningDataModule(
        data_dir=DATA_DIR,
        forget_class=[cifar10_class2idx[c] for c in params["forget_class"]],
        batch_size=BATCH_SIZE,
    )

    dm.setup()

    # Get dataloader for each dataset split
    test_loader = dm.test_dataloader()
    forget_loader = dm.forget_dataloader()

    # Get original trained and unlearned models
    trained_model_path = f"{args.exp_dir}/ckpt/epoch-0.pt"
    unlearned_model_path = f"{args.exp_dir}/ckpt/epoch-20.pt"

    # trained_model = load_resnet18(trained_model_path, device).to(device)
    # unlearned_model = load_resnet18(unlearned_model_path, device).to(device)
    trained_model = torch.load(trained_model_path).to(device)
    unlearned_model = torch.load(unlearned_model_path).to(device)

    # Compute losses (trained)
    forget_loss_trained = per_sample_loss(trained_model, forget_loader, device)
    test_loss_trained = per_sample_loss(trained_model, test_loader, device)

    # Compute losses (unlearned)
    forget_loss_unlearned  = per_sample_loss(unlearned_model, forget_loader, device)
    test_loss_unlearned = per_sample_loss(unlearned_model, test_loader, device)

    # plt.hist(forget_loss_trained, density=True, alpha=0.5, bins=50, label="forget (trained)")
    # plt.hist(test_loss_trained, density=True, alpha=0.5, bins=50, label="test (trained)")
    f, ax = plt.subplots(1, 1, figsize=(12, 8))
    plt.hist(forget_loss_unlearned, density=True, alpha=0.5, bins=50, label="forget (unlearned)")
    plt.hist(test_loss_unlearned, density=True, alpha=0.5, bins=50, label="test (unlearned)")

    plt.title(f"Forget and test loss histogram")
    plt.xlabel("Loss")
    plt.ylabel("Count")
    plt.legend()
    plt.savefig(f"{args.fig_dir}/mia_{params['dataset']}.png")
    
    # Define adversarial model
    attacker = linear_model.LogisticRegression()
    cv = model_selection.StratifiedShuffleSplit(
        n_splits=N_SPLITS, random_state=SEED,
    )
    
    # Test attacker on unlearned losses (forget and test)
    n_test = len(test_loss_unlearned)
    n_forget = len(forget_loss_unlearned)

    if n_test < n_forget:
        np.random.shuffle(forget_loss_unlearned)
        forget_loss_unlearned = forget_loss_unlearned[:n_test]
    else:
        np.random.shuffle(test_loss_unlearned)
        test_loss_unlearned = test_loss_unlearned[:n_forget]

    losses = np.concatenate((forget_loss_unlearned, test_loss_unlearned))
    labels = np.concatenate((np.ones_like(forget_loss_unlearned), np.zeros_like(test_loss_unlearned)))

    shuffle_idx = np.arange(len(losses))
    np.random.shuffle(shuffle_idx)
    losses = losses[shuffle_idx].reshape((-1, 1))
    labels = labels[shuffle_idx]

    score = model_selection.cross_val_score(
        attacker, losses, labels, cv=cv, scoring="accuracy"
    )

    print(f"MIA accuracy: {score.mean()}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--exp_dir", type=str, help="Experiment directory")
    parser.add_argument(
        "--fig_dir", type=str, help="Figure directory", default="./figs"
    )
    args = parser.parse_args()
    main(args)
