import os
from argparse import ArgumentParser
import numpy as np
import torch
from sklearn import linear_model, model_selection
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Subset, DataLoader
from torchvision.datasets import CIFAR10

from models import load_resnet18
from unlearning_datamodule import CIFAR10UnlearningDataModule, cifar10_transform
from utils import per_sample_loss, cifar10_class2idx, read_json

DATA_DIR = "./data"
MODEL_DIR = "./models"
LOG_DIR = "./logs"

N_SPLITS = 10
N_BINS = 40
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
    forget_idx = [cifar10_class2idx[c] for c in params["forget_class"]]
    dm = CIFAR10UnlearningDataModule(
        data_dir=DATA_DIR,
        forget_class=forget_idx,
        batch_size=BATCH_SIZE,
    )

    dm.setup()

    # Get dataloader for each dataset split
    test_loader = dm.test_dataloader()
    forget_loader = dm.forget_dataloader()
    retain_loader = dm.retain_dataloader()

    ds_test = CIFAR10(root=DATA_DIR, download=True, train=False, transform=cifar10_transform)
    test_forget_mask = torch.isin(
        torch.tensor(ds_test.targets), torch.tensor(forget_idx)
    )

    test_forget_idx = test_forget_mask.nonzero().flatten().tolist()
    ds_test_forget = Subset(ds_test, test_forget_idx)
    test_forget_loader = DataLoader(ds_test_forget)

    # Get original trained and unlearned models
    trained_model_path = f"{args.exp_dir}/ckpt/epoch-0.pt"
    unlearned_model_path = f"{args.exp_dir}/ckpt/epoch-20.pt"

    trained_model = torch.load(trained_model_path).to(device)
    unlearned_model = torch.load(unlearned_model_path).to(device)

    # Compute losses (trained)
    forget_loss_trained = per_sample_loss(trained_model, forget_loader, device)
    retain_loss_trained = per_sample_loss(trained_model, retain_loader, device)
    # test_loss_trained = per_sample_loss(trained_model, test_loader, device)
    test_loss_trained = per_sample_loss(trained_model, test_forget_loader, device)

    # Compute losses (unlearned)
    forget_loss_unlearned  = per_sample_loss(unlearned_model, forget_loader, device)
    retain_loss_unlearned = per_sample_loss(unlearned_model, retain_loader, device)
    # test_loss_unlearned = per_sample_loss(unlearned_model, test_loader, device)
    test_loss_unlearned = per_sample_loss(unlearned_model, test_forget_loader, device)

    score = run_mia(forget_loss_unlearned, test_loss_unlearned, N_SPLITS, SEED)
    print(f"MIA accuracy: {score.mean()}")

    plot_loss_hist(
        forget_loss_unlearned,
        retain_loss_unlearned,
        test_loss_unlearned,
        title=f"Unlearned Model Sample Losses\n(MIA accuracy: {score.mean():.4f})",
        filename=f"{args.fig_dir}/mia_{params['dataset']}_unlearned.png",
        bins=N_BINS,
    )


def run_mia(in_train_loss, out_train_loss, n_splits, seed):
    # Define adversarial model
    attacker = linear_model.LogisticRegression()
    cv = model_selection.StratifiedShuffleSplit(
        n_splits=n_splits, random_state=seed,
    )
    
    n_in_train = len(in_train_loss)
    n_out_train = len(out_train_loss)

    if n_out_train < n_in_train:
        np.random.shuffle(in_train_loss)
        in_train_loss = in_train_loss[:n_out_train]
    else:
        np.random.shuffle(out_train_loss)
        out_train_loss = out_train_loss[:n_in_train]

    losses = np.concatenate((in_train_loss, out_train_loss))
    labels = np.concatenate((np.ones_like(in_train_loss), np.zeros_like(out_train_loss)))

    shuffle_idx = np.arange(len(losses))
    np.random.shuffle(shuffle_idx)
    losses = losses[shuffle_idx].reshape((-1, 1))
    labels = labels[shuffle_idx]

    score = model_selection.cross_val_score(
        attacker, losses, labels, cv=cv, scoring="accuracy"
    )

    return score


def plot_loss_hist(forget_loss, retain_loss, test_loss, title, filename, bins, histtype="bar"):
    f, ax = plt.subplots(1, 1, figsize=(12, 8))

    plt.hist(forget_loss, density=True, alpha=0.5, bins=bins, label="forget set", histtype=histtype)
    plt.hist(retain_loss, density=True, alpha=0.5, bins=bins, label="retain set", histtype=histtype)
    plt.hist(test_loss, density=True, alpha=0.5, bins=bins, label="test set (forget)", histtype=histtype)

    plt.title(title)
    plt.xlabel("Cross Entropy Loss")
    plt.ylabel("Density")
    plt.legend()
    plt.savefig(filename)
    

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--exp_dir", type=str, help="Experiment directory")
    parser.add_argument(
        "--fig_dir", type=str, help="Figure directory", default="./figs"
    )
    args = parser.parse_args()
    main(args)
