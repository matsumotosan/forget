import os
from argparse import ArgumentParser
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from utils import read_json, cifar10_idx2class


def main(args):
    os.makedirs(args.fig_dir, exist_ok=True)
    sns.set_theme()
    sns.set_style("ticks")
    sns.set_context("talk")

    params = read_json(f"{args.exp_dir}/params.json")
    metrics = read_json(f"{args.exp_dir}/metrics.json")

    if params["dataset"] == "cifar10":
        idx2class = cifar10_idx2class

    data = []
    for epoch_idx, accuracies in enumerate(metrics["val_acc"]):
        for class_idx, accuracy in enumerate(accuracies):
            data.append(
                {
                    "Unlearning Epoch": epoch_idx,
                    "Class": f"{idx2class[class_idx]}",
                    "Accuracy": accuracy,
                }
            )

    df = pd.DataFrame(data)
    if params["forget_step"] and not params["retain_step"]:
        setting = "forget"
    elif params["retain_step"] and not params["forget_step"]:
        setting = "retain"
    elif params["retain_step"] and params["forget_step"]:
        setting = "forget+retain"
    else:
        raise ValueError("Cannot have neither.")

    f, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.axhline(0.1, color="grey", ls="--")
    sns.lineplot(
        data=df, x="Unlearning Epoch", y="Accuracy", hue="Class", marker="o", ax=ax
    )

    ax.set_title(
        f"{params['dataset'].upper()} Per-Class Validation Accuracy\n({setting} {params['forget_class']})"
    )
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylim([-0.001, 1.05])
    ax.set_ylabel("Accuracy")
    ax.grid()

    # sns.lineplot(x=np.arange(len(metrics["forget_loss"])), y=metrics["forget_loss"], ax=ax[1])
    # sns.lineplot(x=np.arange(len(metrics["forget_loss"])), y=metrics["retain_loss"], ax=ax[1])
    # sns.lineplot(x=np.arange(len(metrics["forget_loss"])), y=metrics["val_loss"], ax=ax[1])
    # ax[1].set_title(f"{params['dataset'].upper()}\nLoss ({setting} {params['forget_class']})")
    # ax[1].set_ylabel("Cross Entropy Loss")
    # ax[1].grid()

    f.tight_layout()

    # plt.show()

    filename = f"{args.fig_dir}/unlearn_{params['dataset']}_{setting}_{os.path.basename(os.path.normpath(args.exp_dir))}.png"
    plt.savefig(filename, dpi=300)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--exp_dir", type=str, help="Experiment directory")
    parser.add_argument(
        "--fig_dir", type=str, help="Figure directory", default="./figs"
    )
    args = parser.parse_args()
    main(args)
