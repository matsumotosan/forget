from argparse import ArgumentParser
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils import read_json


def main(args):
    metrics = read_json(f"{args.exp_dir}/metrics.json")
    params = read_json(f"{args.exp_dir}/params.json")

    data = []
    for epoch_idx, accuracies in enumerate(metrics["val_acc"]):
        for class_idx, accuracy in enumerate(accuracies):
            data.append({'Epoch': epoch_idx, 'Class': f'{class_idx}', 'Accuracy': accuracy})

    df = pd.DataFrame(data)
    if params["forget_step"] and not params["retain_step"]:
        setting = "forget"
    if params["retain_step"] and not params["forget_step"]:
        setting = "retain"
    else:
        setting = "forget+retain"

    fig, ax = plt.subplots(figsize=(15, 8))
    sns.lineplot(data=df, x='Epoch', y='Accuracy', hue='Class', marker='o')

    plt.title(f"MNIST Per-Class Accuracy over Unlearning Epochs ({setting})")
    plt.xticks(df["Epoch"])
    plt.ylabel("Accuracy")
    # plt.tight_layout()
    plt.grid()
    plt.show()
    # plt.savefig(f"{args.fig_dir}/unlearn_mnist_{setting}.png", dpi=300)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--exp_dir", type=str, help="Experiment directory")
    parser.add_argument("--fig_dir", type=str, help="Figure directory", default="./figures")
    args = parser.parse_args()
    main(args)
