import torch

from models import load_resnet18
from unlearning_datamodule import CIFAR10UnlearningDataModule
from utils import evaluate
from sklearn import linear_model

DATASET = "cifar10"

DATA_DIR = "./data"
MODEL_DIR = "./models"
LOG_DIR = "./logs"

FORGET_CLASS = ("airplane", "ship")
BATCH_SIZE = 64
TRAIN_EPOCHS = 3
UNLEARN_EPOCHS = 20
LEARNING_RATE = 1e-3
UNLEARNING_RATE = 1e-3
RETAIN_RATE = 1e-3

FROM_SCRATCH = False
FORGET = True
RETAIN = False

EXPERIMENT_DIR = ""


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize unlearning datamodule
    dm = CIFAR10UnlearningDataModule(
        data_dir=DATA_DIR,
        forget_class=[ds.class_to_idx[c] for c in FORGET_CLASS],
        batch_size=BATCH_SIZE,
    )

    dm.setup()

    # Get dataloader for each dataset split
    val_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()
    forget_loader = dm.forget_dataloader()

    # Get original trained and unlearned models
    trained_model_path = f"{MODEL_DIR}/weights_resnet18_cifar10.pt"
    unlearned_model_path = f"{MODEL_DIR}/weights_resnet18_cifar10.pt"

    # Define adversarial model
    attacker = linear_model.LogisticRegression()

    # Compute loss
    forget_loss_trained = evaluate(trained_model, forget_loader, 10, device)
    forget_loss_unlearned = evaluate(unlearned_model, forget_loader, 10, device)
    
    # Since we have more forget losses than test losses, sub-sample them, to have a class-balanced dataset.
    np.random.shuffle(forget_losses)
    forget_losses = forget_losses[: len(test_losses)]

    samples_mia = np.concatenate((test_losses, forget_losses)).reshape((-1, 1))
    labels_mia = [0] * len(test_losses) + [1] * len(forget_losses)


if __name__ == "__main__":
    main()
