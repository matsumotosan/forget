import os
import requests
import torch
import torch.nn as nn
from torchvision.models import resnet18


class SimpleNet(nn.Module):
    """Basic MLP for MNIST."""
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def load_resnet18(weights_path, device):
    """Load ResNet18 model for CIFAR10."""
    if not os.path.exists(weights_path):
        response = requests.get(
            "https://storage.googleapis.com/unlearning-challenge/weights_resnet18_cifar10.pth"
        )
        open(weights_path, "wb").write(response.content)

    weights = torch.load(weights_path, map_location=device)
    model = resnet18(num_classes=10)
    model.load_state_dict(weights)
    return model


def get_model(model_dir, dataset, device):
    if dataset == "mnist":
        model = SimpleNet().to(device)
    elif dataset == "cifar10":
        model = load_resnet18(f"{model_dir}/weights_resnet18_cifar10.pt", device)
    else:
        raise ValueError(f"Dataset {dataset} not supported.")
    return model
