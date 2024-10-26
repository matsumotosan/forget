import glob
import os
import random
import time
import warnings

# from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms.functional as TF
from sklearn import linear_model, model_selection
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms


def parsing(meta_data):
    image_age_list = []
    # iterate all rows in the metadata file
    for idx, row in meta_data.iterrows():
        image_path = row["image_path"]
        age_class = row["age_class"]
        image_age_list.append([image_path, age_class])
    return image_age_list


class UnlearningDataset(Dataset):
    def __init__(
        self, meta_data, image_directory, transform=None, forget=False, retain=False
    ):
        self.meta_data = meta_data
        self.image_directory = image_directory
        self.transform = transform

        # Process the metadata.
        image_age_list = parsing(meta_data)

        self.image_age_list = image_age_list
        self.age_class_to_label = {
            "a": 0,
            "b": 1,
            "c": 2,
            "d": 3,
            "e": 4,
            "f": 5,
            "g": 6,
            "h": 7,
        }

        # After training the original model, we will do "machine unlearning".
        # The machine unlearning requires two datasets, ① forget dataset and ② retain dataset.
        # In this experiment, we set the first 1,500 images to be forgotten and the rest images to be retained.
        if forget:
            self.image_age_list = self.image_age_list[:1500]
        if retain:
            self.image_age_list = self.image_age_list[1500:]

    def __len__(self):
        return len(self.image_age_list)

    def __getitem__(self, idx):
        image_path, age_class = self.image_age_list[idx]
        img = Image.open(os.path.join(self.image_directory, image_path))
        label = self.age_class_to_label[age_class]

        if self.transform:
            img = self.transform(img)

        return img, label


label_to_age = {
    0: "0-6 years old",
    1: "7-12 years old",
    2: "13-19 years old",
    3: "20-30 years old",
    4: "31-45 years old",
    5: "46-55 years old",
    6: "56-66 years old",
    7: "67-80 years old",
}

train_meta_data_path = (
    "./data/custom_korean_family_dataset_resolution_128/custom_train_dataset.csv"
)
train_meta_data = pd.read_csv(train_meta_data_path)
train_image_directory = "./custom_korean_family_dataset_resolution_128/train_images"

test_meta_data_path = (
    "./data/custom_korean_family_dataset_resolution_128/custom_val_dataset.csv"
)
test_meta_data = pd.read_csv(test_meta_data_path)
test_image_directory = "./data/custom_korean_family_dataset_resolution_128/val_images"

unseen_meta_data_path = (
    "./data/custom_korean_family_dataset_resolution_128/custom_test_dataset.csv"
)
unseen_meta_data = pd.read_csv(unseen_meta_data_path)
unseen_image_directory = (
    "./data/custom_korean_family_dataset_resolution_128/test_images"
)

train_transform = transforms.Compose(
    [
        transforms.Resize(128),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
    ]
)

test_transform = transforms.Compose([transforms.Resize(128), transforms.ToTensor()])
unseen_transform = transforms.Compose([transforms.Resize(128), transforms.ToTensor()])

train_dataset = UnlearningDataset(train_meta_data, train_image_directory, train_transform)
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = UnlearningDataset(test_meta_data, test_image_directory, test_transform)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

unseen_dataset = UnlearningDataset(unseen_meta_data, unseen_image_directory, unseen_transform)
unseen_dataloader = DataLoader(unseen_dataset, batch_size=64, shuffle=False)

model = models.resnet18(pretrained=False)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 8)
# model = model.cuda()
# model_path = f'last_checkpoint_epoch_{num_original_epochs}.pth' # If you train the original model from scratch.
model_path = "./model/last_checkpoint_epoch_30.pth"
model.load_state_dict(torch.load(model_path, map_location='cpu'))

criterion = nn.CrossEntropyLoss()
log_step = 30


def show_images(images, labels, preds=None, nrow=6, save_path=None):
    n_images = len(images)
    nrows = n_images // nrow + (n_images % nrow > 0)

    _, axs = plt.subplots(nrows, nrow, figsize=(14.5, 2.3 * nrows), frameon=False)
    axs = axs.flatten() if n_images > 1 else [axs]

    if preds:
        for idx, (img, label, pred) in enumerate(zip(images, labels, preds)):
            ax = axs[idx]
            img_np = img.numpy().transpose((1, 2, 0))
            ax.imshow(img_np)
            ax.axis('off')

            ax.text(5, 5, label, color='white', fontsize=13,  ha='left', va='top',
                    bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.1'))
            ax.text(5, 10, pred, color='red', fontsize=13,  ha='left', va='top',
                    bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.1'))
    else:
        for idx, (img, label) in enumerate(zip(images, labels)):
            ax = axs[idx]
            img_np = img.numpy().transpose((1, 2, 0))
            ax.imshow(img_np)
            ax.axis('off')

            ax.text(5, 5, label, color='white', fontsize=13,  ha='left', va='top',
                    bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.1'))

    plt.tight_layout(pad=0)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)

    plt.show(block=True)


def test():
    start_time = time.time()
    model.eval()
    total = 0
    running_loss = 0.0
    running_corrects = 0

    for i, batch in enumerate(test_dataloader):
        imgs, labels = batch
        # imgs, labels = imgs.cuda(), labels.cuda()

        with torch.no_grad():
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1) # predictions
            loss = criterion(outputs, labels)

        total += labels.shape[0]
        running_loss += loss.item()
        running_corrects += torch.sum(preds == labels.data)

        if (i == 0) or (i % log_step == log_step - 1):
            print(
                f"[Batch: {i + 1}] running test loss: {running_loss / total}, running test accuracy: {running_corrects / total}"
            )

        label_strs = [label_to_age[label.item()] for label in labels[:12]]
        pred_strs = [label_to_age[pred.item()] for pred in preds[:12]]

        if (i == 0):
            show_images(imgs[:12], label_strs, preds=pred_strs, nrow=6)

    print(f"test loss: {running_loss / total}, accuracy: {running_corrects / total}")
    print("elapsed time:", time.time() - start_time)
    return running_loss / total, (running_corrects / total).item()


test()
