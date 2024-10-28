"""Preprocess images with MTCNN to avoid having to perform face detection during training/validation/testing."""
import os
import glob
from facenet_pytorch import MTCNN
from tqdm.contrib import tenumerate
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torch


def collate_fn(x):
    return x[0]

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

data_dir = "./data/lfw-py/lfw_funneled"
output_dir = "./data/lfw-py/lfw_funneled_preprocessed-mtcnn"

img_paths = glob.glob(f"{data_dir}/*/*.jpg")

dataset = ImageFolder(root=data_dir)
dataloader = DataLoader(dataset, collate_fn=collate_fn)

idx_to_class = {i:c for c, i in dataset.class_to_idx.items()}
idx_to_count = {i: 0 for i in dataset.class_to_idx.values()}

mtcnn = MTCNN(device=device)
thresh = 0.8

for i, (x, y) in tenumerate(dataloader):
    x_detected, prob = mtcnn(x, return_prob=True)
    idx_to_count[y] += 1
    if x_detected is not None:
        path = f"{output_dir}/{idx_to_class[y]}"
        os.makedirs(path, exist_ok=True)
        img = T.ToPILImage()(x_detected)
        img.save(f"{path}/{idx_to_class[y]}_{idx_to_count[y]:04d}.jpg")
    else:
        print(f"{idx_to_class[y]} ({idx_to_count[y]}): not detected")
