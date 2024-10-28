import lightning as L
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T

from face_classifier import FaceClassifier
from lfw_classification_dataset import LFWClassificationDataset

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
num_epochs = 20
lr = 1e-2
batch_size = 32
# pretrained = "vggface2"
pretrained = None

transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    # fixed_image_standardization,
])

dataset = LFWClassificationDataset(
    root="data",
    split="train",
    download=True,
    # min_k_per_person=70,
    people=["Ariel_Sharon", "Colin_Powell", "George_W_Bush"],
    transform=transform,
    preprocessed=True,
)
print(f"dataset.class_to_idx: {dataset.class_to_idx}")
print(f"len(dataset): {len(dataset)}")

# print(dataset.targets)
# unique, counts = np.unique(dataset.targets, return_counts=True)
# print(f"counts: {counts / sum(counts)}")

# img, label = dataset[0]
# print(img.shape)
# plt.imshow(T.ToPILImage()(img))
# plt.show(block=True)

# test_dataset = LFWClassificationDataset(
#     root="data",
#     split="test",
#     download=True,
#     transform=transform,
#     people=list(dataset.class_to_idx.keys())
# )

train_dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
)

model = FaceClassifier(
    num_classes=len(dataset.class_to_idx),
    lr=lr,
    device=device,
    pretrained=pretrained,
)

trainer = L.Trainer(max_epochs=num_epochs)
trainer.fit(model, train_dataloaders=train_dataloader)
