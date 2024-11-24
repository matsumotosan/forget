import torch
from torch import Tensor
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10

mnist_transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)

cifar10_transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)


class UnlearningDataModule:
    """DataModule class for unlearning setting."""

    def __init__(
        self,
        data_dir: str,
        forget_class: Tensor,
        batch_size: int = 32,
        transform=None,
    ) -> None:
        super().__init__()

        self.data_dir = data_dir
        self.forget_class = forget_class
        self.batch_size = batch_size
        self.transform = transform

    def prepare_data(self) -> None:
        raise NotImplementedError("Must implement `prepare_data`.")

    def setup(self) -> None:
        raise NotImplementedError("Must implement `setup`.")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.ds_train, batch_size=self.batch_size)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.ds_val, batch_size=self.batch_size)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.ds_test, batch_size=self.batch_size)

    def forget_dataloader(self) -> DataLoader:
        return DataLoader(self.ds_forget, batch_size=self.batch_size)

    def retain_dataloader(self) -> DataLoader:
        return DataLoader(self.ds_retain, batch_size=self.batch_size)


class MNISTUnlearningDataModule(UnlearningDataModule):
    """MNIST DataModule class for unlearning setting."""

    def __init__(
        self,
        data_dir: str,
        forget_class: Tensor,
        batch_size: int = 32,
        transform=None,
    ) -> None:
        if transform is None:
            transform = mnist_transform

        super().__init__(data_dir, forget_class, batch_size, transform)

    def prepare_data(self) -> None:
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self) -> None:
        self.prepare_data()
        self.ds_train = MNIST(
            root=self.data_dir, train=True, download=True, transform=self.transform
        )

        self.ds_test = MNIST(
            root=self.data_dir, train=False, download=True, transform=self.transform
        )

        self.ds_train, self.ds_val = random_split(self.ds_train, [0.9, 0.1])

        # Split training dataset into retain and forget subsets
        forget_mask = torch.isin(
            self.ds_train.dataset.targets[self.ds_train.indices], self.forget_class
        )

        self.forget_idx = forget_mask.nonzero().flatten().tolist()
        self.retain_idx = (~forget_mask).nonzero().flatten().tolist()

        self.ds_forget = Subset(self.ds_train, self.forget_idx)
        self.ds_retain = Subset(self.ds_train, self.retain_idx)
