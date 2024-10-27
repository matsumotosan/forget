import os
from PIL import Image
from facenet_pytorch import MTCNN
from typing import Any, Callable, Optional
from torchvision.datasets.lfw import _LFW


class LFWClassificationDataset(_LFW):
    """Modified LFW Dataset containing people with a minimum of k images for classifiacation."""
    preprocessed_dir: str = "lfw_funneled_preprocessed-mtcnn"

    def __init__(
        self,
        root: str,
        split: str = "train",
        image_set: str = "funneled",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        min_k_per_person: Optional[int] = None,
        people: Optional[list[str]] = None,
        preprocessed: bool = False,
    ) -> None:
        super().__init__(
            root, split, image_set, "people", transform, target_transform, download
        )

        if min_k_per_person and people:
            raise ValueError("Can only set either min_k_per_person or people")
        if min_k_per_person is None and people is None:
            raise ValueError("Must set either min_k_per_person or people")

        if preprocessed:
            self.images_dir = os.path.join(self.root, self.preprocessed_dir)

        self.min_k_per_person = min_k_per_person
        self.people = people
        self.class_to_idx = self._get_classes()
        self.data, self.targets = self._get_people()

    def _get_people(self) -> tuple[list[str], list[int]]:
        data, targets = [], []
        with open(os.path.join(self.root, self.labels_file)) as f:
            lines = f.readlines()[1:]   # ignore first line (number of samples in set - needed for crossfold val)
            people = [line.strip().split("\t") for line in lines]
            for _, (identity, num_imgs) in enumerate(people):
                if identity in self.class_to_idx:   # include in dataset if has more than k images
                    for num in range(1, int(num_imgs) + 1):
                        img = self._get_path(identity, num)
                        data.append(img)
                        targets.append(self.class_to_idx[identity])

        return data, targets

    def _get_classes(self) -> dict[str, int]:
        """
        Returns:
            dict: dict[name, idx] where idx is the index of the person in the dataset
        """
        with open(os.path.join(self.root, self.names)) as f:
            lines = f.readlines()
            if self.min_k_per_person is not None:
                names = [
                    line.strip().split()[0]
                    for line in lines
                    if int(line.strip().split()[1]) >= self.min_k_per_person
                ]
            elif self.people is not None:
                names = [
                    line.strip().split()[0]
                    for line in lines
                    if line.strip().split()[0] in self.people
                ]
            else:
                raise ValueError("Must set either min_k_per_person or people")
        class_to_idx = {name: idx for idx, name in enumerate(names)}
        return class_to_idx

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target) where target is the identity of the person.
        """
        img = self._loader(self.data[index])
        target = self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def extra_repr(self) -> str:
        return super().extra_repr() + f"\nClasses (identities): {len(self.class_to_idx)}"
