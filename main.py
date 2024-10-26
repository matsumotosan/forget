from torchvision.datasets import LFWPeople


def main():
    # Get dataset
    train_dataset = LFWPeople(root="data", split="train", download=True)

    print("Dataset size:", len(train_dataset))
    print("Dataset labels:", len(train_dataset.names))

    # Get a sample
    print(len(train_dataset.targets))


if __name__ == "__main__":
    main()
