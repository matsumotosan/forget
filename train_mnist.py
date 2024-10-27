from lightning import Trainer
from mnist_classifier import MNISTClassifier
from mnist_datamodule import MNISTDataModule

data_dir = "./data"

model = MNISTClassifier()
datamodule = MNISTDataModule(data_dir)
trainer = Trainer(max_epochs=3)
trainer.fit(model, datamodule)
