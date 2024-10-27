import lightning as L
import torch
import torch.optim as optim
from facenet_pytorch import InceptionResnetV1
from torchmetrics.classification import Accuracy
from typing import Optional


class FaceClassifier(L.LightningModule):
    def __init__(
        self, num_classes: int, lr: float, device, pretrained: Optional[str] = None
    ):
        super().__init__()

        self.model = InceptionResnetV1(
            classify=True,
            pretrained=pretrained,
            num_classes=num_classes,
        ).to(device)

        self.lr = lr
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes, top_k=1)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes, top_k=1)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes, top_k=1)
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        probs = logits.softmax(dim=1)

        loss = self.loss_fn(logits, y)
        self.train_acc.update(probs, y)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "train_acc",
            self.train_acc.compute(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        probs = logits.softmax(dim=1)

        loss = self.loss_fn(logits, y)
        self.val_acc.update(probs, y)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "val_acc",
            self.val_acc.compute(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        probs = logits.softmax(dim=1)

        self.test_acc.update(probs, y)

        self.log(
            "test_acc",
            self.test_acc.compute(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)
