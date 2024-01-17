from transformers import ViTForImageClassification, ViTConfig
import pytorch_lightning as pl
import torch
from pathlib import Path
from torch import nn, optim


class ViT(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        print(config)

        # Load pretrained model
        self.ViT = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k")

        # make conv2d layer converting 1 channel to 3 channels
        self.head = nn.Conv2d(
            config.model["in_channels"],
            3,
            kernel_size=(16, 16),
            stride=1,
            padding="same",
        )

        # Replace the head of the model to match the number of classes
        self.ViT.classifier = nn.Linear(768, config.model["num_classes"])

        # make logSoftmax as output activation
        self.tail = nn.LogSoftmax(dim=1)

        self.criterion = getattr(nn, config.trainer["loss"])()

        self.optimizer = getattr(optim, config.trainer["optimizer"])

    def forward(self, x):
        x = self.head(x)
        x = self.ViT(x).logits
        x = self.tail(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        # Compute accuracy
        preds = torch.argmax(logits, dim=1)
        correct_count = torch.sum(preds == y)
        accuracy = correct_count.float() / y.size(0)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_accuracy", accuracy, prog_bar=True)
        return {"val_loss": loss, "val_accuracy": accuracy}

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr=self.config.trainer["lr"])


if __name__ == "__main__":
    # set seed
    pl.seed_everything(42)
    DATA_PATH = Path(__file__).resolve().parents[2].joinpath("data/processed/train.pt")
    data = torch.load(DATA_PATH)
    image = data[0][0].unsqueeze(0)

    hparams = {
        "num_classes": 4,
        "in_channels": 1,
        "loss": "NLLLoss",
        "optimizer": "Adam",
        "lr": 1e-4,
    }
    model = ViT(hparams)
    ps_logits = model(image)
    ps = torch.exp(ps_logits)
    print(ps)
