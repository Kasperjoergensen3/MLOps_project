import torch
from torch import nn, optim
import pytorch_lightning as pl
from pathlib import Path

class SimpleCNN(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        print(config)
        print(config.model["in_channels"])
        # Define a simple CNN
        self.model = nn.Sequential(
            nn.Conv2d(config.model['in_channels'], 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(64 * 56 * 56, 128), 
            nn.ReLU(),
            nn.Linear(128, config.model['num_classes']),
            nn.LogSoftmax(dim=1)  
        )

        self.criterion = getattr(nn, config.trainer['loss'])()
        self.optimizer = getattr(optim, config.trainer['optimizer'])

    def forward(self, x):
        return self.model(x)

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
        preds = torch.argmax(logits, dim=1)  # Get the predicted class
        correct_count = torch.sum(preds == y)  # Count how many predictions were correct
        accuracy = correct_count.float() / y.size(0)  # Calculate accuracy

        # Log loss and accuracy
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_accuracy", accuracy, prog_bar=True)

        return {"val_loss": loss, "val_accuracy": accuracy}


    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr=self.config.trainer["lr"])

if __name__ == "__main__":
    pl.seed_everything(42)

    hparams = {
        "num_classes": 4,
        "in_channels": 1,
        "loss": "NLLLoss",
        "optimizer": "Adam",
        "lr": 1e-4,
    }

    model = SimpleCNN(hparams)
