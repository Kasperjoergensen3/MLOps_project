from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import torch
from pathlib import Path


class CustomDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.batch_size = config.trainer["batch_size"]
        self.data_path = Path(config.trainer["data_path"])

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = torch.load(self.data_path / "train.pt")
            self.val_dataset = torch.load(self.data_path / "valid.pt")

        if stage == "test" or stage is None:
            self.test_dataset = torch.load(self.data_path / "test.pt")

    def train_dataloader(self):
        # Return the dataloader for training
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        # Return the dataloader for validation
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        # Return the dataloader for testing
        return DataLoader(self.test_dataset, batch_size=self.batch_size)


if __name__ == "__main__":
    # Load the config file
    config = {
        "data_path": Path(__file__).resolve().parents[2].joinpath("data/processed"),
        "batch_size": 32,
    }

    CustomDataModule(config)
