from data_generator import CustomDataModule
from src.models.ViT import ViT
from pathlib import Path
import pytorch_lightning as pl

import hydra

from pytorch_lightning.loggers import WandbLogger
from src.utilities.modules import recursive_find_python_class


@hydra.main(config_path="conf", config_name="default_config.yaml")
def train(config):
    # init datamodule
    if config.trainer.quick_test:
        config.trainer.max_epochs = 2

    dm = CustomDataModule(config)

    # init model
    module = recursive_find_python_class(config.model.name)
    model = module(config)

    # init logger
    if config.trainer.wandb:
        logger = WandbLogger(
            project="MLOps_project", entity="mlops_team29", name="test_run"
        )
    else:
        logger = None

    # callbacks
    callbacks = []

    # model checkpoint
    if config.trainer.checkpoint_callback:
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath="checkpoints",
            filename="best-checkpoint",
            save_top_k=1,
            verbose=True,
            monitor="val_loss",
            mode="min",
            save_last = True,
        )
        callbacks.append(checkpoint_callback)

    # early stopping
    if config.trainer.early_stopping_callback:
        early_stop_callback = pl.callbacks.early_stopping.EarlyStopping(
            monitor="val_loss",
            patience=10,
            verbose=True,
            mode="min",
        )
        callbacks.append(early_stop_callback)

    trainer = pl.Trainer(
        max_epochs=config.trainer.max_epochs,
        logger=logger,
        callbacks=callbacks,
    )

    trainer.fit(model, dm)

if __name__ == "__main__":
    train()
