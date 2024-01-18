import os

from data_generator import CustomDataModule
from src.models.ViT import ViT
from pathlib import Path
import pytorch_lightning as pl
import hydra
import wandb
from omegaconf import OmegaConf
from pytorch_lightning.loggers import WandbLogger

from src.utilities.modules import recursive_find_python_class
from callbacks.plotting_callback import PlotLogger

@hydra.main(config_path="conf", config_name="default_config.yaml")
def train(config):
    # Check if the script is run as part of a W&B sweep
    if 'WANDB_SWEEP_ID' in os.environ:
        wandb.init()
        sweep_config = wandb.config
        # Manually update the Hydra config with the W&B parameters
        for key, val in sweep_config.items():
            if hasattr(config.trainer, key):
                config.trainer[key] = sweep_config[key]
            elif hasattr(config.model, key):
                config.model[key] = sweep_config[key]

    # init datamodule
    if config.trainer.quick_test:
        config.trainer.max_epochs = 5

    dm = CustomDataModule(config)

    # init model
    module = recursive_find_python_class(config.model.name)
    model = module(config)

    # init logger
    if config.trainer.wandb:
        run_name = f"{config.model.name}_{config.trainer.optimizer}_{config.trainer.lr}_loss_{config.trainer.loss}"
        logger = WandbLogger(project="mlops-project", entity="mlops_team29", name=run_name)
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
            save_last=True,
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

    # plotting
    # if config.trainer.plotting_callback:
    #     plot_callback = PlotLogger(model, dm, config=config)
    #     callbacks.append(plot_callback)

    trainer = pl.Trainer(
        max_epochs=config.trainer.max_epochs,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=10,
    )

    trainer.fit(model, dm)


if __name__ == "__main__":
    train()
