import torch
import argparse
from pathlib import Path
import pytorch_lightning as pl
from omegaconf import OmegaConf
from src.utilities.modules import recursive_find_python_class
from src.data_generator import CustomDataModule
from tqdm import tqdm


def load_model(config: OmegaConf, checkpoint_path: Path) -> torch.nn.Module:
    """
    Load model from checkpoint.

    Args:
        config: config object
        checkpoint_path: path to checkpoint

    Returns:
        model
    """

    module = recursive_find_python_class(config.model.name)
    model = module(config)
    state_dict = torch.load(checkpoint_path)["state_dict"]
    model.load_state_dict(state_dict)
    return model


def predict(model: torch.nn.Module, dataloader: pl.LightningDataModule) -> None:
    """
    Run inference on test set.

    Args:
        model: model
        dataloader: dataloader
    """
    equals = torch.zeros(0, dtype=torch.long)
    for batch in tqdm(dataloader):
        x, y = batch
        logits = model(x)
        ps = torch.exp(logits)
        top_p, top_class = ps.topk(1, dim=1)
        equals = torch.cat((equals, top_class == y.view(*top_class.shape)))
    accuracy = torch.mean(equals.type(torch.FloatTensor))
    print(f"Accuracy: {accuracy.item()*100}%")


if __name__ == "__main__":
    # make agument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="models/test_predict/")
    parser.add_argument("--checkpoint", type=str, default="best-checkpoint.ckpt")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    config_path = output_dir.joinpath(".hydra", "config.yaml")
    checkpoint_path = output_dir.joinpath("checkpoints", args.checkpoint)
    config = OmegaConf.load(config_path)

    # load model
    model = load_model(config, checkpoint_path)
    model.eval()

    # load data
    config.trainer.data_path = "data/processed"
    config.trainer.quick_test = False
    dm = CustomDataModule(config)
    dm.setup("test")

    # run inference
    predict(model, dm.test_dataloader())
