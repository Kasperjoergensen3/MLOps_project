import hydra
import wandb
from src.train import train

@hydra.main(config_path="conf/sweep", config_name="my-awesome-sweep.yaml")
def sweep(config):
    # Here, 'config' will be populated based on the chosen sweep configuration file

    sweep_id = wandb.sweep(dict(config), entity="mlops_team29", project="MLOps_project")

    # Run the sweep
    wandb.agent(sweep_id, function=lambda: train(config))

if __name__ == "__main__":
    sweep()
